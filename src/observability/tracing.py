"""
OpenTelemetry distributed tracing for DataShield.
Exports traces to Jaeger via OTLP.

Usage:
    from observability.tracing import setup_tracing, get_tracer
    setup_tracing("datashield")
    tracer = get_tracer()

    with tracer.start_as_current_span("quality_detection") as span:
        span.set_attribute("table_name", table_name)
        span.set_attribute("row_count", len(df))
        result = detector.detect(df)
        span.set_attribute("alerts_found", len(result))
"""

import os
import functools
import contextlib
from typing import Dict, Optional, Callable, Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False

# Module-level tracer provider reference so it can be reused
_tracer_provider: Optional[TracerProvider] = None


def setup_tracing(
    service_name: str,
    otlp_endpoint: str = "http://localhost:4317",
) -> None:
    """
    Initialize the OpenTelemetry TracerProvider for DataShield.

    Attempts to export spans to Jaeger via OTLP gRPC.  Falls back to the
    console exporter if the OTLP package is not installed or the endpoint
    is unreachable (ConnectionError).

    Resource attributes set:
        service.name           – passed-in service_name
        service.version        – "0.3.0"
        deployment.environment – value of env var DEPLOYMENT_ENV (default "development")

    Args:
        service_name:   Logical name of this service, e.g. "datashield".
        otlp_endpoint:  OTLP gRPC collector endpoint, e.g. "http://localhost:4317".
    """
    global _tracer_provider

    deployment_env = os.environ.get("DEPLOYMENT_ENV", "development")

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.3.0",
            "deployment.environment": deployment_env,
        }
    )

    provider = TracerProvider(resource=resource)

    # Attempt OTLP export; fall back to console on failure.
    exporter = _build_exporter(otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _tracer_provider = provider


def _build_exporter(otlp_endpoint: str):
    """
    Build the best available span exporter.

    Returns an OTLPSpanExporter pointed at *otlp_endpoint* when the
    opentelemetry-exporter-otlp-proto-grpc package is present; otherwise
    falls back to ConsoleSpanExporter.  A ConnectionError during OTLP
    exporter construction also triggers the console fallback.
    """
    if not OTLP_AVAILABLE:
        return ConsoleSpanExporter()

    try:
        return OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    except ConnectionError:
        return ConsoleSpanExporter()


def get_tracer(name: str = "datashield") -> trace.Tracer:
    """
    Return a named Tracer from the current global TracerProvider.

    Call setup_tracing() at least once before using the returned tracer so
    that spans are exported to the configured backend.

    Args:
        name: Instrumentation scope name (default "datashield").

    Returns:
        An opentelemetry Tracer instance.
    """
    return trace.get_tracer(name)


@contextlib.contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context-manager / decorator that wraps a block (or function) in an
    OpenTelemetry span.

    As a context manager::

        with trace_span("my_operation", {"key": "value"}):
            do_work()

    As a decorator::

        @trace_span("my_function")
        def my_function():
            ...

    Args:
        name:       Name of the span.
        attributes: Optional key/value attributes to attach to the span.

    Yields:
        The active opentelemetry Span object.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


def _trace_span_decorator(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Return a decorator that wraps the decorated function in a span.

    This overload allows trace_span to double as both a context manager and
    a plain function decorator::

        @trace_span("quality_check")
        def run_quality_check(df):
            ...

    Args:
        name:       Span name.
        attributes: Optional attributes applied to every invocation.

    Returns:
        A decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with trace_span(name, attributes):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Patch trace_span so it can also be used as a plain decorator.
# When called with a callable as the first positional argument (i.e. used
# without parentheses), it wraps that callable directly.
_original_trace_span = trace_span


class _TraceSpanHelper:
    """
    Makes trace_span usable as both a context manager and a plain decorator.

    Usage as context manager (primary)::

        with trace_span("op", {"k": "v"}) as span:
            ...

    Usage as decorator factory::

        @trace_span("op")
        def my_fn():
            ...
    """

    def __call__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        # If ``name`` is actually a callable the user wrote @trace_span without
        # parentheses; wrap it with a default span name.
        if callable(name):
            func = name
            return _trace_span_decorator(func.__name__)(func)
        # Normal usage: return either a context manager or a decorator.
        return _ContextManagerOrDecorator(name, attributes)

    # Allow ``with trace_span("x"):`` after ``trace_span = _TraceSpanHelper()``.
    # This isn't needed for the helper itself but keeps type checkers happy.


class _ContextManagerOrDecorator:
    """
    Returned by trace_span(name, attrs).  Can be used as a context manager
    (``with ...``) **or** called as a decorator (``@trace_span("x")``).
    """

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]]):
        self._name = name
        self._attributes = attributes
        self._cm = None

    # --- context-manager protocol ---

    def __enter__(self):
        self._cm = _original_trace_span(self._name, self._attributes)
        return self._cm.__enter__()

    def __exit__(self, *exc_info):
        return self._cm.__exit__(*exc_info)

    # --- decorator protocol ---

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _original_trace_span(self._name, self._attributes):
                return func(*args, **kwargs)
        return wrapper


trace_span = _TraceSpanHelper()


def record_anomaly_metric(
    table_name: str,
    anomaly_count: int,
    detection_time_ms: float,
) -> None:
    """
    Record an anomaly detection result as a span event on the current span.

    If there is no active span the call is a no-op.

    Args:
        table_name:        Name of the table that was scanned.
        anomaly_count:     Number of anomalies detected.
        detection_time_ms: Wall-clock time spent on detection (milliseconds).
    """
    current_span = trace.get_current_span()
    if current_span is None:
        return

    current_span.add_event(
        "anomaly_detection_result",
        attributes={
            "table_name": table_name,
            "anomaly_count": anomaly_count,
            "detection_time_ms": detection_time_ms,
        },
    )


def record_blast_radius_metric(
    source_table: str,
    affected_count: int,
    computation_ms: float,
) -> None:
    """
    Record a blast radius computation result as a span event on the current span.

    If there is no active span the call is a no-op.

    Args:
        source_table:   Name of the table whose failure was simulated.
        affected_count: Total number of downstream tables affected.
        computation_ms: Wall-clock time spent computing the blast radius (ms).
    """
    current_span = trace.get_current_span()
    if current_span is None:
        return

    current_span.add_event(
        "blast_radius_result",
        attributes={
            "source_table": source_table,
            "affected_count": affected_count,
            "computation_ms": computation_ms,
        },
    )
