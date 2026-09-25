"""Sample data contracts for testing and demonstration."""

from .registry import DataContract, FieldContract


def get_orders_contract() -> DataContract:
    return DataContract(
        table_name="orders",
        version="2.1.0",
        producer="order_service",
        consumers=["ml_features", "revenue_dashboard", "finance_report"],
        description="Core orders table — source of truth for all revenue metrics",
        fields=[
            FieldContract("order_id", "integer", nullable=False, description="Unique order ID"),
            FieldContract("customer_id", "integer", nullable=False, description="FK to customers"),
            FieldContract("amount", "float", nullable=False, constraints={"min": 0, "max": 1_000_000}),
            FieldContract(
                "status", "string", nullable=False,
                constraints={"allowed_values": ["pending", "processing", "shipped", "delivered", "cancelled"]},
            ),
            FieldContract("created_at", "timestamp", nullable=False),
            FieldContract("updated_at", "timestamp", nullable=True),
            FieldContract("region", "string", nullable=False,
                         constraints={"allowed_values": ["US", "EU", "APAC", "LATAM"]}),
            FieldContract("discount_pct", "float", nullable=True,
                         constraints={"min": 0, "max": 100}),
        ],
    )


def get_users_contract() -> DataContract:
    return DataContract(
        table_name="users",
        version="1.0.0",
        producer="user_service",
        consumers=["ml_features", "customer_analytics"],
        description="User profiles — PII fields marked nullable as masking policy",
        fields=[
            FieldContract("user_id", "integer", nullable=False),
            FieldContract("created_at", "timestamp", nullable=False),
            FieldContract("account_type", "string", nullable=False,
                         constraints={"allowed_values": ["free", "pro", "enterprise"]}),
            FieldContract("country_code", "string", nullable=False,
                         constraints={"pattern": r"^[A-Z]{2}$"}),
            FieldContract("email_hash", "string", nullable=True,
                         description="SHA-256 hash — never plain text"),
            FieldContract("is_active", "boolean", nullable=False),
        ],
    )


def get_events_contract() -> DataContract:
    return DataContract(
        table_name="raw_events",
        version="3.0.0",
        producer="event_collector",
        consumers=["session_analytics", "ml_features", "fraud_detection"],
        description="High-volume raw event stream — 50M+ events/day",
        fields=[
            FieldContract("event_id", "string", nullable=False,
                         constraints={"pattern": r"^[a-f0-9-]{36}$"}),
            FieldContract("event_type", "string", nullable=False),
            FieldContract("user_id", "integer", nullable=True,
                         description="Null for anonymous events"),
            FieldContract("session_id", "string", nullable=False),
            FieldContract("timestamp", "timestamp", nullable=False),
            FieldContract("properties", "string", nullable=True,
                         description="JSON-encoded event properties"),
            FieldContract("source", "string", nullable=False,
                         constraints={"allowed_values": ["web", "ios", "android", "api"]}),
        ],
    )
