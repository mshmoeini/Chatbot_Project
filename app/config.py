MODEL_NAME = "llama3.1:latest"
MODEL_TEMPERATURE = 0

SUPPORTED_INTENTS = {
    "general_chat",
    "query_request",
    "chart_request",
    "conversation_meta",
    "unsupported",
}

REQUIRED_FIELDS_BY_INTENT = {
    "chart_request": ["metric", "time_range", "chart_type", "district"],
    "query_request": ["metric", "time_range", "district"],
}

FIELD_QUESTIONS = {
    "metric": "Which metric do you want to analyze?",
    "time_range": "What time range would you like to use?",
    "chart_type": "What chart type would you like, for example bar or line?",
    "district": "Which district would you like to use?",
}

ALLOWED_METRICS = {
    "pressure",
    "leakage",
    "flow",
    "consumption",
}

ALLOWED_CHART_TYPES = {
    "line",
    "bar",
    "scatter",
}

ALLOWED_AGGREGATIONS = {
    "avg",
    "sum",
    "min",
    "max",
    "count",
}

ALLOWED_DISTRICTS = {
    "Marconi",
    "Ponte",
    "Piemonte",
}

QUERY_SPEC_REQUIRED_FIELDS = {
    "query_request": ["metric", "time_range", "district"],
    "chart_request": ["metric", "time_range", "district", "chart_type"],
}

FORBIDDEN_QUERY_KEYWORDS = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
}

MAX_QUERY_LIMIT = 1000

INTERRUPT_TYPES = {
    "clarification",
    "confirmation",
    "human_review",
}