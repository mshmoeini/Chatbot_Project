MODEL_NAME = "llama3.1:latest"
MODEL_TEMPERATURE = 0

SUPPORTED_INTENTS = {
    "general_chat",
    "query_request",
    "chart_request",
    "clarification_answer",
    "conversation_meta",
    "unsupported",
}

REQUIRED_FIELDS_BY_INTENT = {
    "chart_request": ["metric", "time_range", "chart_type", "district"],
    "query_request": ["metric", "time_range", "district"],
}

FIELD_QUESTIONS = {
    "time_range": "What time range would you like to use?",
    "chart_type": "What chart type would you like, for example bar or line?",
    "district": "Which district would you like to check?",
    "metric": "Which metric do you want to analyze?",
}