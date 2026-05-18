from langgraph.types import interrupt

from app.config import FIELD_QUESTIONS
from app.state import AppState


def build_clarification_questions(state: AppState) -> list[str]:
    questions = []

    extracted = state.get("extracted", {})
    resolved_request = state.get("resolved_request", {})

    missing_fields = extracted.get("missing_fields", [])
    for field in missing_fields:
        if field in FIELD_QUESTIONS:
            questions.append(FIELD_QUESTIONS[field])

    metric_mapping = resolved_request.get("metric_mapping", {})
    if metric_mapping.get("status") == "ambiguous":
        questions.append("Which metric do you mean exactly?")
    elif metric_mapping.get("status") == "no_match" and extracted.get("metric"):
        questions.append(
            f"I could not match the metric '{extracted.get('metric')}'. Can you clarify it?"
        )

    district_mapping = resolved_request.get("district_mapping", {})
    if district_mapping.get("status") == "ambiguous":
        questions.append("Which district do you mean exactly?")
    elif district_mapping.get("status") == "no_match" and extracted.get("district"):
        questions.append(
            f"I could not match the district '{extracted.get('district')}'. Can you clarify it?"
        )

    chart_type_mapping = resolved_request.get("chart_type_mapping", {})
    if chart_type_mapping.get("status") == "ambiguous":
        questions.append("Which chart type do you want exactly?")
    elif chart_type_mapping.get("status") == "no_match" and extracted.get("chart_type"):
        questions.append(
            f"I could not match the chart type '{extracted.get('chart_type')}'. Can you clarify it?"
        )

    aggregation_mapping = resolved_request.get("aggregation_mapping", {})
    if aggregation_mapping.get("status") == "ambiguous":
        questions.append("Which aggregation do you want exactly?")
    elif aggregation_mapping.get("status") == "no_match" and extracted.get("aggregation"):
        questions.append(
            f"I could not match the aggregation '{extracted.get('aggregation')}'. Can you clarify it?"
        )

    time_resolution = resolved_request.get("time_resolution", {})
    if time_resolution.get("status") == "ambiguous":
        questions.append("Your requested time range is ambiguous. Can you clarify it?")
    elif time_resolution.get("status") == "no_match" and extracted.get("time_range"):
        questions.append(
            f"I could not resolve the time range '{extracted.get('time_range')}'. Can you clarify it?"
        )

    return questions


def build_confirmation_message(state: AppState) -> str:
    resolved_request = state.get("resolved_request", {})

    intent = resolved_request.get("intent")
    metric_key = resolved_request.get("metric_key")
    district = resolved_request.get("district")
    chart_type = resolved_request.get("chart_type")
    aggregation = resolved_request.get("aggregation")
    time_phrase = resolved_request.get("time_phrase")

    parts = []

    if intent == "chart_request":
        parts.append("You asked for a chart")
    elif intent == "query_request":
        parts.append("You asked for data")
    else:
        parts.append("You made a request")

    if aggregation:
        parts.append(f"with aggregation '{aggregation}'")
    if metric_key:
        parts.append(f"for metric '{metric_key}'")
    if district:
        parts.append(f"in district '{district}'")
    if chart_type:
        parts.append(f"using a '{chart_type}' chart")
    if time_phrase:
        parts.append(f"for time range '{time_phrase}'")

    return " ".join(parts) + ". Is that correct?"


def human_gate_node(state: AppState) -> dict:
    if state.get("error"):
        return {}

    extracted = state.get("extracted", {})
    resolved_request = state.get("resolved_request", {})
    intent = extracted.get("intent")

    if intent in {"general_chat", "conversation_meta", "unsupported"}:
        return {}

    clarification_questions = build_clarification_questions(state)
    validation_errors = resolved_request.get("validation_errors", [])

    if clarification_questions or validation_errors:
        clarification_message = " ".join(clarification_questions + validation_errors)

        user_reply = interrupt(
            {
                "type": "clarification",
                "message": clarification_message,
            }
        )

        return {
            "last_human_gate": "clarification",
            "messages": [user_reply],
        }

    confirmation_message = build_confirmation_message(state)

    user_reply = interrupt(
        {
            "type": "confirmation",
            "message": confirmation_message,
        }
    )

    return {
        "last_human_gate": "confirmation",
        "confirmation_message": confirmation_message,
        "messages": [user_reply],
    }