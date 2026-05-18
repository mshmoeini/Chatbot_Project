import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_interpreter_llm
from app.state import AppState
from app.utils.mapping import map_entity_phrase
from app.utils.time_resolution import resolve_time_phrase
from app.utils.validation import (
    get_missing_fields,
    get_request_validation_errors,
    validate_time_resolution,
)


llm = get_interpreter_llm()


def interpret_request_node(state: AppState) -> dict:
    if not state.get("messages"):
        return {
            "extracted": {},
            "resolved_request": {},
            "error": "No messages found in state.",
        }

    last_message = state["messages"][-1]

    if not isinstance(last_message, HumanMessage):
        return {
            "extracted": {},
            "resolved_request": {},
            "error": "Last message is not a HumanMessage.",
        }

    system_prompt = """
You are an information extraction system.

Your task is to extract structured information from the user's message.

Return ONLY valid JSON.
Do not include markdown fences.
Do not include explanations.
Do not include extra text before or after the JSON.

Use this exact schema:
{
  "intent": "general_chat | query_request | chart_request | conversation_meta | unsupported",
  "metric": "string or null",
  "time_range": "string or null",
  "chart_type": "string or null",
  "district": "string or null",
  "aggregation": "string or null"
}

Rules:
- If the user greets or speaks casually, use "general_chat".
- If the user asks for a chart or visualization, use "chart_request".
- If the user asks for values, analysis, statistics, or data, use "query_request".
- If the user asks about previous prompts, requests, or conversation history, use "conversation_meta".
- If unsupported, use "unsupported".
- Use null for missing fields.
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message.content),
        ]
    )

    raw_output = response.content.strip()

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "extracted": {},
            "resolved_request": {},
            "error": f"Model did not return valid JSON. Raw output: {raw_output}",
        }

    extracted = {
        "intent": parsed.get("intent"),
        "metric": parsed.get("metric"),
        "time_range": parsed.get("time_range"),
        "chart_type": parsed.get("chart_type"),
        "district": parsed.get("district"),
        "aggregation": parsed.get("aggregation"),
    }

    missing_fields = get_missing_fields(extracted)
    extracted["missing_fields"] = missing_fields

    if extracted["intent"] in {"general_chat", "conversation_meta", "unsupported"}:
        return {
            "extracted": extracted,
            "resolved_request": {},
            "error": "",
        }

    metric_mapping = map_entity_phrase("metric", extracted.get("metric"))
    district_mapping = map_entity_phrase("district", extracted.get("district"))
    chart_type_mapping = map_entity_phrase("chart_type", extracted.get("chart_type"))
    aggregation_mapping = map_entity_phrase("aggregation", extracted.get("aggregation"))

    time_resolution = resolve_time_phrase(extracted.get("time_range"))
    time_validation_errors = validate_time_resolution(time_resolution)

    resolved_request = {
        "intent": extracted.get("intent"),
        "metric": extracted.get("metric"),
        "metric_mapping": metric_mapping,
        "metric_key": metric_mapping.get("canonical_value"),
        "metric_definition": metric_mapping.get("definition"),
        "district": district_mapping.get("canonical_value"),
        "district_mapping": district_mapping,
        "chart_type": chart_type_mapping.get("canonical_value"),
        "chart_type_mapping": chart_type_mapping,
        "aggregation": aggregation_mapping.get("canonical_value"),
        "aggregation_mapping": aggregation_mapping,
        "time_phrase": extracted.get("time_range"),
        "time_resolution": time_resolution,
    }

    validation_errors = get_request_validation_errors(
        {
            "intent": resolved_request.get("intent"),
            "metric": resolved_request.get("metric_key"),
            "district": resolved_request.get("district"),
            "chart_type": resolved_request.get("chart_type"),
            "aggregation": resolved_request.get("aggregation"),
        }
    )

    validation_errors.extend(time_validation_errors)
    resolved_request["validation_errors"] = validation_errors

    return {
        "extracted": extracted,
        "resolved_request": resolved_request,
        "error": "",
    }