import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.state import AppState
from app.utils.validation import get_missing_fields


llm = get_llm()


def interpret_request_node(state: AppState) -> dict[str, Any]:
    if not state.get("messages"):
        return {"error": "No messages found in state."}

    last_message = state["messages"][-1]

    if not isinstance(last_message, HumanMessage):
        return {"error": "Last message is not a HumanMessage."}

    if state.get("awaiting_clarification") and state.get("pending_request"):
        pending = state["pending_request"]

        system_prompt = f"""
You are completing a partially filled user request.

Current pending request:
{json.dumps(pending, indent=2)}

The user has provided a follow-up clarification message.
Extract only the new information from the clarification.

Return ONLY valid JSON.

Use this exact schema:
{{
  "metric": "string or null",
  "time_range": "string or null",
  "chart_type": "string or null",
  "district": "string or null"
}}

Rules:
- Only extract what the user is clarifying now.
- Use null for fields not provided in this clarification message.
- Do not repeat old values unless they are explicitly mentioned again.
"""

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message.content),
        ])

        raw_output = response.content.strip()

        try:
            parsed = json.loads(raw_output)

            merged = {
                "intent": pending.get("intent"),
                "metric": pending.get("metric"),
                "time_range": pending.get("time_range"),
                "chart_type": pending.get("chart_type"),
                "district": pending.get("district"),
            }

            for key in ["metric", "time_range", "chart_type", "district"]:
                if parsed.get(key) is not None:
                    merged[key] = parsed[key]

            missing_fields = get_missing_fields(merged)
            merged["needs_clarification"] = len(missing_fields) > 0
            merged["missing_fields"] = missing_fields

            return {
                "extracted": merged,
                "error": ""
            }

        except json.JSONDecodeError:
            return {
                "extracted": {},
                "error": f"Model did not return valid JSON for clarification. Raw output: {raw_output}"
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
  "intent": "general_chat | query_request | chart_request | clarification_answer | conversation_meta | unsupported",
  "metric": "string or null",
  "time_range": "string or null",
  "chart_type": "string or null",
  "district": "string or null"
}

Rules:
- If the user greets or speaks casually, use "general_chat".
- If the user asks for a chart or visualization, use "chart_request".
- If the user asks for values, analysis, statistics, or data, use "query_request".
- If the user asks about previous prompts, requests, or conversation history, use "conversation_meta".
- If unsupported, use "unsupported".
- Use null for missing fields.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message.content),
    ])

    raw_output = response.content.strip()

    try:
        parsed = json.loads(raw_output)

        validated = {
            "intent": parsed.get("intent"),
            "metric": parsed.get("metric"),
            "time_range": parsed.get("time_range"),
            "chart_type": parsed.get("chart_type"),
            "district": parsed.get("district"),
        }

        missing_fields = get_missing_fields(validated)
        validated["needs_clarification"] = len(missing_fields) > 0
        validated["missing_fields"] = missing_fields

        return {
            "extracted": validated,
            "error": ""
        }

    except json.JSONDecodeError:
        return {
            "extracted": {},
            "error": f"Model did not return valid JSON. Raw output: {raw_output}"
        }