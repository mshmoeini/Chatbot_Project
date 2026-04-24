from typing import Any

from app.state import AppState


def mock_execution_node(state: AppState) -> dict[str, Any]:
    resolved_request = state.get("resolved_request", {})
    extracted = state.get("extracted", {})

    request_data = resolved_request or extracted
    intent = request_data.get("intent")

    if intent == "chart_request":
        return {
            "final_response": "Your chart is ready. [This is currently a mock response. No real chart is generated yet.]",
            "last_executed_request": request_data,
            "execution_history": [
                {
                    "request": request_data,
                    "status": "executed",
                    "summary": "Mock chart request executed."
                }
            ]
        }

    if intent == "query_request":
        return {
            "final_response": "Your data is ready. [This is currently a mock response. No real query is executed yet.]",
            "last_executed_request": request_data,
            "execution_history": [
                {
                    "request": request_data,
                    "status": "executed",
                    "summary": "Mock query request executed."
                }
            ]
        }

    return {
        "final_response": "The request has been processed."
    }