from typing import Any

from app.config import FIELD_QUESTIONS
from app.state import AppState


def prepare_user_message_node(state: AppState) -> dict[str, Any]:
    extracted = state.get("extracted", {})
    intent = extracted.get("intent")
    needs_clarification = extracted.get("needs_clarification", False)
    missing_fields = extracted.get("missing_fields", [])

    if intent == "general_chat":
        return {
            "assistant_response": "Hello! How can I help you today?",
            "awaiting_clarification": False,
            "pending_request": {}
        }

    if intent == "unsupported":
        return {
            "assistant_response": "I do not support this type of request yet.",
            "awaiting_clarification": False,
            "pending_request": {}
        }

    if intent == "conversation_meta":
        return {
            "assistant_response": "I understood that you are asking about previous conversation context. This handler is not implemented yet.",
            "awaiting_clarification": False,
            "pending_request": {}
        }

    if needs_clarification:
        questions = [FIELD_QUESTIONS[field] for field in missing_fields if field in FIELD_QUESTIONS]

        if questions:
            return {
                "assistant_response": " ".join(questions),
                "awaiting_clarification": True,
                "pending_request": extracted
            }

        return {
            "assistant_response": "Please provide more details so I can complete your request.",
            "awaiting_clarification": True,
            "pending_request": extracted
        }

    if intent == "chart_request":
        return {
            "assistant_response": "I am preparing your chart.",
            "awaiting_clarification": False,
            "pending_request": {}
        }

    if intent == "query_request":
        return {
            "assistant_response": "I am preparing the requested information.",
            "awaiting_clarification": False,
            "pending_request": {}
        }

    return {
        "assistant_response": "Your request has been received.",
        "awaiting_clarification": False,
        "pending_request": {}
    }