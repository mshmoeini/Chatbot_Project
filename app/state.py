from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class AppState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    extracted: dict[str, Any]
    pending_request: dict[str, Any]
    resolved_request: dict[str, Any]
    last_executed_request: dict[str, Any]
    execution_history: list[dict[str, Any]]

    awaiting_clarification: bool
    awaiting_confirmation: bool

    confirmation_message: str
    assistant_response: str
    final_response: str
    error: str