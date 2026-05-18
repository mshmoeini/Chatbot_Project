from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class AppState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    extracted: dict[str, Any]
    resolved_request: dict[str, Any]

    confirmation_message: str
    last_human_gate: str

    query_draft: str
    approved_query: str

    execution_result: dict[str, Any]
    last_executed_request: dict[str, Any]
    execution_history: list[dict[str, Any]]

    error: str