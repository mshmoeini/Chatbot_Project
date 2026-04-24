from langgraph.graph import StateGraph, START, END

from app.state import AppState
from app.nodes.interpret import interpret_request_node
from app.nodes.response import prepare_user_message_node
from app.nodes.execution import mock_execution_node


def route_after_response(state: AppState) -> str:
    extracted = state.get("extracted", {})
    intent = extracted.get("intent")
    needs_clarification = extracted.get("needs_clarification", False)

    if state.get("error"):
        return "end"

    if intent in ["general_chat", "unsupported", "conversation_meta"]:
        return "end"

    if needs_clarification:
        return "end"

    if intent in ["chart_request", "query_request"]:
        return "execute"

    return "end"


def build_graph():
    graph_builder = StateGraph(AppState)

    graph_builder.add_node("interpret_request", interpret_request_node)
    graph_builder.add_node("prepare_user_message", prepare_user_message_node)
    graph_builder.add_node("mock_execution", mock_execution_node)

    graph_builder.add_edge(START, "interpret_request")
    graph_builder.add_edge("interpret_request", "prepare_user_message")

    graph_builder.add_conditional_edges(
        "prepare_user_message",
        route_after_response,
        {
            "execute": "mock_execution",
            "end": END
        }
    )

    graph_builder.add_edge("mock_execution", END)

    return graph_builder.compile()