from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.state import AppState
from app.nodes.interpret import interpret_request_node
from app.nodes.human_gate import human_gate_node
from app.nodes.query_generation import query_generation_node
from app.nodes.query_approval import query_approval_node
from app.nodes.execution import execution_node
from app.nodes.final_response import final_response_node


def route_after_interpret(state: AppState) -> str:
    if state.get("error"):
        return "end"

    extracted = state.get("extracted", {})
    intent = extracted.get("intent")

    if intent in {"general_chat", "conversation_meta", "unsupported"}:
        return "end"

    return "human_gate"


def route_after_human_gate(state: AppState) -> str:
    if state.get("error"):
        return "end"

    last_human_gate = state.get("last_human_gate", "")

    if last_human_gate == "clarification":
        return "interpret_request"

    if last_human_gate == "confirmation":
        return "query_generation"

    return "end"


def build_graph():
    graph_builder = StateGraph(AppState)

    graph_builder.add_node("interpret_request", interpret_request_node)
    graph_builder.add_node("human_gate", human_gate_node)
    graph_builder.add_node("query_generation", query_generation_node)
    graph_builder.add_node("query_approval", query_approval_node)
    graph_builder.add_node("execution", execution_node)
    graph_builder.add_node("final_response", final_response_node)

    graph_builder.add_edge(START, "interpret_request")

    graph_builder.add_conditional_edges(
        "interpret_request",
        route_after_interpret,
        {
            "human_gate": "human_gate",
            "end": END,
        },
    )

    graph_builder.add_conditional_edges(
        "human_gate",
        route_after_human_gate,
        {
            "interpret_request": "interpret_request",
            "query_generation": "query_generation",
            "end": END,
        },
    )

    graph_builder.add_edge("query_generation", "query_approval")
    graph_builder.add_edge("query_approval", "execution")
    graph_builder.add_edge("execution", "final_response")
    graph_builder.add_edge("final_response", END)

    checkpointer = MemorySaver()

    return graph_builder.compile(checkpointer=checkpointer)