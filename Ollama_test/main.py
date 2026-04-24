import json
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    extracted: dict[str, Any]
    assistant_response: str
    final_response: str
    error: str


llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0
)


def interpret_request_node(state: State):
    if not state.get("messages"):
        return {"error": "No messages found in state."}

    last_message = state["messages"][-1]

    if not isinstance(last_message, HumanMessage):
        return {"error": "Last message is not a HumanMessage."}

    system_prompt = """
You are an information extraction system.

Your task is to extract structured information from the user's message.

Return ONLY valid JSON.
Do not include markdown fences.
Do not include explanations.
Do not include extra text before or after the JSON.

Use this exact schema:
{
  "intent": "general_chat | query_request | chart_request | clarification_answer | unsupported",
  "metric": "string or null",
  "time_range": "string or null",
  "chart_type": "string or null",
  "district": "string or null",
  "needs_clarification": true,
  "missing_fields": ["field1", "field2"]
}

Rules:
- If the user greets or speaks casually, use "general_chat".
- If the user asks for a chart or visualization, use "chart_request".
- If the user asks for values, analysis, statistics, or data, use "query_request".
- If the user answers a previous clarification question, use "clarification_answer".
- If unsupported, use "unsupported".
- If important information is missing, set "needs_clarification" to true.
- Use null for missing fields.
- Use an empty list if no fields are missing.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message.content),
    ])

    raw_output = response.content.strip()

    try:
        parsed = json.loads(raw_output)
        return {
            "extracted": parsed,
            "error": ""
        }
    except json.JSONDecodeError:
        return {
            "extracted": {},
            "error": f"Model did not return valid JSON. Raw output: {raw_output}"
        }


def prepare_user_message_node(state: State):
    extracted = state.get("extracted", {})
    intent = extracted.get("intent")
    needs_clarification = extracted.get("needs_clarification", False)
    missing_fields = extracted.get("missing_fields", [])

    if intent == "general_chat":
        return {
            "assistant_response": "Hello! How can I help you today?"
        }

    if intent == "unsupported":
        return {
            "assistant_response": "I do not support this type of request yet."
        }

    if needs_clarification:
        field_questions = {
            "time_range": "What time range would you like to use?",
            "chart_type": "What chart type would you like, for example bar or line?",
            "district": "Which district would you like to use?",
            "metric": "Which metric do you want to analyze?"
        }

        questions = [field_questions[f] for f in missing_fields if f in field_questions]

        if questions:
            return {
                "assistant_response": " ".join(questions)
            }
        else:
            return {
                "assistant_response": "Please provide more details so I can complete your request."
            }

    if intent == "chart_request":
        return {
            "assistant_response": "I am preparing your chart."
        }

    if intent == "query_request":
        return {
            "assistant_response": "I am preparing the requested information."
        }

    return {
        "assistant_response": "Your request has been received."
    }


def route_request(state: State) -> Literal["end_after_message", "mock_execution"]:
    if state.get("error"):
        return "end_after_message"

    extracted = state.get("extracted", {})
    intent = extracted.get("intent")
    needs_clarification = extracted.get("needs_clarification", False)

    if intent in ["general_chat", "unsupported"]:
        return "end_after_message"

    if needs_clarification:
        return "end_after_message"

    return "mock_execution"


def mock_execution_node(state: State):
    extracted = state.get("extracted", {})
    intent = extracted.get("intent")

    if intent == "chart_request":
        return {
            "final_response": "Your chart is ready. [This is currently a mock response. No real chart is generated yet.]"
        }

    if intent == "query_request":
        return {
            "final_response": "Your data is ready. [This is currently a mock response. No real query is executed yet.]"
        }

    return {
        "final_response": "The request has been processed."
    }


def final_response_node(state: State):
    return {
        "final_response": state.get("final_response", "No final response is available.")
    }


graph_builder = StateGraph(State)

graph_builder.add_node("interpret_request", interpret_request_node)
graph_builder.add_node("prepare_user_message", prepare_user_message_node)
graph_builder.add_node("mock_execution", mock_execution_node)
graph_builder.add_node("final_response", final_response_node)

graph_builder.add_edge(START, "interpret_request")
graph_builder.add_edge("interpret_request", "prepare_user_message")

graph_builder.add_conditional_edges(
    "prepare_user_message",
    route_request,
    {
        "end_after_message": END,
        "mock_execution": "mock_execution"
    }
)

graph_builder.add_edge("mock_execution", "final_response")
graph_builder.add_edge("final_response", END)

graph = graph_builder.compile()


def main():
    print("CLI Demo - LangGraph + Ollama")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("USER: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Bye.")
            break

        if not user_input:
            continue

        initial_state: State = {
            "messages": [HumanMessage(content=user_input)]
        }

        result = graph.invoke(initial_state)

        if result.get("error"):
            print("\nERROR:")
            print(result["error"])
            print()
            continue

        print("\n[EXTRACTED]")
        print(json.dumps(result.get("extracted", {}), indent=2, ensure_ascii=False))

        if result.get("assistant_response"):
            print(f"\nAI (intermediate): {result['assistant_response']}")

        if result.get("final_response"):
            print(f"AI (final): {result['final_response']}")

        print()


if __name__ == "__main__":
    main()