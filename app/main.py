from langgraph.types import Command

from app.graph import build_graph


def main() -> None:
    graph = build_graph()

    print("LangGraph CLI Chat")
    print("Type 'exit' or 'quit' to stop.\n")

    thread_id = input("Thread ID (press Enter for 'default-thread'): ").strip() or "default-thread"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("USER: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        if not user_input:
            continue

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        while "__interrupt__" in result and result["__interrupt__"]:
            interrupt_obj = result["__interrupt__"][0]
            interrupt_payload = interrupt_obj.value

            interrupt_type = interrupt_payload.get("type", "unknown")
            interrupt_message = interrupt_payload.get("message", "The graph is waiting for input.")

            print(f"\n[INTERRUPT: {interrupt_type}]")
            print(interrupt_message)

            resume_input = input("YOUR RESPONSE: ").strip()

            if resume_input.lower() in {"exit", "quit"}:
                print("Bye.")
                return

            result = graph.invoke(
                Command(resume={"role": "user", "content": resume_input}),
                config=config,
            )

        if result.get("error"):
            print(f"\n[ERROR]\n{result['error']}\n")
            continue

        extracted = result.get("extracted", {})
        resolved_request = result.get("resolved_request", {})
        confirmation_message = result.get("confirmation_message", "")
        query_draft = result.get("query_draft", "")
        approved_query = result.get("approved_query", "")
        execution_result = result.get("execution_result", {})
        final_response = result.get("final_response", "")

        if extracted:
            print("\n[EXTRACTED]")
            print(extracted)

        if resolved_request:
            print("\n[RESOLVED REQUEST]")
            print(resolved_request)

        if confirmation_message:
            print("\n[CONFIRMATION MESSAGE]")
            print(confirmation_message)

        if query_draft:
            print("\n[QUERY DRAFT]")
            print(query_draft)

        if approved_query:
            print("\n[APPROVED QUERY]")
            print(approved_query)

        if execution_result:
            print("\n[EXECUTION RESULT]")
            print(execution_result)

        if final_response:
            print("\n[FINAL RESPONSE]")
            print(final_response)

        print()


if __name__ == "__main__":
    main()