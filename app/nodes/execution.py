from app.state import AppState


def run_query(query: str) -> dict:
    """
    Temporary mock execution layer.
    Replace this with real database execution later.
    """
    return {
        "status": "success",
        "rows": [],
        "row_count": 0,
        "query": query,
    }


def execution_node(state: AppState) -> dict:
    if state.get("error"):
        return {}

    approved_query = state.get("approved_query", "")
    resolved_request = state.get("resolved_request", {})
    execution_history = state.get("execution_history", [])

    if not approved_query:
        return {
            "error": "No approved query found for execution."
        }

    execution_result = run_query(approved_query)

    history_item = {
        "request": resolved_request,
        "query": approved_query,
        "status": execution_result.get("status", "unknown"),
        "row_count": execution_result.get("row_count", 0),
        "summary": "Query executed successfully." if execution_result.get("status") == "success" else "Query execution failed.",
    }

    return {
        "execution_result": execution_result,
        "last_executed_request": resolved_request,
        "execution_history": execution_history + [history_item],
        "error": "",
    }