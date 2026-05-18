from app.state import AppState


def build_query_response(state: AppState) -> str:
    resolved_request = state.get("resolved_request", {})
    execution_result = state.get("execution_result", {})

    metric_key = resolved_request.get("metric_key")
    district = resolved_request.get("district")
    aggregation = resolved_request.get("aggregation")
    time_phrase = resolved_request.get("time_phrase")

    status = execution_result.get("status")
    row_count = execution_result.get("row_count", 0)

    if status != "success":
        return "The query could not be executed successfully."

    parts = ["Your data request was executed successfully."]

    if aggregation:
        parts.append(f"Aggregation: {aggregation}.")
    if metric_key:
        parts.append(f"Metric: {metric_key}.")
    if district:
        parts.append(f"District: {district}.")
    if time_phrase:
        parts.append(f"Time range: {time_phrase}.")
    parts.append(f"Returned rows: {row_count}.")

    return " ".join(parts)


def build_chart_response(state: AppState) -> str:
    resolved_request = state.get("resolved_request", {})
    execution_result = state.get("execution_result", {})

    metric_key = resolved_request.get("metric_key")
    district = resolved_request.get("district")
    chart_type = resolved_request.get("chart_type")
    time_phrase = resolved_request.get("time_phrase")

    status = execution_result.get("status")
    row_count = execution_result.get("row_count", 0)

    if status != "success":
        return "The chart request could not be executed successfully."

    parts = ["Your chart request was executed successfully."]

    if chart_type:
        parts.append(f"Chart type: {chart_type}.")
    if metric_key:
        parts.append(f"Metric: {metric_key}.")
    if district:
        parts.append(f"District: {district}.")
    if time_phrase:
        parts.append(f"Time range: {time_phrase}.")
    parts.append(f"Returned rows: {row_count}.")

    return " ".join(parts)


def final_response_node(state: AppState) -> dict:
    if state.get("error"):
        return {}

    resolved_request = state.get("resolved_request", {})
    execution_result = state.get("execution_result", {})

    if not resolved_request:
        return {
            "final_response": "No resolved request was available."
        }

    if not execution_result:
        return {
            "final_response": "No execution result was available."
        }

    intent = resolved_request.get("intent")

    if intent == "query_request":
        return {
            "final_response": build_query_response(state),
            "error": "",
        }

    if intent == "chart_request":
        return {
            "final_response": build_chart_response(state),
            "error": "",
        }

    return {
        "final_response": "The request was processed.",
        "error": "",
    }