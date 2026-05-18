from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_query_generator_llm
from app.schema import DATA_SCHEMA, CANONICAL_METRICS
from app.state import AppState
from app.utils.validation import is_query_spec_ready


llm = get_query_generator_llm()


def build_schema_context_text() -> str:
    table_name = DATA_SCHEMA.get("table_name", "")
    columns = DATA_SCHEMA.get("columns", {})
    time_column = DATA_SCHEMA.get("time_column", "")
    value_column = DATA_SCHEMA.get("value_column", "")
    filterable_columns = DATA_SCHEMA.get("filterable_columns", [])

    lines = [
        f"Table name: {table_name}",
        "Columns:",
    ]

    for column_name, column_type in columns.items():
        lines.append(f"- {column_name}: {column_type}")

    lines.append(f"Time column: {time_column}")
    lines.append(f"Default value column: {value_column}")
    lines.append(f"Filterable columns: {', '.join(filterable_columns)}")

    return "\n".join(lines)


def build_metric_context_text() -> str:
    lines = ["Canonical metrics:"]

    for metric_key, definition in CANONICAL_METRICS.items():
        label = definition.get("label", "")
        description = definition.get("description", "")
        value_column = definition.get("value_column", "")
        filters = definition.get("filters", {})

        lines.append(
            f"- {metric_key}: label='{label}', description='{description}', "
            f"value_column='{value_column}', filters={filters}"
        )

    return "\n".join(lines)


def query_generation_node(state: AppState) -> dict:
    if state.get("error"):
        return {}

    resolved_request = state.get("resolved_request", {})
    if not resolved_request:
        return {
            "error": "No resolved request found for query generation."
        }

    if resolved_request.get("validation_errors"):
        return {
            "error": "Resolved request is not valid for query generation."
        }

    query_spec = {
        "intent": resolved_request.get("intent"),
        "metric_key": resolved_request.get("metric_key"),
        "district": resolved_request.get("district"),
        "chart_type": resolved_request.get("chart_type"),
        "aggregation": resolved_request.get("aggregation"),
        "time_resolution": resolved_request.get("time_resolution"),
    }

    if not is_query_spec_ready(
        {
            "intent": query_spec.get("intent"),
            "metric": query_spec.get("metric_key"),
            "district": query_spec.get("district"),
            "chart_type": query_spec.get("chart_type"),
        }
    ):
        return {
            "error": "Query spec is not ready for query generation."
        }

    schema_context = build_schema_context_text()
    metric_context = build_metric_context_text()

    system_prompt = f"""
You are a query generation assistant.

Your task is to generate a read-only SQL query draft from a resolved user request.

Database context:
{schema_context}

Metric context:
{metric_context}

Return ONLY the SQL query text.
Do not include markdown fences.
Do not include explanations.
Do not include comments.

Rules:
- Generate only a SELECT query.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, GRANT, or REVOKE.
- Use the provided schema and canonical metric definitions only.
- Use UTC timestamps.
- If the request intent is chart_request, generate a query suitable for retrieving chart data.
- If the request intent is query_request, generate a query suitable for retrieving the requested data or aggregation.
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=str(query_spec)),
        ]
    )

    query_draft = response.content.strip()

    return {
        "query_draft": query_draft,
        "error": "",
    }