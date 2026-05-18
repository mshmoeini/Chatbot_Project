from app.state import AppState
from app.utils.validation import contains_forbidden_query_keywords


def is_read_only_query(query: str) -> bool:
    normalized = query.strip().lower()
    return normalized.startswith("select")


def query_approval_node(state: AppState) -> dict:
    if state.get("error"):
        return {}

    query_draft = state.get("query_draft", "")

    if not query_draft:
        return {
            "error": "No query draft found for approval."
        }

    if contains_forbidden_query_keywords(query_draft):
        return {
            "error": "Query draft contains forbidden SQL keywords."
        }

    if not is_read_only_query(query_draft):
        return {
            "error": "Only read-only SELECT queries are allowed."
        }

    return {
        "approved_query": query_draft,
        "error": "",
    }