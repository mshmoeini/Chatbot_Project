from typing import Any

from app.config import REQUIRED_FIELDS_BY_INTENT


def get_required_fields(intent: str | None) -> list[str]:
    if not intent:
        return []

    return REQUIRED_FIELDS_BY_INTENT.get(intent, [])


def get_missing_fields(data: dict[str, Any]) -> list[str]:
    intent = data.get("intent")
    required_fields = get_required_fields(intent)

    missing_fields = []

    for field in required_fields:
        if not data.get(field):
            missing_fields.append(field)

    return missing_fields


def is_request_complete(data: dict[str, Any]) -> bool:
    return len(get_missing_fields(data)) == 0