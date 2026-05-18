from typing import Any
from datetime import datetime, timezone

from app.config import (
    REQUIRED_FIELDS_BY_INTENT,
    QUERY_SPEC_REQUIRED_FIELDS,
    ALLOWED_METRICS,
    ALLOWED_CHART_TYPES,
    ALLOWED_AGGREGATIONS,
    ALLOWED_DISTRICTS,
    FORBIDDEN_QUERY_KEYWORDS,
)


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


def validate_metric(metric: str | None) -> bool:
    if not metric:
        return False

    return metric in ALLOWED_METRICS


def validate_chart_type(chart_type: str | None) -> bool:
    if not chart_type:
        return False

    return chart_type in ALLOWED_CHART_TYPES


def validate_aggregation(aggregation: str | None) -> bool:
    if not aggregation:
        return False

    return aggregation in ALLOWED_AGGREGATIONS


def validate_district(district: str | None) -> bool:
    if not district:
        return False

    return district in ALLOWED_DISTRICTS


def get_request_validation_errors(data: dict[str, Any]) -> list[str]:
    errors = []

    intent = data.get("intent")

    if not intent:
        errors.append("Missing intent.")
        return errors

    missing_fields = get_missing_fields(data)
    for field in missing_fields:
        errors.append(f"Missing required field: {field}")

    metric = data.get("metric")
    if metric and not validate_metric(metric):
        errors.append(f"Invalid metric: {metric}")

    chart_type = data.get("chart_type")
    if chart_type and not validate_chart_type(chart_type):
        errors.append(f"Invalid chart type: {chart_type}")

    aggregation = data.get("aggregation")
    if aggregation and not validate_aggregation(aggregation):
        errors.append(f"Invalid aggregation: {aggregation}")

    district = data.get("district")
    if district and not validate_district(district):
        errors.append(f"Invalid district: {district}")

    return errors


def is_request_valid(data: dict[str, Any]) -> bool:
    return len(get_request_validation_errors(data)) == 0


def get_query_spec_required_fields(intent: str | None) -> list[str]:
    if not intent:
        return []

    return QUERY_SPEC_REQUIRED_FIELDS.get(intent, [])


def get_missing_query_spec_fields(data: dict[str, Any]) -> list[str]:
    intent = data.get("intent")
    required_fields = get_query_spec_required_fields(intent)

    missing_fields = []

    for field in required_fields:
        if not data.get(field):
            missing_fields.append(field)

    return missing_fields


def is_query_spec_ready(data: dict[str, Any]) -> bool:
    return len(get_missing_query_spec_fields(data)) == 0


def contains_forbidden_query_keywords(query: str | None) -> bool:
    if not query:
        return False

    upper_query = query.upper()

    for keyword in FORBIDDEN_QUERY_KEYWORDS:
        if keyword in upper_query:
            return True

    return False


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def validate_time_resolution(time_data: dict[str, Any]) -> list[str]:
    errors = []

    status = time_data.get("status")
    start_time_raw = time_data.get("start_time")
    end_time_raw = time_data.get("end_time_exclusive")

    if status not in {"resolved", "ambiguous", "no_match"}:
        errors.append("Invalid time resolution status.")
        return errors

    if status != "resolved":
        return errors

    start_time = parse_iso_datetime(start_time_raw)
    end_time_exclusive = parse_iso_datetime(end_time_raw)

    if not start_time:
        errors.append("Invalid or missing start_time.")
    if not end_time_exclusive:
        errors.append("Invalid or missing end_time_exclusive.")

    if errors:
        return errors

    if start_time >= end_time_exclusive:
        errors.append("start_time must be earlier than end_time_exclusive.")

    now_utc = datetime.now(timezone.utc)

    if end_time_exclusive > now_utc:
        errors.append("Requested time range cannot end in the future.")

    return errors


def is_time_resolution_valid(time_data: dict[str, Any]) -> bool:
    return len(validate_time_resolution(time_data)) == 0