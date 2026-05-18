import json
from typing import Any
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_interpreter_llm


llm = get_interpreter_llm()


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    return " ".join(text.strip().lower().split())


def build_time_resolution_result(
    original_value: str | None,
    normalized_value: str,
    status: str,
    time_range_key: str | None = None,
    start_time: str | None = None,
    end_time_exclusive: str | None = None,
    resolution_source: str = "llm",
) -> dict[str, Any]:
    return {
        "status": status,
        "original_value": original_value,
        "normalized_value": normalized_value,
        "time_range_key": time_range_key,
        "start_time": start_time,
        "end_time_exclusive": end_time_exclusive,
        "resolution_source": resolution_source,
    }


def get_current_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_time_phrase(time_phrase: str | None, user_timezone: str = "UTC") -> dict[str, Any]:
    normalized_phrase = normalize_text(time_phrase)

    if not normalized_phrase:
        return build_time_resolution_result(
            original_value=time_phrase,
            normalized_value=normalized_phrase,
            status="no_match",
            resolution_source="llm",
        )

    current_utc_timestamp = get_current_utc_timestamp()
    current_year = datetime.now(timezone.utc).year

    system_prompt = f"""
You are a time resolution assistant.

Your task is to convert a user's natural language time phrase into a precise executable UTC time range.

Context:
- Current UTC datetime: {current_utc_timestamp}
- Current year: {current_year}
- User timezone: {user_timezone}
- Database timestamps are stored in UTC.
- The output must be suitable for filtering a UTC timestamp column in a database.

Return ONLY valid JSON.

Use this exact schema:
{{
  "status": "resolved | ambiguous | no_match",
  "time_range_key": "string or null",
  "start_time": "ISO-8601 UTC datetime string or null",
  "end_time_exclusive": "ISO-8601 UTC datetime string or null"
}}

Rules:
- If the phrase can be resolved clearly, return "resolved".
- If the user does not specify a year, assume the current year.
- If that assumption would produce a future-only time range, do NOT silently accept it. Return "ambiguous" instead.
- Do not produce a time range that ends in the future.
- If the phrase needs missing information such as a year or reference context, return "ambiguous".
- If the phrase cannot be interpreted confidently, return "no_match".
- start_time and end_time_exclusive must be in UTC.
- end_time_exclusive must be strictly greater than start_time when resolved.
- Do not include explanations.
- Do not include markdown fences.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=normalized_phrase),
    ])

    raw_output = response.content.strip()

    try:
        parsed = json.loads(raw_output)

        status = parsed.get("status")
        time_range_key = parsed.get("time_range_key")
        start_time = parsed.get("start_time")
        end_time_exclusive = parsed.get("end_time_exclusive")

        if status not in {"resolved", "ambiguous", "no_match"}:
            status = "no_match"

        return build_time_resolution_result(
            original_value=time_phrase,
            normalized_value=normalized_phrase,
            status=status,
            time_range_key=time_range_key,
            start_time=start_time,
            end_time_exclusive=end_time_exclusive,
            resolution_source="llm",
        )

    except json.JSONDecodeError:
        return build_time_resolution_result(
            original_value=time_phrase,
            normalized_value=normalized_phrase,
            status="no_match",
            resolution_source="llm",
        )