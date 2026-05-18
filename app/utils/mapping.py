from typing import Any

from app.llm import get_interpreter_llm
from app.mappings import ENTITY_ALIAS_MAPS
from app.schema import CANONICAL_METRICS


llm = get_interpreter_llm()


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    return " ".join(text.strip().lower().split())


def get_alias_registry(entity_type: str) -> dict[str, str]:
    return ENTITY_ALIAS_MAPS.get(entity_type, {})


def find_alias_match(entity_type: str, phrase: str | None) -> str | None:
    normalized_phrase = normalize_text(phrase)

    if not normalized_phrase:
        return None

    alias_registry = get_alias_registry(entity_type)
    return alias_registry.get(normalized_phrase)


def get_metric_definition(metric_key: str | None) -> dict[str, Any] | None:
    if not metric_key:
        return None

    return CANONICAL_METRICS.get(metric_key)


def build_mapping_result(entity_type: str, phrase: str | None) -> dict[str, Any]:
    normalized_phrase = normalize_text(phrase)
    canonical_value = find_alias_match(entity_type, phrase)

    result = {
        "entity_type": entity_type,
        "status": "no_match",
        "original_value": phrase,
        "normalized_value": normalized_phrase,
        "canonical_value": None,
        "definition": None,
        "mapping_source": "alias_lookup",
    }

    if not canonical_value:
        return result

    result["status"] = "mapped"
    result["canonical_value"] = canonical_value

    if entity_type == "metric":
        result["definition"] = get_metric_definition(canonical_value)

    return result


def build_metric_candidates_text() -> str:
    lines = []

    for metric_key, definition in CANONICAL_METRICS.items():
        label = definition.get("label", "")
        description = definition.get("description", "")
        lines.append(f"- {metric_key}: label='{label}', description='{description}'")

    return "\n".join(lines)


def llm_map_metric_phrase(metric_phrase: str | None) -> dict[str, Any]:
    normalized_phrase = normalize_text(metric_phrase)

    if not normalized_phrase:
        return {
            "entity_type": "metric",
            "status": "no_match",
            "original_value": metric_phrase,
            "normalized_value": normalized_phrase,
            "canonical_value": None,
            "definition": None,
            "mapping_source": "llm_fallback",
        }

    candidates_text = build_metric_candidates_text()

    system_prompt = f"""
You are a metric mapping assistant.

Your task is to map a user metric phrase to one canonical metric key.

Allowed canonical metrics:
{candidates_text}

Return ONLY valid JSON.

Use this exact schema:
{{
  "status": "mapped | ambiguous | no_match",
  "canonical_value": "string or null"
}}

Rules:
- Choose ONLY from the allowed canonical metric keys listed above.
- Do not invent new metric keys.
- If there is one clear best match, return "mapped".
- If there are multiple plausible matches and you are not sure, return "ambiguous".
- If nothing matches confidently, return "no_match".
"""

    response = llm.invoke([
        ("system", system_prompt),
        ("human", normalized_phrase),
    ])

    raw_output = response.content.strip()

    try:
        import json

        parsed = json.loads(raw_output)

        status = parsed.get("status")
        canonical_value = parsed.get("canonical_value")

        if status not in {"mapped", "ambiguous", "no_match"}:
            status = "no_match"

        if canonical_value not in CANONICAL_METRICS:
            canonical_value = None

        definition = get_metric_definition(canonical_value)

        return {
            "entity_type": "metric",
            "status": status,
            "original_value": metric_phrase,
            "normalized_value": normalized_phrase,
            "canonical_value": canonical_value,
            "definition": definition,
            "mapping_source": "llm_fallback",
        }

    except Exception:
        return {
            "entity_type": "metric",
            "status": "no_match",
            "original_value": metric_phrase,
            "normalized_value": normalized_phrase,
            "canonical_value": None,
            "definition": None,
            "mapping_source": "llm_fallback",
        }


def map_entity_phrase(entity_type: str, phrase: str | None) -> dict[str, Any]:
    alias_result = build_mapping_result(entity_type, phrase)

    if alias_result["status"] == "mapped":
        return alias_result

    if entity_type == "metric":
        return llm_map_metric_phrase(phrase)

    return alias_result