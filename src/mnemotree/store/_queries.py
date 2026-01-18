from __future__ import annotations

from typing import Any

from .serialization import normalize_entity_text


def build_entity_set(entities: dict[str, str] | list[str]) -> set[str]:
    entity_names = list(entities.keys()) if isinstance(entities, dict) else list(entities)
    return {normalize_entity_text(entity) for entity in entity_names if entity}


def entity_matches(stored_entities: dict[str, Any], entity_set: set[str]) -> bool:
    if not stored_entities or not entity_set:
        return False
    stored_entity_names = {normalize_entity_text(k) for k in stored_entities}
    exact_match = bool(entity_set & stored_entity_names)
    substring_match = any(
        req in stored
        for req in entity_set
        for stored in stored_entity_names
        if req and stored and len(req) >= 4 and len(stored) >= 4
    )
    return exact_match or substring_match
