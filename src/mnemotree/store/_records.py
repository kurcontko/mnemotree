from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..core.models import MemoryItem, MemoryType
from ..utils.serialization import json_dumps_safe, json_loads_dict
from .serialization import safe_load_context, serialize_datetime, serialize_datetime_list


def build_neo4j_memory_payload(
    memory: MemoryItem,
) -> tuple[dict[str, Any], dict[str, str]]:
    embedding_list = (
        memory.embedding.tolist() if isinstance(memory.embedding, np.ndarray) else memory.embedding
    )
    context_json = json_dumps_safe(memory.context or {})
    valid_entities = {
        str(text): str(etype) for text, etype in (memory.entities or {}).items() if text and etype
    }
    payload = {
        "memory_id": memory.memory_id,
        "conversation_id": memory.conversation_id,
        "user_id": memory.user_id,
        "content": memory.content,
        "summary": memory.summary,
        "author": memory.author,
        "memory_type": memory.memory_type.value,
        "timestamp": serialize_datetime(memory.timestamp),
        "last_accessed": serialize_datetime(memory.last_accessed),
        "access_count": memory.access_count,
        "access_history": serialize_datetime_list(list(memory.access_history)),
        "importance": memory.importance,
        "decay_rate": memory.decay_rate,
        "confidence": memory.confidence,
        "fidelity": memory.fidelity,
        "emotional_valence": memory.emotional_valence,
        "emotional_arousal": memory.emotional_arousal,
        "emotions": memory.emotions,
        "linked_concepts": memory.linked_concepts,
        "previous_event_id": memory.previous_event_id,
        "next_event_id": memory.next_event_id,
        "source": memory.source,
        "credibility": memory.credibility,
        "embedding": embedding_list,
        "context": context_json,
        "entities": json_dumps_safe(valid_entities),
        "entity_mentions": json_dumps_safe(memory.entity_mentions),
    }
    return payload, valid_entities


def parse_neo4j_node_data(
    node_data: dict[str, Any],
    *,
    parse_context: bool = True,
    parse_entities: bool = True,
    parse_entity_mentions: bool = True,
    strict_json: bool = False,
) -> dict[str, Any]:
    parsed = dict(node_data)
    if parse_context:
        parsed["context"] = _load_json_value(parsed.get("context"), strict=strict_json)
    if parse_entities:
        parsed["entities"] = _load_json_value(parsed.get("entities"), strict=strict_json)
    if parse_entity_mentions:
        parsed["entity_mentions"] = _load_json_value(
            parsed.get("entity_mentions"), strict=strict_json
        )
    parsed["memory_type"] = MemoryType(parsed["memory_type"])
    return parsed


def chroma_metadata_from_memory(memory: MemoryItem) -> dict[str, str]:
    return {
        "memory_type": memory.memory_type.value,
        "timestamp": serialize_datetime(memory.timestamp) or "",
        "importance": str(memory.importance),
        "tags": ",".join(memory.tags) if memory.tags else "",
        "emotions": ",".join(str(e) for e in memory.emotions) if memory.emotions else "",
        "confidence": str(memory.confidence),
        "source": memory.source if memory.source is not None else "",
        "context": json_dumps_safe(memory.context) if memory.context else "",
        "last_accessed": serialize_datetime(memory.last_accessed) or "",
        "access_count": str(memory.access_count),
        "access_history": json.dumps(serialize_datetime_list(list(memory.access_history))),
        "entities": json_dumps_safe(memory.entities) if memory.entities else "{}",
        "associations": ",".join(memory.associations) if memory.associations else "",
        "linked_concepts": ",".join(memory.linked_concepts) if memory.linked_concepts else "",
        "conflicts_with": ",".join(memory.conflicts_with) if memory.conflicts_with else "",
        "previous_event_id": memory.previous_event_id if memory.previous_event_id else "",
        "next_event_id": memory.next_event_id if memory.next_event_id else "",
    }


def chroma_memory_from_record(
    *,
    memory_id: str,
    document: str,
    embedding: list[float] | None,
    metadata: dict[str, Any],
    entities_override: dict[str, Any] | None = None,
) -> MemoryItem:
    entities = (
        entities_override
        if entities_override is not None
        else json_loads_dict(metadata.get("entities"))
    )
    memory_data = {
        "memory_id": memory_id,
        "content": document,
        "memory_type": MemoryType(metadata["memory_type"]),
        "timestamp": metadata["timestamp"],
        "last_accessed": metadata.get("last_accessed", metadata["timestamp"]),
        "access_count": int(metadata.get("access_count") or 0),
        "access_history": json.loads(metadata.get("access_history") or "[]"),
        "importance": float(metadata["importance"]),
        "confidence": float(metadata["confidence"]),
        "tags": metadata["tags"].split(",") if metadata["tags"] else [],
        "emotions": metadata["emotions"].split(",") if metadata["emotions"] else [],
        "source": metadata["source"] if metadata["source"] else None,
        "context": safe_load_context(metadata.get("context", "")),
        "embedding": embedding,
        "entities": entities,
        "associations": metadata.get("associations", "").split(",")
        if metadata.get("associations")
        else [],
        "linked_concepts": metadata.get("linked_concepts", "").split(",")
        if metadata.get("linked_concepts")
        else [],
        "conflicts_with": metadata.get("conflicts_with", "").split(",")
        if metadata.get("conflicts_with")
        else [],
        "previous_event_id": metadata.get("previous_event_id") or None,
        "next_event_id": metadata.get("next_event_id") or None,
    }
    return MemoryItem(**memory_data)


def sqlite_record_from_memory(memory: MemoryItem) -> dict[str, Any]:
    return {
        "memory_id": memory.memory_id,
        "conversation_id": memory.conversation_id,
        "user_id": memory.user_id,
        "content": memory.content,
        "summary": memory.summary,
        "author": memory.author,
        "memory_type": memory.memory_type.value,
        "timestamp": serialize_datetime(memory.timestamp),
        "last_accessed": serialize_datetime(memory.last_accessed),
        "access_count": memory.access_count,
        "access_history": json.dumps(serialize_datetime_list(list(memory.access_history))),
        "importance": float(memory.importance),
        "decay_rate": memory.decay_rate,
        "confidence": memory.confidence,
        "fidelity": memory.fidelity,
        "emotional_valence": memory.emotional_valence,
        "emotional_arousal": memory.emotional_arousal,
        "emotions": _serialize_list(memory.emotions) if memory.emotions else "",
        "tags": _serialize_list(memory.tags) if memory.tags else "",
        "source": memory.source,
        "credibility": memory.credibility,
        "context": json_dumps_safe(memory.context) if memory.context else "",
        "metadata": json_dumps_safe(memory.metadata) if memory.metadata else "",
        "embedding": json.dumps(memory.embedding) if memory.embedding is not None else None,
        "entities": json_dumps_safe(memory.entities) if memory.entities else "{}",
        "entity_mentions": json_dumps_safe(memory.entity_mentions)
        if memory.entity_mentions
        else "{}",
        "associations": _serialize_list(memory.associations) if memory.associations else "",
        "linked_concepts": _serialize_list(memory.linked_concepts)
        if memory.linked_concepts
        else "",
        "conflicts_with": _serialize_list(memory.conflicts_with) if memory.conflicts_with else "",
        "previous_event_id": memory.previous_event_id,
        "next_event_id": memory.next_event_id,
    }


def sqlite_memory_from_row(row: sqlite3.Row) -> MemoryItem:
    embedding = None
    if row["embedding"]:
        try:
            embedding = json.loads(row["embedding"])
        except (json.JSONDecodeError, TypeError, ValueError):
            embedding = None

    memory_data = {
        "memory_id": row["memory_id"],
        "conversation_id": row["conversation_id"],
        "user_id": row["user_id"],
        "content": row["content"],
        "summary": row["summary"],
        "author": row["author"],
        "memory_type": MemoryType(row["memory_type"]),
        "timestamp": row["timestamp"],
        "last_accessed": row["last_accessed"] or row["timestamp"],
        "access_count": int(row["access_count"] or 0),
        "access_history": json.loads(row["access_history"] or "[]"),
        "importance": float(row["importance"]),
        "decay_rate": row["decay_rate"] if row["decay_rate"] is not None else 0.01,
        "confidence": float(row["confidence"] or 1.0),
        "fidelity": float(row["fidelity"] or 1.0),
        "emotional_valence": row["emotional_valence"],
        "emotional_arousal": row["emotional_arousal"],
        "emotions": _deserialize_list(row["emotions"]),
        "tags": _deserialize_list(row["tags"]),
        "source": row["source"],
        "credibility": row["credibility"],
        "context": safe_load_context(row["context"]),
        "metadata": json_loads_dict(row["metadata"]),
        "embedding": embedding,
        "entities": json_loads_dict(row["entities"]),
        "entity_mentions": json_loads_dict(row["entity_mentions"]),
        "associations": _deserialize_list(row["associations"]),
        "linked_concepts": _deserialize_list(row["linked_concepts"]),
        "conflicts_with": _deserialize_list(row["conflicts_with"]),
        "previous_event_id": row["previous_event_id"] or None,
        "next_event_id": row["next_event_id"] or None,
    }
    return MemoryItem(**memory_data)


def _load_json_value(value: str | None, *, strict: bool) -> Any:
    if strict:
        if value is None:
            value = "{}"
        return json.loads(value)
    return json_loads_dict(value)


def _serialize_list(values: Iterable[str]) -> str:
    return ",".join(values)


def _deserialize_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(",") if item]
