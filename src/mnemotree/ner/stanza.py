from __future__ import annotations

import asyncio
from typing import Any

from .base import BaseNER, NERResult


class StanzaNER(BaseNER):
    """Stanza NER backend (multilingual, reasonably lightweight)."""

    def __init__(
        self,
        *,
        lang: str = "en",
        processors: str = "tokenize,ner",
        use_gpu: bool = False,
        pipeline_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            lang: Language code (e.g. "en").
            processors: Stanza processors string, must include "ner".
            use_gpu: Whether to attempt GPU usage.
            pipeline_kwargs: Extra kwargs passed to `stanza.Pipeline`.
        """
        try:
            import stanza
        except ImportError as exc:
            raise ImportError(
                "StanzaNER requires optional dependencies. "
                "Install with: pip install 'mnemotree[ner_stanza]'"
            ) from exc

        kwargs: dict[str, Any] = dict(pipeline_kwargs or {})
        if "verbose" not in kwargs:
            kwargs["verbose"] = False

        self._nlp = stanza.Pipeline(
            lang=lang,
            processors=processors,
            use_gpu=use_gpu,
            **kwargs,
        )

    async def extract_entities(self, text: str) -> NERResult:
        doc = await asyncio.to_thread(self._nlp, text)

        entities: dict[str, str] = {}
        mentions: dict[str, list[str]] = {}

        for ent in getattr(doc, "ents", []):
            entity_text = getattr(ent, "text", "").strip()
            if not entity_text:
                continue

            entity_type = getattr(ent, "type", None) or getattr(ent, "label", None) or "ENTITY"
            start = getattr(ent, "start_char", None)
            end = getattr(ent, "end_char", None)

            entities[entity_text] = str(entity_type)
            if isinstance(start, int) and isinstance(end, int):
                context = self._get_context(text, start, end)
                mentions.setdefault(entity_text, []).append(context)

        return NERResult(
            entities=entities,
            mentions=mentions,
        )
