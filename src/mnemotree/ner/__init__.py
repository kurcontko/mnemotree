from __future__ import annotations

from .base import BaseNER, NERResult
from .composite import CompositeNER
from .gliner import GLiNERNER
from .hf_transformers import TransformersNER
from .llm import LangchainLLMNER
from .spacy import SpacyNER
from .spark_nlp import SparkNLPNER
from .stanza import StanzaNER


def create_ner(backend: str, **kwargs: object) -> BaseNER:
    """
    Create an NER backend by name.

    Examples:
        - `create_ner("spacy", model="en_core_web_sm")`
        - `create_ner("transformers", model="dslim/distilbert-NER")`
        - `create_ner("gliner", model_name="urchade/gliner_medium-v2.1")`
    """
    key = backend.strip().lower()
    if key in {"spacy"}:
        return SpacyNER(**kwargs)  # type: ignore[arg-type]
    if key in {"gliner"}:
        return GLiNERNER(**kwargs)  # type: ignore[arg-type]
    if key in {"transformers", "hf", "huggingface"}:
        return TransformersNER(**kwargs)  # type: ignore[arg-type]
    if key in {"stanza"}:
        return StanzaNER(**kwargs)  # type: ignore[arg-type]
    if key in {"spark", "spark-nlp", "sparknlp"}:
        return SparkNLPNER(**kwargs)  # type: ignore[arg-type]
    if key in {"llm", "langchain"}:
        return LangchainLLMNER(**kwargs)
    raise ValueError(f"Unknown NER backend: {backend!r}")


__all__ = [
    "BaseNER",
    "NERResult",
    "SpacyNER",
    "GLiNERNER",
    "TransformersNER",
    "StanzaNER",
    "SparkNLPNER",
    "LangchainLLMNER",
    "CompositeNER",
    "create_ner",
]
