import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mnemotree import ConfigurationError, IngestConflict, InvalidInput, MnemotreeError
from mnemotree.config import (
    Config,
    OpenAIEmbeddingConfig,
    OpenAIExtractionConfig,
    SentenceTransformerEmbeddingConfig,
)
from mnemotree.evidence import Memory, Message, SearchOptions
from mnemotree.evidence import cli as evidence_cli
from mnemotree.evidence.models import EmbeddingSpec

ROOT = Path(__file__).resolve().parents[1]


class CharCounter:
    """Offline token counter: one token per character."""

    signature = "chars"

    def count(self, text: str) -> int:
        return len(text)


@pytest.mark.parametrize(
    "path",
    [
        *sorted((ROOT / "examples" / "evidence").glob("*.json")),
        ROOT / "docker/aml/config.example.json",
    ],
    ids=lambda path: path.name,
)
def test_shipped_configs_parse(path):
    Config.model_validate_json(path.read_text())


def test_embedding_kind_selects_the_provider_config():
    config = Config.model_validate(
        {
            "embedding": {"kind": "sentence-transformers", "model": "m", "revision": "r"},
            "extraction": {"kind": "openai", "model": "m", "revision": "r"},
        }
    )
    assert isinstance(config.embedding, SentenceTransformerEmbeddingConfig)
    assert isinstance(config.extraction, OpenAIExtractionConfig)
    assert config.extraction.api_key_env == "EXTRACTION_API_KEY"


@pytest.mark.parametrize(
    ("section", "body", "message"),
    [
        ("embedding", {"kind": "nope", "model": "m"}, "does not match any of the expected tags"),
        ("embedding", {"kind": "sentence-transformers", "model": "m"}, "revision"),
        (
            "embedding",
            {"kind": "sentence-transformers", "model": "m", "revision": "r", "x": 1},
            "x",
        ),
        ("extraction", {"kind": "anthropic", "model": "m", "revision": "r"}, "'openai'"),
        (
            "embedding",
            {"kind": "openai", "model": "m", "dimensions": 2, "revision": "r", "api_key": "k"},
            "use api_key_env instead",
        ),
        (
            "extraction",
            {"kind": "openai", "model": "m", "revision": "r", "api_key": "k"},
            "api_key_env",
        ),
    ],
)
def test_bad_provider_settings_fail_when_the_config_loads(section, body, message):
    with pytest.raises(ValidationError, match=message):
        Config.model_validate({section: body})


def test_omitted_settings_keep_the_provider_defaults(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    config = OpenAIEmbeddingConfig(
        kind="openai", model="m", dimensions=4, revision="r", api_key_env="MY_KEY", batch_size=3
    )
    embedder = config.build()
    assert embedder.spec == EmbeddingSpec(
        provider="https://api.openai.com/v1", model="m", revision="r", dimensions=4
    )
    assert (embedder.api_key, embedder.batch_size, embedder.timeout) == ("secret", 3, 60.0)


def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    config = OpenAIEmbeddingConfig(kind="openai", model="m", dimensions=4, revision="r")
    with pytest.raises(SystemExit, match="EMBEDDING_API_KEY"):
        config.build()


def test_cli_module_still_exports_the_config_api():
    from mnemotree import config

    assert evidence_cli.Config is config.Config
    assert evidence_cli.open_memory is config.open_memory
    assert evidence_cli.secret is config.secret


def test_errors_are_value_errors():
    for error in (InvalidInput, ConfigurationError, IngestConflict):
        assert issubclass(error, MnemotreeError)
        assert issubclass(error, ValueError)


async def test_invalid_arguments_raise_invalid_input(tmp_path):
    with Memory(tmp_path / "m.db", token_counter=CharCounter()) as memory:
        with pytest.raises(InvalidInput, match="query must not be blank"):
            await memory.search("u", "  ")
        with pytest.raises(InvalidInput, match="namespace"):
            await memory.search(" ", "q")
        with pytest.raises(InvalidInput, match="session_id"):
            await memory.add("u", [Message(content="x")], request_id="r", session_id="")
        with pytest.raises(InvalidInput, match="messages"):
            await memory.add("u", [], request_id="r", session_id="s")


async def test_unservable_requests_raise_configuration_error(tmp_path):
    with Memory(tmp_path / "m.db", token_counter=CharCounter()) as memory:
        with pytest.raises(ConfigurationError, match="indexed embedder"):
            await memory.search("u", "q", options=SearchOptions(mode="dense"))
        with pytest.raises(ConfigurationError, match="extracted corpus"):
            await memory.search(
                "u", "q", options=SearchOptions(mode="lexical", representation="atoms")
            )
    with pytest.raises(ConfigurationError, match="chunk_size"):
        Memory(tmp_path / "m.db", chunk_size=1, chunk_overlap=1)


def test_config_round_trips_through_json():
    data = {"embedding": {"kind": "openai", "model": "m", "dimensions": 2, "revision": "r"}}
    config = Config.model_validate_json(json.dumps(data))
    assert Config.model_validate_json(config.model_dump_json()) == config
