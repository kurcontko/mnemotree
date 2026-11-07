import importlib.machinery
import sys
import threading
import types
from pathlib import Path

import pytest

TOY_MODEL = {
    "model": "toy/embedder",
    "dim": 4,
    "model_file": "onnx/model.onnx",
    "additional_files": ["onnx/model.onnx_data"],
}
TOY_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "onnx/model.onnx",
    "onnx/model.onnx_data",
]


class FakeFastEmbed:
    """Stand-ins for fastembed and huggingface_hub that record what they are asked."""

    def __init__(self) -> None:
        self.downloads: list[dict] = []
        self.loads: list[dict] = []
        self.registry = [dict(TOY_MODEL)]
        self.embedding_size = TOY_MODEL["dim"]
        self.vector_size = TOY_MODEL["dim"]
        self.download_error: Exception | None = None
        # Cleared to hold a load in progress; set by the test to let it finish.
        self.release = threading.Event()
        self.release.set()

    @staticmethod
    def settings(cache_dir) -> dict:
        """An embedding config for the toy model, as the MCP server and CLI take it."""
        return {
            "kind": "fastembed",
            "model": TOY_MODEL["model"],
            "source": "toy/embedder-onnx",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "dimensions": TOY_MODEL["dim"],
            "cache_dir": str(cache_dir),
        }

    def snapshot_download(self, **options):
        self.downloads.append(options)
        path = Path(options["cache_dir"]) / "snapshots" / options["revision"]
        if options.get("local_files_only"):
            if not path.is_dir():
                raise FileNotFoundError("not cached")  # as LocalEntryNotFoundError is
            return str(path)
        if self.download_error is not None:
            raise self.download_error
        for name in options["allow_patterns"]:
            (path / name).parent.mkdir(parents=True, exist_ok=True)
            (path / name).write_text("weights")
        return str(path)

    def text_embedding(self):
        fake = self

        class TextEmbedding:
            @staticmethod
            def list_supported_models():
                return fake.registry

            def __init__(self, model_name, *, cache_dir=None, specific_model_path=None):
                fake.release.wait(10)
                fake.loads.append(
                    {"model": model_name, "cache_dir": cache_dir, "path": specific_model_path}
                )
                self.embedding_size = fake.embedding_size

            def embed(self, documents, batch_size=256):
                for text in documents:
                    # Unnormalized, like the real model: the embedder must normalize.
                    yield [float(len(text)), 3.0] + [0.0] * (fake.vector_size - 2)

        return TextEmbedding


@pytest.fixture
def fake_fastembed(monkeypatch):
    fake = FakeFastEmbed()
    fastembed = types.ModuleType("fastembed")
    fastembed.__spec__ = importlib.machinery.ModuleSpec("fastembed", None)
    fastembed.TextEmbedding = fake.text_embedding()
    hub = types.ModuleType("huggingface_hub")
    hub.__spec__ = importlib.machinery.ModuleSpec("huggingface_hub", None)
    hub.snapshot_download = fake.snapshot_download
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return fake
