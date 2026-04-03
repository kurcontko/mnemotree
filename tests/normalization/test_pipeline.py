"""Tests for NormalizationPipeline and factory."""

from mnemotree.normalization import create_normalization_pipeline
from mnemotree.normalization.base import BaseNormalizer
from mnemotree.normalization.pipeline import NormalizationPipeline


class UpperNormalizer(BaseNormalizer):
    """Test normalizer that uppercases content."""

    async def normalize(self, content, context=None):
        return content.upper()


class SuffixNormalizer(BaseNormalizer):
    """Test normalizer that appends a suffix."""

    async def normalize(self, content, context=None):
        return content + " [normalized]"


async def test_pipeline_chains_normalizers():
    """Pipeline should apply normalizers in sequence."""
    pipeline = NormalizationPipeline([UpperNormalizer(), SuffixNormalizer()])
    result = await pipeline.normalize("hello world")
    assert result == "HELLO WORLD [normalized]"


async def test_pipeline_order_matters():
    """Order of normalizers should affect output."""
    pipeline = NormalizationPipeline([SuffixNormalizer(), UpperNormalizer()])
    result = await pipeline.normalize("hello world")
    assert result == "HELLO WORLD [NORMALIZED]"


async def test_empty_pipeline():
    """Empty pipeline should return content unchanged."""
    pipeline = NormalizationPipeline([])
    result = await pipeline.normalize("hello world")
    assert result == "hello world"


async def test_pipeline_passes_context():
    """Context should be passed to each normalizer."""

    class ContextCapture(BaseNormalizer):
        captured_context = None

        async def normalize(self, content, context=None):
            ContextCapture.captured_context = context
            return content

    pipeline = NormalizationPipeline([ContextCapture()])
    ctx = {"speaker": "Alice"}
    await pipeline.normalize("test", ctx)
    assert ContextCapture.captured_context == ctx


def test_pipeline_bool_empty():
    """Empty pipeline should be falsy."""
    assert not NormalizationPipeline([])


def test_pipeline_bool_nonempty():
    """Non-empty pipeline should be truthy."""
    assert NormalizationPipeline([UpperNormalizer()])


def test_pipeline_len():
    """len() should return number of normalizers."""
    assert len(NormalizationPipeline([])) == 0
    assert len(NormalizationPipeline([UpperNormalizer()])) == 1
    assert len(NormalizationPipeline([UpperNormalizer(), SuffixNormalizer()])) == 2


# --- Factory tests ---


def test_factory_nothing_enabled():
    """Factory returns None when nothing is enabled."""
    result = create_normalization_pipeline()
    assert result is None


def test_factory_coref_only():
    """Factory creates pipeline with coref only."""
    result = create_normalization_pipeline(enable_coref=True)
    assert result is not None
    assert len(result) == 1


def test_factory_temporal_only():
    """Factory creates pipeline with temporal only."""
    result = create_normalization_pipeline(enable_temporal=True)
    assert result is not None
    assert len(result) == 1


def test_factory_both_enabled():
    """Factory creates pipeline with both normalizers."""
    result = create_normalization_pipeline(enable_coref=True, enable_temporal=True)
    assert result is not None
    assert len(result) == 2


# --- Error handling tests ---


class FailingNormalizer(BaseNormalizer):
    """Normalizer that always raises."""

    async def normalize(self, content, context=None):
        raise RuntimeError("normalizer crashed")


async def test_pipeline_skips_failing_normalizer():
    """A failing normalizer should be skipped, not crash the pipeline."""
    pipeline = NormalizationPipeline([UpperNormalizer(), FailingNormalizer(), SuffixNormalizer()])
    result = await pipeline.normalize("hello")
    # UpperNormalizer runs, FailingNormalizer is skipped, SuffixNormalizer runs on uppercase result
    assert result == "HELLO [normalized]"


async def test_pipeline_all_failing_returns_original():
    """If all normalizers fail, return the original content unchanged."""
    pipeline = NormalizationPipeline([FailingNormalizer(), FailingNormalizer()])
    result = await pipeline.normalize("untouched")
    assert result == "untouched"
