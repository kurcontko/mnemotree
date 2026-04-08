"""Tests for TemporalNormalizer."""

from datetime import datetime

import pytest

from mnemotree.normalization.temporal import TemporalNormalizer, _parse_reference_date


@pytest.fixture
def normalizer():
    return TemporalNormalizer()


def test_parse_reference_date_from_context():
    """Reference date should be extracted from context."""
    ctx = {"reference_date": datetime(2023, 5, 8, 13, 0)}
    result = _parse_reference_date("some content", ctx)
    assert result == datetime(2023, 5, 8, 13, 0)


def test_parse_reference_date_from_content():
    """Reference date should be parsed from LoCoMo-style prefix."""
    content = "[1:56 pm on 8 May, 2023] Alice: hello"
    result = _parse_reference_date(content)
    assert result is not None
    assert result.year == 2023
    assert result.month == 5
    assert result.day == 8


def test_parse_reference_date_no_date():
    """No reference date returns None."""
    result = _parse_reference_date("plain text", None)
    assert result is None


async def test_no_reference_date_no_change(normalizer):
    """Without reference date, content should be unchanged."""
    content = "I'll do it tomorrow"
    result = await normalizer.normalize(content)
    assert result == content


async def test_relative_expression_annotated(normalizer):
    """Relative expressions should get ISO date annotations when dateparser available."""
    content = "[1:56 pm on 8 May, 2023] Alice: I'll see you tomorrow"
    context = {"reference_date": datetime(2023, 5, 8, 13, 56)}

    result = await normalizer.normalize(content, context)

    # If dateparser is installed, "tomorrow" should be annotated
    try:
        import dateparser  # noqa: F401

        assert "2023-05-09" in result
    except ImportError:
        # Without dateparser, content unchanged
        assert result == content


async def test_yesterday_annotated(normalizer):
    """'yesterday' should be annotated with correct date."""
    context = {"reference_date": datetime(2023, 5, 8)}
    content = "[1:00 pm on 8 May, 2023] Bob: I did it yesterday"

    result = await normalizer.normalize(content, context)

    try:
        import dateparser  # noqa: F401

        assert "2023-05-07" in result
    except ImportError:
        assert result == content


async def test_n_days_ago_annotated(normalizer):
    """'3 days ago' should be annotated."""
    context = {"reference_date": datetime(2023, 5, 8)}
    content = "[1:00 pm on 8 May, 2023] Alice: We met 3 days ago"

    result = await normalizer.normalize(content, context)

    try:
        import dateparser  # noqa: F401

        assert "2023-05-05" in result
    except ImportError:
        assert result == content


async def test_no_temporal_expressions_unchanged(normalizer):
    """Content without temporal expressions should be unchanged."""
    content = "[1:00 pm on 8 May, 2023] Alice: I love pizza"
    context = {"reference_date": datetime(2023, 5, 8)}

    result = await normalizer.normalize(content, context)
    assert result == content


async def test_empty_content(normalizer):
    """Empty content should return empty."""
    result = await normalizer.normalize("")
    assert result == ""


def test_parse_date_string_malformed():
    """Malformed date strings should return None, not crash."""
    from mnemotree.normalization.temporal import _parse_date_string

    # These should all return None gracefully
    assert _parse_date_string("not-a-date-at-all!!!") is None
    assert _parse_date_string("") is None
    assert _parse_date_string("   ") is None
