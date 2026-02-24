"""Tests for CoreferenceNormalizer."""

import pytest

from mnemotree.normalization.coref import CoreferenceNormalizer


@pytest.fixture
def normalizer():
    return CoreferenceNormalizer(backend="heuristic")


async def test_first_person_replacement_with_context(normalizer):
    """First-person pronouns should be replaced with speaker name."""
    content = "[1:00 pm on 8 May, 2023] Alice: I love my new car and it makes me happy"
    context = {"speaker": "Alice", "other_speaker": "Bob"}

    result = await normalizer.normalize(content, context)

    assert "Alice love Alice's new car" in result
    assert "makes Alice happy" in result


async def test_first_person_parsed_from_content(normalizer):
    """Speaker should be parsed from LoCoMo format when not in context."""
    content = "[1:00 pm on 8 May, 2023] Alice: I went to the store"

    result = await normalizer.normalize(content)

    assert "Alice went to the store" in result


async def test_third_person_replacement_unambiguous(normalizer):
    """Third-person pronouns replaced when single other_speaker and no ambiguity."""
    content = "[1:00 pm on 8 May, 2023] Alice: He went to the park"
    context = {
        "speaker": "Alice",
        "other_speaker": "Bob",
        "recent_entities": [],
    }

    result = await normalizer.normalize(content, context)

    assert "Bob went to the park" in result


async def test_third_person_skipped_when_ambiguous(normalizer):
    """Third-person pronouns NOT replaced when multiple candidates exist."""
    content = "[1:00 pm on 8 May, 2023] Alice: He went to the park"
    context = {
        "speaker": "Alice",
        "other_speaker": "Bob",
        "recent_entities": ["Charlie"],  # extra person creates ambiguity
    }

    result = await normalizer.normalize(content, context)

    # Should NOT replace "He" because there are multiple candidates
    assert "He went to the park" in result


async def test_no_context_no_change(normalizer):
    """Without context, content should be unchanged (except parsed speaker)."""
    content = "Just a plain string without speaker format"

    result = await normalizer.normalize(content)

    assert result == content


async def test_possessive_his_replacement(normalizer):
    """Possessive 'his' should become 'Name's'."""
    content = "[1:00 pm on 8 May, 2023] Alice: His dog is cute"
    context = {
        "speaker": "Alice",
        "other_speaker": "Bob",
        "recent_entities": [],
    }

    result = await normalizer.normalize(content, context)

    assert "Bob's dog is cute" in result


async def test_speaker_with_spaces(normalizer):
    """Speaker names with spaces should work."""
    content = "[1:00 pm on 8 May, 2023] John Smith: I like my job"
    context = {"speaker": "John Smith"}

    result = await normalizer.normalize(content, context)

    assert "John Smith like John Smith's job" in result


async def test_empty_content(normalizer):
    """Empty content should return empty."""
    result = await normalizer.normalize("")
    assert result == ""
