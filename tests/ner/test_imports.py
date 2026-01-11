import importlib


def test_ner_subpackage_imports_without_optional_deps() -> None:
    ner = importlib.import_module("mnemotree.ner")
    assert hasattr(ner, "SpacyNER")
    assert hasattr(ner, "GLiNERNER")
    assert hasattr(ner, "TransformersNER")
    assert hasattr(ner, "StanzaNER")
    assert hasattr(ner, "SparkNLPNER")


def test_create_ner_unknown_backend_raises() -> None:
    ner = importlib.import_module("mnemotree.ner")
    try:
        create_ner = ner.create_ner
    except AttributeError as err:
        raise AssertionError("mnemotree.ner.create_ner is missing") from err
    try:
        create_ner("does-not-exist")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown backend")
