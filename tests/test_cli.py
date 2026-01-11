"""Tests for the CLI entry point."""

import importlib.metadata

from mnemotree import cli


def test_cli_help_prints_usage(capsys):
    result = cli.main([])
    captured = capsys.readouterr()
    assert "usage: mnemotree" in captured.out
    assert result == 0


def test_cli_version_success(monkeypatch, capsys):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.2.3")
    result = cli.main(["--version"])
    captured = capsys.readouterr()
    assert captured.out.strip() == "1.2.3"
    assert result == 0


def test_cli_version_package_not_found(monkeypatch, capsys):
    def _raise_missing(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError("mnemotree")

    monkeypatch.setattr(importlib.metadata, "version", _raise_missing)
    result = cli.main(["--version"])
    captured = capsys.readouterr()
    assert captured.out.strip() == "mnemotree"
    assert result == 0
