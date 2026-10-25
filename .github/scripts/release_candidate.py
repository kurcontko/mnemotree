"""Resolve a main-branch release and create its annotated tag and draft."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from release_prepare import version_key


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def api(endpoint: str, payload: dict | None = None):
    command = ["gh", "api", endpoint]
    if payload is not None:
        command += ["--input", "-"]
    return json.loads(
        subprocess.check_output(command, input=json.dumps(payload) if payload else None, text=True)
    )


def notes(version: str) -> str:
    changelog = Path("CHANGELOG.md").read_text()
    marker = f"## [{version}]"
    if changelog.count(marker) != 1:
        raise ValueError("Release must have exactly one dated changelog section")
    section = changelog.split(marker, 1)[1].split("\n## [", 1)[0]
    if not section.startswith(" — "):
        raise ValueError("Release changelog needs a date")
    return section.split("\n", 1)[1].split("\n[Unreleased]:", 1)[0].strip()


def resolve() -> None:
    if os.environ["GITHUB_REF"] != "refs/heads/main":
        raise ValueError("Release workflows must run from main")
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    tag = os.environ.get("RELEASE_TAG", "")
    automatic = not tag
    if not tag:
        before = os.environ.get("BEFORE_SHA", "")
        old = tomllib.loads(git("show", f"{before}:pyproject.toml"))["project"]["version"]
        if old == version:
            with open(os.environ["GITHUB_OUTPUT"], "a") as out:
                out.write("enabled=false\n")
            return
        if version_key(version) <= version_key(old):
            raise ValueError("Version must increase")
        tag = f"v{version}"
    if not tag.startswith("v"):
        raise ValueError("Release tags must begin with v")
    version_key(tag[1:])
    tags = git("tag", "--list", tag).splitlines()
    source = git("rev-parse", f"{tag}^{{commit}}") if tags else git("rev-parse", "HEAD")
    if automatic and source != git("rev-parse", "HEAD"):
        raise ValueError("Existing release tag does not identify the version-bump commit")
    git("merge-base", "--is-ancestor", source, "HEAD")
    tagged = tomllib.loads(git("show", f"{source}:pyproject.toml"))["project"]["version"]
    if tagged != tag[1:]:
        raise ValueError("Tag and source version disagree")
    with open(os.environ["GITHUB_OUTPUT"], "a") as out:
        out.write(f"enabled=true\ntag={tag}\nsource={source}\n")


def draft() -> None:
    tag, source = os.environ["RELEASE_TAG"], os.environ["SOURCE_SHA"]
    version_key(tag[1:])
    body = notes(tag[1:])
    repo = f"repos/{os.environ['GITHUB_REPOSITORY']}"
    # Checkout fetched all tags; never change an existing tag's target.
    if git("tag", "--list", tag):
        if git("rev-parse", f"{tag}^{{commit}}") != source:
            raise ValueError("Existing tag points elsewhere")
    else:
        obj = api(
            f"{repo}/git/tags",
            {"tag": tag, "message": f"Mnemotree {tag}", "object": source, "type": "commit"},
        )
        api(f"{repo}/git/refs", {"ref": f"refs/tags/{tag}", "sha": obj["sha"]})
    releases = json.loads(
        subprocess.check_output(
            ["gh", "api", "--paginate", "--slurp", f"{repo}/releases?per_page=100"], text=True
        )
    )
    existing = next((r for page in releases for r in page if r["tag_name"] == tag), None)
    if existing:
        if not existing["draft"]:
            raise ValueError("Release is already public; use publication recovery")
        release = existing
    else:
        release = api(
            f"{repo}/releases",
            {
                "tag_name": tag,
                "target_commitish": source,
                "name": f"Mnemotree {tag}",
                "body": body,
                "draft": True,
                "prerelease": version_key(tag[1:])[3] < 3,
            },
        )
    assets = [
        *sorted(Path("release/dist").iterdir()),
        Path("release/constraints.txt"),
        Path("release/manifest.json"),
        Path("release/SHA256SUMS"),
    ]
    attached = {asset["name"]: asset for asset in release["assets"]}
    if attached.keys() - {path.name for path in assets}:
        raise ValueError("Draft contains unexpected assets; reconcile it deliberately")
    missing = []
    for path in assets:
        if path.name not in attached:
            missing.append(str(path))
            continue
        remote = subprocess.check_output(
            [
                "gh",
                "api",
                "-H",
                "Accept: application/octet-stream",
                f"{repo}/releases/assets/{attached[path.name]['id']}",
            ]
        )
        if hashlib.sha256(remote).digest() != hashlib.sha256(path.read_bytes()).digest():
            raise ValueError("Draft contains a different bundle; resume its original candidate run")
    if missing:
        subprocess.run(["gh", "release", "upload", tag, *missing], check=True)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as out:
        out.write(f"Release candidate: {release['html_url']}\n\nSource: `{source}`\n\n")
        out.write(Path("release/SHA256SUMS").read_text())


if __name__ == "__main__":
    {"resolve": resolve, "draft": draft}[sys.argv[1]]()
