"""Validate distributions and bind their digests to the tested source and run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from release_prepare import PACKAGE, version_key

CHECK_PACKAGE = Path(__file__).with_name("check_package.sh")
CONSTRAINTS = "constraints.txt"


def build_tools() -> dict[str, str]:
    """The pins check_package.sh built with, plus the uv that drove it."""
    script = CHECK_PACKAGE.read_text()
    tools = {}
    for name in ("hatchling", "twine"):
        match = re.search(rf'(?m)^{name.upper()}_VERSION="([^"]+)"$', script)
        if not match:
            raise ValueError(f"check_package.sh no longer pins {name}")
        tools[name] = match[1]
    tools["uv"] = subprocess.check_output(["uv", "--version"], text=True).split()[1]
    return tools


def read_metadata(path: Path, version: str) -> bytes:
    """The core metadata file of one wheel or sdist, refusing anything unexpected."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Distribution must be a regular file")
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if names != [f"{PACKAGE}-{version}.dist-info/METADATA"]:
                raise ValueError("Unexpected wheel metadata")
            return archive.read(names[0])
    with tarfile.open(path) as archive:
        member = archive.getmember(f"{PACKAGE}-{version}/PKG-INFO")
        stream = archive.extractfile(member) if member.isfile() else None
        if stream is None:
            raise ValueError("Invalid sdist metadata")
        return stream.read()


def distributions(directory: Path, version: str) -> dict[str, str]:
    version_key(version)
    expected = {f"{PACKAGE}-{version}-py3-none-any.whl", f"{PACKAGE}-{version}.tar.gz"}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError(
            "Expected exactly one universal wheel and one sdist with the release version"
        )
    digests = {}
    for path in sorted(directory.iterdir()):
        parsed = BytesParser().parsebytes(read_metadata(path, version))
        if parsed["Name"] != PACKAGE or parsed["Version"] != version:
            raise ValueError("Distribution metadata disagrees with release version")
        digests[path.name] = digest(path)
    return digests


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bundle: Path, *, tag: str, source: str, run_id: str) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    identity = (
        manifest.get("schema"),
        manifest.get("tag"),
        manifest.get("source_sha"),
        manifest.get("run_id"),
    )
    if identity != (1, tag, source, run_id):
        raise ValueError("Bundle identity does not match the selected release")
    if not re.fullmatch(r"[0-9a-f]{40}", source) or not tag.startswith("v"):
        raise ValueError("Invalid release identity")
    if manifest.get("files") != distributions(bundle / "dist", tag[1:]):
        raise ValueError("Bundle digests do not match")
    if manifest.get("constraints") != digest(bundle / CONSTRAINTS):
        raise ValueError(f"Bundled {CONSTRAINTS} does not match the manifest")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["create", "verify"])
    parser.add_argument("--bundle", type=Path, default=Path("release"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    args = parser.parse_args()
    if args.command == "create":
        manifest = {
            "schema": 1,
            "tag": args.tag,
            "source_sha": args.source,
            "run_id": args.run_id,
            "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
            "repository": os.environ["GITHUB_REPOSITORY"],
            "workflow_ref": os.environ["GITHUB_WORKFLOW_REF"],
            "workflow_sha": os.environ["GITHUB_WORKFLOW_SHA"],
            "files": distributions(args.bundle / "dist", args.tag[1:]),
            "constraints": digest(args.bundle / CONSTRAINTS),
            "build_tools": build_tools(),
        }
        (args.bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        sums = {**manifest["files"], CONSTRAINTS: manifest["constraints"]}
        (args.bundle / "SHA256SUMS").write_text(
            "".join(f"{value}  {name}\n" for name, value in sums.items())
        )
    verify(args.bundle, tag=args.tag, source=args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
