#!/usr/bin/env bash
# Build the sdist and wheel, then install the wheel into clean virtualenvs
# outside the checkout and smoke-test it — bare, and with the mcp extra.
#
# Installs are constrained to uv.lock so a red run means the package broke,
# not that some transitive dependency released overnight; the weekly
# dependency-health workflow covers fresh resolutions.
set -euo pipefail
# Build-tool pins. release_bundle.py records these in a release's manifest,
# so they are plain assignments it can read.
HATCHLING_VERSION="1.32.4"
TWINE_VERSION="7.0.0"
repo=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# The build backend is pinned too: a new hatchling can change the metadata
# version it emits, and twine rejects versions it does not know yet.
echo "hatchling==${HATCHLING_VERSION}" > "$work/build-constraints.txt"
rm -rf dist
uv build --out-dir dist --build-constraint "$work/build-constraints.txt"
# uv drops a .gitignore into the output directory; the release bundle must
# hold exactly one wheel and one sdist.
rm -f dist/.gitignore
uvx --no-build --from "twine==${TWINE_VERSION}" twine check --strict dist/*
uv export --frozen --all-extras --no-emit-project --no-hashes --no-header --no-annotate \
  > "$work/constraints.txt"
wheel=$(find "$repo/dist" -maxdepth 1 -name '*.whl')
for variant in bare mcp; do
  uv venv --quiet "$work/$variant"
  requirement="$wheel"
  if [[ "$variant" == mcp ]]; then requirement="${wheel}[mcp]"; fi
  # --no-build: dependencies install from wheels only, so no third-party
  # setup script runs here. Our own wheel is already built.
  uv pip install --quiet --no-build --python "$work/$variant/bin/python" \
    -c "$work/constraints.txt" "$requirement"
  (
    cd "$work"
    "$work/$variant/bin/python" -I "$repo/.github/scripts/package_smoke.py" "--$variant"
  )
done
