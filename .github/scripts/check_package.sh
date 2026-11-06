#!/usr/bin/env bash
# Build the sdists and wheels of mnemotree and mnemotree-mcp, then install the
# wheels into clean virtualenvs outside the checkout and smoke-test them:
# mnemotree bare and with the serve extra, and mnemotree-mcp on top of the
# freshly built mnemotree wheel.
#
# Installs are constrained to uv.lock so a red run means the package broke,
# not that some transitive dependency released overnight; the weekly
# dependency-health workflow covers fresh resolutions.
set -euo pipefail
repo=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# The build backend is pinned too: a new hatchling can change the metadata
# version it emits, and twine rejects versions it does not know yet.
echo "hatchling==1.32.4" > "$work/build-constraints.txt"
rm -rf dist
uv build --all-packages --out-dir dist --build-constraint "$work/build-constraints.txt"
uvx --no-build --from "twine==7.0.0" twine check --strict dist/*
# Every workspace member's dependencies, fastembed's included.
uv export --frozen --all-packages --all-extras --no-emit-workspace --no-hashes --no-header \
  --no-annotate > "$work/constraints.txt"
wheel=$(find "$repo/dist" -maxdepth 1 -name 'mnemotree-[0-9]*.whl')
mcp_wheel=$(find "$repo/dist" -maxdepth 1 -name 'mnemotree_mcp-[0-9]*.whl')
for variant in bare serve mcp; do
  uv venv --quiet "$work/$variant"
  requirements=("$wheel")
  if [[ "$variant" == serve ]]; then requirements=("${wheel}[serve]"); fi
  # The local mnemotree wheel satisfies mnemotree-mcp's pin, never PyPI's.
  if [[ "$variant" == mcp ]]; then requirements=("$wheel" "$mcp_wheel"); fi
  # --no-build: dependencies install from wheels only, so no third-party
  # setup script runs here. Our own wheels are already built.
  uv pip install --quiet --no-build --python "$work/$variant/bin/python" \
    -c "$work/constraints.txt" "${requirements[@]}"
  (
    cd "$work"
    "$work/$variant/bin/python" -I "$repo/.github/scripts/package_smoke.py" "--$variant"
  )
done

# Check a consumer outside the checkout, after the bare-runtime assertions above.
# assert_type in the smoke file also catches APIs silently degrading to Any.
uv pip install --quiet --no-build --python "$work/bare/bin/python" \
  -c "$work/constraints.txt" mypy
cp "$repo/.github/scripts/typing_smoke.py" "$work/typing_smoke.py"
(
  cd "$work"
  "$work/bare/bin/python" -m mypy --strict --follow-imports=silent \
    --ignore-missing-imports "$work/typing_smoke.py"
)
