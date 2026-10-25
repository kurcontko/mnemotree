#!/usr/bin/env bash
# Prove that ./release holds exactly the bundle a successful release-candidate
# run built for this tag, before anything is smoke-tested or published.
#
#   check_candidate.sh <tag> <candidate-run-id>
#
# Needs GH_TOKEN with actions:read, a checkout that has the tag, and the
# candidate's `release-<tag>` artifact already downloaded into ./release.
set -euo pipefail

tag="$1"
run_id="$2"

if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
  echo "::error title=bad tag::Invalid release tag: $tag"
  exit 1
fi
if [[ ! "$run_id" =~ ^[0-9]+$ ]]; then
  echo "::error title=bad run id::candidate_run_id must be a workflow run number, got: $run_id"
  exit 1
fi

# The run must be a successful release-candidate.yml run on main. Its
# `verify` job is the full CI workflow against the tagged commit, so success
# here is also the gate result; nothing is re-tested or rebuilt below.
run=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run_id")
path=$(jq -r .path <<< "$run")
conclusion=$(jq -r .conclusion <<< "$run")
branch=$(jq -r .head_branch <<< "$run")
if [ "$path" != ".github/workflows/release-candidate.yml" ]; then
  echo "::error title=wrong workflow::run $run_id is $path, not release-candidate.yml"
  exit 1
fi
if [ "$conclusion" != "success" ]; then
  echo "::error title=candidate not green::run $run_id concluded '$conclusion'"
  exit 1
fi
if [ "$branch" != "main" ]; then
  echo "::error title=wrong branch::run $run_id ran on '$branch', not main"
  exit 1
fi

# The bundle's manifest must name this tag, this tag's commit, and this run,
# and every file's digest must match it (release_bundle.py verify).
source_sha=$(git rev-parse "refs/tags/${tag}^{commit}")
python .github/scripts/release_bundle.py verify \
  --bundle release --tag "$tag" --source "$source_sha" --run-id "$run_id"

echo "Candidate $tag verified: run $run_id, source $source_sha"
cat release/SHA256SUMS
