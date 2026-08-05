#!/usr/bin/env bash
# Finalize the zephyr-rose split: publish the extracted standalone repo and convert
# RoSE's in-tree soc/sw/zephyr-rose directory into a submodule that tracks it.
#
# The history-preserving extraction is already done (git subtree split) and lives at:
#   $STAGE       (default: <rose-parent>/zephyr-rose)      -- standalone repo w/ history
#   $BUNDLE      (default: <rose-parent>/zephyr-rose.bundle) -- portable clone-able bundle
#
# This script does the two remote-dependent steps that could not run offline:
#   1. push the standalone repo to $REMOTE (must already exist on GitHub -- create it
#      first: an EMPTY repo, no README/license, so the pushed history is the only content)
#   2. in RoSE: replace the tracked directory with a submodule pointing at $REMOTE
#
# The RoSE build is unaffected: build_zephyr_rose.sh injects the module by PATH
# (-DZEPHYR_EXTRA_MODULES=$ROSE_DIR/soc/sw/zephyr-rose), and the submodule checkout
# occupies that same path. No build-script change is required.
#
# Usage:
#   bash soc/sim/finalize_zephyr_rose_split.sh            # push + swap
#   REMOTE=git@github.com:you/zephyr-rose.git bash ...    # override target
#   SKIP_PUSH=1 bash ...                                  # already pushed; only swap
set -eo pipefail

ROSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARENT="$(dirname "$ROSE_DIR")"
STAGE="${STAGE:-$PARENT/zephyr-rose}"
BUNDLE="${BUNDLE:-$PARENT/zephyr-rose.bundle}"
REMOTE="${REMOTE:-git@github.com:ucb-bar/zephyr-rose.git}"
SUBPATH="soc/sw/zephyr-rose"

echo "ROSE_DIR = $ROSE_DIR"
echo "STAGE    = $STAGE"
echo "REMOTE   = $REMOTE"

if [ ! -d "$STAGE/.git" ]; then
  echo "ERROR: standalone repo not found at $STAGE" >&2
  echo "       (re-run the subtree split, or clone the bundle: git clone $BUNDLE $STAGE)" >&2
  exit 1
fi

# 1. Publish -----------------------------------------------------------------
if [ -z "${SKIP_PUSH:-}" ]; then
  echo "=== pushing $STAGE -> $REMOTE (branch main) ==="
  git -C "$STAGE" push "$REMOTE" main:main
else
  echo "=== SKIP_PUSH set; assuming $REMOTE already has main ==="
fi

# 2. Swap the in-tree dir for a submodule ------------------------------------
cd "$ROSE_DIR"
if git config --file .gitmodules --get "submodule.$SUBPATH.url" >/dev/null 2>&1; then
  echo "=== $SUBPATH is already a submodule; nothing to swap ==="
  exit 0
fi

if [ -n "$(git status --porcelain "$SUBPATH")" ]; then
  echo "ERROR: $SUBPATH has uncommitted changes; commit/stash them first." >&2
  exit 1
fi

echo "=== removing tracked $SUBPATH and re-adding as a submodule ==="
git rm -r --quiet "$SUBPATH"
rm -rf "$SUBPATH"                       # git rm leaves nothing, but be explicit
git submodule add "$REMOTE" "$SUBPATH"
git submodule update --init "$SUBPATH"

echo "=== verifying module.yml resolves through the submodule ==="
test -f "$SUBPATH/zephyr/module.yml" && echo "OK: $SUBPATH/zephyr/module.yml present"

cat <<EOF

Done. Review and commit:
  git status
  git add .gitmodules $SUBPATH
  git commit -m "soc/sw: track zephyr-rose as a submodule (ucb-bar/zephyr-rose)"

RoSE builds are unchanged: build_zephyr_rose.sh still finds the module at $SUBPATH.
EOF
