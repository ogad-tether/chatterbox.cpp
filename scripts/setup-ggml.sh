#!/usr/bin/env bash
# Clone ggml into ./ggml, check out the commit this repo is pinned against,
# and apply the Chatterbox Metal op patch.  Idempotent: safe to re-run.
#
# Update GGML_COMMIT here whenever the patch is re-generated against a newer
# upstream ggml; this file is the single source of truth for the pin.

set -euo pipefail

# -----------------------------------------------------------------------------
# The upstream ggml commit that patches/ggml-metal-chatterbox-ops.patch was
# authored against.  Pin here so fresh clones (and CI) build deterministically.
# -----------------------------------------------------------------------------
GGML_COMMIT="58c38058"
GGML_URL="https://github.com/ggml-org/ggml.git"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "chatterbox.cpp: setting up ggml at pinned commit ${GGML_COMMIT}"

if [ ! -d ggml/.git ]; then
    echo "  → cloning ${GGML_URL}"
    git clone "$GGML_URL" ggml
fi

cd ggml

# Skip if we're already at the pinned commit with the patch already applied.
CURRENT="$(git rev-parse --short=8 HEAD 2>/dev/null || echo '')"
DIRTY_FILES="$(git status --porcelain src/ggml-metal/ 2>/dev/null | wc -l | tr -d ' ')"
if [ "$CURRENT" = "$GGML_COMMIT" ] && [ "$DIRTY_FILES" -ge 1 ]; then
    # Verify the patch would NOT apply cleanly on top — i.e. it's already in.
    if ! git apply --check "$REPO_ROOT/patches/ggml-metal-chatterbox-ops.patch" 2>/dev/null; then
        echo "  → patch already applied on ${GGML_COMMIT}, nothing to do"
        exit 0
    fi
fi

echo "  → checking out ${GGML_COMMIT}"
# Reset any prior partial state first so `git apply` doesn't trip over
# stale diffs from an aborted run.
git checkout -- . 2>/dev/null || true
git checkout "$GGML_COMMIT"

echo "  → applying patches/ggml-metal-chatterbox-ops.patch"
git apply "$REPO_ROOT/patches/ggml-metal-chatterbox-ops.patch"

N_MODIFIED="$(git status --porcelain src/ggml-metal/ | wc -l | tr -d ' ')"
echo "  → ok (${N_MODIFIED} files modified under src/ggml-metal/)"
echo
echo "ggml is ready.  Next:"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON"
echo "    cmake --build build -j\$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
