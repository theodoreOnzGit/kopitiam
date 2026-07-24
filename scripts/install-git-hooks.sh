#!/usr/bin/env sh
# Install KOPITIAM's tracked git hooks for this working copy.
#
# Git hooks are not cloned, so each checkout runs this once. It points
# core.hooksPath at the tracked .githooks/ directory (see .githooks/README.md).
set -eu

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

git config core.hooksPath .githooks
chmod +x .githooks/prepare-commit-msg 2>/dev/null || true

echo "core.hooksPath => $(git config --get core.hooksPath)"
echo "KOPITIAM git hooks installed."
