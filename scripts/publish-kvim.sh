#!/usr/bin/env bash
#
# Publishes the KOPITIAM crates that `kvim` (kopitiam-neovim, the editor)
# depends on, in dependency order, followed by `kopitiam-neovim` itself.
#
# Same idea as publish.sh (which publishes the CLI's tree) but a DIFFERENT set:
# this one is the kvim publish tree. crates.io needs every dependency of a
# published crate to already exist on the registry at a matching version --
# local path deps are not enough -- so these go up one at a time, deps first.
#
# This script does NOT run automatically as part of any workflow, and an agent
# must NEVER run it. Run it yourself, deliberately, after `cargo login <token>`:
#
#   ./scripts/publish-kvim.sh            # publish everything not yet published
#   ./scripts/publish-kvim.sh --dry-run  # show what would happen, publish nothing
#
# Safe to re-run: crates.io rejects re-publishing a version that already exists,
# and this treats that as "already done" and moves on. So if you already
# published ontology/config/semantic at an earlier version, only the NEW ones
# (syntax, snippet, lua, neovim -- and a 0.1.1 bump of the others) go up.
#
# crates.io rate-limits publishing a BRAND-NEW crate name to a burst of 5,
# refilling 1 every 10 minutes. Several crates below are first-time publishes,
# so a full run may pause ~10 min partway through when the 6th new name hits
# the limit -- this script detects the 429 and waits out the Retry-After.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Topological order for the kvim tree: every crate depends only on crates
# earlier in this list (or on nothing internal at all).
CRATES=(
    kopitiam-ontology    # no internal deps
    kopitiam-config      # no internal deps
    kopitiam-syntax      # no internal deps
    kopitiam-snippet     # no internal deps
    kopitiam-lua         # no internal deps (pure-Rust Lua 5.1 VM)
    kopitiam-semantic    # depends on: kopitiam-ontology
    kopitiam-neovim      # depends on: config, lua, semantic, snippet, syntax
)

# All crates share one version via [workspace.package] in the root Cargo.toml.
WORKSPACE_VERSION="$(grep -m1 '^version = ' Cargo.toml | sed -E 's/version = "(.*)"/\1/')"
if [[ -z "$WORKSPACE_VERSION" ]]; then
    echo "error: could not read [workspace.package].version from Cargo.toml" >&2
    exit 1
fi

CRATES_IO_USER_AGENT="kopitiam-publish-script (https://github.com/theodoreOnzGit/kopitiam)"

already_published() {
    local name="$1" version="$2"
    curl -fsS -o /dev/null -H "User-Agent: ${CRATES_IO_USER_AGENT}" \
        "https://crates.io/api/v1/crates/${name}/${version}"
}

# Polls crates.io for up to ~2 min so the NEXT crate (which may depend on this
# one) can resolve it before its own publish.
wait_for_index() {
    local name="$1" version="$2"
    echo "  waiting for ${name}@${version} to appear on crates.io..."
    for _ in $(seq 1 24); do
        if already_published "$name" "$version"; then
            echo "  ${name}@${version} is live."
            return 0
        fi
        sleep 5
    done
    echo "  warning: ${name}@${version} did not appear within 2 minutes; continuing anyway." >&2
}

# Publishes one crate, waiting out crates.io's new-crate rate limit (HTTP 429)
# instead of treating it as fatal. Parses the Retry-After moment from the error.
publish_with_retry() {
    local crate="$1"
    local max_attempts=8
    local attempt output retry_at retry_epoch now_epoch wait_seconds

    for (( attempt = 1; attempt <= max_attempts; attempt++ )); do
        if output=$(cargo publish -p "$crate" 2>&1); then
            printf '%s\n' "$output"
            return 0
        fi
        printf '%s\n' "$output" >&2

        if ! grep -qiE "too many (new crates|requests|updates)|429 Too Many Requests" <<<"$output"; then
            return 1 # a real error -- don't blindly retry
        fi

        retry_at="$(grep -oE 'after [A-Za-z]{3}, [0-9]{2} [A-Za-z]{3} [0-9]{4} [0-9:]{8} GMT' <<<"$output" | sed 's/^after //')"
        wait_seconds=600
        if [[ -n "$retry_at" ]]; then
            retry_epoch="$(date -d "$retry_at" +%s 2>/dev/null || true)"
            now_epoch="$(date -u +%s)"
            if [[ -n "$retry_epoch" ]] && (( retry_epoch > now_epoch )); then
                wait_seconds=$(( retry_epoch - now_epoch + 15 ))
            fi
        fi

        echo "  crates.io's new-crate rate limit kicked in (see https://crates.io/docs/rate-limits)."
        echo "  waiting ${wait_seconds}s before retrying $crate (attempt $attempt/$max_attempts)..."
        sleep "$wait_seconds"
    done

    echo "error: $crate is still rate-limited after $max_attempts attempts; run this script again later." >&2
    return 1
}

for crate in "${CRATES[@]}"; do
    version="$WORKSPACE_VERSION"

    if already_published "$crate" "$version"; then
        echo "== $crate@$version already published, skipping =="
        continue
    fi

    echo "== publishing $crate@$version =="
    if $DRY_RUN; then
        cargo publish -p "$crate" --dry-run
    else
        publish_with_retry "$crate"
        wait_for_index "$crate" "$version"
    fi
done

echo
echo "Done. If this was a real (non---dry-run) run, verify with:"
echo "  cargo install kopitiam-neovim --locked --force   # installs the 'kvim' binary"
