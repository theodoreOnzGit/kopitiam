#!/usr/bin/env bash
#
# Publishes the KOPITIAM crates that `kopitiam` (the CLI, apps/cli) depends
# on, in dependency order, followed by `kopitiam` itself.
#
# crates.io requires every dependency of a published crate to already exist
# on the registry at a matching version -- local path dependencies are not
# enough. That means these 9 crates must go up one at a time, in an order
# where each crate's dependencies are already live before it publishes.
#
# This script does NOT run automatically as part of any other workflow. Run
# it yourself, deliberately, after `cargo login <your-token>`:
#
#   ./scripts/publish.sh            # publish everything not yet published
#   ./scripts/publish.sh --dry-run  # show what would happen, publish nothing
#
# Safe to re-run: crates.io rejects re-publishing a version that already
# exists, and this script treats that as "already done" and moves on,
# rather than treating it as a fatal error.
#
# crates.io rate-limits *publishing a brand-new crate name* to a burst of 5,
# refilling at 1 every 10 minutes (see https://crates.io/docs/rate-limits;
# confirmed against the exact numbers in rust-lang/crates.io's
# src/rate_limiter.rs). Since every crate below is a first-time publish,
# publishing all 9 back-to-back *will* hit that limit after the 5th. This
# script detects the resulting 429 and waits out crates.io's own
# `Retry-After` deadline rather than failing -- expect it to pause for
# stretches of ~10 minutes partway through a full run.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Topological order: every crate here depends only on crates earlier in
# this list (or on nothing internal at all).
CRATES=(
    kopitiam-ontology
    kopitiam-index
    kopitiam-pdf
    kopitiam-workspace   # depends on: kopitiam-index
    kopitiam-document    # depends on: kopitiam-pdf
    kopitiam-semantic    # depends on: kopitiam-ontology
    kopitiam-knowledge   # depends on: kopitiam-ontology
    kopitiam-markdown    # depends on: kopitiam-document
    kopitiam             # depends on: all of the above
)

# All 9 crates share one version via [workspace.package] in the root
# Cargo.toml, so there is exactly one place to read it from -- no need to
# parse `cargo metadata` JSON (and no new tool dependency, e.g. `jq`, for
# a shell script whose whole point is being easy to just run).
WORKSPACE_VERSION="$(grep -m1 '^version = ' Cargo.toml | sed -E 's/version = "(.*)"/\1/')"
if [[ -z "$WORKSPACE_VERSION" ]]; then
    echo "error: could not read [workspace.package].version from Cargo.toml" >&2
    exit 1
fi

# Queries the crates.io HTTP API directly, rather than `cargo info`: run
# from inside this workspace, `cargo info <name>` happily reports the
# *local path* crate as if it were published (it says so right in its
# output -- "version: 0.1.0 (from ./crates/...)"), which made every crate
# look "already published" during testing. Hitting the registry API
# directly has no such ambiguity. crates.io asks API clients to send an
# identifying User-Agent; see https://crates.io/policies#crawlers.
CRATES_IO_USER_AGENT="kopitiam-publish-script (https://github.com/theodoreOnzGit/kopitiam)"

already_published() {
    local name="$1" version="$2"
    curl -fsS -o /dev/null -H "User-Agent: ${CRATES_IO_USER_AGENT}" \
        "https://crates.io/api/v1/crates/${name}/${version}"
}

# Polls the crates.io API for up to ~2 minutes so the *next* crate in the
# list (which may depend on this one) doesn't fail to resolve it. crates.io's
# sparse index is usually near-instant, but this is cheap insurance against
# the rare slow propagation.
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

# Publishes one crate, retrying on crates.io's publish-rate-limit response
# (HTTP 429, "You have published too many new crates in a short period of
# time...") instead of treating it as fatal. crates.io includes the exact
# moment the limit resets in its error text ("Please try again after <HTTP
# date>"); we parse that and sleep until then (plus a small buffer) rather
# than guessing, falling back to the documented 10-minute refill interval
# if the message ever changes shape.
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
echo "  cargo install kopitiam --locked --force"
