#!/usr/bin/env bash
#
# Seeds the FIRST three Kopitiam Runtime / model-layer crate NAMES onto
# crates.io at v0.0.1, as an early name reservation while the Runtime engine is
# still mid-build. This is NOT the real Runtime release -- 0.0.1 sits behind the
# workspace version (0.1.x) on purpose, so the real lockstep release supersedes
# it later. Why seed early: crates.io rate-limits brand-NEW crate names to a
# burst of 5 (refill 1 / 10 min), so claiming names now spreads that cost out and
# stops anyone else grabbing them before the engine is ready.
#
# Seed set (dependency order -- crates.io needs every internal dep already live):
#   kopitiam-core    -- leaf (only thiserror); nothing internal under it
#   kopitiam-tensor  -- depends on kopitiam-core, so core MUST be up first
#   kopitiam-models  -- no internal deps; standalone
#
# These three carry version = "0.0.1" pinned directly in their own Cargo.toml
# (overriding version.workspace), so `cargo publish -p <crate>` picks 0.0.1
# without any version variable here.
#
# Same hard rule as the other publish scripts: an agent must NEVER run this. Run
# it yourself, deliberately, after `cargo login <token>`:
#
#   ./scripts/publish-runtime-seed.sh            # publish anything not yet up
#   ./scripts/publish-runtime-seed.sh --dry-run  # show what would happen, upload nothing
#
# Safe to re-run: crates.io rejects re-publishing a version that already exists,
# and this treats that as "already done" and moves on.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# All three seed crates share this reserved version. Kept as one constant (not
# read from [workspace.package]) precisely because these sit BEHIND the workspace
# version -- reading the workspace version here would try to publish 0.1.x.
SEED_VERSION="0.0.1"

# Dependency order: core before tensor (tensor depends on core); models anywhere.
CRATES=(
    kopitiam-core     # leaf: only thiserror
    kopitiam-tensor   # depends on: kopitiam-core
    kopitiam-models   # no internal deps
)

CRATES_IO_USER_AGENT="kopitiam-publish-script (https://github.com/theodoreOnzGit/kopitiam)"

already_published() {
    local name="$1" version="$2"
    curl -fsS -o /dev/null -H "User-Agent: ${CRATES_IO_USER_AGENT}" \
        "https://crates.io/api/v1/crates/${name}/${version}"
}

# Polls crates.io for up to ~2 min so the NEXT crate (tensor, which needs core)
# can resolve it before its own publish.
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
    version="$SEED_VERSION"

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
echo "Done. These three names are now seeded at ${SEED_VERSION}. The real Runtime"
echo "release goes up later at the workspace version (0.1.x), in lockstep with the"
echo "rest of the tree -- bump the seed crates back to version.workspace then."
