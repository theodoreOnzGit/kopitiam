#!/usr/bin/env bash
#
# Seeds the kopitiam-gpu crate NAME onto crates.io at v0.0.1, as an early name
# reservation while the GPU parallel-compute foundation is still mid-build. This
# is NOT the real release -- 0.0.1 sits BEHIND the workspace version (0.1.x) on
# purpose, same pattern as kopitiam-core/tensor/models (see
# scripts/publish-runtime-seed.sh), so the real lockstep release supersedes it
# later. Why seed early: crates.io rate-limits brand-NEW crate names to a burst
# of 5 (refill 1 / 10 min), so claiming the name now stops anyone else grabbing
# it before the engine is ready.
#
# kopitiam-gpu is STANDALONE -- no internal KOPITIAM dependencies (its public API
# is plain &[f32]/Vec<f32>), only crates.io deps (wgpu, pollster, bytemuck,
# thiserror). So, unlike the runtime seed, there is no dependency ordering and
# nothing to wait for on the index: it is a single self-contained publish.
#
# version = "0.0.1" is pinned directly in crates/kopitiam-gpu/Cargo.toml
# (overriding version.workspace), so `cargo publish -p kopitiam-gpu` picks 0.0.1
# without any version variable here.
#
# HARD RULE, same as every other publish script: an agent must NEVER run this.
# Run it yourself, deliberately, after `cargo login <token>`:
#
#   ./scripts/publish-gpu-seed.sh            # publish if not yet up
#   ./scripts/publish-gpu-seed.sh --dry-run  # package + check, upload nothing
#
# Safe to re-run: crates.io rejects re-publishing a version that already exists,
# and this treats that as "already done".

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

CRATE="kopitiam-gpu"
# Pinned behind the workspace version on purpose; see the crate's Cargo.toml.
SEED_VERSION="0.0.1"

CRATES_IO_USER_AGENT="kopitiam-publish-script (https://github.com/theodoreOnzGit/kopitiam)"

already_published() {
    local name="$1" version="$2"
    curl -fsS -o /dev/null -H "User-Agent: ${CRATES_IO_USER_AGENT}" \
        "https://crates.io/api/v1/crates/${name}/${version}"
}

# Publishes the crate, waiting out crates.io's new-crate rate limit (HTTP 429)
# instead of treating it as fatal. Mirrors publish-runtime-seed.sh.
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

if already_published "$CRATE" "$SEED_VERSION"; then
    echo "== $CRATE@$SEED_VERSION already published, nothing to do =="
    exit 0
fi

echo "== publishing $CRATE@$SEED_VERSION =="
if $DRY_RUN; then
    cargo publish -p "$CRATE" --dry-run
else
    publish_with_retry "$CRATE"
fi

echo
echo "Done. The name '$CRATE' is now seeded at ${SEED_VERSION}. The real GPU-engine"
echo "release goes up later at the workspace version (0.1.x), in lockstep with the"
echo "rest of the tree -- bump the seed crate back to version.workspace then."
