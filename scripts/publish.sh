#!/usr/bin/env bash
#
# Publishes the full `kopitiam` CLI tree: every internal KOPITIAM crate that
# the `kopitiam` binary (apps/cli) transitively depends on, in dependency
# order, followed by `kopitiam` itself.
#
# crates.io requires every dependency of a published crate to already exist
# on the registry at a matching version -- local path dependencies are not
# enough. So these crates go up one at a time, in an order where each crate's
# dependencies are already live before it publishes.
#
# The CRATES list below is the topological order (deps first, `kopitiam`
# last) of the set produced by:
#
#   cargo tree -p kopitiam -e normal --prefix none | grep -oE '^kopitiam[a-z-]*' | sort -u
#
# RE-RUN THAT COMMAND AND DIFF IT AGAINST `CRATES` EVERY TIME YOU RELEASE.
# The list is hand-maintained, so it goes stale the moment somebody adds an
# internal dep, and the failure is only visible at upload time -- one crate
# into a multi-crate, rate-limited run. Bitten once already, at the 0.2.5
# prep: `kopitiam-resource` (plain dep of apps/cli) and `kopitiam-gpu`
# (pulled in by kopitiam-runtime's DEFAULT `gpu` feature) were both in the
# tree but missing here, so `kopitiam` and `kopitiam-runtime` would each have
# been rejected -- crates.io demand every dependency, optional-behind-a-
# default-feature also count, already sit on the registry.
#
# ONE DELIBERATE EXCEPTION to that diff, for the 0.2.5 release: `kopitiam-neovim`
# is commented OUT of the list on purpose. kvim is red on Windows (`:term` freeze
# + the `gf` isfname bug), so the maintainer decided to ship the AI-chat fixes
# WITHOUT the editor. Nothing to change in any manifest to make this work:
# apps/cli already asks for `kopitiam-neovim = "0.2.1"`, i.e. `^0.2.1`, so a
# published `kopitiam` 0.2.5 just resolves it to the newest ON THE REGISTRY --
# 0.2.4, the previous release -- while local builds keep using the path dep.
# When kvim goes green, uncomment the line and it ships again; existing installs
# pick it up automatically because `^0.2.1` already allows it. So the list-vs-tree
# diff SHOULD show kopitiam-neovim missing until then, and that is not drift.
#
# MIXED VERSIONS: the tree no longer shares one version. Most crates release
# in lockstep at the workspace version (0.2.5), but a few sit on their own
# pin: kopitiam-ocr at 0.1.0, and kopitiam-gpu at 0.0.1 (an early name-
# reservation seed, see scripts/publish-gpu-seed.sh). So there is NO single
# WORKSPACE_VERSION constant here: `cargo publish -p <crate>` already uses
# each crate's own manifest version, and the already-published skip-check
# resolves EACH crate's own version via `cargo pkgid`.
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
# src/rate_limiter.rs). Several crates below are first-time publishes, so a
# full run can hit that limit. This script detects the resulting 429 and
# waits out crates.io's own `Retry-After` deadline rather than failing --
# expect it to pause for stretches of ~10 minutes partway through a full run.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Topological order: every crate here depends only on crates earlier in this
# list (or on nothing internal at all). Derived from `cargo tree -p kopitiam`
# (see header). `kopitiam` itself goes last.
CRATES=(
    kopitiam-core        # leaf: only thiserror
    kopitiam-ontology    # leaf
    kopitiam-index       # leaf
    kopitiam-pdf         # leaf
    kopitiam-models      # leaf (net feature -> ureq)
    kopitiam-gpu         # leaf, STANDALONE (wgpu/pollster/bytemuck only), pinned 0.0.1
    kopitiam-resource    # leaf: no internal deps; apps/cli's §6 budget gate
    kopitiam-config      # leaf (editor/kvim)
    kopitiam-lua         # leaf (editor/kvim: pure-Rust Lua interpreter)
    kopitiam-syntax      # leaf (editor/kvim)
    kopitiam-snippet     # leaf (editor/kvim)
    kopitiam-tensor      # depends on: kopitiam-core
    kopitiam-loader      # depends on: kopitiam-core
    kopitiam-tokenizer   # depends on: kopitiam-core
    kopitiam-semantic    # depends on: kopitiam-ontology
    kopitiam-knowledge   # depends on: kopitiam-ontology
    kopitiam-document    # depends on: kopitiam-pdf
    kopitiam-workspace   # depends on: kopitiam-index
    kopitiam-markdown    # depends on: kopitiam-document
    kopitiam-runtime     # depends on: kopitiam-{core,loader,tensor,tokenizer}
    kopitiam-ai          # depends on: kopitiam-{loader,runtime,tokenizer}
    kopitiam-workflow    # depends on: kopitiam-{ai,knowledge,ontology,workspace}
    # kopitiam-neovim    # HELD BACK at 0.2.4 for the 0.2.5 release -- see below.
    kopitiam-ocr         # NEW 0.1.0: Tesseract LSTM OCR port; depends on kopitiam-tensor
    kopitiam             # depends on: all of the above (incl. kopitiam-neovim + kopitiam-ocr)
)

# Resolves a crate's OWN published version straight from Cargo's own view of
# the manifest -- correct whether the crate pins its version explicitly
# (0.1.0) or inherits it from [workspace.package] (0.1.8). `cargo pkgid`
# prints e.g. `path+file:///...#0.1.8` or `path+file:///...#kopitiam@0.1.0`;
# stripping everything up to the last `#` or `@` leaves just the version. No
# single workspace-version constant, and no `jq`.
crate_version() {
    local name="$1"
    cargo pkgid -p "$name" 2>/dev/null | sed -E 's/.*[#@]//'
}

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

# Refuses to publish a crate that keeps vendored reference sources under
# vendor/ (e.g. kopitiam-ai's ~255MB of llama.cpp + OpenFOAM, GPL, local-only)
# unless its Cargo.toml explicitly excludes them with `exclude = ["vendor/"]`.
# Without that line `cargo publish` would try to pack hundreds of MB of
# third-party source into the .crate tarball. Belt-and-braces alongside the
# .gitignore rule: .gitignore keeps vendor/ out of git, this keeps it out of
# the published tarball.
preflight_vendor_check() {
    local name="$1"
    local dir manifest
    dir="$(cargo pkgid -p "$name" 2>/dev/null | sed -E 's#^.*file://##; s/#.*$//')"
    manifest="${dir}/Cargo.toml"
    if [[ -d "${dir}/vendor" ]] && ! grep -qE '^[[:space:]]*exclude[[:space:]]*=.*"vendor/"' "$manifest"; then
        echo "error: $name has a vendor/ dir but its Cargo.toml lacks exclude = [\"vendor/\"];" >&2
        echo "       refusing to publish (would pack vendored third-party sources into the .crate)." >&2
        return 1
    fi
    return 0
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

# Preflight the whole set BEFORE publishing anything, so a vendor/ misconfig
# fails fast rather than partway through a multi-crate (rate-limited) run.
for crate in "${CRATES[@]}"; do
    preflight_vendor_check "$crate"
done

for crate in "${CRATES[@]}"; do
    version="$(crate_version "$crate")"
    if [[ -z "$version" ]]; then
        echo "error: could not resolve a version for $crate via cargo pkgid" >&2
        exit 1
    fi

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
