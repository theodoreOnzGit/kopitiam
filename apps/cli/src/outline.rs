//! The `outline` subcommand: file skeleton / outline mode — token-max Task II-2.
//!
//! `outline <file>` prints an *items-only* skeleton of a Rust file — one line
//! per declaration (module, struct, field, `impl`, fn/method, const, type
//! alias) with the line it begins on and its signature, and **no bodies**. An
//! 811-line file whose outline is ~60 lines lets an agent orient for roughly a
//! tenth of the tokens a full read costs, then (with Task II-1) read only the
//! one function it actually needs (`kopitiam_token_max.md` §11 II-2).
//!
//! All the real work lives in [`kopitiam_semantic::outline`]: it runs a
//! `textDocument/documentSymbol` request and flattens rust-analyzer's
//! hierarchical symbol tree into ordered, depth-marked [`OutlineItem`]s. This
//! command is the thin CLI shell — connect a session, call `outline`, and emit
//! either the human skeleton ([`Outline::to_skeleton`]) or the `--json` form.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use kopitiam_semantic::{Outline, outline};

use crate::syntactic;

/// Options for `kopitiam outline`.
#[derive(Args, Debug)]
pub struct OutlineArgs {
    /// The Rust source file to outline.
    pub file: PathBuf,

    /// Directory containing the workspace `Cargo.toml` that `file` belongs to.
    /// Defaults to the current directory; passed to rust-analyzer as the root.
    #[arg(long, default_value = ".")]
    pub root: PathBuf,

    /// Emit the outline as JSON (the serialized [`Outline`]: `items` with
    /// `line`/`kind`/`name`/`detail`/`depth`) instead of the human skeleton.
    /// Progress notices go to stderr so stdout stays clean JSON (§0.2).
    #[arg(long)]
    pub json: bool,

    /// Skip rust-analyzer entirely and produce the outline by a deterministic,
    /// dependency-free Rust item scan (no cross-file resolution, no server).
    /// Instant, and the guaranteed path on a workspace too large for
    /// rust-analyzer to index in a bounded time.
    #[arg(long = "no-lsp", visible_alias = "syntactic")]
    pub no_lsp: bool,

    /// Require rust-analyzer: on timeout/unavailability fail (non-zero exit)
    /// instead of falling back to the syntactic scan.
    #[arg(long, conflicts_with = "no_lsp")]
    pub lsp: bool,
}

/// Runs `kopitiam outline`.
///
/// Default behaviour is workspace-size aware (P1): on a **large** workspace
/// (where rust-analyzer will not index in bounded time) it goes straight to the
/// instant syntactic item scan with a stderr note; on a **small** workspace it
/// tries rust-analyzer first (real signatures, the server's own symbol tree)
/// and, on timeout/unavailability, falls back automatically to the syntactic
/// scan ([`syntactic::outline_file`]). `--no-lsp`/`--syntactic` forces the scan;
/// `--lsp` forces rust-analyzer regardless of workspace size and makes a failure
/// a hard, non-zero exit rather than a silent empty outline.
pub fn run(args: OutlineArgs) -> Result<()> {
    let outline = resolve_outline(&args)?;
    emit(&outline, args.json);
    Ok(())
}

/// Produces the [`Outline`] per the `--no-lsp`/`--lsp` policy above.
fn resolve_outline(args: &OutlineArgs) -> Result<Outline> {
    if args.no_lsp {
        return syntactic::outline_file(&args.file);
    }

    // Resource gate FIRST: ask whether this box can afford rust-analyzer at all
    // right now (free RAM + cores, via the shared AID-0037 budgeter). This is the
    // axis the file count below cannot see — the same workspace is comfortable on
    // a desktop and fatal on a 4 GB tablet, where RA's indexing gets the process
    // SIGKILLed by the low-memory killer and the CLI just looks frozen. Standing
    // down costs a thinner outline; not standing down can cost the whole process.
    if !args.lsp
        && let crate::ra_guard::RaGate::StandDown(why) = crate::ra_guard::decide(&args.root)
    {
        eprintln!("rust-analyzer stood down: {why}; syntactic outline (--lsp to force)");
        return syntactic::outline_file(&args.file);
    }

    // P1: on a large workspace rust-analyzer will not index in bounded time, so
    // DEFAULT to the instant syntactic scan (rust-analyzer stays opt-in via
    // `--lsp`, which skips this branch and keeps its full-timeout, fail-hard
    // semantics). A one-line stderr note keeps `--json` stdout clean. Kept
    // alongside the resource gate above, not replaced by it: this one catches
    // "nobody indexes this in bounded time" even on a box with RAM to spare.
    if !args.lsp && syntactic::prefer_syntactic_default(&args.root) {
        eprintln!("large workspace: syntactic outline; --lsp for rust-analyzer");
        return syntactic::outline_file(&args.file);
    }

    let ra = (|| -> Result<Outline> {
        let mut session = syntactic::connect_ra(&args.root)?;
        let result = outline(&mut session, &args.file)?;
        let _ = session.shutdown();
        Ok(result)
    })();

    match ra {
        // A non-empty semantic outline is the best answer.
        Ok(result) if !result.items.is_empty() => Ok(result),
        // rust-analyzer answered but with nothing. When forced (`--lsp`), trust
        // it; otherwise this is usually "the index was not ready", so prefer the
        // syntactic scan when *it* finds items (fixing the silent empty-outline
        // exit-0 bug) and only report a truly empty file otherwise.
        Ok(empty) => {
            if args.lsp {
                return Ok(empty);
            }
            let scanned = syntactic::outline_file(&args.file)?;
            if scanned.items.is_empty() {
                Ok(empty)
            } else {
                eprintln!(
                    "rust-analyzer returned no symbols (index likely not ready); using syntactic fallback"
                );
                Ok(scanned)
            }
        }
        Err(err) if args.lsp => {
            Err(err.context("rust-analyzer outline failed (and --lsp forbids the syntactic fallback)"))
        }
        Err(err) => {
            eprintln!("rust-analyzer unavailable/timed out ({err:#}); using syntactic fallback");
            syntactic::outline_file(&args.file)
        }
    }
}

/// Prints the outline as the human skeleton or, with `--json`, the serialized
/// [`Outline`]. `to_skeleton` already newline-terminates each item.
fn emit(outline: &Outline, json: bool) {
    if json {
        match serde_json::to_string_pretty(outline) {
            Ok(s) => println!("{s}"),
            Err(e) => eprintln!("failed to serialize outline JSON: {e}"),
        }
    } else {
        print!("{}", outline.to_skeleton());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use kopitiam_semantic::{Outline, OutlineItem};

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: OutlineArgs,
    }

    #[test]
    fn parses_file_with_default_root() {
        let cli = TestCli::try_parse_from(["t", "src/reconstruction/mod.rs"]).unwrap();
        assert_eq!(cli.args.file, PathBuf::from("src/reconstruction/mod.rs"));
        assert_eq!(cli.args.root, PathBuf::from("."));
        assert!(!cli.args.json);
    }

    #[test]
    fn parses_root_and_json() {
        let cli =
            TestCli::try_parse_from(["t", "a.rs", "--root", "/w", "--json"]).unwrap();
        assert_eq!(cli.args.root, PathBuf::from("/w"));
        assert!(cli.args.json);
    }

    #[test]
    fn parses_no_lsp_and_its_syntactic_alias() {
        let cli = TestCli::try_parse_from(["t", "a.rs", "--no-lsp"]).unwrap();
        assert!(cli.args.no_lsp);
        let cli = TestCli::try_parse_from(["t", "a.rs", "--syntactic"]).unwrap();
        assert!(cli.args.no_lsp, "--syntactic is an alias for --no-lsp");
        assert!(!cli.args.lsp);
    }

    #[test]
    fn lsp_and_no_lsp_conflict() {
        assert!(TestCli::try_parse_from(["t", "a.rs", "--lsp", "--no-lsp"]).is_err());
    }

    #[test]
    fn json_shape_is_the_serialized_outline() {
        // The exact serialization the CLI emits for --json: `items` array with
        // the documented per-item fields. Built from typed OutlineItems so no
        // rust-analyzer is needed.
        let outline = Outline {
            items: vec![
                OutlineItem {
                    line: 3,
                    kind: "struct".into(),
                    name: "Rebuilder".into(),
                    detail: None,
                    depth: 0,
                },
                OutlineItem {
                    line: 9,
                    kind: "fn".into(),
                    name: "new".into(),
                    detail: Some("(config: Config) -> Self".into()),
                    depth: 1,
                },
            ],
        };
        let json: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&outline).unwrap()).unwrap();
        let items = json["items"].as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["line"], 3);
        assert_eq!(items[0]["kind"], "struct");
        assert_eq!(items[1]["name"], "new");
        assert_eq!(items[1]["detail"], "(config: Config) -> Self");
        assert_eq!(items[1]["depth"], 1);
    }
}
