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
use kopitiam_semantic::{RustAnalyzerSession, outline};

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
}

/// Runs `kopitiam outline`: spawn rust-analyzer, request the file's document
/// symbols, and print the body-free skeleton.
pub fn run(args: OutlineArgs) -> Result<()> {
    let root = std::fs::canonicalize(&args.root)?;
    // The progress line goes to stderr, never stdout, so `--json` output is a
    // clean document a caller can pipe straight into `jq`.
    eprintln!(
        "Starting rust-analyzer and waiting for it to index {}...",
        root.display()
    );
    let mut session = RustAnalyzerSession::connect(&root)?;
    let result = outline(&mut session, &args.file)?;
    let _ = session.shutdown();

    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        // `to_skeleton` already ends every item with a newline.
        print!("{}", result.to_skeleton());
    }
    Ok(())
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
