//! The C# language adapter: turns a C# solution/project tree into
//! `kopitiam-ontology` facts, using a real C# language server for symbol-level
//! knowledge and a deterministic native reader for project structure.
//!
//! # What C# looks like, and why that shapes this provider
//!
//! A C# codebase is not "a directory of `.cs` files". It has three nested
//! shapes, and a provider that ignores them produces a graph nobody can query:
//!
//! * A **solution** (`.sln`) is a flat list of projects. It is not a build
//!   unit — it is an IDE grouping — but it is the thing a human names when they
//!   say "the codebase", so it earns an [`EntityKind::Artifact`].
//! * A **project** (`.csproj`) *is* the build unit — roughly a Rust crate. It
//!   declares its target framework(s), its `ProjectReference`s (edges to other
//!   projects in the tree) and its `PackageReference`s (edges to NuGet
//!   packages).
//! * The project's **source files** are, in modern *SDK-style* projects,
//!   **not listed at all**: `<Project Sdk="Microsoft.NET.Sdk">` implicitly
//!   globs every `**/*.cs` under the project directory (minus `bin/` and
//!   `obj/`). Only *legacy* (non-SDK) projects, or projects that set
//!   `<EnableDefaultCompileItems>false</EnableDefaultCompileItems>`, enumerate
//!   their sources with `<Compile Include="..."/>`. A reader that only trusts
//!   `<Compile Include>` therefore sees **zero source files** in the majority
//!   of real projects written since 2017. That fact is the single most
//!   important piece of C#-format knowledge in this file — see
//!   [`CsProj::compiles`].
//!
//! # The .NET runtime dependency, stated honestly
//!
//! Symbol-level facts (namespaces, types, members) come from a real C#
//! language server, exactly as [`crate::RustAnalyzerProvider`] gets Rust
//! symbols from rust-analyzer. Every viable C# language server —
//! **Roslyn LSP** (`Microsoft.CodeAnalysis.LanguageServer`) and **OmniSharp** —
//! is itself a .NET program and requires a .NET runtime on the user's machine.
//!
//! This does **not** violate KOPITIAM's Pure Rust Core: that promise governs
//! what *we compile* (this crate pulls in no C, C++ or MSBuild toolchain, and
//! `cargo build` remains the whole build), not which external tools a user
//! happens to have installed. rust-analyzer is an external binary too. But it
//! does mean C# symbol facts are **unavailable on a machine with no .NET**, and
//! the honest consequence is written into the code rather than papered over:
//! [`CSharpProvider::collect`] detects the server on `PATH` and, when none is
//! present, returns [`ProviderOutput::empty`] with a warning — never an error.
//! Per [`KnowledgeProvider`]'s contract, a provider whose tool is missing must
//! degrade, not fail the whole collection run: a machine without .NET should
//! still be able to index the Rust half of a mixed repository.
//!
//! # Why Roslyn LSP is preferred over OmniSharp
//!
//! Both are detected; Roslyn LSP wins when both are on `PATH`. OmniSharp is the
//! older editor backend (it speaks its own HTTP/stdio protocol natively and
//! only speaks LSP when launched with `-lsp`), and Microsoft's own C# editor
//! tooling migrated off it. `Microsoft.CodeAnalysis.LanguageServer` is what
//! ships in the modern VS Code C# extension, is maintained in the Roslyn
//! repository alongside the compiler itself, and is a native LSP server rather
//! than a protocol bridge. Preferring the actively-maintained one is the same
//! reasoning KOPITIAM applies everywhere: optimize for the decade, not the
//! afternoon.
//!
//! # Why project structure is read natively rather than asked of the server
//!
//! Solutions, projects, references and `using` directives are all recoverable
//! by *deterministic reading* — no language server, no .NET, no network. Asking
//! a model or even a language server for something a 200-line parser can
//! compute exactly is precisely what the Semantic Runtime forbids. It also
//! means the expensive, .NET-dependent part of this provider is confined to one
//! function ([`CSharpProvider::document_symbols`]), which is the only place the
//! runtime dependency actually bites.
//!
//! # Known gap: the live symbol path is blocked on two `LspClient` additions
//!
//! This provider deliberately **reuses** [`crate::lsp_client::LspClient`]
//! rather than writing a second LSP client. As of this writing that client
//! cannot yet drive a C# server, for two concrete reasons:
//!
//! 1. `LspClient::spawn(program, root, timeout)` passes **no arguments** to the
//!    server. OmniSharp speaks LSP only when launched as `OmniSharp -lsp`, and
//!    Roslyn LSP requires `--stdio` plus a log directory. Without argv, neither
//!    server can be started in a mode this client can talk to. Needed:
//!    a `spawn_with_args(program, args, root, timeout)` (or an `args` parameter
//!    on `spawn`).
//! 2. There is no `textDocument/documentSymbol` request. `workspace/symbol`
//!    (the only symbol query `LspClient` exposes) returns *flat*
//!    `SymbolInformation`, which throws away the nesting — method inside class
//!    inside namespace — that this provider models as containment
//!    relationships. Needed, roughly:
//!
//!    ```ignore
//!    /// Hierarchical `DocumentSymbol[]` for one open document.
//!    pub fn document_symbols(&mut self, uri: &str) -> Result<Vec<Value>> {
//!        let result: Value = self.request(
//!            "textDocument/documentSymbol",
//!            json!({ "textDocument": { "uri": uri } }),
//!            Duration::from_secs(60),
//!        )?;
//!        match result { Value::Array(items) => Ok(items), Value::Null => Ok(Vec::new()), other => bail!("...") }
//!    }
//!    ```
//!
//!    (`LspClient::did_open` also hardcodes `"languageId": "rust"`; it needs to
//!    take the language id. Servers mostly key off the URI extension, so this
//!    is the least severe of the three.)
//!
//! Those three edits are one file away, but that file is shared with three
//! other language adapters being written concurrently, so it is not edited
//! here. Everything downstream of the request — the `DocumentSymbol` tree
//! walker, the position conversion, the containment/inheritance edges — is
//! implemented and tested against the exact JSON both servers emit
//! ([`SymbolPass`] and its tests), so closing the gap is a body swap inside
//! [`CSharpProvider::document_symbols`] and nothing else.
//!
//! Until then, `collect` still emits the full **structural** graph (solutions,
//! projects, source files, NuGet packages, `using` edges) whenever a C# server
//! is present, and nothing at all when one is not.
//!
//! # The C# -> ontology mapping
//!
//! Every entity carries `source == "csharp"` (this provider's [`name`]) and
//! `metadata.language == "csharp"`, so a consumer can tell a C# `class` from a
//! Python `class` — while both remain an [`EntityKind::Symbol`], which is the
//! entire point of a shared vocabulary.
//!
//! | C# construct | Ontology |
//! |---|---|
//! | `.sln` solution | [`EntityKind::Artifact`], `metadata.kind = "solution"` |
//! | `.csproj` project | [`EntityKind::Artifact`], `metadata.kind = "project"` |
//! | `.cs` source file | [`EntityKind::Artifact`], `metadata.kind = "source_file"` |
//! | NuGet package | [`EntityKind::Artifact`], `metadata.kind = "package"` |
//! | namespace | [`EntityKind::Symbol`], `metadata.kind = "namespace"` |
//! | class / record / struct / interface / enum / delegate | [`EntityKind::Symbol`], `metadata.kind` = the declaration keyword |
//! | method / property / field / event / constructor / enum member | [`EntityKind::Symbol`], `metadata.kind` = the LSP symbol kind |
//!
//! | Fact | Relationship |
//! |---|---|
//! | project belongs to solution | `project` -[`LocatedIn`]-> `solution` |
//! | file belongs to project | `file` -[`LocatedIn`]-> `project` |
//! | symbol is defined in file | `symbol` -[`LocatedIn`]-> `file` |
//! | member is inside a type/namespace | `member` -[`LocatedIn`]-> `containing symbol` |
//! | `<ProjectReference>` | `project` -[`DependsOn`]-> `project` |
//! | `<PackageReference>` | `project` -[`DependsOn`]-> `package` |
//! | `using Foo.Bar;` | `file` -[`DependsOn`]-> `namespace symbol` |
//! | `class Derived : Base` / `: IFace` | `Base`/`IFace` -[`ImplementedBy`]-> `Derived` |
//!
//! [`LocatedIn`]: RelationshipKind::LocatedIn
//! [`DependsOn`]: RelationshipKind::DependsOn
//! [`ImplementedBy`]: RelationshipKind::ImplementedBy
//! [`name`]: KnowledgeProvider::name
//!
//! Two mapping judgments are worth stating out loud, because four language
//! adapters must agree on this vocabulary:
//!
//! * **Containment is `LocatedIn`, at every level.** Symbol-in-file,
//!   member-in-type, file-in-project, project-in-solution are all the same
//!   relation ("x is inside y"), so they all get the same edge rather than four
//!   near-synonyms. `DependsOn` is reserved for edges that cross a boundary the
//!   build system knows about.
//! * **Inheritance and interface implementation both become `ImplementedBy`,
//!   pointing base -> derived.** The ontology has no `Inherits`/`Extends`
//!   variant, and inventing a `Custom("inherits")` string would only work if
//!   every other language adapter guessed the same string. `ImplementedBy`
//!   ("`IFoo` is implemented by `Foo`") reads exactly right for interfaces and
//!   acceptably for base classes. If the ontology ever gains a distinct
//!   inheritance edge, split these — the `base_types` metadata this provider
//!   records on each type symbol preserves the raw fact either way, so no
//!   knowledge is lost by the merge.
//!
//! Artifacts are named by their path **relative to the collection root**, not
//! by an absolute `file://` URI, so a persisted graph does not bake `/home/you`
//! into every node (the absolute URI is kept in `metadata.uri`). Reproducible,
//! machine-independent facts are worth more than a byte-identical match with
//! any one server's URI spelling.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipKind};
use serde_json::{Value, json};

use crate::position::{PositionEncoding, unit_to_char_col};
use crate::provider::{KnowledgeProvider, ProviderOutput};

/// Which C# language server this provider found, in preference order.
///
/// Both require a .NET runtime (see the module docs). Both are detected by a
/// plain `PATH` lookup rather than by executing them: a server launched with an
/// argument it does not understand may sit waiting on stdin forever, and
/// probing a tool by running it is a poor way to ask "is it installed?".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CSharpServer {
    /// `Microsoft.CodeAnalysis.LanguageServer` — the Roslyn-native LSP server
    /// that modern VS Code C# uses. Preferred: Microsoft maintains it in the
    /// Roslyn repository, and it speaks LSP natively rather than bridging to
    /// another protocol.
    RoslynLsp,
    /// `OmniSharp` — the older editor backend. Speaks LSP only when launched
    /// with `-lsp`; without that flag it speaks the OmniSharp protocol, which
    /// this crate's LSP client cannot read.
    OmniSharp,
}

impl CSharpServer {
    /// Every executable name this server is plausibly installed as. OmniSharp's
    /// casing differs between upstream's release archive (`OmniSharp`) and most
    /// distribution packages (`omnisharp`), and both appear in the wild.
    pub const fn executables(&self) -> &'static [&'static str] {
        match self {
            Self::RoslynLsp => &["Microsoft.CodeAnalysis.LanguageServer"],
            Self::OmniSharp => &["OmniSharp", "omnisharp"],
        }
    }

    /// Stable identifier recorded in every entity's `metadata.server`, so a
    /// consumer can tell which server a symbol fact came from — two servers can
    /// legitimately disagree, and provenance is a hard requirement.
    pub const fn id(&self) -> &'static str {
        match self {
            Self::RoslynLsp => "roslyn-lsp",
            Self::OmniSharp => "omnisharp",
        }
    }

    /// The argv that puts this server into stdio-LSP mode.
    ///
    /// Recorded here rather than at the call site because it is *format
    /// knowledge*, not plumbing: `OmniSharp` without `-lsp` speaks a completely
    /// different protocol, and Roslyn LSP without `--stdio` may choose a named
    /// pipe. Getting this wrong does not produce an error — it produces a
    /// server that never answers, which is far harder to diagnose.
    ///
    /// Not yet consumed: `LspClient::spawn` accepts no argv (see the module
    /// docs' "Known gap"). It is written down now so that closing the gap does
    /// not require rediscovering it.
    pub const fn lsp_args(&self) -> &'static [&'static str] {
        match self {
            Self::RoslynLsp => &["--stdio", "--logLevel", "Warning"],
            Self::OmniSharp => &["-lsp"],
        }
    }

    /// The servers this provider knows about, in preference order.
    pub const ALL: &'static [Self] = &[Self::RoslynLsp, Self::OmniSharp];

    /// Finds the first known server on `path_var` (a `PATH`-style string).
    ///
    /// `path_var` is a parameter rather than an ambient read of the process
    /// environment so that tests can point detection at an isolated directory:
    /// `std::env::set_var` is unsound to call from a multi-threaded test binary.
    pub fn detect_in(path_var: &OsStr) -> Option<(Self, PathBuf)> {
        Self::ALL.iter().find_map(|server| {
            server
                .executables()
                .iter()
                .find_map(|exe| which_in(exe, path_var))
                .map(|path| (*server, path))
        })
    }
}

/// Searches a `PATH`-style string for an executable file. Deliberately a local
/// helper: `kopitiam-neovim` has an equivalent, but this crate must not depend
/// on an editor crate (the Semantic Runtime's dependency direction flows the
/// other way), and a fifteen-line `PATH` walk is not worth a new dependency.
fn which_in(executable: &str, path_var: &OsStr) -> Option<PathBuf> {
    let names: Vec<String> = if cfg!(windows) && !executable.to_ascii_lowercase().ends_with(".exe") {
        vec![executable.to_string(), format!("{executable}.exe")]
    } else {
        vec![executable.to_string()]
    };
    std::env::split_paths(path_var).find_map(|dir| {
        names
            .iter()
            .map(|name| dir.join(name))
            .find(|candidate| is_executable_file(candidate))
    })
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = path.metadata() else { return false };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Facts about a C# codebase: solutions, projects, source files, NuGet
/// dependencies, `using` edges, and (once the language server can be driven —
/// see the module docs) the full nested symbol tree.
///
/// See the module documentation for the .NET-runtime dependency, the
/// degradation contract, and the complete C# -> ontology mapping table.
pub struct CSharpProvider {
    /// The `PATH` used for server detection. `None` means "read the real
    /// process `PATH` at collection time" — the production case. Tests inject a
    /// value instead of mutating process-global state.
    path_var: Option<OsString>,
}

impl CSharpProvider {
    pub fn new() -> Self {
        Self { path_var: None }
    }

    /// A provider that searches `path_var` instead of the process `PATH` when
    /// looking for a C# language server. See [`CSharpServer::detect_in`].
    pub fn with_path(path_var: impl Into<OsString>) -> Self {
        Self {
            path_var: Some(path_var.into()),
        }
    }

    /// The C# language server this provider would use, if any. `None` is a
    /// normal, expected outcome — it means "this machine has no C# server (very
    /// likely: no .NET at all)", and [`Self::collect`] degrades accordingly.
    pub fn detect_server(&self) -> Option<(CSharpServer, PathBuf)> {
        match &self.path_var {
            Some(path_var) => CSharpServer::detect_in(path_var),
            None => CSharpServer::detect_in(std::env::var_os("PATH")?.as_os_str()),
        }
    }

    /// Hierarchical `textDocument/documentSymbol` results for each file, plus
    /// the position encoding the server negotiated.
    ///
    /// **This is the one function that needs a live .NET language server**, and
    /// the only one blocked by the `LspClient` gaps described in the module
    /// docs (no argv on `spawn`, no `documentSymbol` request). It therefore
    /// returns no symbols today, and `collect` continues with the structural
    /// facts rather than failing — the same degradation the trait requires when
    /// a tool is missing entirely.
    ///
    /// The returned [`PositionEncoding`] is UTF-16, which is not a guess: LSP
    /// 3.17 mandates UTF-16 as the encoding in force when a server does not
    /// negotiate another one. Once `LspClient` can be driven, this must return
    /// the *negotiated* encoding (`LspClient::position_encoding()`), because
    /// every symbol column below is converted through it — get it wrong and
    /// every symbol on a line containing a non-ASCII character is silently
    /// misplaced. [`SymbolPass`] already handles all three encodings correctly;
    /// see `symbol_positions_survive_non_ascii_content_in_every_encoding`.
    fn document_symbols(
        &self,
        server: CSharpServer,
        server_path: &Path,
        files: &[PathBuf],
    ) -> (PositionEncoding, HashMap<PathBuf, Vec<Value>>) {
        tracing::warn!(
            server = server.id(),
            server_path = %server_path.display(),
            files = files.len(),
            args = ?server.lsp_args(),
            "C# language server found, but the shared LspClient cannot yet drive it \
             (no argv on spawn, no textDocument/documentSymbol request); emitting structural \
             facts only. See the csharp module docs for the exact additions needed."
        );
        (PositionEncoding::Utf16, HashMap::new())
    }
}

impl Default for CSharpProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeProvider for CSharpProvider {
    fn name(&self) -> &str {
        "csharp"
    }

    fn collect(&self, root: &Path) -> Result<ProviderOutput> {
        let Some((server, server_path)) = self.detect_server() else {
            tracing::warn!(
                candidates = ?CSharpServer::ALL.iter().flat_map(|s| s.executables().iter().copied()).collect::<Vec<_>>(),
                "no C# language server on PATH (both Roslyn LSP and OmniSharp require a .NET runtime); \
                 skipping C# (no facts collected)"
            );
            return Ok(ProviderOutput::empty());
        };

        let root = root
            .canonicalize()
            .with_context(|| format!("resolving C# collection root {}", root.display()))?;
        let tree = CSharpTree::discover(&root)?;
        if tree.is_empty() {
            tracing::info!(root = %root.display(), "no .sln or .csproj found; nothing to collect for C#");
            return Ok(ProviderOutput::empty());
        }

        let mut collector = Collector::new(self.name(), server, &root);
        collector.emit_structure(&tree);

        let files: Vec<PathBuf> = tree.projects.iter().flat_map(|p| p.sources.clone()).collect();
        let (encoding, symbols_by_file) = self.document_symbols(server, &server_path, &files);
        for file in &files {
            let Some(text) = tree.text(file) else { continue };
            let doc_symbols = symbols_by_file.get(file).map(Vec::as_slice).unwrap_or(&[]);
            collector.emit_symbols(file, text, doc_symbols, encoding);
            collector.emit_usings(file, text);
        }
        collector.resolve_base_types();

        Ok(collector.finish())
    }
}

// ---------------------------------------------------------------------------
// The C# project model: solutions, projects, source files.
//
// Everything below is deterministic and needs no .NET, no MSBuild and no
// network. It is the half of C# knowledge that must never depend on a runtime
// the user might not have.
// ---------------------------------------------------------------------------

/// Directories that never contain source worth indexing. `bin`/`obj` are
/// MSBuild's build outputs and are *explicitly excluded from the implicit
/// `**/*.cs` glob by the SDK itself* — indexing them would double every symbol
/// in the graph with a copy from a generated `AssemblyInfo.cs`.
const IGNORED_DIRS: &[&str] = &["bin", "obj", ".git", ".vs", ".vscode", ".idea", "node_modules"];

/// A parsed C# tree: what was found on disk, before any of it becomes an
/// [`Entity`].
#[derive(Debug, Default)]
struct CSharpTree {
    solutions: Vec<Solution>,
    projects: Vec<Project>,
    /// Source text, read once and keyed by path — both the symbol pass (which
    /// needs line text to convert positions) and the `using` scan read it, and
    /// reading a file twice risks seeing two different versions of it.
    texts: HashMap<PathBuf, String>,
}

#[derive(Debug)]
struct Solution {
    path: PathBuf,
    /// Absolute paths of the `.csproj` files this solution lists. A `.sln` may
    /// reference projects that do not exist (a stale entry); those are dropped
    /// here rather than becoming phantom nodes in the graph.
    projects: Vec<PathBuf>,
}

#[derive(Debug)]
struct Project {
    path: PathBuf,
    manifest: CsProj,
    /// Absolute paths of the `.cs` files this project compiles, resolved
    /// through the implicit-glob rules (see [`CsProj::compiles`]).
    sources: Vec<PathBuf>,
}

impl CSharpTree {
    fn is_empty(&self) -> bool {
        self.solutions.is_empty() && self.projects.is_empty()
    }

    fn text(&self, file: &Path) -> Option<&str> {
        self.texts.get(file).map(String::as_str)
    }

    /// Walks `root` once, parses every `.sln` and `.csproj` found, and assigns
    /// each `.cs` file to the project that compiles it.
    fn discover(root: &Path) -> Result<Self> {
        let mut sln_paths = Vec::new();
        let mut csproj_paths = Vec::new();
        let mut cs_paths = Vec::new();
        walk(root, &mut |path| {
            match path.extension().and_then(OsStr::to_str) {
                Some("sln") => sln_paths.push(path.to_path_buf()),
                Some("csproj") => csproj_paths.push(path.to_path_buf()),
                Some("cs") => cs_paths.push(path.to_path_buf()),
                _ => {}
            };
        })?;

        let mut tree = Self::default();

        for path in &sln_paths {
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            let dir = path.parent().unwrap_or(root);
            let projects: Vec<PathBuf> = parse_sln(&text)
                .into_iter()
                .map(|rel| normalize_relative(dir, &rel))
                .filter(|p| p.exists())
                .collect();
            // A solution may list a project outside the walked root; keep it.
            for project in &projects {
                if !csproj_paths.contains(project) {
                    csproj_paths.push(project.clone());
                }
            }
            tree.solutions.push(Solution {
                path: path.clone(),
                projects,
            });
        }
        csproj_paths.sort();

        for path in &csproj_paths {
            let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            let manifest = CsProj::parse(&text);
            let dir = path.parent().unwrap_or(root).to_path_buf();
            // A `.cs` file belongs to the *deepest* project directory that
            // contains it: nested projects (`src/App/` inside `src/`) must not
            // both claim the same file.
            let owned: Vec<PathBuf> = cs_paths
                .iter()
                .filter(|cs| {
                    cs.starts_with(&dir)
                        && csproj_paths
                            .iter()
                            .filter(|other| *other != path)
                            .filter_map(|other| other.parent())
                            .all(|other_dir| !(cs.starts_with(other_dir) && other_dir.starts_with(&dir) && other_dir != dir))
                })
                .cloned()
                .collect();
            let sources = manifest.compiles(&dir, &owned);
            tree.projects.push(Project {
                path: path.clone(),
                manifest,
                sources,
            });
        }

        for project in &tree.projects {
            for source in &project.sources {
                if !tree.texts.contains_key(source) {
                    let text = std::fs::read_to_string(source).with_context(|| format!("reading {}", source.display()))?;
                    tree.texts.insert(source.clone(), text);
                }
            }
        }

        Ok(tree)
    }
}

/// Recursively visits every file under `dir`, skipping [`IGNORED_DIRS`].
/// Entries are sorted before descending: `read_dir` order is filesystem
/// dependent, and a knowledge graph whose contents depend on inode order is not
/// reproducible.
fn walk(dir: &Path, visit: &mut impl FnMut(&Path)) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading directory {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect();
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            let name = entry.file_name().and_then(OsStr::to_str).unwrap_or_default();
            if IGNORED_DIRS.contains(&name) {
                continue;
            }
            walk(&entry, visit)?;
        } else if entry.is_file() {
            visit(&entry);
        }
    }
    Ok(())
}

/// Resolves an MSBuild-style relative path (which uses `\` separators even on
/// Unix, because Visual Studio wrote it on Windows) against `base`.
fn normalize_relative(base: &Path, relative: &str) -> PathBuf {
    let mut path = base.to_path_buf();
    for segment in relative.replace('\\', "/").split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                path.pop();
            }
            other => path.push(other),
        }
    }
    path
}

/// The `.csproj` fields this provider needs. Not a general MSBuild evaluator —
/// property functions, imports, conditions and item transforms are all ignored,
/// and deliberately so: the goal is a faithful *dependency and source* picture,
/// not a reimplementation of MSBuild in Rust.
#[derive(Debug, Default, PartialEq)]
struct CsProj {
    /// `<Project Sdk="Microsoft.NET.Sdk">`. Its presence is what makes a
    /// project "SDK-style", which is what turns on the implicit source glob.
    sdk: Option<String>,
    /// `<EnableDefaultCompileItems>`, if set explicitly. `None` means "not set",
    /// which the SDK treats as `true`.
    enable_default_compile_items: Option<bool>,
    target_frameworks: Vec<String>,
    compile_includes: Vec<String>,
    compile_removes: Vec<String>,
    project_references: Vec<String>,
    package_references: Vec<(String, Option<String>)>,
}

impl CsProj {
    fn parse(text: &str) -> Self {
        let mut parsed = Self::default();
        for element in parse_xml_elements(text) {
            match element.name.as_str() {
                "Project" => parsed.sdk = element.attr("Sdk").map(str::to_string),
                "EnableDefaultCompileItems" => {
                    parsed.enable_default_compile_items = Some(element.text.trim().eq_ignore_ascii_case("true"));
                }
                // `<TargetFrameworks>` (plural) is a `;`-separated list; the
                // singular form holds exactly one. Both spellings are common.
                "TargetFramework" | "TargetFrameworks" => {
                    parsed.target_frameworks.extend(
                        element
                            .text
                            .split(';')
                            .map(str::trim)
                            .filter(|tfm| !tfm.is_empty())
                            .map(str::to_string),
                    );
                }
                "Compile" => {
                    if let Some(include) = element.attr("Include") {
                        parsed.compile_includes.push(include.to_string());
                    }
                    // `Remove` is the SDK-style spelling; `Exclude` the legacy
                    // one. Both mean "not part of this project".
                    for key in ["Remove", "Exclude"] {
                        if let Some(removed) = element.attr(key) {
                            parsed.compile_removes.push(removed.to_string());
                        }
                    }
                }
                "ProjectReference" => {
                    if let Some(include) = element.attr("Include") {
                        parsed.project_references.push(include.to_string());
                    }
                }
                "PackageReference" => {
                    if let Some(include) = element.attr("Include") {
                        let version = element
                            .attr("Version")
                            .map(str::to_string)
                            .or_else(|| element.attr("version").map(str::to_string));
                        parsed.package_references.push((include.to_string(), version));
                    }
                }
                _ => {}
            }
        }
        parsed
    }

    /// True when this project's sources are implicitly globbed rather than
    /// listed.
    ///
    /// **This is the C# format fact worth remembering.** Since the .NET SDK
    /// (2017), `<Project Sdk="Microsoft.NET.Sdk">` compiles every `**/*.cs`
    /// under the project directory *without listing any of them*, excluding
    /// `bin/` and `obj/`. A tool that only reads `<Compile Include>` sees an
    /// empty project and concludes, wrongly, that there is no code. Legacy
    /// (non-SDK) projects, and any project that sets
    /// `<EnableDefaultCompileItems>false</EnableDefaultCompileItems>` (which is
    /// how a modern project opts back into explicit listing), are the only ones
    /// where `<Compile Include>` tells the whole story.
    fn globs_sources_implicitly(&self) -> bool {
        match self.enable_default_compile_items {
            Some(explicit) => explicit,
            None => self.sdk.is_some(),
        }
    }

    /// The `.cs` files this project compiles, given `dir` (the project
    /// directory) and `candidates` (every `.cs` file found beneath it).
    fn compiles(&self, dir: &Path, candidates: &[PathBuf]) -> Vec<PathBuf> {
        let mut sources: Vec<PathBuf> = if self.globs_sources_implicitly() {
            candidates.to_vec()
        } else {
            Vec::new()
        };

        for include in &self.compile_includes {
            // An explicit `<Compile Include>` may be a literal path or an
            // MSBuild glob (`**/*.cs`). Both are honoured; a literal path
            // outside `candidates` (e.g. a linked file from a sibling
            // directory) is still added if it exists on disk.
            if include.contains('*') {
                sources.extend(candidates.iter().filter(|c| glob_matches(include, dir, c)).cloned());
            } else {
                let path = normalize_relative(dir, include);
                if path.exists() {
                    sources.push(path);
                }
            }
        }

        sources.retain(|source| {
            !self
                .compile_removes
                .iter()
                .any(|pattern| glob_matches(pattern, dir, source))
        });
        sources.sort();
        sources.dedup();
        sources
    }
}

/// Matches an MSBuild item pattern (`Excluded/**/*.cs`, `Legacy/Old.cs`)
/// against `path`, both interpreted relative to `dir`.
///
/// Supports the two wildcards that actually appear in `.csproj` files: `*`
/// (anything within one path segment) and `**` (any number of segments). Full
/// MSBuild item semantics — `$(Property)` expansion, `Condition`, exclude
/// ordering — are out of scope, and this provider says so rather than
/// pretending otherwise.
fn glob_matches(pattern: &str, dir: &Path, path: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(dir) else { return false };
    let Some(relative) = relative.to_str() else { return false };
    let path_segments: Vec<&str> = relative.split('/').collect();
    let normalized = pattern.replace('\\', "/");
    let pattern_segments: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty() && *s != ".").collect();
    glob_segments(&pattern_segments, &path_segments)
}

fn glob_segments(pattern: &[&str], path: &[&str]) -> bool {
    match (pattern.first(), path.first()) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some(&"**"), _) => {
            // `**` matches zero or more segments: try every split point.
            (0..=path.len()).any(|skip| glob_segments(&pattern[1..], &path[skip..]))
        }
        (Some(_), None) => false,
        (Some(segment), Some(name)) => segment_matches(segment, name) && glob_segments(&pattern[1..], &path[1..]),
    }
}

/// Matches a single path segment against a pattern segment containing zero or
/// more `*` wildcards (`*.cs`, `Foo*Bar.cs`).
fn segment_matches(pattern: &str, name: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else { return true };
    if !name.starts_with(first) {
        return false;
    }
    let mut rest = &name[first.len()..];
    let parts: Vec<&str> = parts.collect();
    let Some((last, middle)) = parts.split_last() else {
        // No `*` at all: an exact match.
        return rest.is_empty();
    };
    for part in middle {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    rest.len() >= last.len() && rest.ends_with(last)
}

/// One XML element, flattened: its name, its attributes, and its immediate text.
#[derive(Debug)]
struct XmlElement {
    name: String,
    attributes: Vec<(String, String)>,
    text: String,
}

impl XmlElement {
    fn attr(&self, key: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(key))
            .map(|(_, value)| value.as_str())
    }
}

/// A deliberately minimal XML reader: enough to pull six attribute lookups and
/// three element texts out of a `.csproj`, and no more.
///
/// KOPITIAM avoids unnecessary dependencies, and a full XML crate to read
/// `Include="..."` would be one. The trade-off is stated plainly: this handles
/// comments, self-closing tags, single- and double-quoted attributes, and
/// namespace prefixes; it does not handle CDATA, entity references beyond the
/// five predefined ones, or DTDs — none of which occur in a `.csproj` written
/// by any tool in the .NET ecosystem. If one ever does, the failure mode is a
/// missed reference, not a crash or a wrong fact.
fn parse_xml_elements(text: &str) -> Vec<XmlElement> {
    let text = strip_xml_comments(text);
    let bytes: Vec<char> = text.chars().collect();
    let mut elements = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != '<' {
            i += 1;
            continue;
        }
        i += 1;
        // Closing tags, declarations (`<?xml`) and doctypes carry nothing we want.
        if matches!(bytes.get(i), Some('/') | Some('?') | Some('!')) {
            while i < bytes.len() && bytes[i] != '>' {
                i += 1;
            }
            continue;
        }
        let name_start = i;
        while i < bytes.len() && !bytes[i].is_whitespace() && bytes[i] != '>' && bytes[i] != '/' {
            i += 1;
        }
        let raw_name: String = bytes[name_start..i].iter().collect();
        // Strip any XML namespace prefix: legacy `.csproj` files put every
        // element in the `msbuild` namespace, SDK-style ones use none.
        let name = raw_name.rsplit(':').next().unwrap_or(&raw_name).to_string();

        let mut attributes = Vec::new();
        let mut self_closing = false;
        while i < bytes.len() {
            while i < bytes.len() && bytes[i].is_whitespace() {
                i += 1;
            }
            match bytes.get(i) {
                None => break,
                Some('>') => {
                    i += 1;
                    break;
                }
                Some('/') => {
                    self_closing = true;
                    i += 1;
                    continue;
                }
                Some(_) => {}
            }
            let key_start = i;
            while i < bytes.len() && bytes[i] != '=' && !bytes[i].is_whitespace() && bytes[i] != '>' {
                i += 1;
            }
            let key: String = bytes[key_start..i].iter().collect();
            while i < bytes.len() && (bytes[i].is_whitespace() || bytes[i] == '=') {
                i += 1;
            }
            let Some(&quote) = bytes.get(i) else { break };
            if quote != '"' && quote != '\'' {
                continue;
            }
            i += 1;
            let value_start = i;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            let value: String = bytes[value_start..i].iter().collect();
            i += 1;
            if !key.is_empty() {
                attributes.push((key, unescape_xml(&value)));
            }
        }

        let mut element_text = String::new();
        if !self_closing {
            let text_start = i;
            while i < bytes.len() && bytes[i] != '<' {
                i += 1;
            }
            element_text = unescape_xml(&bytes[text_start..i].iter().collect::<String>());
        }
        elements.push(XmlElement {
            name,
            attributes,
            text: element_text,
        });
    }
    elements
}

fn strip_xml_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<!--") {
        out.push_str(&rest[..start]);
        match rest[start..].find("-->") {
            Some(end) => rest = &rest[start + end + 3..],
            None => return out, // unterminated comment: everything after is a comment
        }
    }
    out.push_str(rest);
    out
}

fn unescape_xml(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Extracts the project paths a `.sln` lists.
///
/// A solution file is not XML — it is a line-oriented Visual Studio format:
///
/// ```text
/// Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "App", "src\App\App.csproj", "{9A19103F-...}"
/// ```
///
/// The quoted fields are, in order: the *project type* GUID, the display name,
/// the path, and the project GUID. Solution *folders* use the same line shape
/// but their "path" is just the folder name, so filtering on a `.csproj`
/// extension is what separates real projects from folders. (The newer XML
/// `.slnx` format is not yet handled; it is rare in the wild as of writing.)
fn parse_sln(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| line.trim_start().starts_with("Project("))
        .filter_map(|line| {
            let quoted: Vec<&str> = line.split('"').skip(1).step_by(2).collect();
            quoted.get(2).copied().map(str::to_string)
        })
        .filter(|path| path.to_ascii_lowercase().ends_with(".csproj"))
        .collect()
}

// ---------------------------------------------------------------------------
// Fact emission: the C# model above, expressed in the shared ontology.
// ---------------------------------------------------------------------------

/// Accumulates entities and relationships, and keeps the name -> id indexes
/// needed to connect facts discovered at different times (a `using` seen in one
/// file must point at the namespace symbol declared in another).
struct Collector<'a> {
    source: &'a str,
    server: CSharpServer,
    root: &'a Path,
    entities: Vec<Entity>,
    relationships: Vec<Relationship>,
    /// Artifact ids by absolute path (projects and source files).
    artifacts: HashMap<PathBuf, EntityId>,
    /// Namespace symbols by fully-qualified name — created either by a
    /// declaration or by a `using` that names them, whichever is seen first, so
    /// that `using MyApp.Models;` and `namespace MyApp.Models` converge on one
    /// node instead of two.
    namespaces: HashMap<String, EntityId>,
    /// Type symbols, indexed under *both* their fully-qualified and simple
    /// names, so a base type written as `Base` resolves to `MyApp.Base`.
    types: HashMap<String, Vec<EntityId>>,
    /// `(derived type, base type as written)`, resolved once every file has been
    /// seen — a class can inherit from a type declared in a file read later.
    pending_bases: Vec<(EntityId, String)>,
}

impl<'a> Collector<'a> {
    fn new(source: &'a str, server: CSharpServer, root: &'a Path) -> Self {
        Self {
            source,
            server,
            root,
            entities: Vec::new(),
            relationships: Vec::new(),
            artifacts: HashMap::new(),
            namespaces: HashMap::new(),
            types: HashMap::new(),
            pending_bases: Vec::new(),
        }
    }

    fn finish(self) -> ProviderOutput {
        ProviderOutput {
            entities: self.entities,
            relationships: self.relationships,
        }
    }

    fn push(&mut self, entity: Entity) -> EntityId {
        let id = entity.id;
        self.entities.push(entity);
        id
    }

    fn relate(&mut self, from: EntityId, to: EntityId, kind: RelationshipKind) {
        self.relationships.push(Relationship::new(from, to, kind));
    }

    /// The path a fact is named by: relative to the collection root, with `/`
    /// separators on every platform, so the graph is portable between machines.
    fn relative(&self, path: &Path) -> String {
        path.strip_prefix(self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    }

    fn uri(&self, path: &Path) -> String {
        url::Url::from_file_path(path)
            .map(|url| url.to_string())
            .unwrap_or_else(|()| format!("file://{}", path.display()))
    }

    fn artifact(&mut self, path: &Path, kind: &str, extra: Value) -> EntityId {
        if let Some(&id) = self.artifacts.get(path) {
            return id;
        }
        let mut metadata = json!({
            "kind": kind,
            "language": "csharp",
            "server": self.server.id(),
            "path": self.relative(path),
            "uri": self.uri(path),
        });
        merge_metadata(&mut metadata, extra);
        let id = self.push(Entity::new(EntityKind::Artifact, self.relative(path), self.source).with_metadata(metadata));
        self.artifacts.insert(path.to_path_buf(), id);
        id
    }

    /// Solutions, projects, source files, packages, and the containment and
    /// dependency edges between them.
    fn emit_structure(&mut self, tree: &CSharpTree) {
        for project in &tree.projects {
            let manifest = &project.manifest;
            self.artifact(
                &project.path,
                "project",
                json!({
                    "sdk": manifest.sdk,
                    "sdk_style": manifest.sdk.is_some(),
                    "implicit_source_glob": manifest.globs_sources_implicitly(),
                    "target_frameworks": manifest.target_frameworks,
                    "source_count": project.sources.len(),
                }),
            );
        }

        for solution in &tree.solutions {
            let solution_id = self.artifact(&solution.path, "solution", json!({ "project_count": solution.projects.len() }));
            for project_path in &solution.projects {
                if let Some(&project_id) = self.artifacts.get(project_path) {
                    self.relate(project_id, solution_id, RelationshipKind::LocatedIn);
                }
            }
        }

        // NuGet packages are deduplicated by `name@version`: two projects
        // referencing `Newtonsoft.Json 13.0.3` depend on the *same* package.
        let mut packages: HashMap<String, EntityId> = HashMap::new();
        for project in &tree.projects {
            let Some(&project_id) = self.artifacts.get(&project.path) else { continue };

            for reference in &project.manifest.project_references {
                let referenced = normalize_relative(project.path.parent().unwrap_or(self.root), reference);
                if let Some(&referenced_id) = self.artifacts.get(&referenced) {
                    self.relate(project_id, referenced_id, RelationshipKind::DependsOn);
                }
            }

            for (package, version) in &project.manifest.package_references {
                let key = format!("{package}@{}", version.as_deref().unwrap_or("*"));
                let package_id = match packages.get(&key) {
                    Some(&id) => id,
                    None => {
                        let entity = Entity::new(EntityKind::Artifact, package.clone(), self.source).with_metadata(json!({
                            "kind": "package",
                            "language": "csharp",
                            "ecosystem": "nuget",
                            "server": self.server.id(),
                            "version": version,
                        }));
                        let id = self.push(entity);
                        packages.insert(key, id);
                        id
                    }
                };
                self.relate(project_id, package_id, RelationshipKind::DependsOn);
            }

            for source in &project.sources {
                let file_id = self.artifact(source, "source_file", Value::Null);
                self.relate(file_id, project_id, RelationshipKind::LocatedIn);
            }
        }
    }

    /// Walks one file's hierarchical `DocumentSymbol` tree into symbols and
    /// containment edges. See [`SymbolPass`].
    fn emit_symbols(&mut self, file: &Path, text: &str, doc_symbols: &[Value], encoding: PositionEncoding) {
        let Some(&file_id) = self.artifacts.get(file) else { return };
        let lines: Vec<&str> = text.lines().collect();
        let uri = self.uri(file);
        let mut pass = SymbolPass {
            collector: self,
            file_id,
            uri,
            lines,
            encoding,
        };
        for symbol in doc_symbols {
            pass.visit(symbol, None, "");
        }
    }

    /// `using` directives: an edge from the file to each namespace (or, for
    /// `using static`, type) it imports. This is the cheapest cross-file
    /// dependency signal C# has, and it needs no language server at all.
    fn emit_usings(&mut self, file: &Path, text: &str) {
        let Some(&file_id) = self.artifacts.get(file) else { return };
        for directive in using_directives(text) {
            let target_id = self.import_target(&directive);
            self.relate(file_id, target_id, RelationshipKind::DependsOn);
        }
    }

    /// The symbol a `using` names, reusing the declared namespace or type when
    /// this project declares it, and creating an external symbol (`System`,
    /// `Newtonsoft.Json`) when it does not.
    fn import_target(&mut self, directive: &UsingDirective) -> EntityId {
        if let Some(&id) = self.namespaces.get(&directive.target) {
            return id;
        }
        if let Some(ids) = self.types.get(&directive.target)
            && ids.len() == 1
        {
            return ids[0];
        }
        let kind = if directive.is_static { "type" } else { "namespace" };
        let entity = Entity::new(EntityKind::Symbol, directive.target.clone(), self.source).with_metadata(json!({
            "kind": kind,
            "language": "csharp",
            "server": self.server.id(),
            "imported": true,
            "static": directive.is_static,
            "alias": directive.alias,
            "global": directive.is_global,
        }));
        let id = self.push(entity);
        if !directive.is_static {
            self.namespaces.insert(directive.target.clone(), id);
        }
        id
    }

    /// Turns every `class Derived : Base` seen during the symbol pass into
    /// `Base -ImplementedBy-> Derived`, but **only** when `Base` resolves
    /// unambiguously to a type this project declares.
    ///
    /// Unresolvable bases (`System.Exception`, `IDisposable`) are deliberately
    /// *not* materialized as entities: a `using`-derived guess at what
    /// `IDisposable` refers to would be an inference, not a fact, and the raw
    /// base-type list is already preserved in each type's `base_types`
    /// metadata. Ambiguous simple names (two `Base` types in different
    /// namespaces) are skipped for the same reason — resolving them correctly
    /// needs the compiler's name binding, which is exactly what the language
    /// server would give us once it can be driven.
    fn resolve_base_types(&mut self) {
        let pending = std::mem::take(&mut self.pending_bases);
        for (derived_id, base) in pending {
            let head = base.split('<').next().unwrap_or(&base).trim().to_string();
            let Some(candidates) = self.types.get(&head) else { continue };
            let mut unique: Vec<EntityId> = candidates.iter().copied().filter(|&id| id != derived_id).collect();
            unique.dedup();
            if unique.len() == 1 {
                self.relate(unique[0], derived_id, RelationshipKind::ImplementedBy);
            }
        }
    }
}

/// Merges `extra`'s object fields into `base`. `Value::Null` extras are a no-op,
/// which keeps the call sites free of `Option` noise.
fn merge_metadata(base: &mut Value, extra: Value) {
    let (Some(base), Value::Object(extra)) = (base.as_object_mut(), extra) else { return };
    base.extend(extra);
}

// ---------------------------------------------------------------------------
// The symbol pass: LSP `DocumentSymbol` tree -> ontology symbols.
// ---------------------------------------------------------------------------

/// Walks one document's hierarchical `DocumentSymbol` tree.
///
/// Hierarchical `textDocument/documentSymbol` is used rather than flat
/// `workspace/symbol` precisely because C# nests: a method lives in a class,
/// which lives in a namespace. `SymbolInformation` flattens that into a
/// `containerName` string; `DocumentSymbol` keeps the tree, and this pass turns
/// the tree back into [`RelationshipKind::LocatedIn`] edges — real graph
/// structure a query can traverse, rather than a string a consumer would have
/// to re-parse.
struct SymbolPass<'a, 'b> {
    collector: &'a mut Collector<'b>,
    file_id: EntityId,
    uri: String,
    /// The file's lines, needed to convert LSP wire columns into `char` columns
    /// and to read declaration headers (modifiers, base types) that
    /// `DocumentSymbol` does not carry.
    lines: Vec<&'a str>,
    encoding: PositionEncoding,
}

impl SymbolPass<'_, '_> {
    /// Emits one `DocumentSymbol` and recurses into its children.
    ///
    /// `container` is the entity a nested symbol is contained by (a type, or a
    /// namespace); `qualifier` is the dotted path that prefixes its name
    /// (`MyApp.Models`), which is what makes fully-qualified lookup — and hence
    /// base-type resolution across files — possible.
    fn visit(&mut self, symbol: &Value, container: Option<EntityId>, qualifier: &str) {
        let Some(name) = symbol.get("name").and_then(Value::as_str) else { return };
        if name.is_empty() {
            return;
        }
        let lsp_kind = symbol.get("kind").and_then(Value::as_u64).unwrap_or(0);
        let range = self.position(symbol.pointer("/range/start"));
        let range_end = self.position(symbol.pointer("/range/end"));
        // `selectionRange` is the identifier itself; `range` is the whole
        // declaration including modifiers and attributes. The gap between them
        // is the declaration header, which is where C#'s accessibility and its
        // `class`/`record`/`struct` keyword live — neither of which the LSP
        // symbol kind can tell us (`record` has no SymbolKind of its own).
        let name_position = self.position(symbol.pointer("/selectionRange/start")).or(range);
        let name_end = self.position(symbol.pointer("/selectionRange/end"));

        let header = match (range, name_position) {
            (Some(start), Some(end)) => self.slice(start, end),
            _ => String::new(),
        };
        let declared = DeclarationHeader::parse(&header);
        let kind = declared
            .keyword
            .clone()
            .unwrap_or_else(|| lsp_symbol_kind_name(lsp_kind).to_string());

        // An enum's `: byte` is its *underlying type*, not a base class, and an
        // enum can implement nothing — reading it as inheritance would be a
        // fabricated fact.
        let base_types = if is_inheritable(&kind) {
            name_end.map(|end| self.base_types(end)).unwrap_or_default()
        } else {
            Vec::new()
        };

        let qualified = if qualifier.is_empty() {
            name.to_string()
        } else {
            format!("{qualifier}.{name}")
        };

        let entity = Entity::new(EntityKind::Symbol, name, self.collector.source).with_metadata(json!({
            "kind": kind,
            "lsp_kind": lsp_kind,
            "language": "csharp",
            "server": self.collector.server.id(),
            "uri": self.uri,
            "line": name_position.map(|(line, _)| line),
            "character": name_position.map(|(_, character)| character),
            "end_line": range_end.map(|(line, _)| line),
            "qualified_name": qualified,
            "container": (!qualifier.is_empty()).then(|| qualifier.to_string()),
            "accessibility": declared.accessibility,
            "modifiers": declared.modifiers,
            "base_types": base_types,
            "detail": symbol.get("detail").and_then(Value::as_str),
        }));
        let id = self.collector.push(entity);

        // Containment: a nested symbol is located in its container, and every
        // symbol is located in its file. Both edges are emitted — "which type is
        // this method in" and "which file is this method in" are both questions
        // the graph should answer without a join.
        self.collector.relate(id, self.file_id, RelationshipKind::LocatedIn);
        if let Some(container) = container {
            self.collector.relate(id, container, RelationshipKind::LocatedIn);
        }

        if kind == "namespace" {
            self.collector.namespaces.insert(qualified.clone(), id);
        } else if is_type(&kind) {
            // Indexed under both spellings so a base written as `Base` finds
            // `MyApp.Base`. A top-level type's qualified name *is* its simple
            // name, so guard against indexing the same id twice — a duplicate
            // would later look like two candidate types with the same name and
            // be discarded as ambiguous.
            for key in [qualified.clone(), name.to_string()] {
                let candidates = self.collector.types.entry(key).or_default();
                if !candidates.contains(&id) {
                    candidates.push(id);
                }
            }
        }
        for base in base_types {
            self.collector.pending_bases.push((id, base));
        }

        if let Some(children) = symbol.get("children").and_then(Value::as_array) {
            for child in children {
                self.visit(child, Some(id), &qualified);
            }
        }
    }

    /// An LSP `Position` as `(line, char column)`.
    ///
    /// The `character` on the wire is in the encoding the server negotiated —
    /// bytes for `"utf-8"`, code units for `"utf-16"`, characters for
    /// `"utf-32"`. Converting it here, once, is what keeps every symbol on a
    /// line containing a `£`, a `日`, or an emoji from being silently
    /// misplaced. See [`crate::position`] for the full rundown.
    fn position(&self, position: Option<&Value>) -> Option<(u32, u32)> {
        let position = position?;
        let line = position.get("line").and_then(Value::as_u64)? as u32;
        let unit = position.get("character").and_then(Value::as_u64)? as u32;
        let text = self.lines.get(line as usize).copied().unwrap_or("");
        Some((line, unit_to_char_col(text, unit, self.encoding)))
    }

    /// The text between two `(line, char column)` positions.
    fn slice(&self, start: (u32, u32), end: (u32, u32)) -> String {
        let mut out = String::new();
        for line in start.0..=end.0 {
            let text = self.lines.get(line as usize).copied().unwrap_or("");
            let from = if line == start.0 { start.1 as usize } else { 0 };
            let to = if line == end.0 { end.1 as usize } else { text.chars().count() };
            out.extend(text.chars().skip(from).take(to.saturating_sub(from)));
            if line != end.0 {
                out.push('\n');
            }
        }
        out
    }

    /// The base class and interfaces of a type declared with its name ending at
    /// `after` — read from the source, because `DocumentSymbol` does not carry
    /// them and a full type hierarchy request would cost a second round trip
    /// per symbol.
    ///
    /// C# puts them after a `:` that follows the name, its generic parameters,
    /// and (C# 12) a primary-constructor parameter list:
    ///
    /// ```text
    /// public sealed record Solver<T>(int Steps) : SolverBase<T>, IDisposable where T : struct
    ///                                           ^-- from here to `where`/`{`/`;`
    /// ```
    ///
    /// The scan therefore skips balanced `<...>` and `(...)` before looking for
    /// the `:`, and stops at `where` (a generic *constraint*, whose `:` means
    /// something else entirely — misreading it would invent inheritance edges
    /// that do not exist).
    fn base_types(&self, after: (u32, u32)) -> Vec<String> {
        const MAX_HEADER_CHARS: usize = 512;
        let text: String = self.slice(after, (after.0 + 4, 0)).chars().take(MAX_HEADER_CHARS).collect();
        let mut chars = text.chars().peekable();
        let mut depth = 0i32;
        // Skip generic parameters and any primary-constructor parameter list.
        loop {
            let Some(&c) = chars.peek() else { return Vec::new() };
            match c {
                '<' | '(' | '[' => depth += 1,
                '>' | ')' | ']' => depth -= 1,
                ':' if depth == 0 => break,
                '{' | ';' | '=' if depth == 0 => return Vec::new(),
                c if c.is_whitespace() || depth > 0 => {}
                // Any other token at depth 0 before a `:` (e.g. `where`) means
                // there is no base list.
                _ => return Vec::new(),
            }
            chars.next();
        }
        chars.next(); // consume ':'

        let mut bases = Vec::new();
        let mut current = String::new();
        let mut depth = 0i32;
        for c in chars {
            match c {
                '<' | '(' | '[' => {
                    depth += 1;
                    current.push(c);
                }
                '>' | ')' | ']' => {
                    depth -= 1;
                    current.push(c);
                }
                ',' if depth == 0 => {
                    push_base(&mut bases, &mut current);
                }
                '{' | ';' if depth == 0 => break,
                _ => current.push(c),
            }
            if depth == 0 && current.trim_end().ends_with("where") && current.trim() != "where" {
                // `... IDisposable where T : struct` — the constraint clause
                // begins; drop it and stop.
                let trimmed = current.trim_end();
                current = trimmed[..trimmed.len() - "where".len()].to_string();
                break;
            }
        }
        push_base(&mut bases, &mut current);
        bases.retain(|base| base != "where");
        bases
    }
}

fn push_base(bases: &mut Vec<String>, current: &mut String) {
    let base = current.trim().trim_end_matches("where").trim().to_string();
    current.clear();
    if !base.is_empty() {
        bases.push(base);
    }
}

/// What a C# declaration header (`[Obsolete] public sealed partial class`)
/// tells us that the LSP symbol kind cannot.
#[derive(Debug, Default, PartialEq)]
struct DeclarationHeader {
    /// `public`, `internal`, `protected internal`, ... or `None` when the
    /// declaration states none. `None` is recorded honestly rather than filled
    /// in with C#'s default (`private` for members, `internal` for top-level
    /// types): "not declared" is the fact, and a consumer that wants the
    /// language default can apply it knowing it is a rule, not an observation.
    accessibility: Option<String>,
    /// `static`, `abstract`, `sealed`, `partial`, `async`, ... in declaration order.
    modifiers: Vec<String>,
    /// The declaration keyword: `class`, `record`, `record struct`, `struct`,
    /// `interface`, `enum`, `delegate`, `namespace`. This is the *only* way to
    /// tell a `record` from a `class` — LSP reports both as `SymbolKind.Class`,
    /// and a translation engine that cannot see the difference would translate
    /// C#'s value-semantics record into a mutable Rust struct.
    keyword: Option<String>,
}

const ACCESS_MODIFIERS: &[&str] = &["public", "private", "protected", "internal", "file"];
const OTHER_MODIFIERS: &[&str] = &[
    "static", "abstract", "sealed", "virtual", "override", "readonly", "const", "async", "partial", "extern", "unsafe",
    "new", "required", "volatile", "implicit", "explicit", "ref", "event",
];
const DECLARATION_KEYWORDS: &[&str] = &["namespace", "class", "struct", "interface", "enum", "record", "delegate"];

impl DeclarationHeader {
    fn parse(header: &str) -> Self {
        // Attributes (`[Obsolete("...")]`) sit inside the declaration range but
        // are not modifiers; drop them, brackets and all.
        let mut cleaned = String::with_capacity(header.len());
        let mut depth = 0i32;
        for c in header.chars() {
            match c {
                '[' => depth += 1,
                ']' => depth -= 1,
                c if depth == 0 => cleaned.push(c),
                _ => {}
            }
        }

        let tokens: Vec<&str> = cleaned.split_whitespace().collect();
        let mut parsed = Self::default();
        let mut access: Vec<&str> = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            if ACCESS_MODIFIERS.contains(token) {
                access.push(token);
            } else if OTHER_MODIFIERS.contains(token) {
                parsed.modifiers.push((*token).to_string());
            } else if DECLARATION_KEYWORDS.contains(token) && parsed.keyword.is_none() {
                // `record struct` / `record class` are two tokens naming one
                // construct; keep them together. Taking only the *first*
                // declaration keyword is what stops the `struct` in
                // `record struct` from overwriting the `record` that qualifies
                // it — a header never declares two constructs.
                parsed.keyword = Some(match (*token, tokens.get(i + 1).copied()) {
                    ("record", Some(next @ ("struct" | "class"))) => format!("record {next}"),
                    (other, _) => other.to_string(),
                });
            }
        }
        if !access.is_empty() {
            parsed.accessibility = Some(access.join(" "));
        }
        parsed
    }
}

/// True for constructs that can declare a base type or interface list.
fn is_inheritable(kind: &str) -> bool {
    matches!(kind, "class" | "struct" | "interface" | "record" | "record struct" | "record class")
}

/// True for constructs that name a type (and can therefore be the *target* of
/// an inheritance edge, or of a `using static`).
fn is_type(kind: &str) -> bool {
    is_inheritable(kind) || matches!(kind, "enum" | "delegate")
}

/// The LSP 3.17 `SymbolKind` enumeration, as the name C# would use.
///
/// Used only as a fallback: the declaration keyword (see [`DeclarationHeader`])
/// is more precise where both are available, because LSP has no `record` and no
/// `delegate`.
fn lsp_symbol_kind_name(kind: u64) -> &'static str {
    match kind {
        2 => "module",
        3 => "namespace",
        4 => "package",
        5 => "class",
        6 => "method",
        7 => "property",
        8 => "field",
        9 => "constructor",
        10 => "enum",
        11 => "interface",
        12 => "function",
        13 => "variable",
        14 => "constant",
        22 => "enum_member",
        23 => "struct",
        24 => "event",
        25 => "operator",
        26 => "type_parameter",
        _ => "symbol",
    }
}

// ---------------------------------------------------------------------------
// `using` directives.
// ---------------------------------------------------------------------------

/// One `using` directive: what it imports, and in which of C#'s four flavours.
#[derive(Debug, PartialEq)]
struct UsingDirective {
    /// The namespace (or, for `using static`, the type) being imported.
    target: String,
    /// `using Json = Newtonsoft.Json;` — the local alias, if any.
    alias: Option<String>,
    /// `using static System.Math;` imports a *type's* static members, not a namespace.
    is_static: bool,
    /// `global using System;` (C# 10) applies to every file in the project. It
    /// is recorded as a fact on the declaring file rather than fanned out to
    /// every other file: the graph should record what the source says, and a
    /// consumer can apply the language rule.
    is_global: bool,
}

/// Extracts the `using` *directives* from a C# file.
///
/// The parse is deliberately conservative, because C# spells three unrelated
/// things with the same keyword:
///
/// * `using System.Text;`            — a directive (what we want)
/// * `using (var f = File.Open(p))`  — a `using` *statement*
/// * `using var f = File.Open(p);`   — a `using` *declaration* (C# 8)
///
/// The discriminator used here is that a directive's target is always a plain
/// dotted identifier path. `new X()` and `File.Open(p)` are not, so the two
/// statement forms are rejected without needing to know whether we are inside a
/// method body. Block comments are tracked across lines so that a commented-out
/// `using` never becomes a dependency edge.
fn using_directives(text: &str) -> Vec<UsingDirective> {
    let mut directives = Vec::new();
    let mut in_block_comment = false;

    for raw_line in text.lines() {
        let mut line = raw_line;
        if in_block_comment {
            match line.find("*/") {
                Some(end) => {
                    in_block_comment = false;
                    line = &line[end + 2..];
                }
                None => continue,
            }
        }
        // Strip a trailing block comment opener and any line comment.
        let mut code = line.to_string();
        if let Some(start) = code.find("/*") {
            match code[start..].find("*/") {
                Some(end) => code.replace_range(start..start + end + 2, " "),
                None => {
                    in_block_comment = true;
                    code.truncate(start);
                }
            }
        }
        if let Some(start) = code.find("//") {
            code.truncate(start);
        }

        let statement = code.trim();
        let Some(directive) = parse_using(statement) else { continue };
        if !directives.contains(&directive) {
            directives.push(directive);
        }
    }
    directives
}

fn parse_using(statement: &str) -> Option<UsingDirective> {
    let rest = statement.strip_suffix(';')?.trim_end();
    let (is_global, rest) = match rest.strip_prefix("global ") {
        Some(rest) => (true, rest.trim_start()),
        None => (false, rest),
    };
    let rest = rest.strip_prefix("using")?;
    // `usingFoo` is an identifier, not a directive.
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();
    let (is_static, rest) = match rest.strip_prefix("static ") {
        Some(rest) => (true, rest.trim_start()),
        None => (false, rest),
    };

    let (alias, target) = match rest.split_once('=') {
        Some((alias, target)) => {
            let alias = alias.trim();
            if !is_identifier(alias) {
                return None; // `using var f = ...` and friends
            }
            (Some(alias.to_string()), target.trim())
        }
        None => (None, rest),
    };
    if !is_dotted_identifier(target) {
        return None;
    }
    Some(UsingDirective {
        target: target.to_string(),
        alias,
        is_static,
        is_global,
    })
}

fn is_identifier(text: &str) -> bool {
    !text.is_empty()
        && text.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_' || c == '@')
        && text.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '@')
}

/// `System`, `System.Collections.Generic`, `global::System.IO`. Deliberately
/// rejects anything with parentheses, operators or generics — those spellings
/// mean the `using` was a statement, not a directive.
fn is_dotted_identifier(text: &str) -> bool {
    let text = text.strip_prefix("global::").unwrap_or(text);
    !text.is_empty() && text.split('.').all(is_identifier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // -- Fixtures -----------------------------------------------------------

    /// A `PATH` containing exactly one directory, holding fake executables with
    /// the given names. Injecting a `PATH` (rather than calling
    /// `std::env::set_var`, which is unsound in a multi-threaded test binary)
    /// is what lets these tests assert on server detection without .NET
    /// installed — and without disturbing the three other language-adapter test
    /// suites running in the same process.
    fn path_with(executables: &[&str]) -> (tempfile::TempDir, OsString) {
        let dir = tempfile::tempdir().expect("tempdir");
        for name in executables {
            let path = dir.path().join(name);
            let mut file = std::fs::File::create(&path).expect("create fake server");
            file.write_all(b"#!/bin/sh\nexit 0\n").expect("write fake server");
            drop(file);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
            }
        }
        let path_var = std::env::join_paths([dir.path()]).expect("join_paths");
        (dir, path_var)
    }

    fn write(root: &Path, relative: &str, contents: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(&path, contents).expect("write");
    }

    /// A synthetic but realistic C# solution: an SDK-style library with a
    /// namespace, an interface, a class implementing it (with methods,
    /// properties and a field), a NuGet reference and a project reference — plus
    /// build output under `obj/` that must never be indexed.
    fn synthetic_solution(root: &Path) {
        write(
            root,
            "Kopitiam.sln",
            r#"Microsoft Visual Studio Solution File, Format Version 12.00
Project("{9A19103F-16F7-4668-BE54-9A1E7A4F7556}") = "Solver", "src\Solver\Solver.csproj", "{11111111-1111-1111-1111-111111111111}"
EndProject
Project("{9A19103F-16F7-4668-BE54-9A1E7A4F7556}") = "Core", "src\Core\Core.csproj", "{22222222-2222-2222-2222-222222222222}"
EndProject
Project("{2150E333-8FDC-42A3-9474-1A3956D46DE8}") = "solution items", "solution items", "{33333333-3333-3333-3333-333333333333}"
EndProject
"#,
        );
        write(
            root,
            "src/Core/Core.csproj",
            r#"<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
</Project>
"#,
        );
        write(
            root,
            "src/Core/IThermalModel.cs",
            "namespace Kopitiam.Core;\n\npublic interface IThermalModel\n{\n    double Conductivity { get; }\n}\n",
        );
        write(
            root,
            "src/Solver/Solver.csproj",
            r#"<Project Sdk="Microsoft.NET.Sdk">
  <!-- A comment mentioning <PackageReference Include="NotReal" /> must be ignored. -->
  <PropertyGroup>
    <TargetFrameworks>net8.0;net9.0</TargetFrameworks>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="..\Core\Core.csproj" />
    <PackageReference Include="MathNet.Numerics" Version="5.0.0" />
  </ItemGroup>
</Project>
"#,
        );
        write(
            root,
            "src/Solver/HeatSolver.cs",
            r#"using System;
using System.Collections.Generic;
using Kopitiam.Core;
using static System.Math;
using Vec = System.Numerics.Vector3;

namespace Kopitiam.Solver
{
    public sealed class HeatSolver : IThermalModel
    {
        private readonly double _conductivity;

        public double Conductivity => _conductivity;

        public double Step(double dt)
        {
            return Sqrt(dt);
        }
    }
}
"#,
        );
        // Build output: the SDK excludes bin/ and obj/ from the implicit glob,
        // and so must we -- otherwise every symbol is duplicated by its
        // generated copy.
        write(root, "src/Solver/obj/Debug/Solver.AssemblyInfo.cs", "// generated\nclass Generated {}\n");
        write(root, "src/Solver/bin/Release/Leftover.cs", "class Leftover {}\n");
    }

    /// A hierarchical `DocumentSymbol` response of the shape Roslyn LSP and
    /// OmniSharp both return for `src/Solver/HeatSolver.cs` above: namespace >
    /// class > members. Positions are UTF-16 (the LSP default) and match the
    /// real file text, so the position conversion is exercised for real.
    fn heat_solver_document_symbols() -> Vec<Value> {
        json!([{
            "name": "Kopitiam.Solver",
            "kind": 3,
            "range": { "start": { "line": 6, "character": 0 }, "end": { "line": 19, "character": 1 } },
            "selectionRange": { "start": { "line": 6, "character": 10 }, "end": { "line": 6, "character": 25 } },
            "children": [{
                "name": "HeatSolver",
                "kind": 5,
                "range": { "start": { "line": 8, "character": 4 }, "end": { "line": 18, "character": 5 } },
                "selectionRange": { "start": { "line": 8, "character": 24 }, "end": { "line": 8, "character": 34 } },
                "children": [
                    {
                        "name": "_conductivity",
                        "kind": 8,
                        "range": { "start": { "line": 10, "character": 8 }, "end": { "line": 10, "character": 45 } },
                        "selectionRange": { "start": { "line": 10, "character": 31 }, "end": { "line": 10, "character": 44 } },
                        "children": []
                    },
                    {
                        "name": "Conductivity",
                        "kind": 7,
                        "detail": "double",
                        "range": { "start": { "line": 12, "character": 8 }, "end": { "line": 12, "character": 51 } },
                        "selectionRange": { "start": { "line": 12, "character": 22 }, "end": { "line": 12, "character": 34 } },
                        "children": []
                    },
                    {
                        "name": "Step",
                        "kind": 6,
                        "detail": "double Step(double dt)",
                        "range": { "start": { "line": 14, "character": 8 }, "end": { "line": 17, "character": 9 } },
                        "selectionRange": { "start": { "line": 14, "character": 22 }, "end": { "line": 14, "character": 26 } },
                        "children": []
                    }
                ]
            }]
        }])
        .as_array()
        .expect("array")
        .clone()
    }

    /// Runs a collection over `root` with a fake Roslyn LSP on the injected
    /// `PATH`, then splices in `doc_symbols` for `file` as though the server had
    /// answered `textDocument/documentSymbol` — the one thing the shared
    /// `LspClient` cannot yet do (see the module docs). Everything downstream of
    /// the request is the real code path.
    fn collect_with_symbols(root: &Path, symbols: &[(&str, Vec<Value>)]) -> ProviderOutput {
        let (_dir, path_var) = path_with(&["Microsoft.CodeAnalysis.LanguageServer"]);
        let provider = CSharpProvider::with_path(path_var);
        let root = root.canonicalize().expect("canonicalize root");
        let tree = CSharpTree::discover(&root).expect("discover");
        let mut collector = Collector::new(provider.name(), CSharpServer::RoslynLsp, &root);
        collector.emit_structure(&tree);
        for project in &tree.projects {
            for file in &project.sources {
                let text = tree.text(file).expect("text");
                let relative = file.strip_prefix(&root).expect("relative").to_string_lossy().replace('\\', "/");
                let doc_symbols = symbols
                    .iter()
                    .find(|(name, _)| *name == relative)
                    .map(|(_, symbols)| symbols.clone())
                    .unwrap_or_default();
                collector.emit_symbols(file, text, &doc_symbols, PositionEncoding::Utf16);
                collector.emit_usings(file, text);
            }
        }
        collector.resolve_base_types();
        collector.finish()
    }

    fn named<'a>(output: &'a ProviderOutput, name: &str) -> Option<&'a Entity> {
        output.entities.iter().find(|entity| entity.name == name)
    }

    fn kind_of(entity: &Entity) -> &str {
        entity.metadata.get("kind").and_then(Value::as_str).unwrap_or("")
    }

    fn related(output: &ProviderOutput, from: EntityId, to: EntityId, kind: RelationshipKind) -> bool {
        output
            .relationships
            .iter()
            .any(|r| r.from == from && r.to == to && r.kind == kind)
    }

    // -- Server detection and the .NET degradation path ----------------------

    #[test]
    fn roslyn_lsp_is_preferred_when_both_servers_are_installed() {
        let (dir, path_var) = path_with(&["Microsoft.CodeAnalysis.LanguageServer", "OmniSharp"]);
        let (server, path) = CSharpProvider::with_path(path_var).detect_server().expect("a server");
        assert_eq!(server, CSharpServer::RoslynLsp, "Roslyn LSP is what Microsoft maintains today");
        assert_eq!(path, dir.path().join("Microsoft.CodeAnalysis.LanguageServer"));
        assert_eq!(server.lsp_args(), &["--stdio", "--logLevel", "Warning"]);
    }

    #[test]
    fn omnisharp_is_used_when_it_is_the_only_server_present() {
        let (_dir, path_var) = path_with(&["omnisharp"]);
        let (server, _) = CSharpProvider::with_path(path_var).detect_server().expect("a server");
        assert_eq!(server, CSharpServer::OmniSharp);
        // OmniSharp speaks its own protocol unless launched with `-lsp`.
        assert_eq!(server.lsp_args(), &["-lsp"]);
    }

    /// The contract that matters on a machine with no .NET, which is most
    /// machines: no server means no facts, **not** an error and not a panic.
    /// A missing C# toolchain must never fail the collection run for the Rust
    /// half of a mixed repository.
    #[test]
    fn without_a_csharp_server_on_path_collection_degrades_to_empty_not_an_error() {
        let root = tempfile::tempdir().expect("tempdir");
        synthetic_solution(root.path());
        let (_dir, path_var) = path_with(&["some-unrelated-tool"]);

        let output = CSharpProvider::with_path(path_var)
            .collect(root.path())
            .expect("a missing language server must not be an error");

        assert!(output.entities.is_empty(), "no server, no facts");
        assert!(output.relationships.is_empty());
    }

    #[test]
    fn an_empty_path_degrades_rather_than_panicking() {
        let root = tempfile::tempdir().expect("tempdir");
        let output = CSharpProvider::with_path(OsString::new())
            .collect(root.path())
            .expect("an empty PATH must not be an error");
        assert!(output.entities.is_empty());
    }

    #[test]
    fn a_root_with_no_csharp_project_yields_nothing_even_when_a_server_exists() {
        let root = tempfile::tempdir().expect("tempdir");
        write(root.path(), "README.md", "no C# here");
        let (_dir, path_var) = path_with(&["Microsoft.CodeAnalysis.LanguageServer"]);

        let output = CSharpProvider::with_path(path_var).collect(root.path()).expect("collect");
        assert!(output.entities.is_empty(), "a server being installed does not conjure a C# project");
    }

    // -- Project structure ---------------------------------------------------

    #[test]
    fn solution_projects_files_and_packages_become_artifacts_with_containment_edges() {
        let root = tempfile::tempdir().expect("tempdir");
        synthetic_solution(root.path());
        let output = collect_with_symbols(root.path(), &[]);

        let solution = named(&output, "Kopitiam.sln").expect("solution artifact");
        let solver = named(&output, "src/Solver/Solver.csproj").expect("project artifact");
        let core = named(&output, "src/Core/Core.csproj").expect("project artifact");
        let heat = named(&output, "src/Solver/HeatSolver.cs").expect("source file artifact");
        let package = named(&output, "MathNet.Numerics").expect("package artifact");

        for entity in [solution, solver, core, heat, package] {
            assert_eq!(entity.kind, EntityKind::Artifact);
            assert_eq!(entity.source, "csharp", "provenance is a hard requirement");
        }
        assert_eq!(kind_of(solution), "solution");
        assert_eq!(kind_of(solver), "project");
        assert_eq!(kind_of(heat), "source_file");
        assert_eq!(kind_of(package), "package");
        assert_eq!(package.metadata.get("ecosystem").and_then(Value::as_str), Some("nuget"));
        assert_eq!(package.metadata.get("version").and_then(Value::as_str), Some("5.0.0"));

        // Containment: project -> solution, file -> project.
        assert!(related(&output, solver.id, solution.id, RelationshipKind::LocatedIn));
        assert!(related(&output, core.id, solution.id, RelationshipKind::LocatedIn));
        assert!(related(&output, heat.id, solver.id, RelationshipKind::LocatedIn));

        // Dependencies: project -> project, project -> package.
        assert!(related(&output, solver.id, core.id, RelationshipKind::DependsOn));
        assert!(related(&output, solver.id, package.id, RelationshipKind::DependsOn));

        // The `.sln`'s "solution items" folder shares the `Project(...)` line
        // shape but is not a project.
        assert!(named(&output, "solution items").is_none());

        // Multi-targeting is a fact worth keeping: it changes what a translation
        // of this project must support.
        let frameworks = solver.metadata.get("target_frameworks").and_then(Value::as_array).expect("tfms");
        assert_eq!(frameworks.len(), 2, "TargetFrameworks is a `;`-separated list");
    }

    #[test]
    fn sdk_style_projects_glob_their_sources_implicitly_and_never_index_build_output() {
        let root = tempfile::tempdir().expect("tempdir");
        synthetic_solution(root.path());
        let output = collect_with_symbols(root.path(), &[]);

        // HeatSolver.cs is listed nowhere in Solver.csproj -- it is compiled
        // only because SDK-style projects glob `**/*.cs`. This is the single
        // most consequential C# format fact in this provider.
        assert!(named(&output, "src/Solver/HeatSolver.cs").is_some());
        assert!(named(&output, "src/Core/IThermalModel.cs").is_some());

        // ...but `obj/` and `bin/` are excluded from that glob by the SDK.
        assert!(named(&output, "src/Solver/obj/Debug/Solver.AssemblyInfo.cs").is_none());
        assert!(named(&output, "src/Solver/bin/Release/Leftover.cs").is_none());
        assert!(!output.entities.iter().any(|e| e.name == "Generated" || e.name == "Leftover"));

        let solver = named(&output, "src/Solver/Solver.csproj").expect("project");
        assert_eq!(solver.metadata.get("implicit_source_glob").and_then(Value::as_bool), Some(true));
    }

    #[test]
    fn legacy_and_opted_out_projects_compile_only_their_listed_files() {
        let root = tempfile::tempdir().expect("tempdir");
        // A non-SDK project: no `Sdk` attribute, sources listed explicitly.
        write(
            root.path(),
            "Legacy/Legacy.csproj",
            r#"<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup>
    <Compile Include="Included.cs" />
  </ItemGroup>
</Project>
"#,
        );
        write(root.path(), "Legacy/Included.cs", "class Included {}\n");
        write(root.path(), "Legacy/NotListed.cs", "class NotListed {}\n");
        // A modern project that opts back out of the implicit glob.
        write(
            root.path(),
            "OptOut/OptOut.csproj",
            r#"<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <EnableDefaultCompileItems>false</EnableDefaultCompileItems>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="Kept.cs" />
  </ItemGroup>
</Project>
"#,
        );
        write(root.path(), "OptOut/Kept.cs", "class Kept {}\n");
        write(root.path(), "OptOut/Dropped.cs", "class Dropped {}\n");

        let output = collect_with_symbols(root.path(), &[]);
        assert!(named(&output, "Legacy/Included.cs").is_some());
        assert!(named(&output, "Legacy/NotListed.cs").is_none(), "a legacy project compiles only what it lists");
        assert!(named(&output, "OptOut/Kept.cs").is_some());
        assert!(
            named(&output, "OptOut/Dropped.cs").is_none(),
            "EnableDefaultCompileItems=false turns the implicit glob back off"
        );
    }

    #[test]
    fn compile_remove_globs_exclude_matching_sources() {
        let root = tempfile::tempdir().expect("tempdir");
        write(
            root.path(),
            "App/App.csproj",
            r#"<Project Sdk="Microsoft.NET.Sdk">
  <ItemGroup>
    <Compile Remove="Excluded\**\*.cs" />
  </ItemGroup>
</Project>
"#,
        );
        write(root.path(), "App/Kept.cs", "class Kept {}\n");
        write(root.path(), "App/Excluded/Deep/Gone.cs", "class Gone {}\n");

        let output = collect_with_symbols(root.path(), &[]);
        assert!(named(&output, "App/Kept.cs").is_some());
        assert!(
            named(&output, "App/Excluded/Deep/Gone.cs").is_none(),
            "`Excluded\\**\\*.cs` must match through nested directories, and `\\` is a separator"
        );
    }

    // -- Symbols -------------------------------------------------------------

    #[test]
    fn document_symbols_become_nested_symbols_with_containment_edges() {
        let root = tempfile::tempdir().expect("tempdir");
        synthetic_solution(root.path());
        let output = collect_with_symbols(root.path(), &[("src/Solver/HeatSolver.cs", heat_solver_document_symbols())]);

        let file = named(&output, "src/Solver/HeatSolver.cs").expect("file");
        let namespace = named(&output, "Kopitiam.Solver").expect("namespace symbol");
        let class = named(&output, "HeatSolver").expect("class symbol");
        let method = named(&output, "Step").expect("method symbol");
        let property = named(&output, "Conductivity").expect("property symbol");
        let field = named(&output, "_conductivity").expect("field symbol");

        for entity in [namespace, class, method, property, field] {
            assert_eq!(entity.kind, EntityKind::Symbol, "a C# class and a Python class are both Symbols");
            assert_eq!(entity.source, "csharp");
            assert_eq!(entity.metadata.get("language").and_then(Value::as_str), Some("csharp"));
            assert_eq!(entity.metadata.get("server").and_then(Value::as_str), Some("roslyn-lsp"));
            // Every symbol is located in its file, at every nesting depth.
            assert!(related(&output, entity.id, file.id, RelationshipKind::LocatedIn));
        }

        assert_eq!(kind_of(namespace), "namespace");
        assert_eq!(kind_of(class), "class");
        assert_eq!(kind_of(method), "method");
        assert_eq!(kind_of(property), "property");
        assert_eq!(kind_of(field), "field");

        // The nesting the flat `workspace/symbol` response would have thrown
        // away: method -> class -> namespace.
        assert!(related(&output, class.id, namespace.id, RelationshipKind::LocatedIn));
        assert!(related(&output, method.id, class.id, RelationshipKind::LocatedIn));
        assert!(related(&output, property.id, class.id, RelationshipKind::LocatedIn));
        assert!(related(&output, field.id, class.id, RelationshipKind::LocatedIn));

        assert_eq!(
            method.metadata.get("qualified_name").and_then(Value::as_str),
            Some("Kopitiam.Solver.HeatSolver.Step")
        );
        assert_eq!(method.metadata.get("container").and_then(Value::as_str), Some("Kopitiam.Solver.HeatSolver"));
        assert_eq!(method.metadata.get("line").and_then(Value::as_u64), Some(14));
        assert_eq!(method.metadata.get("detail").and_then(Value::as_str), Some("double Step(double dt)"));

        // Accessibility and modifiers come from the declaration header, which
        // sits between `range.start` and `selectionRange.start`.
        assert_eq!(class.metadata.get("accessibility").and_then(Value::as_str), Some("public"));
        let modifiers = class.metadata.get("modifiers").and_then(Value::as_array).expect("modifiers");
        assert_eq!(modifiers, &[json!("sealed")]);
        assert_eq!(field.metadata.get("accessibility").and_then(Value::as_str), Some("private"));
    }

    #[test]
    fn a_base_type_declared_in_this_project_becomes_an_implemented_by_edge() {
        let root = tempfile::tempdir().expect("tempdir");
        synthetic_solution(root.path());
        let interface_symbols = json!([{
            "name": "IThermalModel",
            "kind": 11,
            "range": { "start": { "line": 2, "character": 0 }, "end": { "line": 5, "character": 1 } },
            "selectionRange": { "start": { "line": 2, "character": 17 }, "end": { "line": 2, "character": 30 } },
            "children": []
        }])
        .as_array()
        .expect("array")
        .clone();

        let output = collect_with_symbols(
            root.path(),
            &[
                ("src/Core/IThermalModel.cs", interface_symbols),
                ("src/Solver/HeatSolver.cs", heat_solver_document_symbols()),
            ],
        );

        let interface = named(&output, "IThermalModel").expect("interface");
        let class = named(&output, "HeatSolver").expect("class");

        // `public sealed class HeatSolver : IThermalModel` -- read from the
        // declaration header, since DocumentSymbol does not carry base types.
        let bases = class.metadata.get("base_types").and_then(Value::as_array).expect("base_types");
        assert_eq!(bases, &[json!("IThermalModel")]);

        // The edge points base -> derived: "IThermalModel is implemented by
        // HeatSolver". Resolved across files, and only because the interface is
        // declared in this project.
        assert!(related(&output, interface.id, class.id, RelationshipKind::ImplementedBy));
    }

    #[test]
    fn an_unresolvable_base_type_is_recorded_as_metadata_but_never_invented_as_an_entity() {
        let root = tempfile::tempdir().expect("tempdir");
        write(root.path(), "App/App.csproj", r#"<Project Sdk="Microsoft.NET.Sdk"></Project>"#);
        write(
            root.path(),
            "App/Widget.cs",
            "using System;\n\npublic class Widget : IDisposable\n{\n    public void Dispose() {}\n}\n",
        );
        let symbols = json!([{
            "name": "Widget",
            "kind": 5,
            "range": { "start": { "line": 2, "character": 0 }, "end": { "line": 5, "character": 1 } },
            "selectionRange": { "start": { "line": 2, "character": 13 }, "end": { "line": 2, "character": 19 } },
            "children": []
        }])
        .as_array()
        .expect("array")
        .clone();

        let output = collect_with_symbols(root.path(), &[("App/Widget.cs", symbols)]);
        let widget = named(&output, "Widget").expect("class");
        let bases = widget.metadata.get("base_types").and_then(Value::as_array).expect("base_types");
        assert_eq!(bases, &[json!("IDisposable")], "the fact is recorded verbatim");
        assert!(
            named(&output, "IDisposable").is_none(),
            "guessing that IDisposable means System.IDisposable is an inference, not a fact -- \
             the language server's name binding is what would settle it"
        );
        assert!(!output.relationships.iter().any(|r| r.kind == RelationshipKind::ImplementedBy));
    }

    /// `record` is invisible to LSP — Roslyn reports it as `SymbolKind.Class`
    /// (5). Losing the distinction would matter: a C# `record` has value
    /// semantics and an immutable-by-default shape, and translating one into a
    /// mutable Rust struct would silently change the program's meaning.
    #[test]
    fn a_record_is_distinguished_from_a_class_by_its_declaration_keyword() {
        let root = tempfile::tempdir().expect("tempdir");
        write(root.path(), "App/App.csproj", r#"<Project Sdk="Microsoft.NET.Sdk"></Project>"#);
        write(
            root.path(),
            "App/Types.cs",
            "public record Measurement(double Value);\n\npublic record struct Point(int X, int Y) : IPoint;\n\npublic enum Phase : byte { Solid, Liquid }\n",
        );
        let symbols = json!([
            {
                "name": "Measurement",
                "kind": 5,
                "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 0, "character": 40 } },
                "selectionRange": { "start": { "line": 0, "character": 14 }, "end": { "line": 0, "character": 25 } },
                "children": []
            },
            {
                "name": "Point",
                "kind": 23,
                "range": { "start": { "line": 2, "character": 0 }, "end": { "line": 2, "character": 49 } },
                "selectionRange": { "start": { "line": 2, "character": 21 }, "end": { "line": 2, "character": 26 } },
                "children": []
            },
            {
                "name": "Phase",
                "kind": 10,
                "range": { "start": { "line": 4, "character": 0 }, "end": { "line": 4, "character": 42 } },
                "selectionRange": { "start": { "line": 4, "character": 12 }, "end": { "line": 4, "character": 17 } },
                "children": []
            }
        ])
        .as_array()
        .expect("array")
        .clone();

        let output = collect_with_symbols(root.path(), &[("App/Types.cs", symbols)]);
        assert_eq!(kind_of(named(&output, "Measurement").expect("record")), "record");
        assert_eq!(kind_of(named(&output, "Point").expect("record struct")), "record struct");
        assert_eq!(kind_of(named(&output, "Phase").expect("enum")), "enum");

        // A primary constructor's parameter list sits between the name and the
        // base list; the scan must step over it.
        let point = named(&output, "Point").expect("record struct");
        let bases = point.metadata.get("base_types").and_then(Value::as_array).expect("base_types");
        assert_eq!(bases, &[json!("IPoint")]);

        // `enum Phase : byte` declares an *underlying type*, not a base class.
        // Reading it as inheritance would fabricate a relationship.
        let phase = named(&output, "Phase").expect("enum");
        let phase_bases = phase.metadata.get("base_types").and_then(Value::as_array).expect("base_types");
        assert!(phase_bases.is_empty(), "`: byte` is an underlying type, not a base type");
    }

    #[test]
    fn generic_constraints_are_not_mistaken_for_base_types() {
        let root = tempfile::tempdir().expect("tempdir");
        write(root.path(), "App/App.csproj", r#"<Project Sdk="Microsoft.NET.Sdk"></Project>"#);
        write(
            root.path(),
            "App/Generic.cs",
            "public sealed class Cache<TKey, TValue> : Store<TKey>, IEnumerable<TValue> where TKey : notnull\n{\n}\n",
        );
        let symbols = json!([{
            "name": "Cache",
            "kind": 5,
            "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 2, "character": 1 } },
            "selectionRange": { "start": { "line": 0, "character": 20 }, "end": { "line": 0, "character": 25 } },
            "children": []
        }])
        .as_array()
        .expect("array")
        .clone();

        let output = collect_with_symbols(root.path(), &[("App/Generic.cs", symbols)]);
        let cache = named(&output, "Cache").expect("class");
        let bases = cache.metadata.get("base_types").and_then(Value::as_array).expect("base_types");
        assert_eq!(
            bases,
            &[json!("Store<TKey>"), json!("IEnumerable<TValue>")],
            "the `where` constraint's `:` must not be read as another base type"
        );
    }

    /// Getting this wrong silently misplaces every symbol on any line
    /// containing a non-ASCII character — and scientific C# is full of `°`,
    /// `Δ` and `μ` in identifiers and comments. The same symbol is described
    /// here in all three LSP position encodings; all three must land on the
    /// same `char` column.
    #[test]
    fn symbol_positions_survive_non_ascii_content_in_every_encoding() {
        let root = tempfile::tempdir().expect("tempdir");
        write(root.path(), "App/App.csproj", r#"<Project Sdk="Microsoft.NET.Sdk"></Project>"#);
        // `/* Δ温度 */ public class Kühler` -- the name is preceded on its own
        // line by characters that are 1 char, 1-2 UTF-16 units and 2-3 UTF-8
        // bytes each, so the three encodings disagree about every column after
        // them.
        let line = "/* Δ温度 */ public class Kühler : IWärmetauscher { }";
        write(root.path(), "App/Kuehler.cs", &format!("{line}\n"));
        let root_path = root.path().canonicalize().expect("canonicalize");
        let file = root_path.join("App/Kuehler.cs");

        // Char columns, counted the way this crate's public contract counts:
        // Unicode scalar values, not bytes and not UTF-16 units.
        let char_col_of = |needle: &str| -> u32 {
            let byte = line.find(needle).unwrap_or_else(|| panic!("{needle} in the fixture line"));
            line[..byte].chars().count() as u32
        };
        let name_char_col = char_col_of("Kühler");
        let range_start_chars = char_col_of("public");

        for encoding in [PositionEncoding::Utf8, PositionEncoding::Utf16, PositionEncoding::Utf32] {
            let wire = |char_col: u32| -> u32 {
                crate::position::char_col_to_unit(line, char_col, encoding)
            };
            let symbols = json!([{
                "name": "Kühler",
                "kind": 5,
                "range": {
                    "start": { "line": 0, "character": wire(range_start_chars) },
                    "end": { "line": 0, "character": wire(line.chars().count() as u32) }
                },
                "selectionRange": {
                    "start": { "line": 0, "character": wire(name_char_col) },
                    "end": { "line": 0, "character": wire(name_char_col + "Kühler".chars().count() as u32) }
                },
                "children": []
            }])
            .as_array()
            .expect("array")
            .clone();

            let (_dir, path_var) = path_with(&["Microsoft.CodeAnalysis.LanguageServer"]);
            let provider = CSharpProvider::with_path(path_var);
            let tree = CSharpTree::discover(&root_path).expect("discover");
            let mut collector = Collector::new(provider.name(), CSharpServer::RoslynLsp, &root_path);
            collector.emit_structure(&tree);
            let text = tree.text(&file).expect("text");
            collector.emit_symbols(&file, text, &symbols, encoding);
            let output = collector.finish();

            let class = output
                .entities
                .iter()
                .find(|entity| entity.name == "Kühler")
                .unwrap_or_else(|| panic!("class symbol missing under {encoding:?}"));

            assert_eq!(
                class.metadata.get("character").and_then(Value::as_u64),
                Some(name_char_col as u64),
                "the name's char column must be identical in every encoding ({encoding:?})"
            );
            // And the declaration header read back from that position must still
            // be the real one -- a mis-converted column would slice mid-word.
            assert_eq!(kind_of(class), "class", "a misplaced column would slice the header apart ({encoding:?})");
            assert_eq!(class.metadata.get("accessibility").and_then(Value::as_str), Some("public"));
            let bases = class.metadata.get("base_types").and_then(Value::as_array).expect("base_types");
            assert_eq!(bases, &[json!("IWärmetauscher")], "base list under {encoding:?}");
        }
    }

    // -- `using` directives --------------------------------------------------

    #[test]
    fn using_directives_become_depends_on_edges_from_the_file_to_the_namespace() {
        let root = tempfile::tempdir().expect("tempdir");
        synthetic_solution(root.path());
        let output = collect_with_symbols(root.path(), &[("src/Solver/HeatSolver.cs", heat_solver_document_symbols())]);

        let file = named(&output, "src/Solver/HeatSolver.cs").expect("file");
        for namespace in ["System", "System.Collections.Generic"] {
            let symbol = named(&output, namespace).unwrap_or_else(|| panic!("{namespace} symbol"));
            assert_eq!(symbol.kind, EntityKind::Symbol);
            assert_eq!(kind_of(symbol), "namespace");
            assert!(related(&output, file.id, symbol.id, RelationshipKind::DependsOn));
        }

        // `using static System.Math;` imports a type, not a namespace.
        let math = named(&output, "System.Math").expect("static import");
        assert_eq!(kind_of(math), "type");
        assert_eq!(math.metadata.get("static").and_then(Value::as_bool), Some(true));
        assert!(related(&output, file.id, math.id, RelationshipKind::DependsOn));

        // An alias records the local name it was bound to.
        let alias = named(&output, "System.Numerics.Vector3").expect("alias import");
        assert_eq!(alias.metadata.get("alias").and_then(Value::as_str), Some("Vec"));

        // `using Kopitiam.Core;` must converge on the *declared* namespace
        // symbol from the other project, not create a second node for it.
        let declared: Vec<&Entity> = output.entities.iter().filter(|e| e.name == "Kopitiam.Core").collect();
        assert_eq!(declared.len(), 1, "a used namespace and a declared namespace are one node");
    }

    #[test]
    fn using_statements_and_declarations_are_not_mistaken_for_directives() {
        let source = r#"
global using System.IO;
using System.Text;
// using Commented.Out;
/* using Also.Commented; */
using Alias = System.Collections.Generic.List<int>;

class C
{
    void M()
    {
        using var stream = File.OpenRead("x");
        using (var other = File.OpenRead("y"))
        {
        }
    }
}
"#;
        let directives = using_directives(source);
        let targets: Vec<&str> = directives.iter().map(|d| d.target.as_str()).collect();
        assert_eq!(targets, ["System.IO", "System.Text"]);
        assert!(directives[0].is_global, "`global using` is a directive too");
        assert!(
            !targets.iter().any(|t| t.contains("Commented")),
            "a commented-out using must never become a dependency edge"
        );
        // `using Alias = System.Collections.Generic.List<int>;` aliases a
        // constructed generic type, which is not a dotted identifier path --
        // rejected rather than recorded as a namespace that does not exist.
        assert!(!targets.iter().any(|t| t.contains("List")));
    }

    // -- Cross-cutting -------------------------------------------------------

    #[test]
    fn every_entity_carries_this_provider_as_its_source() {
        let root = tempfile::tempdir().expect("tempdir");
        synthetic_solution(root.path());
        let output = collect_with_symbols(root.path(), &[("src/Solver/HeatSolver.cs", heat_solver_document_symbols())]);

        assert!(!output.entities.is_empty());
        for entity in &output.entities {
            assert_eq!(entity.source, "csharp", "{} lost its provenance", entity.name);
            assert_eq!(
                entity.metadata.get("language").and_then(Value::as_str),
                Some("csharp"),
                "{} must be attributable to C# in a multi-language graph",
                entity.name
            );
        }
    }

    #[test]
    fn collection_is_deterministic() {
        let root = tempfile::tempdir().expect("tempdir");
        synthetic_solution(root.path());
        let symbols = [("src/Solver/HeatSolver.cs", heat_solver_document_symbols())];

        let first = collect_with_symbols(root.path(), &symbols);
        let second = collect_with_symbols(root.path(), &symbols);

        // Entity ids are fresh UUIDs by design, so compare the facts themselves:
        // same names, same kinds, same order, run after run. A graph whose
        // contents depend on `read_dir` order is not reproducible.
        let shape = |output: &ProviderOutput| -> Vec<(String, String)> {
            output
                .entities
                .iter()
                .map(|e| (e.name.clone(), kind_of(e).to_string()))
                .collect()
        };
        assert_eq!(shape(&first), shape(&second));
        assert_eq!(first.relationships.len(), second.relationships.len());
    }

    // -- Unit tests for the C#-format readers --------------------------------

    #[test]
    fn csproj_parsing_reads_references_frameworks_and_the_sdk_attribute() {
        let parsed = CsProj::parse(
            r#"<Project Sdk="Microsoft.NET.Sdk">
  <!-- <PackageReference Include="Ignored" Version="1.0.0" /> -->
  <PropertyGroup>
    <TargetFrameworks>net8.0;net9.0</TargetFrameworks>
  </PropertyGroup>
  <ItemGroup>
    <ProjectReference Include="..\Core\Core.csproj" />
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
    <PackageReference Include="Serilog" />
  </ItemGroup>
</Project>"#,
        );
        assert_eq!(parsed.sdk.as_deref(), Some("Microsoft.NET.Sdk"));
        assert!(parsed.globs_sources_implicitly());
        assert_eq!(parsed.target_frameworks, ["net8.0", "net9.0"]);
        assert_eq!(parsed.project_references, [r"..\Core\Core.csproj"]);
        assert_eq!(
            parsed.package_references,
            [
                ("Newtonsoft.Json".to_string(), Some("13.0.3".to_string())),
                ("Serilog".to_string(), None),
            ],
            "a PackageReference may carry its version in Directory.Packages.props instead"
        );
    }

    #[test]
    fn sln_parsing_keeps_projects_and_drops_solution_folders() {
        let projects = parse_sln(
            r#"
Project("{9A19103F-16F7-4668-BE54-9A1E7A4F7556}") = "App", "src\App\App.csproj", "{AAAA}"
EndProject
Project("{2150E333-8FDC-42A3-9474-1A3956D46DE8}") = "docs", "docs", "{BBBB}"
EndProject
Project("{778DAE3C-4631-46EA-AA77-85C1314464D9}") = "Legacy", "Legacy\Legacy.vbproj", "{CCCC}"
EndProject
"#,
        );
        assert_eq!(projects, [r"src\App\App.csproj"], "solution folders and non-C# projects are not C# projects");
    }

    #[test]
    fn declaration_headers_yield_accessibility_modifiers_and_the_construct_keyword() {
        let header = DeclarationHeader::parse("[Obsolete(\"use Solver\")]\n    protected internal static partial class ");
        assert_eq!(header.accessibility.as_deref(), Some("protected internal"));
        assert_eq!(header.modifiers, ["static", "partial"]);
        assert_eq!(header.keyword.as_deref(), Some("class"));

        // No declared accessibility is recorded as absent, not guessed at.
        let implicit = DeclarationHeader::parse("    record ");
        assert_eq!(implicit.accessibility, None);
        assert_eq!(implicit.keyword.as_deref(), Some("record"));
    }

    #[test]
    fn glob_matching_handles_the_two_wildcards_that_appear_in_csproj_files() {
        let dir = Path::new("/p");
        assert!(glob_matches("**/*.cs", dir, Path::new("/p/a/b/C.cs")));
        assert!(glob_matches("*.cs", dir, Path::new("/p/C.cs")));
        assert!(!glob_matches("*.cs", dir, Path::new("/p/a/C.cs")), "`*` does not cross a directory boundary");
        assert!(glob_matches(r"Excluded\**\*.cs", dir, Path::new("/p/Excluded/Deep/C.cs")));
        assert!(!glob_matches("Excluded/**/*.cs", dir, Path::new("/p/Kept/C.cs")));
        assert!(glob_matches("Old.cs", dir, Path::new("/p/Old.cs")));
        assert!(!glob_matches("Old.cs", dir, Path::new("/p/Older.cs")));
    }

    // -- Live integration ----------------------------------------------------

    /// A real end-to-end run against whatever C# server is installed.
    ///
    /// `#[ignore]`d: it needs a .NET runtime plus Roslyn LSP or OmniSharp, and
    /// **neither is installed on the development machine this was written on**
    /// (`dotnet` is present; no C# language server is). Nothing is downloaded
    /// to make it pass — a test that installs a toolchain behind the
    /// maintainer's back is worse than a test that honestly skips.
    ///
    /// Run with: `cargo test --release -p kopitiam-semantic -- --ignored`
    ///
    /// Note that until the two `LspClient` additions described in the module
    /// docs land, this exercises the structural half only.
    #[test]
    #[ignore = "requires a .NET runtime and a C# language server (Roslyn LSP or OmniSharp) on PATH"]
    fn live_collection_against_a_real_csharp_server() {
        let provider = CSharpProvider::new();
        let Some((server, path)) = provider.detect_server() else {
            panic!("no C# language server on PATH; this test cannot run here");
        };
        eprintln!("using {} at {}", server.id(), path.display());

        let root = tempfile::tempdir().expect("tempdir");
        synthetic_solution(root.path());
        let output = provider.collect(root.path()).expect("collection must succeed with a real server");

        assert!(output.entities.iter().any(|e| e.kind == EntityKind::Artifact), "expected project artifacts");
        assert!(
            output.entities.iter().any(|e| e.name == "src/Solver/HeatSolver.cs"),
            "expected the implicitly-globbed source file"
        );
    }
}
