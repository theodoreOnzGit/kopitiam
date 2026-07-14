//! C and C++ knowledge provider, backed by **clangd**.
//!
//! C++ is one of KOPITIAM's primary translation *sources*: a long-term goal
//! named in `CLAUDE.md` is to carry large legacy C++ codebases into idiomatic
//! Rust while preserving intent. This provider is the front half of that
//! pipeline, and equally it is how KOPITIAM builds intelligence about any
//! sizeable C++ project. Everything downstream — the knowledge graph, the
//! translation platform, the context builder — sees C++ only through the facts
//! emitted here, so what this module fails to notice, KOPITIAM never knows.
//!
//! # Why clangd, and why not a parser of our own
//!
//! C++ cannot be parsed without being *compiled*. `#include` pulls in arbitrary
//! text, `-D` flags select entire code paths, and templates mean the same token
//! sequence denotes different things in different translation units. Any "C++
//! parser" that does not run the preprocessor with the real compile flags is
//! guessing, and guessing is what CLAUDE.md's Deterministic Facts principle
//! forbids. clangd is the reference C/C++ language server: it runs a real Clang
//! frontend, and it is a plain binary we drive over stdio — so no C or C++
//! *build* dependency enters KOPITIAM. (The Pure Rust Core promise is about
//! what this workspace compiles and links, not about what tools it shells out
//! to; [`crate::providers::rust_analyzer`] makes exactly the same trade.)
//!
//! If clangd is not installed, [`CppProvider::collect`] degrades to
//! [`ProviderOutput::empty`] rather than failing the collection run, as
//! [`KnowledgeProvider`]'s contract requires.
//!
//! # The compilation database is the whole ball game
//!
//! clangd needs a **compilation database** — the include paths and `-D` defines
//! used to compile each translation unit. Without one it does not refuse to
//! work, which would be the merciful outcome. It falls back to a guessed
//! command line, fails to find the project's headers, and then *recovers*:
//! unknown types silently become `int`, and it hands back a symbol tree that
//! looks perfectly healthy.
//!
//! That was measured, not assumed. Given this fragment, with no compilation
//! database present, clangd reports the field `root_` with `detail: "int"`:
//!
//! ```cpp
//! #include "widget.hpp"
//! class Panel { Widget root_; };
//! ```
//!
//! A confident, specific, wrong fact — and a downstream translator would
//! faithfully render a `Widget` field as a C `int`. This is precisely the
//! failure CLAUDE.md's Scientific Standards exist to prevent, so this provider
//! never lets it pass silently:
//!
//! * [`CompilationDatabase::discover`] looks for `compile_commands.json` in the
//!   project root and in the conventional `build/` and `out/` directories, then
//!   for clangd's simpler `compile_flags.txt` fallback at the root.
//! * When neither exists, the provider still emits what it can — a partial graph
//!   beats no graph — but marks the project artifact `"degraded": true`, gives a
//!   `"degradation"` string saying why in words, counts `"unresolved_includes"`,
//!   and logs a warning. Every consumer can see, *from the facts themselves*,
//!   that it is holding a degraded picture.
//! * Per file, `"diagnostic_errors"` records how many error-severity diagnostics
//!   clangd raised. A file whose diagnostics say `'widget.hpp' file not found` has
//!   not really been understood, and now says so.
//!
//! ## When the build system emits no compilation database
//!
//! Not every C++ build writes a `compile_commands.json`. CMake does (with
//! `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`), but a project built with hand-written
//! Makefiles, a bespoke build script, or any system that does not export one
//! lands, out of the box, in exactly the case this module degrades on. Anyone
//! pointing KOPITIAM at such a tree must generate a compilation database first
//! — e.g. with [Bear](https://github.com/rizsotto/Bear) (`bear -- <build
//! command>`), which intercepts the compiler invocations the build makes and
//! writes them out. There is no way around this: without the real `-I` include
//! paths and `-D` defines the project is compiled with, no tool on earth can
//! say what a project-specific type resolves to. Hand-writing a
//! `compile_flags.txt` with the right `-I` set is a workable second best for
//! reading one library subtree.
//!
//! # A clangd limitation that template-heavy C++ walks straight into
//!
//! Base classes come from `typeHierarchy/supertypes` (see
//! [`Collector::collect_bases`]) — and that request **cannot resolve a base that
//! is a template instantiation**:
//!
//! ```cpp
//! class Derived : public Base               {}; // supertypes -> [Base]  ✔
//! class Buffer  : public Container<double>  {}; // supertypes -> []      ✘
//! ```
//!
//! `prepareTypeHierarchy` *does* report that `Buffer` has one parent — its
//! opaque `data.parents` carries a symbol ID — but `supertypes` then fails to
//! resolve that ID to a location and returns nothing. This was measured with the
//! background index both off and on, so it is clangd's behaviour, not a
//! consequence of how this module starts it: an implicit template specialization
//! is not an indexed symbol.
//!
//! This is not a footnote. Template-heavy C++ codebases are full of exactly this
//! shape — a concrete class deriving from an instantiation, as in
//! `class Buffer : public Container<double>` — so a meaningful share of their
//! hierarchy is invisible to the tool. Rather than silently drop those edges,
//! this provider counts them: a class whose
//! declared parents outnumber its resolved ones gets `"unresolved_bases": n` in
//! its metadata, and the project artifact carries the total. The graph therefore
//! says "there is a base class here that I could not name", which is a fact,
//! instead of "this class has no base", which would be a lie.
//!
//! Recovering those bases needs a different mechanism (a clangd AST request, or
//! reading the base-clause from the source range). Neither is done here, and the
//! counter is what tells you how much it is worth.
//!
//! # Positions
//!
//! LSP 3.17 lets a server choose the unit of `Position.character`, and clangd
//! chooses **`"utf-8"` — which per the spec means *byte* offsets**, not `char`
//! offsets. (rust-analyzer, by contrast, chooses `"utf-16"`.) A provider that
//! assumed characters would silently misplace every symbol sitting after
//! non-ASCII text on its line. This module therefore routes every column it
//! emits through [`crate::position`], the crate's single source of truth for
//! that conversion, so a symbol's `"column"` is always a Unicode scalar value
//! offset no matter what the server negotiated.
//!
//! # Why this module drives clangd itself instead of reusing [`LspClient`]
//!
//! [`crate::lsp_client::LspClient`] is this crate's rust-analyzer driver, and
//! reusing it was the intent. It cannot serve clangd, for three structural
//! reasons:
//!
//! 1. **It cannot pass process arguments.** clangd must be started with
//!    `--background-index=false`, because its background index *writes*
//!    `.cache/clangd/index/` into the user's source tree — and a provider whose
//!    job is to *describe* a project must never modify it. It also needs
//!    `--compile-commands-dir=` when the database was found somewhere clangd's
//!    own search would not look (e.g. `out/`).
//! 2. **Its start-up blocks on a rust-analyzer-shaped readiness signal**: a
//!    `$/progress` "end" event for an indexing token. With the background index
//!    off, clangd never sends one, so `LspClient::spawn` would stall for its
//!    full timeout on every run.
//! 3. **It exposes only `workspace/symbol`.** This provider needs
//!    `textDocument/documentSymbol` (hierarchical, so nesting survives),
//!    `textDocument/documentLink` (to resolve `#include`s using the real compile
//!    flags), and `textDocument/prepareTypeHierarchy` + `typeHierarchy/supertypes`
//!    (the only way clangd will tell you a base class). `LspClient::request` is
//!    private to its own module.
//!
//! The duplication is deliberately confined to ~80 lines of `Content-Length`
//! framing. The part that is genuinely hard to get right — position encoding —
//! is *not* duplicated: [`ClangdClient`] uses [`crate::position`] exactly as
//! [`crate::session::RustAnalyzerSession`] does. The right long-term shape is to
//! generalize `LspClient` (process arguments, an opt-out for the indexing wait,
//! and the four requests above) and delete [`ClangdClient`].

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipKind};
use serde_json::{Value, json};

use crate::position::{self, PositionEncoding};
use crate::provider::{KnowledgeProvider, ProviderOutput};

/// The provenance stamped on every entity this provider emits, and the value of
/// [`CppProvider::name`]. It names the *tool*, not the language, in keeping with
/// [`crate::providers::rust_analyzer`]: a consumer weighing how far to trust a
/// fact wants to know what computed it.
const SOURCE: &str = "clangd";

/// C and C++ **source** (translation unit) extensions.
///
/// Case is significant, deliberately: `.C` is a C++ source file by the
/// long-standing GNU convention (used by GCC's own sources and many older C++
/// projects for every implementation file), while `.c` is C. Folding case here
/// would quietly reclassify a whole codebase's sources as C.
const SOURCE_EXTENSIONS: &[&str] = &["c", "cc", "cpp", "cxx", "c++", "C"];

/// C and C++ **header** extensions. `.H` is the GNU / older-C++ header
/// convention — the uppercase counterpart to `.C`. `.inl`/`.ipp`/`.tpp`/`.tcc`
/// are the usual names for files holding template definitions meant to be
/// `#include`d — which, in a template-heavy codebase, is where the real
/// algorithms live.
const HEADER_EXTENSIONS: &[&str] = &["h", "hh", "hpp", "hxx", "h++", "H", "inl", "ipp", "tpp", "tcc"];

/// Directories skipped when walking for sources: version-control metadata,
/// build outputs, dependency dumps. `build/` and `out/` are skipped as *source*
/// locations while still being searched for a compilation database — generated
/// code under a build directory is a build artifact, not project knowledge.
const SKIPPED_DIRS: &[&str] = &[".git", ".svn", ".hg", ".cache", "target", "node_modules", "build", "out"];

/// How clangd will — or won't — learn the compile flags for this project.
///
/// See the module docs: this enum is the difference between a real semantic
/// graph and a confident fiction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationDatabase {
    /// A `compile_commands.json`: the real thing, one entry per translation unit
    /// carrying the exact command line used to compile it.
    CompileCommands(PathBuf),
    /// A `compile_flags.txt`: clangd's simpler fallback — one flag per line,
    /// applied to *every* file. Adequate for a uniform subtree and far better
    /// than nothing, but it cannot express per-file differences.
    CompileFlags(PathBuf),
    /// Neither was found. clangd will guess a command line, fail to resolve the
    /// project's own headers, and recover from the resulting errors by inventing
    /// types. Facts collected in this state are marked degraded.
    Missing,
}

impl CompilationDatabase {
    /// Looks for a compilation database for the project rooted at `root`.
    ///
    /// Search order, most authoritative first:
    ///
    /// 1. `<root>/compile_commands.json` — often a symlink into the build tree.
    /// 2. `<root>/build/compile_commands.json`, `<root>/out/compile_commands.json`
    ///    — where CMake and the common `-B build` conventions leave it.
    /// 3. `<root>/compile_flags.txt` — clangd's flat per-project flag list.
    ///
    /// First hit wins, so a real database always beats the flat fallback.
    pub fn discover(root: &Path) -> Self {
        const DATABASE_DIRS: &[&str] = &["", "build", "out"];

        for dir in DATABASE_DIRS {
            let candidate = root.join(dir).join("compile_commands.json");
            if candidate.is_file() {
                return Self::CompileCommands(candidate);
            }
        }
        let flags = root.join("compile_flags.txt");
        if flags.is_file() {
            return Self::CompileFlags(flags);
        }
        Self::Missing
    }

    /// True when clangd will be working without real compile flags, so every
    /// fact derived from it must be treated as provisional.
    pub fn is_degraded(&self) -> bool {
        matches!(self, Self::Missing)
    }

    /// The directory to hand clangd as `--compile-commands-dir`, if any.
    ///
    /// clangd finds a `compile_commands.json` at the root (or in a `build/`
    /// subdirectory) by walking up from each file it opens, but it will not look
    /// in `out/`. Passing the directory explicitly makes the behaviour identical
    /// wherever *we* found the database, instead of depending on clangd's search
    /// heuristics happening to match ours.
    fn compile_commands_dir(&self) -> Option<&Path> {
        match self {
            Self::CompileCommands(path) => path.parent(),
            Self::CompileFlags(_) | Self::Missing => None,
        }
    }

    /// A short, machine-readable account of the project's compile-flag
    /// situation, recorded in the project artifact so that someone reading these
    /// facts months from now can see how far to trust them.
    fn describe(&self) -> Value {
        match self {
            Self::CompileCommands(path) => json!({
                "kind": "compile_commands.json",
                "path": path.display().to_string(),
            }),
            Self::CompileFlags(path) => json!({
                "kind": "compile_flags.txt",
                "path": path.display().to_string(),
            }),
            Self::Missing => json!({ "kind": "none" }),
        }
    }
}

/// Turns a C or C++ project into `kopitiam-ontology` facts by driving clangd.
///
/// ```no_run
/// use std::path::Path;
/// use kopitiam_semantic::{KnowledgeProvider, providers::cpp::CppProvider};
///
/// let facts = CppProvider::new().collect(Path::new("/src/myproject"))?;
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// Every knob is a builder method, and the defaults are the ones a normal
/// project wants. See the module docs for what happens when the project has no
/// compilation database — short version: you still get facts, and they are
/// stamped degraded.
pub struct CppProvider {
    /// The clangd executable: a bare name resolved against `PATH`, or a path.
    binary: String,
    /// The `PATH` to resolve `binary` against; `None` means the process's own.
    /// Overridable so tests can simulate "clangd is not installed" without
    /// mutating global process state, which is not safe to do while other tests
    /// run on parallel threads.
    path: Option<OsString>,
    /// Upper bound on the number of files analysed, if any.
    ///
    /// Defaults to `None` — analyse everything — because a fact collector that
    /// silently stops early is a fact collector that lies. Callers needing a
    /// bounded run set this explicitly, and can see `"files_truncated": true` in
    /// the project metadata when it bit. Be aware that clangd builds a full AST
    /// per file: on a million-line tree that is an hours-long job, and the
    /// caller, not this provider, is the right place to decide about it.
    max_files: Option<usize>,
    /// How long to wait for any single LSP request. A pathological translation
    /// unit can take a while to parse; a hung one must not wedge the whole run.
    request_timeout: Duration,
    /// Whether to spend two extra LSP round-trips per class recovering base
    /// classes. On by default: in most C++ codebases the architecture *is* its
    /// class hierarchies, and an inheritance edge is among the highest-value
    /// facts a translator can be handed. Turn it off for a fast structural-only
    /// pass.
    type_hierarchy: bool,
}

impl CppProvider {
    pub fn new() -> Self {
        Self {
            binary: SOURCE.to_string(),
            path: None,
            max_files: None,
            request_timeout: Duration::from_secs(60),
            type_hierarchy: true,
        }
    }

    /// Uses a specific clangd executable: a bare name (resolved against `PATH`)
    /// or a path to a binary.
    pub fn with_binary(mut self, binary: impl Into<String>) -> Self {
        self.binary = binary.into();
        self
    }

    /// Resolves the binary against `path` instead of the process's own `PATH`.
    pub fn with_path(mut self, path: impl Into<OsString>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// Analyses at most `max` files. See [`Self::max_files`] for why there is no
    /// default limit.
    pub fn with_max_files(mut self, max: usize) -> Self {
        self.max_files = Some(max);
        self
    }

    /// Sets the per-request timeout. See [`Self::request_timeout`].
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Enables or disables base-class recovery. See [`Self::type_hierarchy`].
    pub fn with_type_hierarchy(mut self, enabled: bool) -> Self {
        self.type_hierarchy = enabled;
        self
    }

    /// Resolves [`Self::binary`] to a concrete executable, or `None` when clangd
    /// is not installed.
    ///
    /// Resolution is done here rather than left to the OS so that [`Self::path`]
    /// can override it: mutating `PATH` in the process environment to exercise
    /// the not-installed path would race every other test in the binary.
    fn resolve_binary(&self) -> Option<PathBuf> {
        let candidate = Path::new(&self.binary);
        if candidate.components().count() > 1 {
            return candidate.is_file().then(|| candidate.to_path_buf());
        }
        let search_path = self.path.clone().or_else(|| std::env::var_os("PATH"))?;
        std::env::split_paths(&search_path)
            .map(|dir| dir.join(&self.binary))
            .find(|candidate| candidate.is_file())
    }
}

impl Default for CppProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeProvider for CppProvider {
    fn name(&self) -> &str {
        SOURCE
    }

    fn collect(&self, root: &Path) -> Result<ProviderOutput> {
        let Some(binary) = self.resolve_binary() else {
            tracing::warn!(
                binary = %self.binary,
                "clangd not found on PATH; skipping C/C++ analysis (no facts collected)"
            );
            return Ok(ProviderOutput::empty());
        };

        let database = CompilationDatabase::discover(root);
        if database.is_degraded() {
            tracing::warn!(
                root = %root.display(),
                "no compile_commands.json or compile_flags.txt: clangd cannot resolve this project's \
                 include paths or defines. Facts will still be emitted, but they are DEGRADED — \
                 unknown types are silently recovered as `int`. If the build system produces no \
                 compilation database, generate one with a tool like Bear (`bear -- <build command>`)."
            );
        }

        let mut files = discover_sources(root);
        files.sort();
        let discovered = files.len();
        let truncated = self.max_files.is_some_and(|max| discovered > max);
        if let Some(max) = self.max_files {
            files.truncate(max);
        }
        if files.is_empty() {
            tracing::info!(root = %root.display(), "no C/C++ sources found");
            return Ok(ProviderOutput::empty());
        }

        let mut client = ClangdClient::spawn(&binary, root, &database, self.request_timeout)?;
        let mut collector = Collector::new(root, &database);
        collector.declare_files(&files);

        for file in &files {
            if let Err(error) = collector.collect_file(&mut client, file, self.type_hierarchy) {
                // One unparseable translation unit must not cost us the other
                // nine hundred. Record the loss, keep going, and let the counters
                // on the project artifact carry it to the caller.
                tracing::warn!(file = %file.display(), %error, "clangd failed on this file; skipping it");
                collector.failed_files += 1;
            }
        }

        collector.resolve_inheritance();
        let clean_shutdown = client.shutdown().is_ok();
        collector.finish(discovered, truncated, clean_shutdown)
    }
}

// ---------------------------------------------------------------------------
// Fact assembly
// ---------------------------------------------------------------------------

/// A base class clangd told us about, recorded during the per-file pass and
/// resolved to an [`EntityId`] once every file has been seen.
///
/// Resolution is deferred because a derived class in `a.H` routinely inherits
/// from a base declared in `b.H`, which the walk may not have reached yet. The
/// base's LSP location is a stable join key: `typeHierarchy/supertypes` reports
/// a class's `selectionRange` identically to `textDocument/documentSymbol`, so
/// the two views of the same class agree exactly.
struct PendingBase {
    derived: EntityId,
    /// `(uri, line, character)` of the base's *name*, in the server's wire
    /// encoding. Both sides of the join speak the same units, so no conversion
    /// is wanted here.
    location: (String, u64, u64),
    name: String,
    /// clangd's `detail` for a supertype is its fully-qualified name, e.g.
    /// `"myproject::Widget"` — the most useful string in the whole response.
    qualified_name: Option<String>,
}

/// Accumulates entities and relationships across a run.
struct Collector<'a> {
    root: &'a Path,
    database: &'a CompilationDatabase,
    entities: Vec<Entity>,
    relationships: Vec<Relationship>,
    /// URI — or `include:<spelling>` for an unresolvable one — to artifact id.
    artifacts: HashMap<String, EntityId>,
    /// URI to index into `entities`, for the project's own files, whose metadata
    /// is finalised once their diagnostics are known.
    file_slots: HashMap<String, usize>,
    /// `(uri, line, character)` of a symbol's name to its id: the join key for
    /// inheritance. See [`PendingBase`].
    symbols_by_location: HashMap<(String, u64, u64), EntityId>,
    /// Entity id to index into `entities`, so a symbol's metadata can be amended
    /// after the fact (an unresolved base class is only discovered once its
    /// derived class has already been emitted).
    symbol_slots: HashMap<EntityId, usize>,
    /// `(qualified name, signature)` to `(id, is_in_header)`. Used to link a
    /// header declaration to the out-of-line definition implementing it; the
    /// signature is what keeps overloads apart.
    by_signature: HashMap<(String, String), Vec<(EntityId, bool)>>,
    pending_bases: Vec<PendingBase>,
    failed_files: usize,
    files_with_errors: usize,
    unresolved_includes: usize,
    /// Base classes clangd knew existed but could not name — almost always a
    /// template instantiation. See the module docs.
    unresolved_bases: usize,
}

impl<'a> Collector<'a> {
    fn new(root: &'a Path, database: &'a CompilationDatabase) -> Self {
        Self {
            root,
            database,
            entities: Vec::new(),
            relationships: Vec::new(),
            artifacts: HashMap::new(),
            file_slots: HashMap::new(),
            symbols_by_location: HashMap::new(),
            symbol_slots: HashMap::new(),
            by_signature: HashMap::new(),
            pending_bases: Vec::new(),
            failed_files: 0,
            files_with_errors: 0,
            unresolved_includes: 0,
            unresolved_bases: 0,
        }
    }

    fn push(&mut self, entity: Entity) -> EntityId {
        let id = entity.id;
        self.symbol_slots.insert(id, self.entities.len());
        self.entities.push(entity);
        id
    }

    fn relate(&mut self, from: EntityId, to: EntityId, kind: RelationshipKind) {
        self.relationships.push(Relationship::new(from, to, kind));
    }

    /// Creates the artifact for every project file up front, so that an
    /// `#include` of a project header resolves to the *same* entity the header
    /// itself is described by, whatever order the walk happens to visit them in.
    fn declare_files(&mut self, files: &[PathBuf]) {
        for file in files {
            let Ok(uri) = path_to_uri(file) else { continue };
            let entity = Entity::new(EntityKind::Artifact, uri.clone(), SOURCE).with_metadata(json!({
                "role": file_role(file),
                "language": file_language(file),
                "path": file.display().to_string(),
                "relative_path": relative_to(self.root, file),
            }));
            let id = entity.id;
            self.file_slots.insert(uri.clone(), self.entities.len());
            self.artifacts.insert(uri, id);
            self.entities.push(entity);
        }
    }

    /// The artifact for `uri`, creating one for a file outside the project (a
    /// system or third-party header we only ever meet as an `#include` target)
    /// if it is new.
    fn external_artifact(&mut self, uri: &str) -> EntityId {
        if let Some(id) = self.artifacts.get(uri) {
            return *id;
        }
        let path = uri_to_path(uri);
        let entity = Entity::new(EntityKind::Artifact, uri.to_string(), SOURCE).with_metadata(json!({
            "role": "header",
            "external": true,
            "path": path.as_ref().map(|path| path.display().to_string()),
        }));
        let id = entity.id;
        self.artifacts.insert(uri.to_string(), id);
        self.entities.push(entity);
        id
    }

    /// The artifact standing in for an `#include` clangd could not resolve.
    ///
    /// These are not noise to be dropped. An unresolved include is the clearest
    /// evidence there is that the compilation database is missing or wrong, and
    /// the graph should say so out loud rather than quietly omit the edge and
    /// look complete.
    fn unresolved_include_artifact(&mut self, spelling: &str, system: bool) -> EntityId {
        let key = format!("include:{spelling}");
        if let Some(id) = self.artifacts.get(&key) {
            return *id;
        }
        let entity = Entity::new(EntityKind::Artifact, spelling.to_string(), SOURCE).with_metadata(json!({
            "role": "header",
            "resolved": false,
            "system": system,
            "note": "clangd could not resolve this #include: the compilation database is missing or \
                     lacks the right -I flags. Facts from files including it are suspect.",
        }));
        let id = entity.id;
        self.artifacts.insert(key, id);
        self.entities.push(entity);
        id
    }

    /// Runs the LSP requests for one file and turns the answers into facts.
    fn collect_file(&mut self, client: &mut ClangdClient, file: &Path, type_hierarchy: bool) -> Result<()> {
        let text = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
        let uri = path_to_uri(file)?;
        let Some(file_id) = self.artifacts.get(&uri).copied() else {
            bail!("{} was not declared before collection", file.display());
        };
        let lines: Vec<&str> = text.lines().collect();

        client.did_open(&uri, &text, file_language(file))?;

        let symbols = client.document_symbols(&uri)?;
        let links = client.document_links(&uri).unwrap_or_default();
        let errors = client.take_error_count(&uri);
        if errors > 0 {
            self.files_with_errors += 1;
        }

        self.collect_includes(file_id, &text, &links);
        for symbol in &symbols {
            self.collect_symbol(symbol, &uri, file_id, None, &[], client.encoding, &lines);
        }
        if type_hierarchy {
            self.collect_bases(client, &uri, &symbols)?;
        }

        // Bounding clangd's memory matters: a million-line run then holds one
        // AST at a time rather than ten thousand.
        client.did_close(&uri)?;

        if let Some(slot) = self.file_slots.get(&uri) {
            let metadata = &mut self.entities[*slot].metadata;
            metadata["diagnostic_errors"] = json!(errors);
            metadata["degraded"] = json!(errors > 0 || self.database.is_degraded());
        }
        Ok(())
    }

    /// Emits one `DependsOn` edge per `#include` directive in `text`.
    ///
    /// Two sources are joined here, and neither would do on its own. The
    /// *spelling* (`"widget.hpp"` versus `<vector>`) exists only in the text, and
    /// clangd never reports it. The *resolution* — which file on disk that
    /// spelling actually names, once the compile flags' `-I` paths are applied —
    /// only clangd can compute, via `textDocument/documentLink`; reimplementing
    /// header search here would be exactly the guessing this module exists to
    /// avoid. The two join on line number.
    fn collect_includes(&mut self, file_id: EntityId, text: &str, links: &HashMap<u64, String>) {
        for (line, spelling, system) in scan_includes(text) {
            let target = match links.get(&line) {
                Some(uri) => self.external_artifact(uri),
                None => {
                    self.unresolved_includes += 1;
                    self.unresolved_include_artifact(&spelling, system)
                }
            };
            self.relate(file_id, target, RelationshipKind::DependsOn);
        }
    }

    /// Walks one `DocumentSymbol` subtree, emitting an [`EntityKind::Symbol`] per
    /// node and preserving clangd's nesting as containment edges.
    ///
    /// `scope` is the chain of enclosing symbol names (`["myproject", "Widget"]`),
    /// which is what makes the *qualified* name computable — and the qualified
    /// name is what later lets a declaration in a header and its out-of-line
    /// definition in a `.C` be recognised as the same thing.
    #[allow(clippy::too_many_arguments)]
    fn collect_symbol(
        &mut self,
        symbol: &Value,
        uri: &str,
        file_id: EntityId,
        parent: Option<(EntityId, u64)>,
        scope: &[String],
        encoding: PositionEncoding,
        lines: &[&str],
    ) {
        let Some(name) = symbol.get("name").and_then(Value::as_str) else {
            return;
        };
        if name.is_empty() {
            return;
        }
        let lsp_kind = symbol.get("kind").and_then(Value::as_u64).unwrap_or(0);
        let detail = symbol.get("detail").and_then(Value::as_str);
        let parent_kind = parent.map(|(_, kind)| kind);

        // The *selection* range is the identifier itself; the plain `range` is
        // the whole declaration, body and all. A symbol's identity lives on its
        // name, and `typeHierarchy` reports base classes by their selection range
        // too — using it here is what makes the two views joinable.
        let line = symbol.pointer("/selectionRange/start/line").and_then(Value::as_u64).unwrap_or(0);
        let wire_column = symbol
            .pointer("/selectionRange/start/character")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let end_line = symbol.pointer("/range/end/line").and_then(Value::as_u64);

        let kind = canonical_kind(lsp_kind, detail, parent_kind);
        let is_template = detail.is_some_and(|detail| detail.starts_with("template"));
        let signature = signature_of(kind, detail);
        let qualified = qualified_name(scope, name);

        let entity = Entity::new(EntityKind::Symbol, name, SOURCE).with_metadata(json!({
            "uri": uri,
            "line": line,
            "column": to_char_column(lines, line, wire_column, encoding),
            "end_line": end_line,
            "kind": kind,
            "lsp_kind": lsp_kind,
            "is_template": is_template,
            "signature": signature,
            "detail": detail,
            "container": (!scope.is_empty()).then(|| scope.join("::")),
            "qualified_name": qualified,
            "static": (lsp_kind == LSP_PROPERTY).then_some(true),
        }));
        let id = self.push(entity);

        // Every symbol is located in its file; a nested one is *also* located in
        // the symbol containing it. Both are `LocatedIn`: a member function's
        // home is its class in exactly the sense a class's home is its file.
        self.relate(id, file_id, RelationshipKind::LocatedIn);
        if let Some((parent_id, _)) = parent {
            self.relate(id, parent_id, RelationshipKind::LocatedIn);
        }

        self.symbols_by_location
            .insert((uri.to_string(), line, wire_column), id);
        if let Some(signature) = signature {
            self.by_signature
                .entry((qualified.clone(), signature.to_string()))
                .or_default()
                .push((id, is_header_uri(uri)));
        }

        let mut child_scope = scope.to_vec();
        child_scope.push(name.to_string());
        if let Some(children) = symbol.get("children").and_then(Value::as_array) {
            for child in children {
                self.collect_symbol(child, uri, file_id, Some((id, lsp_kind)), &child_scope, encoding, lines);
            }
        }
    }

    /// Asks clangd for the base classes of every class-like symbol in this file,
    /// recording them for resolution once the whole project has been walked.
    ///
    /// This is the one fact in this module that `textDocument/documentSymbol`
    /// simply does not carry: clangd reports `class D : public C` with no trace
    /// of `C` whatsoever. Inheritance must be pulled out with a separate
    /// `prepareTypeHierarchy` + `supertypes` pair, per class. It is worth the
    /// round-trips — a C++ codebase's whole architecture is often expressed as
    /// class hierarchies, and a translator that cannot see them is translating
    /// unrelated fragments.
    ///
    /// The two halves of the request disagree in one important case, and the
    /// disagreement is itself the fact worth keeping. `prepareTypeHierarchy`
    /// reports how many parents a class has (in its opaque `data.parents`), while
    /// `supertypes` names them — and `supertypes` cannot name a parent that is a
    /// template instantiation (`class Buffer : public Container<double>`; see the
    /// module docs). Where the counts differ, the shortfall is recorded on the
    /// derived class as `"unresolved_bases"` rather than dropped: "this class has
    /// a base I could not name" is true, and "this class has no base" would not be.
    fn collect_bases(&mut self, client: &mut ClangdClient, uri: &str, symbols: &[Value]) -> Result<()> {
        for (line, character) in class_positions(symbols) {
            let Some(derived) = self.symbols_by_location.get(&(uri.to_string(), line, character)).copied() else {
                continue;
            };
            let mut declared = 0usize;
            let mut resolved = 0usize;

            for item in client.prepare_type_hierarchy(uri, line, character)? {
                declared += item
                    .pointer("/data/parents")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len);

                for base in client.supertypes(&item)? {
                    let (Some(base_uri), Some(name)) = (
                        base.get("uri").and_then(Value::as_str),
                        base.get("name").and_then(Value::as_str),
                    ) else {
                        continue;
                    };
                    let base_line = base
                        .pointer("/selectionRange/start/line")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    let base_column = base
                        .pointer("/selectionRange/start/character")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    resolved += 1;
                    self.pending_bases.push(PendingBase {
                        derived,
                        location: (base_uri.to_string(), base_line, base_column),
                        name: name.to_string(),
                        qualified_name: base.get("detail").and_then(Value::as_str).map(str::to_string),
                    });
                }
            }

            if let Some(missing) = declared.checked_sub(resolved).filter(|missing| *missing > 0) {
                self.unresolved_bases += missing;
                if let Some(slot) = self.symbol_slots.get(&derived) {
                    self.entities[*slot].metadata["unresolved_bases"] = json!(missing);
                }
            }
        }
        Ok(())
    }

    /// Turns the deferred base-class records into edges, now that every file's
    /// symbols are known.
    ///
    /// A base living outside the project — a standard-library or third-party
    /// class — has no symbol entity of its own, since this provider only opens
    /// the project's own files. Rather than dropping the edge, which would
    /// silently lose exactly the relationship a translator most needs, the base
    /// gets an `"external": true` stub: the hierarchy stays connected and its
    /// boundary stays visible.
    fn resolve_inheritance(&mut self) {
        for pending in std::mem::take(&mut self.pending_bases) {
            let base_id = match self.symbols_by_location.get(&pending.location) {
                Some(id) => *id,
                None => {
                    let (uri, line, _) = &pending.location;
                    let entity = Entity::new(EntityKind::Symbol, &pending.name, SOURCE).with_metadata(json!({
                        "uri": uri,
                        "line": line,
                        "kind": "class",
                        "external": true,
                        "qualified_name": pending.qualified_name,
                    }));
                    let id = self.push(entity);
                    self.symbols_by_location.insert(pending.location.clone(), id);
                    id
                }
            };
            // `Custom("inherits")` rather than a bespoke ontology variant:
            // `kopitiam-ontology` has no `Inherits` today, and adding one is a
            // cross-language decision (C++ bases, C# base types *and* interfaces,
            // Python's MRO) that belongs to the ontology, not to one provider.
            // This is the honest placeholder until it gets one.
            self.relate(pending.derived, base_id, RelationshipKind::Custom("inherits".to_string()));
        }
    }

    /// Links each header declaration to the out-of-line definition implementing
    /// it, using the ontology's existing `ImplementedBy` edge.
    ///
    /// C++ splits a symbol across two files: `double Circle::area() const;` in
    /// the header, and `double Circle::area() const { ... }` in the `.C`. clangd
    /// reports them as two unrelated symbols in two unrelated documents. They are
    /// the same thing, and a translator handed only the header has the signature
    /// but not the mathematics. Joining on `(qualified name, signature)` reunites
    /// them — the signature is what keeps overloads apart, since by definition
    /// they share a qualified name. An ambiguous match (the same name and
    /// signature declared in two headers) is left unlinked rather than guessed at.
    fn link_declarations_to_definitions(&mut self) {
        let groups: Vec<Vec<(EntityId, bool)>> = self.by_signature.values().cloned().collect();
        for group in groups {
            let declarations: Vec<EntityId> = group.iter().filter(|(_, header)| *header).map(|(id, _)| *id).collect();
            let definitions: Vec<EntityId> = group.iter().filter(|(_, header)| !*header).map(|(id, _)| *id).collect();
            if declarations.len() != 1 || definitions.len() != 1 {
                continue;
            }
            self.relate(declarations[0], definitions[0], RelationshipKind::ImplementedBy);
        }
    }

    /// Emits the project artifact — the one entity carrying the truth about how
    /// far to trust all the others — and returns the run's facts.
    ///
    /// Note what `"degraded"` deliberately does *not* include: `unresolved_bases`.
    /// A template-instantiation base clangd cannot name is a limitation of the
    /// tool, not a defect in the project's setup, and it afflicts essentially
    /// every real C++ codebase. Folding it into `degraded` would make the flag
    /// permanently true and therefore worthless, when what it needs to mean is
    /// "something here is fixable, and until you fix it these facts may be
    /// wrong". The count is reported alongside instead, where it can be read
    /// without diluting the alarm.
    fn finish(mut self, discovered: usize, truncated: bool, clean_shutdown: bool) -> Result<ProviderOutput> {
        self.link_declarations_to_definitions();

        let analysed = self.file_slots.len();
        let degraded = self.database.is_degraded() || self.files_with_errors > 0 || self.failed_files > 0 || truncated;
        let project = Entity::new(EntityKind::Artifact, self.root.display().to_string(), SOURCE).with_metadata(json!({
            "role": "project",
            "language": "cpp",
            "compilation_database": self.database.describe(),
            "degraded": degraded,
            "degradation": degradation_reason(self.database, self.files_with_errors, self.failed_files, truncated),
            "files_discovered": discovered,
            "files_analysed": analysed,
            "files_truncated": truncated,
            "files_failed": self.failed_files,
            "files_with_errors": self.files_with_errors,
            "unresolved_includes": self.unresolved_includes,
            "unresolved_bases": self.unresolved_bases,
            "clean_shutdown": clean_shutdown,
        }));
        let project_id = self.push(project);

        let file_ids: Vec<EntityId> = self.file_slots.values().map(|slot| self.entities[*slot].id).collect();
        for file_id in file_ids {
            self.relate(file_id, project_id, RelationshipKind::LocatedIn);
        }

        Ok(ProviderOutput {
            entities: self.entities,
            relationships: self.relationships,
        })
    }
}

/// The prose the project artifact carries when something is wrong — `None` when
/// the run was clean, because an empty string would be a fact and there is none.
fn degradation_reason(
    database: &CompilationDatabase,
    files_with_errors: usize,
    failed_files: usize,
    truncated: bool,
) -> Option<String> {
    let mut reasons = Vec::new();
    if database.is_degraded() {
        reasons.push(
            "no compile_commands.json or compile_flags.txt: clangd guessed the compile flags, so \
             project headers did not resolve and unknown types were silently recovered as `int`. \
             Types, signatures and inheritance in these facts may be WRONG, not merely absent."
                .to_string(),
        );
    }
    if files_with_errors > 0 {
        reasons.push(format!(
            "{files_with_errors} file(s) produced clang errors (usually unresolved #includes)"
        ));
    }
    if failed_files > 0 {
        reasons.push(format!("{failed_files} file(s) could not be analysed at all"));
    }
    if truncated {
        reasons.push("the file list was truncated by `max_files`".to_string());
    }
    (!reasons.is_empty()).then(|| reasons.join("; "))
}

// ---------------------------------------------------------------------------
// C++ -> ontology mapping
// ---------------------------------------------------------------------------

/// LSP `SymbolKind::Class`. clangd also uses it for structs, unions and type
/// aliases, distinguishing them only in `detail`.
const LSP_CLASS: u64 = 5;
/// LSP `SymbolKind::Property`. In C++ clangd means one specific thing by it: a
/// **static data member**. (In C# the same number means a real property — which
/// is why the raw LSP number is preserved in metadata alongside our own name for
/// it, and why each language provider maps it for itself.)
const LSP_PROPERTY: u64 = 7;
/// LSP `SymbolKind::Enum`. clangd uses it for enumerators as well as enums,
/// leaving nesting as the only way to tell the two apart.
const LSP_ENUM: u64 = 10;
/// LSP `SymbolKind::Struct`. clangd does not emit it for C++ (it says `Class`
/// with `detail: "struct"`), but the spec allows it and other servers use it.
const LSP_STRUCT: u64 = 23;

/// Maps clangd's answer onto the vocabulary shared by every KOPITIAM language
/// provider.
///
/// That shared vocabulary is the **LSP `SymbolKind` name**, lowercased and
/// snake_cased: `"class"`, `"method"`, `"field"`, `"enum_member"`, and so on.
/// This is not arbitrary. Every language server speaks LSP, so every provider in
/// this crate can reach the same vocabulary without inventing a translation
/// table, and a Rust `struct`, a C++ `struct` and a C# `struct` all land on the
/// same string — which is the entire point of a common semantic model.
///
/// Two refinements go beyond the raw number, both because clangd knows more than
/// `SymbolKind` can express:
///
/// * `detail` separates the four things clangd calls `Class`: `class`, `struct`,
///   `union`, and `type alias` (`typedef`/`using`). Collapsing a `typedef` into
///   "class" would be a lie a translator would act on.
/// * An enum member is an `Enum` nested inside an `Enum`. Nothing else in C++ is,
///   so the parent's kind disambiguates it exactly.
fn canonical_kind(lsp_kind: u64, detail: Option<&str>, parent_kind: Option<u64>) -> &'static str {
    match lsp_kind {
        2 => "module",
        3 => "namespace",
        LSP_CLASS => match detail.map(strip_template_prefix) {
            Some("struct") => "struct",
            Some("union") => "union",
            Some("type alias") => "type_alias",
            _ => "class",
        },
        6 => "method",
        LSP_PROPERTY => "field", // a static data member is a field, not a C#-style property
        8 => "field",
        9 => "constructor",
        LSP_ENUM if parent_kind == Some(LSP_ENUM) => "enum_member",
        LSP_ENUM => "enum",
        11 => "interface",
        12 => "function",
        13 => "variable",
        14 => "constant",
        22 => "enum_member",
        LSP_STRUCT => "struct",
        25 => "operator",
        26 => "type_parameter",
        _ => "unknown",
    }
}

/// `"template class"` -> `"class"`.
///
/// clangd prefixes a template's `detail` with `template `, for types
/// (`"template struct"`) *and* for functions (`"template T (T)"`) — which is
/// also how `is_template` is derived, from one rule rather than two. That bit is
/// the most load-bearing thing in this whole mapping for a translator: a C++
/// template and a Rust generic are the same idea, and a C++ template and a Rust
/// concrete type are not.
fn strip_template_prefix(detail: &str) -> &str {
    detail.strip_prefix("template ").unwrap_or(detail)
}

/// The signature of a callable, or `None` for anything else.
///
/// clangd overloads `detail`: for a function it is the signature
/// (`"double (int, double) const"`), but for a class it is the word `"class"`,
/// and for a field it is the field's *type*. Only the first is a signature, and
/// only the first is recorded as one — a field's type is kept too, but under
/// `detail`, where nobody will mistake it for something it is not.
fn signature_of<'a>(kind: &str, detail: Option<&'a str>) -> Option<&'a str> {
    matches!(kind, "function" | "method" | "constructor" | "operator")
        .then_some(detail)
        .flatten()
}

/// `["myproject", "Widget"]` + `"draw"` -> `"myproject::Widget::draw"`.
///
/// Note that clangd already qualifies an out-of-line definition in a `.C` file
/// (it names the symbol `Widget::draw` and nests it under namespace `myproject`),
/// while the declaration in the header is named `draw` and nested under
/// `myproject` *and* `Widget`. Both therefore produce the identical qualified
/// name — which is exactly what
/// [`Collector::link_declarations_to_definitions`] relies on.
fn qualified_name(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        return name.to_string();
    }
    format!("{}::{}", scope.join("::"), name)
}

/// Converts an LSP column, in whatever unit the server negotiated, to the `char`
/// offset this crate promises its callers. See [`crate::position`].
fn to_char_column(lines: &[&str], line: u64, wire_column: u64, encoding: PositionEncoding) -> u32 {
    let text = lines.get(line as usize).copied().unwrap_or("");
    position::unit_to_char_col(text, wire_column as u32, encoding)
}

/// Every `(line, character)` at which a class-like symbol's name starts: the
/// positions to fire `prepareTypeHierarchy` at.
///
/// Type aliases are excluded. They are `SymbolKind::Class` to clangd but have no
/// base classes, and asking anyway would waste two round-trips per alias on a
/// codebase that has thousands of them.
fn class_positions(symbols: &[Value]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut stack: Vec<&Value> = symbols.iter().rev().collect();
    while let Some(symbol) = stack.pop() {
        let kind = symbol.get("kind").and_then(Value::as_u64).unwrap_or(0);
        let detail = symbol.get("detail").and_then(Value::as_str);
        let class_like =
            (kind == LSP_CLASS || kind == LSP_STRUCT) && detail.map(strip_template_prefix) != Some("type alias");
        if class_like {
            let line = symbol.pointer("/selectionRange/start/line").and_then(Value::as_u64);
            let character = symbol.pointer("/selectionRange/start/character").and_then(Value::as_u64);
            if let (Some(line), Some(character)) = (line, character) {
                out.push((line, character));
            }
        }
        if let Some(children) = symbol.get("children").and_then(Value::as_array) {
            stack.extend(children.iter().rev());
        }
    }
    out
}

/// Every `#include` directive in `text`, as `(0-based line, spelling, is_system)`.
///
/// This is a lexical scan, and deliberately so: it recovers the *spelling* the
/// author wrote, which is the one thing clangd will not tell us (see
/// [`Collector::collect_includes`]). It handles the whitespace C permits
/// (`#  include  <x>`) and ignores computed includes (`#include MACRO`), which
/// carry no spelling to record.
///
/// It does **not** evaluate conditional compilation: an `#include` inside a
/// false `#ifdef` is still reported. That over-approximation is deliberate — for
/// a dependency edge, "this file mentions that header" is the useful fact, and
/// pretending to run the preprocessor without the real defines would be exactly
/// the guessing this module exists to avoid.
fn scan_includes(text: &str) -> Vec<(u64, String, bool)> {
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let Some(rest) = line.trim_start().strip_prefix('#') else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix("include") else {
            continue;
        };
        let rest = rest.trim_start();
        let (open, close, system) = match rest.chars().next() {
            Some('<') => ('<', '>', true),
            Some('"') => ('"', '"', false),
            _ => continue, // `#include MACRO`, or an identifier merely starting with "include"
        };
        let body = &rest[open.len_utf8()..];
        let Some(end) = body.find(close) else { continue };
        let spelling = &body[..end];
        if !spelling.is_empty() {
            out.push((index as u64, spelling.to_string(), system));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

/// Recursively finds every C/C++ source and header under `root`, skipping
/// [`SKIPPED_DIRS`]. Symlinked directories are not followed, which keeps a
/// cyclic link from turning the walk into a hang.
fn discover_sources(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_symlink() {
                continue;
            }
            let path = entry.path();
            if kind.is_dir() {
                if !SKIPPED_DIRS.iter().any(|skip| *skip == entry.file_name()) {
                    stack.push(path);
                }
            } else if kind.is_file() && is_cpp_file(&path) {
                out.push(path);
            }
        }
    }
    out
}

fn extension_of(path: &Path) -> Option<&str> {
    path.extension().and_then(|extension| extension.to_str())
}

fn is_cpp_file(path: &Path) -> bool {
    extension_of(path)
        .is_some_and(|extension| SOURCE_EXTENSIONS.contains(&extension) || HEADER_EXTENSIONS.contains(&extension))
}

fn is_header(path: &Path) -> bool {
    extension_of(path).is_some_and(|extension| HEADER_EXTENSIONS.contains(&extension))
}

fn file_role(path: &Path) -> &'static str {
    if is_header(path) { "header" } else { "source" }
}

/// The language of a file, as far as its *name* can honestly say.
///
/// `.c` is C, and `.cpp`/`.C`/`.hpp`/`.H` are C++, but a bare `.h` is genuinely
/// ambiguous — it is the C header extension and also what half the C++ world
/// uses. Nothing but the translation unit that includes it, compiled with the
/// real flags, can settle that, so this reports `"unknown"` rather than picking a
/// side. clangd decides for itself from the compile command, and its answer, not
/// ours, is what shapes the symbols.
fn file_language(path: &Path) -> &'static str {
    match extension_of(path) {
        Some("c") => "c",
        Some("h") => "unknown",
        _ => "cpp",
    }
}

fn is_header_uri(uri: &str) -> bool {
    uri_to_path(uri).as_deref().is_some_and(is_header)
}

fn relative_to(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).unwrap_or(path).display().to_string()
}

fn path_to_uri(path: &Path) -> Result<String> {
    let absolute = path
        .canonicalize()
        .with_context(|| format!("resolving {}", path.display()))?;
    url::Url::from_file_path(&absolute)
        .map(|url| url.to_string())
        .map_err(|()| anyhow::anyhow!("could not build a file:// URI for {}", absolute.display()))
}

fn uri_to_path(uri: &str) -> Option<PathBuf> {
    url::Url::parse(uri).ok()?.to_file_path().ok()
}

// ---------------------------------------------------------------------------
// The clangd driver
// ---------------------------------------------------------------------------

/// A live clangd process, driven over stdio with JSON-RPC.
///
/// See the module docs for why this is not [`crate::lsp_client::LspClient`]. It
/// is the smaller half of that client — no rename, no code actions, no workspace
/// symbols — plus the four requests clangd needs and `LspClient` does not offer,
/// and it borrows [`crate::position`] for the part that actually matters.
struct ClangdClient {
    child: Child,
    stdin: ChildStdin,
    incoming: Receiver<Value>,
    next_id: i64,
    timeout: Duration,
    /// The unit `Position.character` is measured in, as negotiated during
    /// `initialize`. clangd picks `"utf-8"` — **byte offsets** — which is why
    /// this cannot be ignored. See [`crate::position`].
    encoding: PositionEncoding,
    /// How many *error*-severity diagnostics clangd has published, per URI. This
    /// is how a caller learns that a file did not really parse.
    errors: HashMap<String, usize>,
}

impl ClangdClient {
    fn spawn(binary: &Path, root: &Path, database: &CompilationDatabase, timeout: Duration) -> Result<Self> {
        let mut command = Command::new(binary);
        command
            // Without this, clangd indexes the whole project in the background
            // and writes the result into `<project>/.cache/clangd/`. A provider
            // whose job is to *describe* a project must not write to it. We do
            // not need the index either: `documentSymbol` and `supertypes` are
            // both answered from the AST of the currently open file.
            .arg("--background-index=false")
            .arg("--log=error")
            .arg("--pch-storage=memory");
        if let Some(dir) = database.compile_commands_dir() {
            command.arg(format!("--compile-commands-dir={}", dir.display()));
        }

        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to spawn `{}`", binary.display()))?;
        let stdin = child.stdin.take().context("clangd has no stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().context("clangd has no stdout")?);

        let (sender, incoming) = mpsc::channel();
        thread::spawn(move || {
            while let Ok(Some(message)) = read_message(&mut stdout) {
                if sender.send(message).is_err() {
                    break;
                }
            }
        });

        let mut client = Self {
            child,
            stdin,
            incoming,
            next_id: 1,
            timeout,
            encoding: PositionEncoding::Utf16,
            errors: HashMap::new(),
        };
        client.initialize(root)?;
        Ok(client)
    }

    fn initialize(&mut self, root: &Path) -> Result<()> {
        let root_uri = url::Url::from_file_path(root)
            .map_err(|()| anyhow::anyhow!("project root is not an absolute path: {}", root.display()))?
            .to_string();
        let result = self.request(
            "initialize",
            json!({
                "processId": std::process::id(),
                "rootUri": root_uri,
                "capabilities": {
                    "textDocument": {
                        // Ask for the *hierarchical* reply. Without this, a server
                        // is entitled to answer with a flat `SymbolInformation[]`,
                        // and the nesting — which class a method belongs to — is
                        // exactly the structure we came for.
                        "documentSymbol": { "hierarchicalDocumentSymbolSupport": true },
                        "typeHierarchy": { "dynamicRegistration": false },
                        "documentLink": { "dynamicRegistration": false },
                        "publishDiagnostics": { "relatedInformation": false },
                    },
                    "window": { "workDoneProgress": false },
                    // Advertise all three encodings and honour whichever comes
                    // back. clangd answers `"utf-8"`, i.e. byte offsets.
                    "general": { "positionEncodings": ["utf-8", "utf-16", "utf-32"] },
                },
            }),
            Duration::from_secs(30),
        )?;
        self.encoding = PositionEncoding::from_capability(
            result.pointer("/capabilities/positionEncoding").and_then(Value::as_str),
        );
        self.notify("initialized", json!({}))
    }

    fn write(&mut self, message: &Value) -> Result<()> {
        let body = serde_json::to_vec(message)?;
        write!(self.stdin, "Content-Length: {}\r\n\r\n", body.len())?;
        self.stdin.write_all(&body)?;
        self.stdin.flush()?;
        Ok(())
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.write(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
    }

    /// Sends a request and blocks for its response, absorbing everything else
    /// that arrives meanwhile: diagnostics are tallied (see [`Self::errors`]),
    /// server-initiated requests are answered so clangd is never left waiting on
    /// us, and other notifications are dropped.
    fn request(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        self.write(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;

        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let message = self
                .incoming
                .recv_timeout(remaining)
                .with_context(|| format!("timed out waiting for clangd to answer `{method}`"))?;

            if message.get("id").and_then(Value::as_i64) == Some(id) && message.get("method").is_none() {
                if let Some(error) = message.get("error") {
                    bail!("clangd rejected `{method}`: {error}");
                }
                return Ok(message.get("result").cloned().unwrap_or(Value::Null));
            }
            self.absorb(&message)?;
        }
    }

    /// Handles a message that was not the response we were waiting for.
    fn absorb(&mut self, message: &Value) -> Result<()> {
        match message.get("method").and_then(Value::as_str) {
            Some("textDocument/publishDiagnostics") => {
                let uri = message.pointer("/params/uri").and_then(Value::as_str).unwrap_or_default();
                let errors = message
                    .pointer("/params/diagnostics")
                    .and_then(Value::as_array)
                    .map(|diagnostics| {
                        diagnostics
                            .iter()
                            .filter(|diagnostic| diagnostic.get("severity").and_then(Value::as_u64) == Some(1))
                            .count()
                    })
                    .unwrap_or(0);
                // Last publication wins: clangd republishes a document's full
                // diagnostic set every time it reparses it.
                self.errors.insert(uri.to_string(), errors);
            }
            // A server-initiated request (it carries an `id`) must be answered or
            // clangd blocks. We register nothing dynamically, so `null` is both
            // spec-correct and sufficient.
            Some(_) => {
                if let Some(id) = message.get("id").cloned() {
                    self.write(&json!({ "jsonrpc": "2.0", "id": id, "result": Value::Null }))?;
                }
            }
            None => {}
        }
        Ok(())
    }

    fn did_open(&mut self, uri: &str, text: &str, language: &str) -> Result<()> {
        // clangd takes the dialect from the compile command rather than from this
        // field, but sending the truth costs nothing and keeps us honest against
        // a server that does read it.
        let language_id = if language == "c" { "c" } else { "cpp" };
        self.notify(
            "textDocument/didOpen",
            json!({
                "textDocument": { "uri": uri, "languageId": language_id, "version": 1, "text": text },
            }),
        )
    }

    fn did_close(&mut self, uri: &str) -> Result<()> {
        self.notify("textDocument/didClose", json!({ "textDocument": { "uri": uri } }))
    }

    /// The hierarchical `DocumentSymbol[]` for an open document.
    ///
    /// Should a server ignore `hierarchicalDocumentSymbolSupport` and reply with
    /// a flat `SymbolInformation[]`, those entries carry `location` rather than
    /// `selectionRange`; the caller's lookups then miss and it emits nothing,
    /// rather than emitting nonsense. clangd does honour the capability.
    fn document_symbols(&mut self, uri: &str) -> Result<Vec<Value>> {
        let result = self.request(
            "textDocument/documentSymbol",
            json!({ "textDocument": { "uri": uri } }),
            self.timeout,
        )?;
        Ok(as_array(result))
    }

    /// `#include` targets, as a map from 0-based line to resolved file URI. A
    /// directive clangd could not resolve simply has no entry — the signal
    /// [`Collector::collect_includes`] turns into an unresolved-include artifact.
    fn document_links(&mut self, uri: &str) -> Result<HashMap<u64, String>> {
        let result = self.request(
            "textDocument/documentLink",
            json!({ "textDocument": { "uri": uri } }),
            self.timeout,
        )?;
        Ok(as_array(result)
            .into_iter()
            .filter_map(|link| {
                let line = link.pointer("/range/start/line").and_then(Value::as_u64)?;
                let target = link.get("target").and_then(Value::as_str)?.to_string();
                Some((line, target))
            })
            .collect())
    }

    fn prepare_type_hierarchy(&mut self, uri: &str, line: u64, character: u64) -> Result<Vec<Value>> {
        let result = self.request(
            "textDocument/prepareTypeHierarchy",
            json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": character },
            }),
            self.timeout,
        )?;
        Ok(as_array(result))
    }

    /// The direct base classes of a `TypeHierarchyItem`.
    ///
    /// The item must be passed back verbatim: clangd stashes the symbol's index
    /// ID in its opaque `data` field and needs it to answer.
    fn supertypes(&mut self, item: &Value) -> Result<Vec<Value>> {
        let result = self.request("typeHierarchy/supertypes", json!({ "item": item }), self.timeout)?;
        Ok(as_array(result))
    }

    /// How many *errors* clangd reported for `uri`, first draining anything it
    /// has published that we have not yet read.
    ///
    /// Diagnostics arrive asynchronously. In practice clangd publishes them while
    /// building the AST — i.e. before it can answer `documentSymbol` — so by the
    /// time this is called they are already in the channel; the short drain below
    /// covers the case where they are still in flight. This is best-effort
    /// telemetry: a missed diagnostic understates the damage, and no *fact* in
    /// this module depends on it.
    fn take_error_count(&mut self, uri: &str) -> usize {
        while let Ok(message) = self.incoming.recv_timeout(Duration::from_millis(50)) {
            let _ = self.absorb(&message);
        }
        self.errors.get(uri).copied().unwrap_or(0)
    }

    fn shutdown(&mut self) -> Result<()> {
        self.request("shutdown", Value::Null, Duration::from_secs(10))?;
        self.notify("exit", Value::Null)
    }
}

impl Drop for ClangdClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn as_array(value: Value) -> Vec<Value> {
    match value {
        Value::Array(items) => items,
        _ => Vec::new(),
    }
}

/// Reads one `Content-Length`-framed JSON-RPC message. `Ok(None)` means EOF: the
/// server exited.
fn read_message(stdout: &mut BufReader<ChildStdout>) -> Result<Option<Value>> {
    let mut length: Option<usize> = None;
    loop {
        let mut line = String::new();
        if stdout.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some(value) = line.strip_prefix("Content-Length:") {
            length = Some(value.trim().parse()?);
        }
    }
    let length = length.context("a message from clangd had no Content-Length header")?;
    let mut body = vec![0u8; length];
    stdout.read_exact(&mut body)?;
    Ok(Some(serde_json::from_slice(&body)?))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use tempfile::TempDir;

    use super::*;

    /// A namespace, a base class with a virtual method and a member, a derived
    /// class, a class template, a typedef, an enum, a free function, out-of-line
    /// definitions, and both flavours of `#include` — one of each thing the
    /// mapping claims to handle.
    const HEADER: &str = r#"#pragma once
#include <vector>
#include "missing_header.hpp"

namespace geom {

class Shape {
public:
    virtual ~Shape();
    virtual double area() const;
protected:
    double scale_;
};

class Circle : public Shape {
public:
    explicit Circle(double r);
    double area() const;
    double radius() const;
private:
    double radius_;
};

template <class T>
class Field : public Shape {
public:
    T value() const;
};

typedef double scalar;
enum class Kind { A, B };

double freeFunction(int n);

}  // namespace geom
"#;

    const SOURCE_FILE: &str = r#"#include "shape.hpp"

namespace geom {

Shape::~Shape() {}
double Shape::area() const { return 0.0; }

Circle::Circle(double r) : radius_(r) {}
double Circle::area() const { return 3.14159 * radius_ * radius_; }
double Circle::radius() const { return radius_; }

double freeFunction(int n) { return n * 2.0; }

}  // namespace geom
"#;

    /// Writes the sample project. `with_database` controls the *only* thing that
    /// separates a trustworthy run from a degraded one.
    fn sample_project(with_database: bool) -> TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("shape.hpp"), HEADER).unwrap();
        std::fs::write(dir.path().join("shape.cpp"), SOURCE_FILE).unwrap();
        if with_database {
            let root = dir.path().display();
            std::fs::write(
                dir.path().join("compile_commands.json"),
                format!(
                    r#"[{{"directory": "{root}", "file": "{root}/shape.cpp",
                          "command": "clang++ -std=c++17 -c shape.cpp -o shape.o"}}]"#
                ),
            )
            .unwrap();
        }
        dir
    }

    fn clangd_installed() -> bool {
        CppProvider::new().resolve_binary().is_some()
    }

    /// Skips (rather than fails) when clangd is absent, matching
    /// [`crate::providers::rust_analyzer`]'s convention: an environment without
    /// the tool must not turn into a red test suite.
    macro_rules! require_clangd {
        () => {
            if !clangd_installed() {
                eprintln!("skipping: clangd is not installed in this environment");
                return;
            }
        };
    }

    fn symbols<'a>(output: &'a ProviderOutput, name: &str) -> Vec<&'a Entity> {
        output
            .entities
            .iter()
            .filter(|entity| entity.kind == EntityKind::Symbol && entity.name == name)
            .collect()
    }

    fn project(output: &ProviderOutput) -> &Entity {
        output
            .entities
            .iter()
            .find(|entity| entity.metadata.get("role").and_then(Value::as_str) == Some("project"))
            .expect("every run emits exactly one project artifact")
    }

    /// The names of the entities `from` points at with a relationship of `kind`.
    fn targets(output: &ProviderOutput, from: EntityId, kind: &RelationshipKind) -> HashSet<String> {
        output
            .relationships
            .iter()
            .filter(|relationship| relationship.from == from && relationship.kind == *kind)
            .filter_map(|relationship| output.entities.iter().find(|entity| entity.id == relationship.to))
            .map(|entity| entity.name.clone())
            .collect()
    }

    fn inherits() -> RelationshipKind {
        RelationshipKind::Custom("inherits".to_string())
    }

    // -- Degradation: the part that must never lie ---------------------------

    #[test]
    fn degrades_to_empty_when_clangd_is_not_installed() {
        let output = CppProvider::new()
            .with_binary("kopitiam-definitely-not-clangd")
            .collect(&std::env::temp_dir())
            .expect("a missing tool must not fail the collection run");
        assert!(output.entities.is_empty());
        assert!(output.relationships.is_empty());
    }

    /// The same check, but proving `PATH` resolution is what does the work — and
    /// doing it without mutating the process environment, which would race every
    /// other test in this binary.
    #[test]
    fn degrades_to_empty_when_clangd_is_not_on_the_injected_path() {
        let empty = tempfile::tempdir().unwrap();
        let project = sample_project(true);
        let output = CppProvider::new()
            .with_path(empty.path().as_os_str())
            .collect(project.path())
            .expect("a missing tool must not fail the collection run");
        assert!(
            output.entities.is_empty(),
            "no clangd on PATH must mean no facts, not fabricated ones"
        );
    }

    #[test]
    fn finds_compile_commands_in_the_root_and_in_build_and_out() {
        for dir in ["", "build", "out"] {
            let root = tempfile::tempdir().unwrap();
            let holder = root.path().join(dir);
            std::fs::create_dir_all(&holder).unwrap();
            std::fs::write(holder.join("compile_commands.json"), "[]").unwrap();

            let database = CompilationDatabase::discover(root.path());
            assert_eq!(
                database,
                CompilationDatabase::CompileCommands(holder.join("compile_commands.json")),
                "a compile_commands.json in `{dir}/` must be found"
            );
            assert!(!database.is_degraded());
            assert_eq!(database.compile_commands_dir(), Some(holder.as_path()));
        }
    }

    #[test]
    fn falls_back_to_compile_flags_txt_but_prefers_a_real_database() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("compile_flags.txt"), "-std=c++17\n").unwrap();
        let flags = CompilationDatabase::discover(root.path());
        assert_eq!(flags, CompilationDatabase::CompileFlags(root.path().join("compile_flags.txt")));
        assert!(!flags.is_degraded());
        assert_eq!(flags.compile_commands_dir(), None, "clangd finds compile_flags.txt itself");

        std::fs::write(root.path().join("compile_commands.json"), "[]").unwrap();
        assert!(
            matches!(CompilationDatabase::discover(root.path()), CompilationDatabase::CompileCommands(_)),
            "a real compilation database must win over the flat fallback"
        );
    }

    #[test]
    fn reports_a_missing_compilation_database_as_degraded() {
        let root = tempfile::tempdir().unwrap();
        let database = CompilationDatabase::discover(root.path());
        assert_eq!(database, CompilationDatabase::Missing);
        assert!(database.is_degraded());
        assert_eq!(database.describe(), json!({ "kind": "none" }));
    }

    /// The heart of the honesty contract. Without a compilation database clangd
    /// *still returns symbols* — plausible ones, with wrong types. If this
    /// provider handed those back looking like a clean result, a translator would
    /// act on fiction. It must emit what it has AND say, in the facts themselves,
    /// that the picture is degraded.
    #[test]
    fn without_a_compilation_database_it_emits_facts_but_flags_them_degraded() {
        require_clangd!();
        let dir = sample_project(false);
        let output = CppProvider::new().collect(dir.path()).expect("collection");

        assert!(
            !symbols(&output, "Circle").is_empty(),
            "clangd still parses standalone files, so we emit what we can rather than nothing"
        );

        let project = project(&output);
        assert_eq!(project.metadata["degraded"], json!(true));
        assert_eq!(project.metadata["compilation_database"]["kind"], json!("none"));
        let reason = project.metadata["degradation"].as_str().expect("a reason, in words");
        assert!(
            reason.contains("compile_commands.json"),
            "the reason must name the missing thing: {reason}"
        );
        assert!(
            project.metadata["unresolved_includes"].as_u64().unwrap() >= 1,
            "`missing_header.hpp` cannot resolve without a database, and that must be counted"
        );

        // And the unresolvable include is in the graph as evidence, not dropped.
        let include = output
            .entities
            .iter()
            .find(|entity| entity.name == "missing_header.hpp")
            .expect("an unresolved #include must still appear in the graph");
        assert_eq!(include.kind, EntityKind::Artifact);
        assert_eq!(include.metadata["resolved"], json!(false));
    }

    #[test]
    fn with_a_compilation_database_nothing_is_marked_degraded() {
        require_clangd!();
        let dir = sample_project(true);
        let output = CppProvider::new().collect(dir.path()).expect("collection");
        let project = project(&output);
        assert_eq!(
            project.metadata["compilation_database"]["kind"],
            json!("compile_commands.json")
        );
        assert_eq!(project.metadata["files_discovered"], json!(2));
        assert_eq!(project.metadata["files_analysed"], json!(2));
    }

    // -- The mapping ---------------------------------------------------------

    #[test]
    fn emits_the_expected_symbols_with_the_expected_kinds() {
        require_clangd!();
        let dir = sample_project(true);
        let output = CppProvider::new().collect(dir.path()).expect("collection");

        let kinds = |name: &str| -> Vec<String> {
            symbols(&output, name)
                .iter()
                .filter_map(|entity| entity.metadata["kind"].as_str().map(str::to_string))
                .collect()
        };
        assert!(kinds("geom").contains(&"namespace".to_string()));
        assert!(kinds("Shape").contains(&"class".to_string()));
        assert!(kinds("Circle").contains(&"class".to_string()));
        assert!(kinds("Field").contains(&"class".to_string()));
        assert!(kinds("scale_").contains(&"field".to_string()));
        assert!(kinds("scalar").contains(&"type_alias".to_string()), "a typedef is not a class");
        assert!(kinds("Kind").contains(&"enum".to_string()));
        assert!(kinds("A").contains(&"enum_member".to_string()), "an enumerator nested in an enum");
        assert!(kinds("freeFunction").contains(&"function".to_string()));
        assert!(kinds("area").contains(&"method".to_string()));

        // Provenance is not optional: CLAUDE.md's Scientific Standards.
        assert!(output.entities.iter().all(|entity| entity.source == "clangd"));
    }

    /// The single most important bit for translation: is this a template? A C++
    /// template maps to a Rust generic; a concrete class does not.
    #[test]
    fn marks_templates_as_templates() {
        require_clangd!();
        let dir = sample_project(true);
        let output = CppProvider::new().collect(dir.path()).expect("collection");

        assert!(
            symbols(&output, "Field")
                .iter()
                .any(|entity| entity.metadata["is_template"] == json!(true)),
            "`template <class T> class Field` must be marked a template"
        );
        assert!(
            symbols(&output, "Field")
                .iter()
                .any(|entity| entity.metadata["detail"] == json!("template class")),
            "and clangd's own word for it is preserved verbatim"
        );
        assert!(
            symbols(&output, "Circle")
                .iter()
                .all(|entity| entity.metadata["is_template"] == json!(false)),
            "a plain class must not be"
        );
    }

    #[test]
    fn records_signatures_for_callables_and_not_for_types() {
        require_clangd!();
        let dir = sample_project(true);
        let output = CppProvider::new().collect(dir.path()).expect("collection");

        assert!(
            symbols(&output, "freeFunction")
                .iter()
                .any(|entity| entity.metadata["signature"] == json!("double (int)")),
            "a function carries its signature"
        );
        assert!(
            symbols(&output, "Shape")
                .iter()
                .all(|entity| entity.metadata["signature"].is_null()),
            "`detail: \"class\"` is not a signature and must not be recorded as one"
        );
    }

    /// Inheritance: the fact much C++ architecture is built out of, and the one
    /// `documentSymbol` does not carry at all.
    #[test]
    fn captures_inheritance_including_from_a_template() {
        require_clangd!();
        let dir = sample_project(true);
        let output = CppProvider::new().collect(dir.path()).expect("collection");

        let circle = symbols(&output, "Circle")
            .into_iter()
            .find(|entity| entity.metadata["kind"] == json!("class"))
            .expect("class Circle");
        assert!(
            targets(&output, circle.id, &inherits()).contains("Shape"),
            "`class Circle : public Shape` must produce an inheritance edge"
        );

        let field = symbols(&output, "Field").into_iter().next().expect("class Field");
        assert!(
            targets(&output, field.id, &inherits()).contains("Shape"),
            "a class template's base class must be captured too"
        );

        let shape = symbols(&output, "Shape")
            .into_iter()
            .find(|entity| entity.metadata["kind"] == json!("class"))
            .expect("class Shape");
        assert!(
            targets(&output, shape.id, &inherits()).is_empty(),
            "a root class must not acquire a base out of nowhere"
        );
    }

    #[test]
    fn nests_members_inside_their_class_and_locates_everything_in_its_file() {
        require_clangd!();
        let dir = sample_project(true);
        let output = CppProvider::new().collect(dir.path()).expect("collection");

        let radius = symbols(&output, "radius_").into_iter().next().expect("the field radius_");
        let located = targets(&output, radius.id, &RelationshipKind::LocatedIn);
        assert!(located.contains("Circle"), "a member is located in its class: {located:?}");
        assert!(
            located.iter().any(|name| name.ends_with("shape.hpp")),
            "and in its file too: {located:?}"
        );
    }

    #[test]
    fn links_a_header_declaration_to_its_out_of_line_definition() {
        require_clangd!();
        let dir = sample_project(true);
        let output = CppProvider::new().collect(dir.path()).expect("collection");

        // `double Circle::area() const;` in shape.hpp is implemented by
        // `double Circle::area() const { ... }` in shape.cpp. clangd reports them
        // as two unrelated symbols; the graph must not.
        let declaration = symbols(&output, "area")
            .into_iter()
            .find(|entity| {
                entity.metadata["qualified_name"] == json!("geom::Circle::area")
                    && entity.metadata["uri"].as_str().is_some_and(|uri| uri.ends_with("shape.hpp"))
            })
            .expect("the declaration in the header");
        assert_eq!(
            targets(&output, declaration.id, &RelationshipKind::ImplementedBy),
            HashSet::from(["Circle::area".to_string()]),
            "the declaration must point at the definition, which clangd names `Circle::area`"
        );
    }

    #[test]
    fn records_include_dependencies_resolved_and_unresolved() {
        require_clangd!();
        let dir = sample_project(true);
        let output = CppProvider::new().collect(dir.path()).expect("collection");

        let header = output
            .entities
            .iter()
            .find(|entity| entity.kind == EntityKind::Artifact && entity.name.ends_with("shape.hpp"))
            .expect("the header artifact");
        let dependencies = targets(&output, header.id, &RelationshipKind::DependsOn);
        assert!(
            dependencies.iter().any(|name| name.ends_with("vector")),
            "`#include <vector>` resolves to a real system header: {dependencies:?}"
        );
        assert!(
            dependencies.contains("missing_header.hpp"),
            "an unresolvable include is still a recorded dependency, not a silent gap: {dependencies:?}"
        );

        let source = output
            .entities
            .iter()
            .find(|entity| entity.kind == EntityKind::Artifact && entity.name.ends_with("shape.cpp"))
            .expect("the source artifact");
        assert!(
            targets(&output, source.id, &RelationshipKind::DependsOn)
                .iter()
                .any(|name| name.ends_with("shape.hpp")),
            "`#include \"shape.hpp\"` must link the .cpp to the header entity that already exists"
        );
    }

    /// clangd negotiates `"utf-8"`, i.e. *byte* offsets. A symbol sitting after
    /// non-ASCII text on its line is where that stops being a technicality.
    #[test]
    fn reports_char_columns_not_byte_offsets_on_non_ascii_lines() {
        require_clangd!();
        let dir = tempfile::tempdir().unwrap();
        // `Ω` is two bytes and one char, so `Δ`'s byte column and char column
        // differ — which is the entire point of the test.
        let text = "enum Greek { \u{03A9}, \u{0394} };\n";
        std::fs::write(dir.path().join("greek.hpp"), text).unwrap();
        std::fs::write(dir.path().join("compile_flags.txt"), "-std=c++17\n").unwrap();

        let output = CppProvider::new().collect(dir.path()).expect("collection");
        let delta = symbols(&output, "\u{0394}")
            .into_iter()
            .next()
            .expect("the enumerator Δ");

        let line = text.lines().next().unwrap();
        let expected = line.chars().position(|c| c == '\u{0394}').unwrap() as u64;
        let byte_offset = line.find('\u{0394}').unwrap() as u64;
        assert_ne!(expected, byte_offset, "the test is only meaningful if the two differ");
        assert_eq!(
            delta.metadata["column"],
            json!(expected),
            "the column must be a char offset ({expected}), not clangd's byte offset ({byte_offset})"
        );
    }

    #[test]
    fn max_files_truncates_and_says_so_rather_than_quietly_stopping() {
        require_clangd!();
        let dir = sample_project(true);
        let output = CppProvider::new()
            .with_max_files(1)
            .collect(dir.path())
            .expect("collection");

        let project = project(&output);
        assert_eq!(project.metadata["files_discovered"], json!(2));
        assert_eq!(project.metadata["files_analysed"], json!(1));
        assert_eq!(project.metadata["files_truncated"], json!(true));
        assert_eq!(project.metadata["degraded"], json!(true), "a truncated run is a degraded run");
    }

    // -- Pure units ----------------------------------------------------------

    #[test]
    fn scans_include_directives_including_awkward_spacing() {
        let text = "#include <vector>\n#  include \"a/b.H\"\n  # include\t<cmath>\n#include MACRO\n#define X 1\nint x;\n";
        assert_eq!(
            scan_includes(text),
            vec![
                (0, "vector".to_string(), true),
                (1, "a/b.H".to_string(), false),
                (2, "cmath".to_string(), true),
            ],
            "computed includes (`#include MACRO`) carry no spelling and are skipped"
        );
    }

    #[test]
    fn maps_clangd_symbol_kinds_onto_the_shared_vocabulary() {
        assert_eq!(canonical_kind(3, None, None), "namespace");
        assert_eq!(canonical_kind(5, Some("class"), None), "class");
        assert_eq!(canonical_kind(5, Some("template class"), None), "class");
        assert_eq!(canonical_kind(5, Some("struct"), None), "struct");
        assert_eq!(canonical_kind(5, Some("template struct"), None), "struct");
        assert_eq!(canonical_kind(5, Some("union"), None), "union");
        assert_eq!(canonical_kind(5, Some("type alias"), None), "type_alias");
        assert_eq!(canonical_kind(6, Some("void ()"), None), "method");
        assert_eq!(canonical_kind(7, Some("int"), None), "field", "a static data member is a field");
        assert_eq!(canonical_kind(8, Some("double"), None), "field");
        assert_eq!(canonical_kind(9, Some("(double)"), None), "constructor");
        assert_eq!(canonical_kind(10, Some("enum"), None), "enum");
        assert_eq!(canonical_kind(10, Some("Kind"), Some(LSP_ENUM)), "enum_member");
        assert_eq!(canonical_kind(12, Some("template T (T)"), None), "function");
        assert_eq!(canonical_kind(13, Some("const int"), None), "variable");
        assert_eq!(canonical_kind(99, None, None), "unknown");
    }

    #[test]
    fn a_template_is_recognised_from_clangds_detail_prefix_for_types_and_functions() {
        // clangd prefixes both, which is what makes one rule enough.
        assert!("template class".starts_with("template"));
        assert!("template T (T)".starts_with("template"));
        assert_eq!(strip_template_prefix("template struct"), "struct");
        assert_eq!(strip_template_prefix("class"), "class");
    }

    #[test]
    fn signatures_belong_to_callables_only() {
        assert_eq!(signature_of("function", Some("double (int)")), Some("double (int)"));
        assert_eq!(signature_of("method", Some("void () const")), Some("void () const"));
        assert_eq!(signature_of("class", Some("class")), None);
        assert_eq!(signature_of("field", Some("double")), None, "a field's type is not a signature");
    }

    #[test]
    fn qualified_names_reunite_a_declaration_and_its_out_of_line_definition() {
        // How clangd names the declaration in the header ...
        let from_header = qualified_name(&["geom".to_string(), "Circle".to_string()], "area");
        // ... and how it names the definition in the .cpp: different nesting,
        // different `name`, identical qualified name. That is the join.
        let from_source = qualified_name(&["geom".to_string()], "Circle::area");
        assert_eq!(from_header, "geom::Circle::area");
        assert_eq!(from_header, from_source);
    }

    #[test]
    fn classifies_uppercase_c_and_h_as_cpp_with_case_respected() {
        assert!(is_cpp_file(Path::new("widget.C")), "an uppercase `.C` is a C++ source");
        assert!(is_header(Path::new("widget.H")), "and an uppercase `.H` is a C++ header");
        assert!(!is_header(Path::new("widget.C")));
        assert_eq!(file_language(Path::new("widget.C")), "cpp");
        assert_eq!(file_language(Path::new("legacy.c")), "c", "lowercase `.c` is C");
        assert_eq!(
            file_language(Path::new("ambiguous.h")),
            "unknown",
            "a bare `.h` could be either, and nothing but the compile command can say"
        );
        assert!(!is_cpp_file(Path::new("README.md")));
    }

    #[test]
    fn the_file_walk_skips_build_output_and_vcs_directories() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("a.cpp"), "").unwrap();
        for skipped in ["build", ".git", "node_modules"] {
            let dir = root.path().join(skipped);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("generated.cpp"), "").unwrap();
        }
        std::fs::create_dir_all(root.path().join("src")).unwrap();
        std::fs::write(root.path().join("src").join("b.H"), "").unwrap();

        let mut found = discover_sources(root.path());
        found.sort();
        let names: Vec<String> = found
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["a.cpp".to_string(), "b.H".to_string()]);
    }

    /// The full pipeline against a deliberately deep C++ hierarchy: an abstract
    /// base, a polymorphic hierarchy three deep, a class template over an element
    /// type, and members whose types come from a header.
    ///
    /// `#[ignore]`d because it drives a real clangd end to end; run it with
    /// `cargo test --release -p kopitiam-semantic -- --ignored --nocapture`.
    #[test]
    #[ignore = "drives a real clangd process end to end"]
    fn live_clangd_on_a_deep_template_backed_hierarchy() {
        require_clangd!();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("buffer.H"),
            r#"#pragma once
namespace app {
template <class Type> class Container { public: Type* data_; };
class DoubleBuffer : public Container<double> {};
}
"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("component.H"),
            r#"#pragma once
#include "buffer.H"
namespace app {
class Component {
public:
    virtual ~Component();
    virtual void update() = 0;
    virtual double value() const = 0;
protected:
    DoubleBuffer buffer_;
};
class BasicComponent : public Component {
public:
    void update() override;
};
class Button : public BasicComponent {
public:
    void update() override;
    double value() const override;
private:
    double weight_;
};
}
"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("compile_flags.txt"), "-std=c++17\n-xc++\n").unwrap();

        let output = CppProvider::new().collect(dir.path()).expect("collection");

        let button = symbols(&output, "Button").into_iter().next().expect("Button");
        assert!(targets(&output, button.id, &inherits()).contains("BasicComponent"));

        let basic = symbols(&output, "BasicComponent").into_iter().next().expect("BasicComponent");
        assert!(
            targets(&output, basic.id, &inherits()).contains("Component"),
            "the chain Button -> BasicComponent -> Component must be walkable"
        );

        // And now the limitation, pinned as a test so that it is knowledge rather
        // than a surprise. `class DoubleBuffer : public Container<double>` inherits
        // from a template *instantiation*, which clangd's `supertypes` cannot
        // name — with the background index on or off. Template-heavy C++ codebases
        // are full of exactly this shape. We do not fabricate the edge; we record
        // that one is missing.
        //
        // If clangd ever learns to resolve these, this assertion fails — which is
        // the point. Update the module docs when it does.
        let double_buffer = symbols(&output, "DoubleBuffer").into_iter().next().expect("DoubleBuffer");
        assert!(
            targets(&output, double_buffer.id, &inherits()).is_empty(),
            "clangd cannot name a template-instantiation base; if this now passes, clangd improved"
        );
        assert_eq!(
            double_buffer.metadata["unresolved_bases"],
            json!(1),
            "but the graph must still say a base class is there and could not be named"
        );
        assert_eq!(project(&output).metadata["unresolved_bases"], json!(1));

        eprintln!(
            "entities={} relationships={} degraded={} unresolved_bases={}",
            output.entities.len(),
            output.relationships.len(),
            project(&output).metadata["degraded"],
            project(&output).metadata["unresolved_bases"],
        );
    }
}
