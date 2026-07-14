//! Python knowledge provider: turns a Python project into `kopitiam-ontology`
//! facts.
//!
//! # What this provider is for
//!
//! Python is not (yet) a translation *source* for KOPITIAM the way C++ or
//! Fortran are — it is the language a great deal of the software world writes
//! its glue, its drivers, its automation and an increasing share of its
//! application logic in (`requests`, `numpy`, `pandas`, and countless project
//! packages). Indexing it is what lets the Knowledge Engine answer "which
//! module drives this entry point", "what does this class inherit from", "what
//! does this package depend on" without a model re-reading the repository.
//!
//! # The mapping onto the shared ontology
//!
//! Every language adapter in [`crate::providers`] emits *the same* semantic
//! vocabulary — that sameness is the entire point of `kopitiam-ontology`. A
//! Python `class`, a Visual Basic `Class`, a C# `class` and a Rust `struct` must
//! all arrive in the knowledge graph as an [`EntityKind::Symbol`], or nothing
//! above this layer can reason across languages. Concretely, this provider
//! emits:
//!
//! | Python concept | Ontology |
//! |---|---|
//! | the project (a directory with `pyproject.toml` / `setup.py` / `.py` files) | [`EntityKind::Artifact`], `artifact_kind: "project"` |
//! | a package (a directory with `__init__.py`) | [`EntityKind::Artifact`], `artifact_kind: "package"` |
//! | a module (a `.py` file) | [`EntityKind::Artifact`], `artifact_kind: "module"` |
//! | an imported module we do not own (`numpy.linalg`) | [`EntityKind::Artifact`], `artifact_kind: "external_module"` |
//! | class, function, method, module-level constant | [`EntityKind::Symbol`], `symbol_kind: "class" \| "function" \| "method" \| "constant" \| "variable"` |
//! | a base class we cannot resolve inside the project (`torch.nn.Module`) | [`EntityKind::Symbol`], `external: true` |
//!
//! and these relationships:
//!
//! | Edge | Kind | Meaning |
//! |---|---|---|
//! | module/package -> parent package (or project) | [`RelationshipKind::LocatedIn`] | the artifact containment tree |
//! | symbol -> module | [`RelationshipKind::LocatedIn`] | "which file is this symbol in" — emitted for *every* symbol, including nested ones, so that question is one hop for all of them |
//! | symbol -> enclosing symbol | [`RelationshipKind::LocatedIn`] | a method inside a class, a closure inside a function |
//! | module -> imported module | [`RelationshipKind::DependsOn`] | the import graph, internal and external |
//! | class -> base class | `Custom("inherits")` | Python's MRO, first-order |
//!
//! `LocatedIn` carries both containment questions, which is not an overload: the
//! two are told apart by the *kind of the target*. A `LocatedIn` edge to an
//! [`EntityKind::Artifact`] means "in this file"; one to an
//! [`EntityKind::Symbol`] means "inside this class/function". This is also what
//! the C++, C# and Visual Basic adapters do, and four languages agreeing on one
//! encoding is worth more than a fifth spelling of the same fact.
//!
//! Inheritance is the one place no first-class variant exists, so it uses
//! [`RelationshipKind::Custom`] — the ontology's own sanctioned escape hatch —
//! with the same `"inherits"` string and the same derived -> base direction the
//! C++ and Visual Basic adapters chose. `DependsOn` would have been wrong: "is a"
//! is not "uses", and a translation workflow needs to tell them apart. An
//! `Inherits` variant is what `kopitiam-ontology` should grow; when it does, one
//! constant below changes and every language adapter benefits at once.
//!
//! # Where the facts come from
//!
//! Two derivations, and every [`Entity`] says which one produced it
//! (`metadata.derivation`), because a fact you cannot attribute is a fact you
//! cannot trust (CLAUDE.md, Scientific Standards):
//!
//! * `"language-server"` — a real `pyright` or `pylsp` process, driven over
//!   stdio by [`LspClient`]. Authoritative: a type checker understands `__all__`,
//!   re-exports and conditional definitions, which no scanner does.
//! * `"source-scan"` — the deterministic, dependency-free scanner in this module.
//!   Python is indentation-structured, which makes *declarations* (`class`,
//!   `def`, module-level `NAME = ...`) reliably recoverable from the token stream
//!   without a full parse. That is a property of Python's grammar, not a lucky
//!   heuristic — and it is emphatically not true of C++.
//!
//! The source scan is not a stand-in for the language server; it is what makes
//! this provider satisfy CLAUDE.md's **Offline First** rule. Neither `pyright` (a
//! Node.js program) nor `pylsp` (a Python program) is installable with `cargo`,
//! so a purely LSP-driven Python provider emits *nothing* on a machine that has
//! neither — and the import graph, the single most useful thing to know about a
//! Python project, is not something either server would tell us anyway: LSP has
//! no request for "what does this module import". So the scan earns its keep even
//! when a server is present. [`SymbolSource::LanguageServerOnly`] is there for a
//! caller who wants the strict, tool-only reading of the [`KnowledgeProvider`]
//! contract instead.
//!
//! # Known limits of the source scan
//!
//! Written down here rather than rediscovered later. The scanner:
//!
//! * reads only the first line of a `class`/`def` header, so bases and
//!   signatures split across lines are truncated (the symbol itself is still
//!   correct);
//! * tracks triple-quoted strings well enough to ignore an `import` inside a
//!   docstring — a real hazard, since scientific module docstrings routinely
//!   contain usage examples — but it does not tokenize Python, so a `"""` inside
//!   a `#` comment will confuse it;
//! * treats module-level `UPPER_SNAKE = ...` as a constant and ignores other
//!   module-level bindings, to keep the graph signal-dense;
//! * cannot see anything created at runtime (`type()`, `setattr`, metaclass
//!   magic). That is exactly what `pyright` is for, and why it is preferred
//!   whenever it is on `PATH`.
//!
//! The long-term answer is a real Python parse (`rustpython-parser` is pure Rust
//! and would keep the Pure Rust Core promise); until then, this is deterministic,
//! dependency-free, and honest about what it does not know.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use kopitiam_ontology::{Entity, EntityId, EntityKind, Relationship, RelationshipKind};
use serde_json::{Value, json};

use crate::lsp_client::LspClient;
use crate::position::{self, PositionEncoding};
use crate::provider::{KnowledgeProvider, ProviderOutput};

/// The `Entity::source` stamped on every fact this provider emits.
///
/// Deliberately the *language*, not the tool. Unlike Rust — where rust-analyzer
/// is the only game in town and doubles as a useful provenance string — Python
/// has several interchangeable servers, and which one happened to be on `PATH`
/// is a fact about the machine, not about the code. A consumer asking "what do
/// we know about the Python side of this project" must not have to enumerate
/// server names. Which server (or the scanner) produced a given fact is recorded
/// in `metadata.derivation` / `metadata.server` instead, where it belongs.
const SOURCE: &str = "python";

/// Directories never worth walking. Not merely an optimisation: a `.venv`
/// contains thousands of *other people's* modules, and indexing them as though
/// they were part of the project would poison the graph with facts that are true
/// of NumPy and irrelevant to the user.
const SKIPPED_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".eggs",
    "node_modules",
    "site-packages",
    "build",
    "dist",
];

/// Files whose presence marks a directory as the root of a Python project.
/// Python has no single manifest — `pyproject.toml` is only the most recent of
/// several conventions, and a great deal of scientific Python predates all of
/// them and is simply a directory of `.py` files, which is why a project with no
/// marker at all is still a project here.
const PROJECT_MARKERS: &[&str] = &[
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "environment.yml",
    "tox.ini",
];

// ---------------------------------------------------------------------------
// Language servers
// ---------------------------------------------------------------------------

/// A Python language server this provider knows how to drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PythonServer {
    /// Microsoft's pyright, invoked as `pyright-langserver --stdio`. The primary
    /// choice: it is a full type checker, so its symbol table reflects what
    /// Python will actually do at runtime rather than what the source looks like.
    Pyright,
    /// `python-lsp-server` (`pylsp`), the community Jedi-based server. Slower and
    /// less precise, but it is the one already present in most scientific users'
    /// environments, and it speaks stdio with no arguments at all.
    Pylsp,
}

impl PythonServer {
    /// The stable string recorded in `metadata.server`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pyright => "pyright",
            Self::Pylsp => "pylsp",
        }
    }

    /// The executable to look for on `PATH`.
    pub fn program(self) -> &'static str {
        match self {
            Self::Pyright => "pyright-langserver",
            Self::Pylsp => "pylsp",
        }
    }

    /// The argv the server needs in order to speak LSP over stdio.
    ///
    /// `pyright-langserver` refuses to start without an explicit transport flag;
    /// `pylsp` defaults to stdio and needs none. That asymmetry is load-bearing
    /// today — see [`PythonProvider::collect_lsp_symbols`].
    pub fn args(self) -> &'static [&'static str] {
        match self {
            Self::Pyright => &["--stdio"],
            Self::Pylsp => &[],
        }
    }
}

/// The servers this provider will try, in preference order.
const SERVERS: [PythonServer; 2] = [PythonServer::Pyright, PythonServer::Pylsp];

/// A language server found on `PATH`, resolved to an absolute path.
///
/// Absolute matters: detection may run against an *injected* `PATH` (see
/// [`PythonProvider::with_path`]), and spawning by bare name would then resolve
/// against the real process environment instead — silently running a different
/// binary than the one we detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedServer {
    pub server: PythonServer,
    pub program: PathBuf,
}

/// Searches `path_var` (in `PATH`'s platform-native separated syntax) for an
/// executable file named `executable`.
///
/// Takes the `PATH` string as a parameter rather than reading the process
/// environment, so tests can point detection at an isolated directory without
/// mutating global state — `std::env::set_var` is unsound to call from a
/// multi-threaded test binary. The pattern (and the Windows `.exe` handling) is
/// the one `kopitiam-neovim`'s `lsp::registry::which_in` established; it is
/// re-derived here rather than shared because `kopitiam-semantic` sits *below*
/// `kopitiam-neovim` in CLAUDE.md's dependency direction and must not depend on
/// it. Thirty lines of `which` is a cheaper price than an inverted edge in the
/// crate graph.
fn which_in(executable: &str, path_var: &OsStr) -> Option<PathBuf> {
    let names: Vec<String> = if cfg!(windows) && !executable.to_ascii_lowercase().ends_with(".exe") {
        vec![executable.to_string(), format!("{executable}.exe")]
    } else {
        vec![executable.to_string()]
    };
    for dir in std::env::split_paths(path_var) {
        for name in &names {
            let candidate = dir.join(name);
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// Exists, is a regular file, and (on Unix) has an executable bit. On Windows
/// there is no equivalent bit to check, and existence is the whole test — the
/// same rule `cmd.exe` itself applies.
fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = path.metadata() else {
        return false;
    };
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

// ---------------------------------------------------------------------------
// The provider
// ---------------------------------------------------------------------------

/// Where a run's [`EntityKind::Symbol`] facts are allowed to come from.
///
/// Structural facts — project, package and module artifacts, and the import
/// graph — are unaffected by this: no language server reports them, so they
/// always come from the deterministic scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SymbolSource {
    /// Use a language server if one is on `PATH`; fall back to the source scan if
    /// none is, or if the one that is fails to answer. The default, and the right
    /// choice for `kopitiam scan`: as good as the machine allows, never worse
    /// than nothing.
    #[default]
    Auto,
    /// Accept only facts a real language server produced. If none is on `PATH`,
    /// [`PythonProvider::collect`] returns [`ProviderOutput::empty`] — the strict
    /// reading of the [`KnowledgeProvider`] contract ("a provider whose tool is
    /// unavailable must degrade to empty rather than fail the whole collection
    /// run"), for callers who would rather have no facts than heuristic ones.
    LanguageServerOnly,
    /// Never spawn a language server. Fully hermetic, fully offline, fast — and
    /// the mode most tests below use, so their results do not depend on what
    /// happens to be installed on the machine running them.
    SourceScanOnly,
}

/// Facts about a Python project, for KOPITIAM's Semantic Runtime.
///
/// See the [module docs](self) for the ontology mapping and the provenance
/// rules. In the common case you want [`PythonProvider::new`].
///
/// ```no_run
/// use kopitiam_semantic::KnowledgeProvider;
/// use kopitiam_semantic::providers::python::PythonProvider;
///
/// # fn main() -> anyhow::Result<()> {
/// let facts = PythonProvider::new().collect(std::path::Path::new("."))?;
/// println!("{} entities", facts.entities.len());
/// # Ok(())
/// # }
/// ```
pub struct PythonProvider {
    /// The `PATH` to search for a language server, or `None` to use the process
    /// environment's. Injectable purely so tests can prove the
    /// no-server-installed path without mutating process env.
    path: Option<OsString>,
    symbol_source: SymbolSource,
    lsp_timeout: Duration,
}

impl PythonProvider {
    pub fn new() -> Self {
        Self {
            path: None,
            symbol_source: SymbolSource::Auto,
            // Fifteen seconds is not an arbitrary number, and it is not
            // generosity. `LspClient::spawn` waits for a `$/progress` "end" event
            // whose token name contains "index" — which is rust-analyzer's token,
            // and which neither pyright nor pylsp ever sends. So this wait always
            // runs to its full length, and the value is a flat cost paid on every
            // collect. Fifteen seconds is enough for either server to finish
            // `initialize` and get its analysis under way on a project of any
            // realistic size, without stalling a scan for minutes. The proper fix
            // belongs in `LspClient` (return as soon as the server is idle, not
            // only on a rust-analyzer-shaped token) and is noted for the
            // maintainer.
            lsp_timeout: Duration::from_secs(15),
        }
    }

    /// Restricts symbol facts to those a real language server produced. See
    /// [`SymbolSource::LanguageServerOnly`].
    pub fn language_server_only() -> Self {
        Self {
            symbol_source: SymbolSource::LanguageServerOnly,
            ..Self::new()
        }
    }

    /// Never spawns a language server. See [`SymbolSource::SourceScanOnly`].
    pub fn source_scan_only() -> Self {
        Self {
            symbol_source: SymbolSource::SourceScanOnly,
            ..Self::new()
        }
    }

    /// Overrides the `PATH` used to detect a language server. A test seam: see
    /// the field docs.
    pub fn with_path(mut self, path: impl Into<OsString>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn with_symbol_source(mut self, symbol_source: SymbolSource) -> Self {
        self.symbol_source = symbol_source;
        self
    }

    pub fn with_lsp_timeout(mut self, timeout: Duration) -> Self {
        self.lsp_timeout = timeout;
        self
    }

    /// The first server from [`SERVERS`] found on this provider's `PATH`, or
    /// `None` if neither is installed — which is not an error (see
    /// [`KnowledgeProvider`]'s contract), just a less capable run.
    pub fn detect_server(&self) -> Option<DetectedServer> {
        let path = match &self.path {
            Some(path) => path.clone(),
            None => std::env::var_os("PATH")?,
        };
        SERVERS
            .iter()
            .find_map(|&server| which_in(server.program(), &path).map(|program| DetectedServer { server, program }))
    }

    /// Drives the detected language server and returns its symbols, keyed by the
    /// file they live in.
    ///
    /// # Why `workspace/symbol` and not `textDocument/documentSymbol`
    ///
    /// `documentSymbol` is the better request: it returns a *hierarchical*
    /// `DocumentSymbol` tree, so methods arrive already nested inside their class,
    /// with the nesting computed by the server's own parse instead of
    /// reconstructed by us. [`lsp_symbols_to_tree`] parses exactly that shape and
    /// is tested against it, because it is where this provider is going. It cannot
    /// be *issued* today: [`LspClient`] exposes `workspace/symbol` and nothing
    /// else, and that file is shared with three other language adapters being
    /// written concurrently, so growing it is the maintainer's call rather than
    /// this file's. `workspace/symbol` returns the flat `SymbolInformation` shape
    /// with a `containerName`, from which [`lsp_symbols_to_tree`] rebuilds the
    /// nesting — a fair approximation, and the same code path either way.
    ///
    /// The same constraint is why pyright is *detected* but not yet *spawnable*:
    /// `LspClient::spawn` passes no argv and `pyright-langserver` will not start
    /// without `--stdio`. Rather than silently pretend otherwise, this returns an
    /// error naming the missing capability, which [`PythonProvider::collect`]
    /// downgrades to the source scan.
    fn collect_lsp_symbols(
        &self,
        root: &Path,
        project: &PythonProject,
        detected: &DetectedServer,
    ) -> Result<HashMap<PathBuf, Vec<PySymbol>>> {
        if !detected.server.args().is_empty() {
            anyhow::bail!(
                "`{}` needs argv {:?} to speak LSP over stdio, and `LspClient::spawn` passes none \
                 (it needs a `spawn_with_args`)",
                detected.server.program(),
                detected.server.args(),
            );
        }

        let program = detected.program.to_string_lossy().into_owned();
        let mut client = LspClient::spawn(&program, root, self.lsp_timeout)?;
        let raw = client.workspace_symbols("");
        let encoding = client.position_encoding();
        let _ = client.shutdown();

        // Group by the file each symbol claims to be in: the tree builder resolves
        // `containerName` within a file (a `containerName` of "Solver" means the
        // `Solver` in *this* file), and position conversion needs that file's line
        // text to hand.
        let mut by_file: BTreeMap<PathBuf, Vec<Value>> = BTreeMap::new();
        for symbol in raw? {
            let uri = symbol
                .pointer("/location/uri")
                .or_else(|| symbol.get("uri"))
                .and_then(Value::as_str);
            let Some(path) = uri.and_then(uri_to_path) else {
                continue;
            };
            by_file.entry(path).or_default().push(symbol);
        }

        let sources: HashMap<&Path, &str> = project
            .modules
            .iter()
            .map(|module| (module.path.as_path(), module.source.as_str()))
            .collect();

        Ok(by_file
            .into_iter()
            .filter_map(|(path, symbols)| {
                // A symbol in a file we did not discover — a typeshed stub, or a
                // dependency inside a virtualenv the server indexed anyway — is not
                // part of this project. Dropping it is what keeps the graph honest.
                let source = sources.get(path.as_path())?;
                let lines: Vec<&str> = source.lines().collect();
                Some((path, lsp_symbols_to_tree(&symbols, encoding, &lines)))
            })
            .collect())
    }
}

impl Default for PythonProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeProvider for PythonProvider {
    fn name(&self) -> &str {
        SOURCE
    }

    fn collect(&self, root: &Path) -> Result<ProviderOutput> {
        let project = PythonProject::discover(root);
        if project.is_empty() {
            // Not a Python project at all. Emitting a bare "project" artifact for
            // every directory KOPITIAM ever scans would be noise, not knowledge.
            return Ok(ProviderOutput::empty());
        }

        let server = match self.symbol_source {
            SymbolSource::SourceScanOnly => None,
            SymbolSource::Auto | SymbolSource::LanguageServerOnly => self.detect_server(),
        };

        if server.is_none() && self.symbol_source == SymbolSource::LanguageServerOnly {
            tracing::warn!(
                "no Python language server (pyright-langserver, pylsp) on PATH; skipping (no facts collected)"
            );
            return Ok(ProviderOutput::empty());
        }

        // A failure from the server is logged and downgraded, never propagated: an
        // unavailable *or misbehaving* tool must not fail the whole collection run.
        let lsp_symbols = server.as_ref().and_then(|detected| {
            match self.collect_lsp_symbols(root, &project, detected) {
                Ok(symbols) => Some(symbols),
                Err(error) => {
                    tracing::warn!(
                        server = detected.server.as_str(),
                        %error,
                        "Python language server did not answer; falling back to the source scan"
                    );
                    None
                }
            }
        });

        let derivation = match (&lsp_symbols, self.symbol_source) {
            (Some(_), _) => Derivation::LanguageServer(server.as_ref().map(|detected| detected.server)),
            // The server was found but could not answer, and the caller asked for
            // language-server facts only: emit the structure, and no symbols.
            (None, SymbolSource::LanguageServerOnly) => Derivation::None,
            (None, _) => Derivation::SourceScan,
        };

        let symbols: HashMap<PathBuf, Vec<PySymbol>> = match derivation {
            Derivation::LanguageServer(_) => lsp_symbols.unwrap_or_default(),
            Derivation::SourceScan => project
                .modules
                .iter()
                .map(|module| (module.path.clone(), scan_symbols(&module.source)))
                .collect(),
            Derivation::None => HashMap::new(),
        };

        Ok(Emitter::new(derivation).emit(&project, &symbols))
    }
}

/// Which derivation produced a run's symbols. Recorded on every symbol entity —
/// see the module docs on provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Derivation {
    LanguageServer(Option<PythonServer>),
    SourceScan,
    /// Structure only: a server was required, was found, and then failed.
    None,
}

impl Derivation {
    fn as_str(self) -> &'static str {
        match self {
            Self::LanguageServer(_) => "language-server",
            Self::SourceScan => "source-scan",
            Self::None => "none",
        }
    }

    fn server(self) -> Option<&'static str> {
        match self {
            Self::LanguageServer(server) => server.map(PythonServer::as_str),
            _ => None,
        }
    }
}

fn uri_to_path(uri: &str) -> Option<PathBuf> {
    url::Url::parse(uri).ok()?.to_file_path().ok()
}

// ---------------------------------------------------------------------------
// Project discovery
// ---------------------------------------------------------------------------

/// One `.py` file, under the dotted module path Python itself would import it by.
#[derive(Debug, Clone)]
struct PyModule {
    /// `"solver.mesh.grid"` — the import path, not the file path. This is the name
    /// the ontology sees, because it is the name an `import` statement refers to,
    /// and the import graph is the point.
    dotted: String,
    path: PathBuf,
    /// True if this file is a package's `__init__.py`, in which case `dotted` names
    /// the package rather than a module inside it.
    is_package: bool,
    source: String,
}

/// The Python project rooted at some directory: its shape, and every module in it.
#[derive(Debug, Default)]
struct PythonProject {
    root: PathBuf,
    /// Which of [`PROJECT_MARKERS`] were found, in the order listed there.
    markers: Vec<String>,
    /// `"src-layout"` (packages under `src/`) or `"flat"`. Recorded because it is
    /// what tells a translation workflow where generated code belongs.
    layout: &'static str,
    /// Sorted by path, so two runs over an unchanged tree produce identical output.
    /// A hard requirement of an index that is meant to be *rebuildable* rather than
    /// synchronised (CLAUDE.md, Semantic Runtime).
    modules: Vec<PyModule>,
}

impl PythonProject {
    fn discover(root: &Path) -> Self {
        let markers: Vec<String> = PROJECT_MARKERS
            .iter()
            .filter(|marker| root.join(marker).is_file())
            .map(|marker| (*marker).to_string())
            .collect();

        let mut files = Vec::new();
        collect_python_files(root, &mut files);
        files.sort();

        let modules: Vec<PyModule> = files
            .iter()
            .filter_map(|path| {
                let dotted = module_path(path, root)?;
                let source = std::fs::read_to_string(path).ok()?;
                Some(PyModule {
                    dotted,
                    is_package: path.file_name() == Some(OsStr::new("__init__.py")),
                    path: path.clone(),
                    source,
                })
            })
            .collect();

        let src = root.join("src");
        let layout = if src.is_dir() && modules.iter().any(|module| module.path.starts_with(&src)) {
            "src-layout"
        } else {
            "flat"
        };

        Self {
            root: root.to_path_buf(),
            markers,
            layout,
            modules,
        }
    }

    /// A directory with no Python in it and no Python project marker is not a
    /// Python project, and this provider has nothing to say about it.
    fn is_empty(&self) -> bool {
        self.modules.is_empty() && self.markers.is_empty()
    }

    /// The dotted names this project defines — how an internal import is told
    /// apart from an external one.
    fn known_modules(&self) -> BTreeSet<&str> {
        self.modules.iter().map(|module| module.dotted.as_str()).collect()
    }
}

/// Recursively collects `.py` files, skipping [`SKIPPED_DIRS`]. Symlinks are not
/// followed: a symlinked directory is the classic way to walk a tree forever, and
/// a symlinked `.py` is a duplicate of a file we will visit anyway.
fn collect_python_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else { continue };
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if SKIPPED_DIRS.contains(&name.as_ref()) || name.ends_with(".egg-info") {
                continue;
            }
            collect_python_files(&path, out);
        } else if path.extension() == Some(OsStr::new("py")) {
            out.push(path);
        }
    }
}

/// The dotted module path `file` would be imported under, given the project `root`.
///
/// This is real Python semantics, not a path-to-dots substitution: a directory is
/// part of the import path only while it is a *package*, i.e. contains an
/// `__init__.py`. So `root/src/solver/mesh/grid.py` is `solver.mesh.grid` — not
/// `src.solver.mesh.grid` — because `src/` has no `__init__.py` and is therefore a
/// source root, not a package. Getting this wrong would break every import edge in
/// the graph, since the names in an `import` statement are exactly these dotted
/// paths and nothing else.
fn module_path(file: &Path, root: &Path) -> Option<String> {
    let stem = file.file_stem()?.to_str()?;
    let mut packages: Vec<String> = Vec::new();
    let mut dir = file.parent()?;
    loop {
        if !dir.starts_with(root) || !dir.join("__init__.py").is_file() {
            break;
        }
        packages.push(dir.file_name()?.to_str()?.to_string());
        if dir == root {
            break;
        }
        dir = dir.parent()?;
    }
    packages.reverse();

    if stem == "__init__" {
        if packages.is_empty() {
            // `root/__init__.py`: the root directory is itself the package.
            return root.file_name()?.to_str().map(str::to_string);
        }
        return Some(packages.join("."));
    }
    packages.push(stem.to_string());
    Some(packages.join("."))
}

// ---------------------------------------------------------------------------
// The lexical layer shared by the import scan and the symbol scan
// ---------------------------------------------------------------------------

/// Yields each line of `source` with string bodies and comments removed, paired
/// with its 0-indexed line number and its raw text.
///
/// This exists so that the import scan does not attribute a dependency to a module
/// because the words `import numpy` appeared inside a docstring — a real hazard in
/// scientific Python, where module docstrings routinely contain worked examples.
/// It is a state machine over quote delimiters, not a tokenizer; see the module
/// docs for exactly where that breaks down.
struct CodeLines<'a> {
    lines: std::iter::Enumerate<std::str::Lines<'a>>,
    /// The triple-quote delimiter we are currently inside, if any.
    inside: Option<&'static str>,
}

impl<'a> CodeLines<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            lines: source.lines().enumerate(),
            inside: None,
        }
    }
}

impl<'a> Iterator for CodeLines<'a> {
    /// `(line number, code, raw line)`. `code` has string bodies and comments
    /// stripped; `raw` is the untouched line, which the symbol scanner needs in
    /// order to compute a *character* column into the real text.
    type Item = (u32, String, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (index, raw) = self.lines.next()?;
            let mut rest = raw;
            let mut code = String::new();

            // Close an open docstring first: everything up to the delimiter is
            // string body, everything after it is code again.
            if let Some(delimiter) = self.inside {
                match rest.find(delimiter) {
                    Some(at) => {
                        self.inside = None;
                        rest = &rest[at + delimiter.len()..];
                    }
                    None => continue,
                }
            }

            // Then walk what is left, so that a `#` inside a string is not mistaken
            // for a comment and a `"""` in code opens a docstring.
            loop {
                let triple = ["\"\"\"", "'''"]
                    .into_iter()
                    .filter_map(|delimiter| rest.find(delimiter).map(|at| (at, delimiter)))
                    .min_by_key(|(at, _)| *at);
                let hash = rest.find('#');
                match (triple, hash) {
                    (Some((at, delimiter)), hash) if hash.is_none_or(|h| at < h) => {
                        code.push_str(&rest[..at]);
                        rest = &rest[at + delimiter.len()..];
                        match rest.find(delimiter) {
                            // Opened and closed on one line: an inline string. Skip
                            // its body and keep reading code after it.
                            Some(end) => rest = &rest[end + delimiter.len()..],
                            None => {
                                self.inside = Some(delimiter);
                                break;
                            }
                        }
                    }
                    (_, Some(at)) => {
                        code.push_str(&rest[..at]);
                        break;
                    }
                    (_, None) => {
                        code.push_str(rest);
                        break;
                    }
                }
            }

            return Some((index as u32, code, raw));
        }
    }
}

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

/// One resolved import target: an absolute dotted module name, and the line the
/// `import` statement sits on.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct PyImport {
    module: String,
    line: u32,
}

/// Every module `source` imports, with relative imports resolved against `current`
/// (the dotted name of the module doing the importing).
///
/// `known` is the set of modules the project defines. It is needed because `from .
/// import grid` is ambiguous in the source alone: it may pull the *name* `grid` out
/// of the package's `__init__`, or it may import the *submodule* `pkg.grid`. If
/// `pkg.grid` exists, both readings create a real dependency on it and the edge is
/// emitted; if it does not, only the edge to the package itself is real.
fn scan_imports(source: &str, current: &str, is_package: bool, known: &BTreeSet<&str>) -> Vec<PyImport> {
    // The package a relative import is relative to: for `pkg/__init__.py` that is
    // `pkg` itself; for the module `pkg.mod` it is `pkg`.
    let package: Vec<&str> = if is_package {
        current.split('.').collect()
    } else {
        let mut parts: Vec<&str> = current.split('.').collect();
        parts.pop();
        parts
    };

    let mut imports = BTreeSet::new();
    for (line, code, _) in CodeLines::new(source) {
        let code = code.trim();
        if let Some(rest) = code.strip_prefix("import ") {
            for clause in rest.split(',') {
                if let Some(module) = import_clause_target(clause) {
                    imports.insert(PyImport { module, line });
                }
            }
        } else if let Some(rest) = code.strip_prefix("from ") {
            let Some((head, tail)) = rest.split_once(" import ") else {
                continue;
            };
            let Some(base) = resolve_relative(head.trim(), &package) else {
                continue;
            };
            for name in imported_names(tail) {
                let submodule = format!("{base}.{name}");
                if known.contains(submodule.as_str()) {
                    imports.insert(PyImport {
                        module: submodule,
                        line,
                    });
                }
            }
            imports.insert(PyImport { module: base, line });
        }
    }
    imports.into_iter().collect()
}

/// `"numpy.linalg as la"` -> `"numpy.linalg"`.
fn import_clause_target(clause: &str) -> Option<String> {
    let target = clause.split_whitespace().next()?;
    if target.is_empty() || !target.chars().all(is_dotted_identifier_char) {
        return None;
    }
    Some(target.to_string())
}

/// The names bound by the tail of `from X import a, b as c, (d, e)` — used only to
/// spot submodule imports.
fn imported_names(tail: &str) -> Vec<&str> {
    tail.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim_end_matches('\\')
        .split(',')
        .filter_map(|part| part.split_whitespace().next())
        .filter(|name| !name.is_empty() && *name != "*")
        .collect()
}

/// Turns the module part of a `from ... import` into an absolute dotted name.
///
/// Python's rule: `n` leading dots means "go up `n - 1` levels from the current
/// package". `from . import x` inside `pkg.sub` is `pkg.sub`; `from ..other import
/// x` is `pkg.other`. Returns `None` if the import climbs above the project root —
/// which means either the source is broken or the file is being read outside its
/// real package context, and in both cases inventing an edge is worse than emitting
/// none.
fn resolve_relative(head: &str, package: &[&str]) -> Option<String> {
    let dots = head.chars().take_while(|c| *c == '.').count();
    let rest = head[dots..].trim();
    if dots == 0 {
        return if !rest.is_empty() && rest.chars().all(is_dotted_identifier_char) {
            Some(rest.to_string())
        } else {
            None
        };
    }
    let keep = package.len().checked_sub(dots - 1)?;
    let mut parts: Vec<&str> = package[..keep].to_vec();
    if !rest.is_empty() {
        parts.extend(rest.split('.'));
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("."))
}

fn is_dotted_identifier_char(c: char) -> bool {
    c == '.' || c == '_' || c.is_alphanumeric()
}

// ---------------------------------------------------------------------------
// Symbols
// ---------------------------------------------------------------------------

/// What a symbol *is*, in the vocabulary `metadata.symbol_kind` carries.
///
/// Deliberately small: the intersection of what every language adapter can
/// meaningfully say, not the union of Python's grammar. A cross-language knowledge
/// graph is only useful to the extent its vocabulary means the same thing in every
/// language that feeds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PySymbolKind {
    Class,
    /// A `def` at module level, or nested inside another `def`.
    Function,
    /// A `def` whose immediate parent is a class.
    Method,
    /// A module-level `UPPER_SNAKE = ...`.
    Constant,
    /// Anything else a language server reports as a value binding.
    Variable,
}

impl PySymbolKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Class => "class",
            Self::Function => "function",
            Self::Method => "method",
            Self::Constant => "constant",
            Self::Variable => "variable",
        }
    }
}

/// A symbol and the symbols nested inside it.
///
/// The single internal model that *both* derivations produce and the emitter
/// consumes, so the Python -> ontology mapping is written exactly once. A pyright
/// run and an offline scan are then guaranteed to agree on the shape of what lands
/// in the graph, differing only in how much they knew.
#[derive(Debug, Clone, PartialEq)]
struct PySymbol {
    name: String,
    kind: PySymbolKind,
    /// 0-indexed line of the declaration.
    line: u32,
    /// 0-indexed **character** (Unicode scalar value) column of the *name* — this
    /// crate's public position unit, per [`crate::position`]. Never a byte offset
    /// and never a UTF-16 code unit, whatever the wire happened to say.
    character: u32,
    /// 0-indexed last line of the declaration's body.
    end_line: u32,
    is_async: bool,
    decorators: Vec<String>,
    /// Base classes as written (`["Grid", "abc.ABC"]`). Empty for non-classes.
    bases: Vec<String>,
    /// The parameter list as written on the declaration line, if any.
    signature: Option<String>,
    /// The raw LSP `SymbolKind` integer, when a language server produced this.
    lsp_kind: Option<u64>,
    children: Vec<PySymbol>,
}

/// A declaration and the indent it was found at — the intermediate form the
/// scanner's passes agree on.
struct Decl {
    indent: usize,
    symbol: PySymbol,
}

/// Recovers the symbol tree from Python source, without a language server.
///
/// Indentation *is* Python's block structure, which is what makes this honest
/// rather than a regex hack: a `def` at column 4 immediately under a `class` at
/// column 0 is a method, by the language's own grammar. The scanner keeps a stack
/// of open declarations by indent, so nesting to any depth — a closure in a method
/// in a class in a factory function — comes out right.
fn scan_symbols(source: &str) -> Vec<PySymbol> {
    let raw_lines: Vec<&str> = source.lines().collect();
    let mut decls: Vec<Decl> = Vec::new();
    let mut decorators: Vec<String> = Vec::new();

    // Pass 1: every declaration, with its indent, in source order.
    for (line, code, raw) in CodeLines::new(source) {
        let trimmed = code.trim();
        if trimmed.is_empty() {
            continue;
        }
        let indent = code.len() - code.trim_start().len();

        if let Some(decorator) = trimmed.strip_prefix('@') {
            let name = decorator.split(['(', ' ']).next().unwrap_or("").trim();
            if !name.is_empty() {
                decorators.push(name.to_string());
            }
            continue;
        }

        let (is_async, header) = match trimmed.strip_prefix("async ") {
            Some(rest) => (true, rest.trim_start()),
            None => (false, trimmed),
        };

        let symbol = if let Some(rest) = header.strip_prefix("def ") {
            let rest = rest.trim_start();
            let name = identifier_prefix(rest);
            if name.is_empty() {
                continue;
            }
            let signature = rest
                .find('(')
                .map(|at| rest[at..].trim().trim_end_matches(':').trim().to_string());
            PySymbol {
                character: name_column(raw, &name),
                name,
                // Fixed up in pass 3, once the parent is known: a `def` under a class
                // is a method, a `def` under anything else is a function.
                kind: PySymbolKind::Function,
                line,
                end_line: line,
                is_async,
                decorators: std::mem::take(&mut decorators),
                bases: Vec::new(),
                signature,
                lsp_kind: None,
                children: Vec::new(),
            }
        } else if let Some(rest) = header.strip_prefix("class ") {
            let rest = rest.trim_start();
            let name = identifier_prefix(rest);
            if name.is_empty() {
                continue;
            }
            PySymbol {
                character: name_column(raw, &name),
                name,
                kind: PySymbolKind::Class,
                line,
                end_line: line,
                is_async: false,
                decorators: std::mem::take(&mut decorators),
                bases: parse_bases(rest),
                signature: None,
                lsp_kind: None,
                children: Vec::new(),
            }
        } else if indent == 0 {
            let Some(name) = module_level_constant(trimmed) else {
                decorators.clear();
                continue;
            };
            PySymbol {
                character: name_column(raw, &name),
                name,
                kind: PySymbolKind::Constant,
                line,
                end_line: line,
                is_async: false,
                decorators: std::mem::take(&mut decorators),
                bases: Vec::new(),
                signature: None,
                lsp_kind: None,
                children: Vec::new(),
            }
        } else {
            decorators.clear();
            continue;
        };

        decls.push(Decl { indent, symbol });
    }

    // Pass 2: a declaration's body runs until the next line — blanks aside — that
    // carries code at or before its own indent.
    for decl in decls.iter_mut() {
        let mut end = decl.symbol.line as usize;
        for (index, raw) in raw_lines.iter().enumerate().skip(decl.symbol.line as usize + 1) {
            if raw.trim().is_empty() {
                continue;
            }
            let indent = raw.len() - raw.trim_start().len();
            if indent <= decl.indent {
                break;
            }
            end = index;
        }
        decl.symbol.end_line = end as u32;
    }

    // Pass 3: assemble the tree by indent.
    let mut roots: Vec<PySymbol> = Vec::new();
    // The currently open ancestors, innermost last: the indent each was declared
    // at, and the path of child-indices that reaches it from `roots`.
    let mut open: Vec<(usize, Vec<usize>)> = Vec::new();

    for Decl { indent, mut symbol } in decls {
        while open.last().is_some_and(|(open_indent, _)| *open_indent >= indent) {
            open.pop();
        }
        let parent_is_class = open
            .last()
            .is_some_and(|(_, path)| symbol_at(&roots, path).kind == PySymbolKind::Class);
        if parent_is_class && symbol.kind == PySymbolKind::Function {
            symbol.kind = PySymbolKind::Method;
        }

        let mut path = open.last().map(|(_, path)| path.clone()).unwrap_or_default();
        let siblings = match open.last() {
            Some((_, parent)) => &mut symbol_at_mut(&mut roots, parent).children,
            None => &mut roots,
        };
        path.push(siblings.len());
        siblings.push(symbol);
        open.push((indent, path));
    }

    roots
}

fn symbol_at<'a>(roots: &'a [PySymbol], path: &[usize]) -> &'a PySymbol {
    let (first, rest) = path.split_first().expect("a tree path is never empty");
    let mut symbol = &roots[*first];
    for index in rest {
        symbol = &symbol.children[*index];
    }
    symbol
}

fn symbol_at_mut<'a>(roots: &'a mut [PySymbol], path: &[usize]) -> &'a mut PySymbol {
    let (first, rest) = path.split_first().expect("a tree path is never empty");
    let mut symbol = &mut roots[*first];
    for index in rest {
        symbol = &mut symbol.children[*index];
    }
    symbol
}

/// The leading Python identifier of `s` (`"solve(x):"` -> `"solve"`).
///
/// Identifiers may be non-ASCII: `def résumé` and `def 温度` are both legal Python 3
/// (PEP 3131), and in this project's domain a `λ` or a `Δt` is not a curiosity.
fn identifier_prefix(s: &str) -> String {
    s.chars().take_while(|c| *c == '_' || c.is_alphanumeric()).collect()
}

/// The **character** column at which `name` begins in `raw`.
///
/// `str::find` returns a *byte* offset; the ontology promises a `char` offset (see
/// [`PySymbol::character`]). Those differ on any line carrying a Greek variable, a
/// CJK identifier, or an em-dash in a preceding string — all of which occur freely
/// in scientific Python. Converting here, once, is what spares every downstream
/// consumer from having to know.
fn name_column(raw: &str, name: &str) -> u32 {
    raw.find(name)
        .map(|byte| raw[..byte].chars().count() as u32)
        .unwrap_or(0)
}

/// The base classes on a `class Foo(Bar, metaclass=ABCMeta):` header. Keyword
/// arguments (`metaclass=`, `total=`) are bindings, not bases.
fn parse_bases(rest: &str) -> Vec<String> {
    let Some(open) = rest.find('(') else {
        return Vec::new();
    };
    let tail = &rest[open + 1..];
    let close = tail.rfind(')').unwrap_or(tail.len());
    tail[..close]
        .split(',')
        .map(str::trim)
        .filter(|base| !base.is_empty() && !base.contains('='))
        .map(str::to_string)
        .collect()
}

/// `"MAX_ITERATIONS = 100"` -> `Some("MAX_ITERATIONS")`; `"solver = Solver()"` ->
/// `None`.
///
/// Only screaming-snake bindings are taken to be constants. Every other
/// module-level binding is an alias, an instance, or a side effect, and admitting
/// them all would bury the real symbols — a knowledge graph's value is its signal
/// density, not its row count.
fn module_level_constant(code: &str) -> Option<String> {
    let (lhs, _) = code.split_once('=')?;
    // Exclude `==`, `!=`, `+=` and friends. A type-annotated `X: int = 1` keeps its
    // name to the left of the colon.
    let lhs = lhs.trim().trim_end_matches(['!', '<', '>', '+', '-', '*', '/', '%', '|', '&', '^']);
    let name = lhs.split(':').next()?.trim();
    if name.is_empty() || name.contains(|c: char| c.is_whitespace() || c == '.' || c == '[') {
        return None;
    }
    if !name.chars().all(|c| c.is_uppercase() || c.is_numeric() || c == '_') {
        return None;
    }
    Some(name.to_string())
}

// ---------------------------------------------------------------------------
// LSP symbols
// ---------------------------------------------------------------------------

/// Builds the same [`PySymbol`] tree from whatever an LSP server sent back.
///
/// Handles both response shapes the protocol allows, because which one arrives
/// depends on the request *and* the server:
///
/// * `DocumentSymbol` — hierarchical, with `children`, a `range` (the whole
///   declaration) and a `selectionRange` (the name). This is what
///   `textDocument/documentSymbol` returns, and the one we want: the nesting is the
///   server's, from a real parse.
/// * `SymbolInformation` — flat, with a `location` and an optional `containerName`.
///   This is what `workspace/symbol` returns. The nesting is rebuilt by matching
///   `containerName` against symbols already seen in the same file, which is right
///   in every case except a file holding two same-named containers.
///
/// `lines` is the file's text, needed because `character` arrives in the server's
/// negotiated [`PositionEncoding`] and must be converted to this crate's
/// `char`-offset contract. Skipping that conversion is not a theoretical bug: under
/// `"utf-8"` — which LSP 3.17 defines as *byte* offsets — every symbol appearing
/// after a Greek letter or a CJK string literal on its line would be silently
/// placed at the wrong column.
fn lsp_symbols_to_tree(items: &[Value], encoding: PositionEncoding, lines: &[&str]) -> Vec<PySymbol> {
    let hierarchical = items.iter().any(|item| item.get("location").is_none());
    if hierarchical {
        return items
            .iter()
            .filter_map(|item| document_symbol(item, encoding, lines, false))
            .collect();
    }

    // Flat `SymbolInformation`: sort by position so a container is always seen
    // before the things it contains, then attach by `containerName`.
    let mut flat: Vec<&Value> = items.iter().collect();
    flat.sort_by_key(|item| {
        item.pointer("/location/range/start/line")
            .and_then(Value::as_u64)
            .unwrap_or(0)
    });

    let mut roots: Vec<PySymbol> = Vec::new();
    // Container name -> the path into `roots` that reaches it.
    let mut containers: HashMap<String, Vec<usize>> = HashMap::new();

    for item in flat {
        let parent_path = item
            .get("containerName")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .and_then(|name| containers.get(name))
            .cloned();
        let parent_is_class = parent_path
            .as_deref()
            .is_some_and(|path| symbol_at(&roots, path).kind == PySymbolKind::Class);
        let Some(symbol) = document_symbol(item, encoding, lines, parent_is_class) else {
            continue;
        };
        let name = symbol.name.clone();

        let mut path = parent_path.clone().unwrap_or_default();
        let siblings = match parent_path.as_deref() {
            Some(parent) => &mut symbol_at_mut(&mut roots, parent).children,
            None => &mut roots,
        };
        path.push(siblings.len());
        siblings.push(symbol);
        containers.insert(name, path);
    }

    roots
}

/// One `DocumentSymbol` or `SymbolInformation` entry, and everything under it.
///
/// `parent_is_class` is how a `SymbolKind.Function` becomes a `method`: LSP's
/// `SymbolKind` does distinguish the two, but servers are inconsistent about it
/// (pylsp reports class-level `def`s as `Function`), and containment we can see is
/// more reliable than a label the server may have got wrong. The raw integer is
/// preserved in `lsp_kind` either way, so nothing is lost.
fn document_symbol(item: &Value, encoding: PositionEncoding, lines: &[&str], parent_is_class: bool) -> Option<PySymbol> {
    let name = item.get("name").and_then(Value::as_str)?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let lsp_kind = item.get("kind").and_then(Value::as_u64);

    // `selectionRange` (the name) if the server sent one, else the declaration's
    // `range`, else `SymbolInformation`'s `location`.
    let start = item
        .pointer("/selectionRange/start")
        .or_else(|| item.pointer("/range/start"))
        .or_else(|| item.pointer("/location/range/start"))?;
    let line = start.get("line").and_then(Value::as_u64).unwrap_or(0) as u32;
    let wire_character = start.get("character").and_then(Value::as_u64).unwrap_or(0) as u32;
    let end_line = item
        .pointer("/range/end/line")
        .or_else(|| item.pointer("/location/range/end/line"))
        .and_then(Value::as_u64)
        .unwrap_or(u64::from(line)) as u32;

    let line_text = lines.get(line as usize).copied().unwrap_or("");
    let character = position::unit_to_char_col(line_text, wire_character, encoding);

    let mut kind = match lsp_kind {
        // Class(5), Enum(10), Interface(11), Struct(23).
        Some(5 | 10 | 11 | 23) => PySymbolKind::Class,
        // Method(6), Property(7), Constructor(9).
        Some(6 | 7 | 9) => PySymbolKind::Method,
        Some(12) => PySymbolKind::Function,
        Some(14) => PySymbolKind::Constant,
        _ => PySymbolKind::Variable,
    };
    if kind == PySymbolKind::Function && parent_is_class {
        kind = PySymbolKind::Method;
    } else if kind == PySymbolKind::Method && !parent_is_class {
        kind = PySymbolKind::Function;
    }

    // The server says where the symbol is; the source says what LSP's data model
    // has no field for. `async`, decorators and base classes are all right there on
    // the declaration line, and reading them costs nothing — we already hold the
    // file. In scientific Python a decorator is frequently the most important thing
    // about a function: it is what makes it compiled (`@numba.njit`), cached, or a
    // test.
    let is_class = kind == PySymbolKind::Class;
    let declaration = lines.get(line as usize).copied().unwrap_or("");
    let is_async = declaration.trim_start().starts_with("async ");
    let decorators = decorators_above(lines, line);
    let bases = if is_class { parse_bases(declaration) } else { Vec::new() };
    let signature = if is_class {
        None
    } else {
        declaration
            .find('(')
            .map(|at| declaration[at..].trim().trim_end_matches(':').trim().to_string())
    };

    let children = item
        .get("children")
        .and_then(Value::as_array)
        .map(|children| {
            children
                .iter()
                .filter_map(|child| document_symbol(child, encoding, lines, is_class))
                .collect()
        })
        .unwrap_or_default();

    Some(PySymbol {
        name,
        kind,
        line,
        character,
        end_line,
        is_async,
        decorators,
        bases,
        signature,
        lsp_kind,
        children,
    })
}

/// The decorators immediately above `line`, outermost first.
fn decorators_above(lines: &[&str], line: u32) -> Vec<String> {
    let mut decorators = Vec::new();
    let mut index = line as usize;
    while index > 0 {
        index -= 1;
        let trimmed = lines[index].trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some(decorator) = trimmed.strip_prefix('@') else {
            break;
        };
        let name = decorator.split(['(', ' ']).next().unwrap_or("").trim();
        if !name.is_empty() {
            decorators.push(name.to_string());
        }
    }
    decorators.reverse();
    decorators
}

// ---------------------------------------------------------------------------
// Emission into the ontology
// ---------------------------------------------------------------------------

/// Class inheritance, derived -> base. The one relationship with no first-class
/// ontology variant; see the module docs. The string and the direction match the
/// C++ and Visual Basic adapters exactly — a shared graph is only shared if
/// everyone spells the edge the same way.
const INHERITS: &str = "inherits";

/// Turns a discovered project and its symbols into ontology facts.
///
/// Everything the graph will ever learn about Python is created here, in one place,
/// from one internal model — so "what does a Python fact look like" has a single
/// answer, readable in one sitting and comparable against the other language
/// adapters.
struct Emitter {
    derivation: Derivation,
    entities: Vec<Entity>,
    relationships: Vec<Relationship>,
    /// Dotted module name -> its artifact, for resolving import targets.
    module_ids: HashMap<String, EntityId>,
    /// External (not-in-this-project) modules and base classes, deduplicated:
    /// `numpy` imported by nine modules is one node with nine edges, not nine nodes.
    external_modules: HashMap<String, EntityId>,
    external_symbols: HashMap<String, EntityId>,
    /// Simple class name -> its symbol entity, for resolving base classes.
    class_ids: HashMap<String, EntityId>,
    /// `(class, base as written)`, resolved only once every class is known — a class
    /// may inherit from one declared later in the file, or in a module not yet
    /// walked.
    pending_bases: Vec<(EntityId, String)>,
}

impl Emitter {
    fn new(derivation: Derivation) -> Self {
        Self {
            derivation,
            entities: Vec::new(),
            relationships: Vec::new(),
            module_ids: HashMap::new(),
            external_modules: HashMap::new(),
            external_symbols: HashMap::new(),
            class_ids: HashMap::new(),
            pending_bases: Vec::new(),
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

    fn emit(mut self, project: &PythonProject, symbols: &HashMap<PathBuf, Vec<PySymbol>>) -> ProviderOutput {
        let project_name = project
            .root
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| project.root.display().to_string());

        let project_id = self.push(
            Entity::new(EntityKind::Artifact, project_name, SOURCE).with_metadata(json!({
                "language": "python",
                "artifact_kind": "project",
                "path": project.root.display().to_string(),
                "markers": project.markers,
                "layout": project.layout,
                "modules": project.modules.len(),
            })),
        );

        // Every module artifact first, before any edge between them: an import may
        // point forward, or around a cycle — which Python permits and scientific
        // codebases contain — and an edge to a module we have not created yet would
        // be an edge to nothing.
        for module in &project.modules {
            let id = self.push(
                Entity::new(EntityKind::Artifact, &module.dotted, SOURCE).with_metadata(json!({
                    "language": "python",
                    "artifact_kind": if module.is_package { "package" } else { "module" },
                    "module": module.dotted,
                    "path": module.path.display().to_string(),
                    "uri": path_to_uri(&module.path),
                })),
            );
            self.module_ids.insert(module.dotted.clone(), id);
        }

        for module in &project.modules {
            let module_id = self.module_ids[&module.dotted];

            // The artifact containment tree: `solver.mesh.grid` is located in
            // `solver.mesh`, located in `solver`, located in the project. `LocatedIn`
            // carries this without needing a new variant.
            let parent = module
                .dotted
                .rsplit_once('.')
                .and_then(|(parent, _)| self.module_ids.get(parent).copied())
                .unwrap_or(project_id);
            if parent != module_id {
                self.relate(module_id, parent, RelationshipKind::LocatedIn);
            }

            for symbol in symbols.get(&module.path).map(Vec::as_slice).unwrap_or_default() {
                self.emit_symbol(symbol, module, module_id, None);
            }
        }

        // Imports, now that every module artifact exists.
        let known = project.known_modules();
        for module in &project.modules {
            let module_id = self.module_ids[&module.dotted];
            for import in scan_imports(&module.source, &module.dotted, module.is_package, &known) {
                let target = self.module_or_external(&import.module);
                if target != module_id {
                    self.relate(module_id, target, RelationshipKind::DependsOn);
                }
            }
        }

        // Base classes, now that every class symbol exists.
        for (class_id, base) in std::mem::take(&mut self.pending_bases) {
            let target = match self.class_ids.get(base.as_str()).copied() {
                Some(id) => id,
                None => self.external_symbol(&base),
            };
            if target != class_id {
                self.relate(class_id, target, RelationshipKind::Custom(INHERITS.to_string()));
            }
        }

        ProviderOutput {
            entities: self.entities,
            relationships: self.relationships,
        }
    }

    /// Emits `symbol` and everything nested inside it.
    ///
    /// Note the two edges a nested symbol gets, and why. Every symbol, at any depth,
    /// is `LocatedIn` its **module** — so "which file is this in" is always one hop,
    /// never a walk up a containment chain. A *nested* symbol is additionally
    /// `LocatedIn` its **enclosing symbol**. The two are distinguished by the target's
    /// [`EntityKind`], not by the edge, which is exactly how the C++, C# and Visual
    /// Basic adapters encode the same thing.
    fn emit_symbol(&mut self, symbol: &PySymbol, module: &PyModule, module_id: EntityId, parent: Option<EntityId>) {
        let id = self.push(
            Entity::new(EntityKind::Symbol, &symbol.name, SOURCE).with_metadata(json!({
                "language": "python",
                "symbol_kind": symbol.kind.as_str(),
                "module": module.dotted,
                "path": module.path.display().to_string(),
                "uri": path_to_uri(&module.path),
                "line": symbol.line,
                "character": symbol.character,
                "end_line": symbol.end_line,
                "is_async": symbol.is_async,
                "decorators": symbol.decorators,
                "bases": symbol.bases,
                "signature": symbol.signature,
                "lsp_kind": symbol.lsp_kind,
                "derivation": self.derivation.as_str(),
                "server": self.derivation.server(),
            })),
        );

        self.relate(id, module_id, RelationshipKind::LocatedIn);
        if let Some(parent) = parent {
            self.relate(id, parent, RelationshipKind::LocatedIn);
        }

        if symbol.kind == PySymbolKind::Class {
            // First definition wins on a name collision. Python allows two classes
            // with the same name in different modules, so resolving a base by simple
            // name is a heuristic — one that is *marked* on the edge's target
            // (`resolved: false` when it escapes the project), never silently guessed.
            self.class_ids.entry(symbol.name.clone()).or_insert(id);
            for base in &symbol.bases {
                self.pending_bases.push((id, base.clone()));
            }
        }

        for child in &symbol.children {
            self.emit_symbol(child, module, module_id, Some(id));
        }
    }

    /// The artifact for `module`: the project's own if we have it, otherwise a
    /// deduplicated external one.
    ///
    /// External module nodes are not clutter. `solver.pressure -DependsOn->
    /// numpy.linalg` is precisely the fact that tells a translation workflow it will
    /// need a linear-algebra crate, and tells a literature workflow whose conventions
    /// the code follows.
    fn module_or_external(&mut self, module: &str) -> EntityId {
        if let Some(id) = self.module_ids.get(module) {
            return *id;
        }
        if let Some(id) = self.external_modules.get(module) {
            return *id;
        }
        let distribution = module.split('.').next().unwrap_or(module).to_string();
        let id = self.push(
            Entity::new(EntityKind::Artifact, module, SOURCE).with_metadata(json!({
                "language": "python",
                "artifact_kind": "external_module",
                "module": module,
                "distribution": distribution,
                "external": true,
            })),
        );
        self.external_modules.insert(module.to_string(), id);
        id
    }

    /// A stub for a base class defined outside this project (`torch.nn.Module`,
    /// `abc.ABC`, `object`). `resolved: false` says plainly that we know the name and
    /// nothing else — an honest gap is worth more than a confident guess.
    fn external_symbol(&mut self, name: &str) -> EntityId {
        if let Some(id) = self.external_symbols.get(name) {
            return *id;
        }
        let id = self.push(
            Entity::new(EntityKind::Symbol, name, SOURCE).with_metadata(json!({
                "language": "python",
                "symbol_kind": "class",
                "external": true,
                "resolved": false,
                "derivation": self.derivation.as_str(),
            })),
        );
        self.external_symbols.insert(name.to_string(), id);
        id
    }
}

/// A `file://` URI for `path`, or the path itself if it cannot be expressed as one.
/// Recorded alongside the path because a URI is what LSP — and therefore every
/// editor client — speaks in.
fn path_to_uri(path: &Path) -> String {
    url::Url::from_file_path(path)
        .map(|url| url.to_string())
        .unwrap_or_else(|()| path.display().to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A miniature but realistic scientific Python project: a `src`-layout package,
    /// a subpackage, a class with methods and inheritance, a module-level constant,
    /// an async method, a decorator, a closure, a relative import, an absolute
    /// intra-project import, and third-party imports.
    fn synthetic_project() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        write(root, "pyproject.toml", "[project]\nname = \"solver\"\n");
        write(
            root,
            "src/solver/__init__.py",
            "\"\"\"Solver package.\n\nExample:\n    import numpy as np\n\"\"\"\n\nfrom .grid import Grid\n\nVERSION = \"0.1.0\"\n",
        );
        write(
            root,
            "src/solver/grid.py",
            "import numpy as np\n\nCELL_COUNT = 128\n\n\nclass Grid:\n    \"\"\"A structured grid.\"\"\"\n\n    def spacing(self):\n        return 1.0 / CELL_COUNT\n",
        );
        write(
            root,
            "src/solver/schemes/__init__.py",
            "from solver.grid import Grid\nfrom . import upwind\n",
        );
        write(
            root,
            "src/solver/schemes/upwind.py",
            "from ..grid import Grid\nimport scipy.sparse\n\n\nclass Upwind(Grid, abc.ABC):\n    @staticmethod\n    def flux(left, right):\n        def limiter(r):\n            return max(0.0, min(1.0, r))\n\n        return limiter(left / right)\n\n    async def solve(self, dt: float) -> float:\n        return dt\n\n\ndef make_scheme() -> Upwind:\n    return Upwind()\n",
        );
        // Must be ignored entirely: a virtualenv full of other people's code, and a
        // bytecode cache full of our own, twice.
        write(root, ".venv/lib/site.py", "class NotOurs:\n    pass\n");
        write(root, "src/solver/__pycache__/grid.py", "class Cached:\n    pass\n");
        dir
    }

    fn write(root: &Path, relative: &str, contents: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(&path, contents).expect("write");
    }

    fn entity<'a>(output: &'a ProviderOutput, kind: EntityKind, name: &str) -> Option<&'a Entity> {
        output
            .entities
            .iter()
            .find(|entity| entity.kind == kind && entity.name == name)
    }

    fn meta<'a>(entity: &'a Entity, key: &str) -> &'a Value {
        entity.metadata.get(key).unwrap_or(&Value::Null)
    }

    fn has_edge(output: &ProviderOutput, from: &Entity, to: &Entity, kind: RelationshipKind) -> bool {
        output
            .relationships
            .iter()
            .any(|rel| rel.from == from.id && rel.to == to.id && rel.kind == kind)
    }

    fn inherits() -> RelationshipKind {
        RelationshipKind::Custom(INHERITS.to_string())
    }

    /// The core test: a real project on disk, through the real provider, with no
    /// language server anywhere near it — and the whole ontology mapping comes out
    /// the other side.
    #[test]
    fn maps_a_python_project_onto_the_shared_ontology() {
        let dir = synthetic_project();
        let output = PythonProvider::source_scan_only()
            .collect(dir.path())
            .expect("collection must succeed");

        // Provenance, on every single fact. Non-negotiable (CLAUDE.md, Scientific
        // Standards).
        assert!(
            output.entities.iter().all(|entity| entity.source == "python"),
            "every entity must be attributed to this provider"
        );

        // Artifacts: the project, its packages, its modules.
        let project_name = dir.path().file_name().unwrap().to_str().unwrap();
        let project = entity(&output, EntityKind::Artifact, project_name).expect("project artifact");
        assert_eq!(meta(project, "artifact_kind"), "project");
        assert_eq!(meta(project, "layout"), "src-layout");
        assert_eq!(meta(project, "markers"), &json!(["pyproject.toml"]));

        let package = entity(&output, EntityKind::Artifact, "solver").expect("package artifact");
        assert_eq!(meta(package, "artifact_kind"), "package");
        let subpackage = entity(&output, EntityKind::Artifact, "solver.schemes").expect("subpackage artifact");
        assert_eq!(meta(subpackage, "artifact_kind"), "package");
        let grid = entity(&output, EntityKind::Artifact, "solver.grid").expect("module artifact");
        assert_eq!(meta(grid, "artifact_kind"), "module");
        let upwind = entity(&output, EntityKind::Artifact, "solver.schemes.upwind").expect("module artifact");

        // `src/` has no `__init__.py`, so it is a source root, not a package: no
        // `src.solver` may exist anywhere in the graph.
        assert!(
            !output.entities.iter().any(|e| e.name.starts_with("src.")),
            "a directory without __init__.py is not part of the import path"
        );
        // The virtualenv and the bytecode cache are not this project's code.
        assert!(entity(&output, EntityKind::Symbol, "NotOurs").is_none());
        assert!(entity(&output, EntityKind::Symbol, "Cached").is_none());

        // The artifact containment tree.
        assert!(has_edge(&output, grid, package, RelationshipKind::LocatedIn));
        assert!(has_edge(&output, upwind, subpackage, RelationshipKind::LocatedIn));
        assert!(has_edge(&output, subpackage, package, RelationshipKind::LocatedIn));
        assert!(has_edge(&output, package, project, RelationshipKind::LocatedIn));

        // Symbols, with Python's own facts preserved as metadata.
        let grid_class = entity(&output, EntityKind::Symbol, "Grid").expect("class symbol");
        assert_eq!(meta(grid_class, "symbol_kind"), "class");
        assert_eq!(meta(grid_class, "derivation"), "source-scan");
        assert_eq!(meta(grid_class, "module"), "solver.grid");
        assert_eq!(meta(grid_class, "line"), 5);

        let spacing = entity(&output, EntityKind::Symbol, "spacing").expect("method symbol");
        assert_eq!(meta(spacing, "symbol_kind"), "method", "a def inside a class is a method");
        assert_eq!(meta(spacing, "signature"), "(self)");

        let flux = entity(&output, EntityKind::Symbol, "flux").expect("decorated method");
        assert_eq!(meta(flux, "decorators"), &json!(["staticmethod"]));

        let solve = entity(&output, EntityKind::Symbol, "solve").expect("async method");
        assert_eq!(meta(solve, "is_async"), &json!(true));
        assert_eq!(meta(solve, "signature"), "(self, dt: float) -> float");

        let make_scheme = entity(&output, EntityKind::Symbol, "make_scheme").expect("function symbol");
        assert_eq!(
            meta(make_scheme, "symbol_kind"),
            "function",
            "a def at module level is a function, not a method"
        );

        let cell_count = entity(&output, EntityKind::Symbol, "CELL_COUNT").expect("constant symbol");
        assert_eq!(meta(cell_count, "symbol_kind"), "constant");
        let version = entity(&output, EntityKind::Symbol, "VERSION").expect("constant in a package __init__");
        assert_eq!(meta(version, "module"), "solver");

        // Nesting is preserved, not flattened: a closure inside a static method
        // inside a class. Containment is `LocatedIn` at a *symbol*...
        let limiter = entity(&output, EntityKind::Symbol, "limiter").expect("nested function");
        let upwind_class = entity(&output, EntityKind::Symbol, "Upwind").expect("class symbol");
        assert!(has_edge(&output, limiter, flux, RelationshipKind::LocatedIn));
        assert!(has_edge(&output, flux, upwind_class, RelationshipKind::LocatedIn));
        // ...and "which file" is `LocatedIn` at an *artifact*, one hop from every
        // symbol regardless of how deeply it is nested. Same edge kind, told apart
        // by the target's EntityKind — the convention all four language adapters use.
        assert!(has_edge(&output, limiter, upwind, RelationshipKind::LocatedIn));
        assert!(has_edge(&output, upwind_class, upwind, RelationshipKind::LocatedIn));

        // Inheritance: resolved inside the project, stubbed outside it.
        assert!(has_edge(&output, upwind_class, grid_class, inherits()));
        let abc = entity(&output, EntityKind::Symbol, "abc.ABC").expect("external base stub");
        assert_eq!(meta(abc, "external"), &json!(true));
        assert_eq!(meta(abc, "resolved"), &json!(false));
        assert!(has_edge(&output, upwind_class, abc, inherits()));

        // The import graph: third-party, absolute, and relative.
        let numpy = entity(&output, EntityKind::Artifact, "numpy").expect("external module");
        assert_eq!(meta(numpy, "artifact_kind"), "external_module");
        assert!(has_edge(&output, grid, numpy, RelationshipKind::DependsOn));

        let scipy = entity(&output, EntityKind::Artifact, "scipy.sparse").expect("external module");
        assert_eq!(meta(scipy, "distribution"), "scipy");
        assert!(has_edge(&output, upwind, scipy, RelationshipKind::DependsOn));

        // `from ..grid import Grid` in solver.schemes.upwind -> solver.grid.
        assert!(has_edge(&output, upwind, grid, RelationshipKind::DependsOn));
        // `from .grid import Grid` in the package __init__ -> solver.grid.
        assert!(has_edge(&output, package, grid, RelationshipKind::DependsOn));
        // `from solver.grid import Grid` -> the same module, named absolutely.
        assert!(has_edge(&output, subpackage, grid, RelationshipKind::DependsOn));
        // `from . import upwind` -> the submodule, because that submodule exists.
        assert!(has_edge(&output, subpackage, upwind, RelationshipKind::DependsOn));

        // `solver/__init__.py`'s docstring contains `import numpy as np`. It is
        // prose, not a dependency.
        assert!(
            !has_edge(&output, package, numpy, RelationshipKind::DependsOn),
            "an import inside a docstring is not an import"
        );
    }

    /// The [`KnowledgeProvider`] contract, taken literally: the tool is not
    /// installed, so the provider produces nothing — and does not error, and does
    /// not panic. The `PATH` is injected rather than the process environment
    /// mutated, because `std::env::set_var` is unsound in a multi-threaded test
    /// binary.
    #[test]
    fn language_server_only_degrades_to_empty_when_no_server_is_on_path() {
        let dir = synthetic_project();
        let provider = PythonProvider::language_server_only().with_path(OsString::new());
        assert_eq!(provider.detect_server(), None);

        let output = provider.collect(dir.path()).expect("a missing tool must not be an error");
        assert!(output.entities.is_empty(), "no server, no language-server facts");
        assert!(output.relationships.is_empty());
    }

    /// ...and the reason that is not the default. `Auto` on a machine with no Python
    /// server still knows the whole shape of the project, because the structure and
    /// the import graph never came from a language server in the first place. This
    /// is CLAUDE.md's Offline First rule, tested.
    #[test]
    fn auto_falls_back_to_the_source_scan_when_no_server_is_on_path() {
        let dir = synthetic_project();
        let output = PythonProvider::new()
            .with_path(OsString::new())
            .collect(dir.path())
            .expect("a missing tool must not be an error");

        assert!(entity(&output, EntityKind::Symbol, "Grid").is_some());
        assert!(
            output
                .entities
                .iter()
                .filter(|e| e.kind == EntityKind::Symbol && meta(e, "external") != &json!(true))
                .all(|e| meta(e, "derivation") == "source-scan"),
            "no symbol may claim a language server that never ran"
        );
    }

    /// A directory that is not a Python project at all yields nothing, rather than a
    /// lonely artifact for every directory KOPITIAM ever scans.
    #[test]
    fn a_directory_with_no_python_in_it_yields_no_facts() {
        let dir = tempfile::tempdir().expect("tempdir");
        write(dir.path(), "README.md", "# not python\n");
        let output = PythonProvider::source_scan_only().collect(dir.path()).expect("ok");
        assert!(output.entities.is_empty());
        assert!(output.relationships.is_empty());
    }

    /// Two runs over an unchanged tree must produce the same facts in the same
    /// order. An index that is rebuilt rather than synchronised is only trustworthy
    /// if rebuilding is deterministic.
    #[test]
    fn collection_is_deterministic() {
        let dir = synthetic_project();
        let provider = PythonProvider::source_scan_only();
        let first = provider.collect(dir.path()).expect("ok");
        let second = provider.collect(dir.path()).expect("ok");

        let names = |output: &ProviderOutput| -> Vec<(EntityKind, String)> {
            output
                .entities
                .iter()
                .map(|e| (e.kind, e.name.clone()))
                .collect()
        };
        assert_eq!(names(&first), names(&second));
        assert_eq!(first.relationships.len(), second.relationships.len());
    }

    /// Detection prefers pyright and falls back to pylsp, against an injected `PATH`.
    #[test]
    fn detects_pyright_first_then_pylsp() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bin = dir.path().to_path_buf();
        let path: OsString = bin.clone().into();

        assert_eq!(
            PythonProvider::new().with_path(path.clone()).detect_server(),
            None,
            "an empty directory on PATH holds no servers"
        );

        fake_executable(&bin, "pylsp");
        let detected = PythonProvider::new().with_path(path.clone()).detect_server();
        assert_eq!(detected.as_ref().map(|d| d.server), Some(PythonServer::Pylsp));
        assert_eq!(
            detected.map(|d| d.program),
            Some(bin.join("pylsp")),
            "detection must resolve to an absolute path, or spawning would re-resolve \
             the name against the real process PATH"
        );

        fake_executable(&bin, "pyright-langserver");
        let detected = PythonProvider::new().with_path(path).detect_server();
        assert_eq!(
            detected.map(|d| d.server),
            Some(PythonServer::Pyright),
            "pyright is primary: a full type checker beats a heuristic one"
        );
    }

    /// A file that exists but cannot be executed must not satisfy detection —
    /// otherwise a stray `pylsp` directory, or a shim without `+x`, would send us
    /// spawning something that cannot run.
    #[test]
    fn a_non_executable_file_does_not_satisfy_detection() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("pylsp"), "not executable").expect("write");
        let path: OsString = dir.path().to_path_buf().into();
        assert_eq!(PythonProvider::new().with_path(path).detect_server(), None);
    }

    fn fake_executable(dir: &Path, name: &str) {
        let path = dir.join(name);
        std::fs::write(&path, "#!/bin/sh\nexit 0\n").expect("write");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        }
    }

    /// PEP 3131: Python identifiers are Unicode. A Greek variable name is not an edge
    /// case in this project's domain, it is Tuesday — and `str::find` returns a
    /// *byte* offset, which is not what the ontology promises.
    #[test]
    fn non_ascii_source_yields_char_columns_not_byte_offsets() {
        let source = "Δt = 0.01\n\n\nclass Schéma:\n    # δu/δt — the update\n    def résumé(self, λ):\n        return λ\n";
        let symbols = scan_symbols(source);

        let schema = &symbols[0];
        assert_eq!(schema.name, "Schéma");
        assert_eq!(schema.line, 3);
        assert_eq!(schema.character, 6, "'class ' is six characters");

        let resume = &schema.children[0];
        assert_eq!(resume.name, "résumé");
        assert_eq!(resume.kind, PySymbolKind::Method);
        assert_eq!(resume.line, 5);
        assert_eq!(resume.character, 8, "'    def ' is eight characters");
        assert_eq!(resume.signature.as_deref(), Some("(self, λ)"));

        // The case where byte and char offsets actually diverge: a name preceded on
        // its own line by multi-byte text. `str::find` would say 12 (bytes); the
        // ontology's answer is 10 (chars), because `Δ` and `θ` are two bytes each.
        let source = "class Δθ_Solver:\n    pass\n";
        let symbols = scan_symbols(source);
        assert_eq!(symbols[0].name, "Δθ_Solver");
        assert_eq!(symbols[0].character, 6);

        let source = "def résumé(x): pass\n";
        let symbols = scan_symbols(source);
        assert_eq!(symbols[0].character, 4, "'def ' is four characters and four bytes");

        // ...and the one that would have been silently wrong: a symbol after
        // multi-byte text on the same line.
        let source = "class A:\n    Δ = 1\n    def λf(self): pass\n";
        let symbols = scan_symbols(source);
        let lambda_f = &symbols[0].children[0];
        assert_eq!(lambda_f.name, "λf");
        assert_eq!(lambda_f.character, 8, "char column 8, not byte offset 8 by luck");
    }

    /// The wire-encoding conversion, which is where non-ASCII silently corrupts
    /// positions if you get it wrong. Under LSP 3.17's `"utf-8"` a position is a
    /// *byte* offset; under `"utf-16"` it is a code-unit offset. The same symbol has
    /// a different number on the wire in each — and exactly one `char` column in the
    /// ontology.
    #[test]
    fn lsp_positions_convert_from_every_wire_encoding_to_char_columns() {
        let lines = vec!["class Café:", "    def résumé(self):"];
        let hierarchical = json!([{
            "name": "Café",
            "kind": 5,
            "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 1, "character": 24 } },
            "selectionRange": { "start": { "line": 0, "character": 6 }, "end": { "line": 0, "character": 10 } },
            "children": [{
                "name": "résumé",
                "kind": 12,
                "range": { "start": { "line": 1, "character": 4 }, "end": { "line": 1, "character": 24 } },
                "selectionRange": { "start": { "line": 1, "character": 8 }, "end": { "line": 1, "character": 14 } },
            }],
        }]);
        let tree = lsp_symbols_to_tree(hierarchical.as_array().unwrap(), PositionEncoding::Utf16, &lines);
        assert_eq!(tree.len(), 1, "the hierarchical shape must not be flattened");
        assert_eq!(tree[0].name, "Café");
        assert_eq!(tree[0].character, 6);
        assert_eq!(tree[0].end_line, 1, "the class body's last line, from `range`");
        assert_eq!(tree[0].children[0].name, "résumé");
        assert_eq!(tree[0].children[0].character, 8);
        assert_eq!(
            tree[0].children[0].kind,
            PySymbolKind::Method,
            "SymbolKind.Function inside a class is a method"
        );

        // The case that actually diverges between encodings: a symbol *after*
        // multi-byte text on its line. `Δ` is two UTF-8 bytes but one UTF-16 unit,
        // so the very same column is 8 on a `"utf-8"` wire and 7 on a `"utf-16"` one.
        let lines = vec!["Δ = 1; NAME = 2"];
        let flat = json!([{
            "name": "NAME",
            "kind": 14,
            "location": { "uri": "file:///x.py", "range": {
                "start": { "line": 0, "character": 8 },
                "end": { "line": 0, "character": 12 },
            } },
        }]);
        let items = flat.as_array().unwrap();

        let as_utf8 = lsp_symbols_to_tree(items, PositionEncoding::Utf8, &lines);
        assert_eq!(as_utf8[0].character, 7, "byte 8 is char 7: 'Δ' is two bytes");
        let as_utf16 = lsp_symbols_to_tree(items, PositionEncoding::Utf16, &lines);
        assert_eq!(as_utf16[0].character, 8, "'Δ' is one UTF-16 unit, so this column is unchanged");
        let as_utf32 = lsp_symbols_to_tree(items, PositionEncoding::Utf32, &lines);
        assert_eq!(as_utf32[0].character, 8, "utf-32 *is* char offsets");
        assert_ne!(
            as_utf8[0].character, as_utf16[0].character,
            "if these agreed, this test would be proving nothing"
        );
    }

    /// `workspace/symbol` returns the flat `SymbolInformation` shape, so the nesting
    /// has to be rebuilt from `containerName` — and a class-level `def` must still
    /// come out a `method` even when the server labels it `Function` (12), as pylsp
    /// does.
    #[test]
    fn flat_symbol_information_is_renested_by_container_name() {
        let lines = vec!["class Solver:", "    def step(self):", "        pass"];
        let items = json!([
            {
                "name": "step",
                "kind": 12,
                "containerName": "Solver",
                "location": { "uri": "file:///s.py", "range": {
                    "start": { "line": 1, "character": 8 }, "end": { "line": 2, "character": 12 } } },
            },
            {
                "name": "Solver",
                "kind": 5,
                "location": { "uri": "file:///s.py", "range": {
                    "start": { "line": 0, "character": 6 }, "end": { "line": 2, "character": 12 } } },
            },
        ]);
        let tree = lsp_symbols_to_tree(items.as_array().unwrap(), PositionEncoding::Utf16, &lines);

        assert_eq!(tree.len(), 1, "`step` must be nested, not left as a second root");
        assert_eq!(tree[0].name, "Solver");
        assert_eq!(tree[0].kind, PySymbolKind::Class);
        assert_eq!(tree[0].children.len(), 1);
        let step = &tree[0].children[0];
        assert_eq!(step.name, "step");
        assert_eq!(
            step.kind,
            PySymbolKind::Method,
            "containment we can see beats a label the server got wrong"
        );
        assert_eq!(step.lsp_kind, Some(12), "the raw LSP kind is preserved regardless");
    }

    /// Relative imports resolve with Python's own rule: against the importing
    /// module's *package*, not its module.
    #[test]
    fn relative_imports_resolve_against_the_package() {
        let known = BTreeSet::from(["pkg", "pkg.sub", "pkg.sub.mod", "pkg.other"]);

        let imports = scan_imports("from . import mod\n", "pkg.sub", true, &known);
        let modules: Vec<&str> = imports.iter().map(|i| i.module.as_str()).collect();
        assert!(modules.contains(&"pkg.sub"), "`from .` inside a package is that package");
        assert!(modules.contains(&"pkg.sub.mod"), "and `import mod` names a real submodule");

        // From a *module* inside the package, `.` still means the package.
        let imports = scan_imports("from .. import other\n", "pkg.sub.mod", false, &known);
        let modules: Vec<&str> = imports.iter().map(|i| i.module.as_str()).collect();
        assert!(modules.contains(&"pkg"), "`..` from pkg.sub.mod climbs to pkg");
        assert!(modules.contains(&"pkg.other"));

        // Climbing above the root is not an edge we are willing to invent.
        assert!(scan_imports("from ..... import nothing\n", "pkg.sub", false, &known).is_empty());
    }

    /// Every way Python spells an import — and several ways it does not.
    #[test]
    fn the_import_scanner_handles_the_real_syntax() {
        let known = BTreeSet::new();
        let source = concat!(
            "\"\"\"Docstring.\n",
            "\n",
            "    import not_a_dependency\n",
            "\"\"\"\n",
            "import os\n",
            "import numpy as np, scipy.linalg\n",
            "from typing import (\n",
            "    Iterator,\n",
            ")\n",
            "# import commented_out\n",
            "def f():\n",
            "    import lazily_imported  # deferred, but still a dependency\n",
            "EXAMPLE = \"import quoted\"\n",
        );
        let imports = scan_imports(source, "m", false, &known);
        let modules: Vec<&str> = imports.iter().map(|i| i.module.as_str()).collect();

        assert!(modules.contains(&"os"));
        assert!(modules.contains(&"numpy"));
        assert!(modules.contains(&"scipy.linalg"), "a comma-separated import is two imports");
        assert!(modules.contains(&"typing"));
        assert!(
            modules.contains(&"lazily_imported"),
            "an import inside a function is deferred, not fictional"
        );
        assert!(!modules.contains(&"not_a_dependency"), "prose in a docstring is not an import");
        assert!(!modules.contains(&"commented_out"));
        assert!(!modules.contains(&"quoted"), "a string that reads like an import is not one");
    }

    /// A `def` several levels deep still knows what it is. Indentation is Python's
    /// grammar, so this is a parse, not a guess.
    #[test]
    fn the_source_scanner_nests_to_arbitrary_depth() {
        let source = concat!(
            "def factory():\n",
            "    class Inner:\n",
            "        def method(self):\n",
            "            def closure():\n",
            "                pass\n",
            "            return closure\n",
            "    return Inner\n",
        );
        let symbols = scan_symbols(source);
        assert_eq!(symbols.len(), 1);

        let factory = &symbols[0];
        assert_eq!(factory.kind, PySymbolKind::Function);
        assert_eq!(factory.end_line, 6, "the body runs to the last more-indented line");

        let inner = &factory.children[0];
        assert_eq!(inner.kind, PySymbolKind::Class);
        let method = &inner.children[0];
        assert_eq!(method.kind, PySymbolKind::Method, "a def inside a class");
        let closure = &method.children[0];
        assert_eq!(closure.kind, PySymbolKind::Function, "a def inside a def is not a method");
    }

    /// Only screaming-snake module-level bindings are constants.
    #[test]
    fn module_level_constants_are_told_apart_from_every_other_binding() {
        assert_eq!(module_level_constant("MAX_ITER = 100"), Some("MAX_ITER".to_string()));
        assert_eq!(module_level_constant("TOLERANCE: float = 1e-9"), Some("TOLERANCE".to_string()));
        assert_eq!(module_level_constant("solver = Solver()"), None);
        assert_eq!(module_level_constant("if x == 1"), None);
        assert_eq!(module_level_constant("obj.ATTR = 1"), None);
        assert_eq!(module_level_constant("TABLE[0] = 1"), None);
    }

    /// The integration test. Ignored by default: it needs a real pyright or pylsp on
    /// `PATH`, which a build may not assume and a test may not go and install. Run it
    /// with `cargo test --release -p kopitiam-semantic -- --ignored`.
    ///
    /// Neither server was installed on the machine this provider was written on,
    /// which is precisely why the source scan exists.
    #[test]
    #[ignore = "requires pyright-langserver or pylsp on PATH"]
    fn a_real_language_server_produces_language_server_facts() {
        let provider = PythonProvider::new().with_lsp_timeout(Duration::from_secs(60));
        let Some(detected) = provider.detect_server() else {
            eprintln!("skipping: no Python language server on PATH");
            return;
        };
        eprintln!("using {} at {}", detected.server.as_str(), detected.program.display());

        let dir = synthetic_project();
        let output = provider.collect(dir.path()).expect("collection must succeed");

        let grid = entity(&output, EntityKind::Symbol, "Grid").expect("the server must find `Grid`");
        assert_eq!(
            meta(grid, "derivation"),
            "language-server",
            "a server was on PATH, so its facts are the ones that must have landed"
        );
        assert_eq!(meta(grid, "server"), detected.server.as_str());
        assert_eq!(meta(grid, "symbol_kind"), "class");
    }
}
