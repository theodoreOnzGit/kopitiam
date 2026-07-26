// `kmux latex <file.tex>` — the LaTeX live-preview workflow.
//
// This is a KOPITIAM extension subcommand, wired into `cli::run` alongside the
// other extensions (`claude`, `capabilities`, `diagnose`). It opens a two-pane
// kmux split — `kvim <file.tex>` on the left, a PDF view of the compiled output
// on the right — and recompiles the document whenever the `.tex` file is saved,
// refreshing the viewer pane so the new PDF shows.
//
// # Why a subcommand, not a config preset
//
// kmux inherits tmux's declarative layout surface (`split-window`,
// `select-layout`, config files), but none of those can express "watch a file
// and re-run a command on change". The watcher is the load-bearing part of this
// feature, and it needs a live process. A subcommand that *drives* kmux's own
// existing session/split machinery — by re-invoking the `kmux` binary as a
// subprocess, exactly as `cli/claude_launcher.rs` already does — is the only
// entry point that fits. It adds no new dependency (the async `rmux-sdk` grid
// builder is a dev-dependency only, and pulling it into the binary would be a
// new runtime dependency plus a Tokio reactor this command does not otherwise
// need) and reuses the same, already-tested `new-session` / `split-window` /
// `respawn-pane` / `attach-session` command surface.
//
// # Compile-on-save watcher
//
// A background thread polls the `.tex` file's modification time (std only, no
// `notify` crate — Termux-safe). A [`SaveDebouncer`] collapses a burst of saves
// into a single compile once the file has been quiet for a debounce window. On
// each debounced change the thread runs `latexmk` (falling back to `pdflatex`
// when `latexmk` is absent) and then respawns the viewer pane so the fresh PDF
// is picked up.
//
// # Graceful degradation
//
// The viewer pane runs `kmux __latex-view <pdf>`, an internal wrapper that
// launches `kopitiam view <pdf>` when it is available and otherwise prints a
// placeholder ("run `cargo install kopitiam`") and holds the pane open. If
// neither `latexmk` nor `pdflatex` is installed, the watcher simply performs no
// compile — it never crashes the session. `kopitiam view` itself is delivered by
// a parallel wave; this workflow only shells out to it at arm's length.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{self, Command as ProcessCommand, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use super::ExitFailure;

/// Internal subcommand name for the viewer wrapper spawned in the right pane.
const INTERNAL_VIEWER_COMMAND: &str = "__latex-view";

/// Environment variables that would make a nested `kmux` refuse to build or
/// attach a session. Cleared before every child `kmux` invocation, mirroring
/// `cli/claude_launcher.rs`.
const OUTER_MUX_ENV: &[&str] = &["RMUX", "TMUX", "RMUX_PANE", "TMUX_PANE"];

/// How often the watcher samples the `.tex` file's modification time.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// How long the `.tex` file must be quiet (unchanged mtime) before a compile is
/// triggered. Collapses a burst of editor saves into a single build.
const DEBOUNCE: Duration = Duration::from_millis(400);

/// A parsed `kmux latex` / `kmux __latex-view` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum LatexInvocation {
    /// `kmux latex <file.tex>` — run the full live-preview workflow.
    Workflow { tex: PathBuf },
    /// `kmux __latex-view <file.pdf>` — internal viewer wrapper for the right
    /// pane.
    Viewer { pdf: PathBuf },
}

/// Recognises `latex` and the internal `__latex-view` command after the
/// top-level socket flags. Returns `Ok(None)` for any other command.
pub(super) fn parse_invocation(
    arguments: &[OsString],
) -> Result<Option<LatexInvocation>, ExitFailure> {
    let Some(command_index) = split_top_level_prefix(arguments) else {
        return Ok(None);
    };
    let Some(command) = arguments.get(command_index).and_then(|value| value.to_str()) else {
        return Ok(None);
    };
    let rest = &arguments[command_index + 1..];
    match command {
        "latex" => Ok(Some(parse_workflow(rest)?)),
        INTERNAL_VIEWER_COMMAND => Ok(Some(parse_viewer(rest)?)),
        _ => Ok(None),
    }
}

fn parse_workflow(arguments: &[OsString]) -> Result<LatexInvocation, ExitFailure> {
    let mut file: Option<PathBuf> = None;
    for argument in arguments {
        match argument.to_str() {
            Some("--help" | "-h") => {
                return Err(ExitFailure::new_stdout(
                    0,
                    "usage: kmux latex <file.tex>\n\n\
                     Opens a two-pane split: kvim on the left, the compiled PDF on the\n\
                     right, and recompiles with latexmk (or pdflatex) on every save.",
                ));
            }
            Some(other) if other.starts_with('-') && other != "-" => {
                return Err(ExitFailure::new(
                    1,
                    format!("kmux latex: unknown argument '{other}'"),
                ));
            }
            _ => {
                if file.is_some() {
                    return Err(ExitFailure::new(
                        1,
                        "kmux latex: expected exactly one .tex file",
                    ));
                }
                file = Some(PathBuf::from(argument));
            }
        }
    }

    let file = file.ok_or_else(|| {
        ExitFailure::new(1, "kmux latex: expected a .tex file (usage: kmux latex <file.tex>)")
    })?;
    if !is_tex_file(&file) {
        return Err(ExitFailure::new(
            1,
            format!(
                "kmux latex: expected a .tex file, got '{}'",
                file.display()
            ),
        ));
    }
    Ok(LatexInvocation::Workflow { tex: file })
}

fn parse_viewer(arguments: &[OsString]) -> Result<LatexInvocation, ExitFailure> {
    let pdf = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| ExitFailure::new(1, "kmux __latex-view: expected a pdf path"))?;
    Ok(LatexInvocation::Viewer { pdf })
}

/// Dispatches a parsed invocation.
pub(super) fn run(invocation: LatexInvocation) -> Result<i32, ExitFailure> {
    match invocation {
        LatexInvocation::Workflow { tex } => run_workflow(tex),
        LatexInvocation::Viewer { pdf } => run_viewer(&pdf),
    }
}

// ---------------------------------------------------------------------------
// Pure, unit-tested logic
// ---------------------------------------------------------------------------

/// Returns `true` if `path` names a `.tex` file (case-insensitive extension).
fn is_tex_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("tex"))
}

/// Derives the compiled PDF path from a `.tex` path by swapping the extension:
/// `chapters/intro.tex` -> `chapters/intro.pdf`.
fn derived_pdf_path(tex: &Path) -> PathBuf {
    tex.with_extension("pdf")
}

/// The LaTeX engine used for a compile pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompileEngine {
    /// `latexmk` — preferred; handles reruns and bibliography automatically.
    Latexmk,
    /// `pdflatex` — fallback used when `latexmk` is not installed.
    Pdflatex,
}

impl CompileEngine {
    /// Engines tried in order until one is found on `PATH`.
    const ORDER: [CompileEngine; 2] = [CompileEngine::Latexmk, CompileEngine::Pdflatex];
}

/// A fully-constructed compile command: the program name plus its arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CompileCommand {
    program: &'static str,
    args: Vec<String>,
}

/// Builds the compile command for `engine` targeting `tex` (a file name or
/// path). Both engines run non-interactively and stop on the first error so a
/// broken document cannot hang the watcher waiting on a `?` prompt.
fn compile_command(engine: CompileEngine, tex: &str) -> CompileCommand {
    match engine {
        CompileEngine::Latexmk => CompileCommand {
            program: "latexmk",
            args: vec![
                "-pdf".to_owned(),
                "-interaction=nonstopmode".to_owned(),
                "-halt-on-error".to_owned(),
                tex.to_owned(),
            ],
        },
        CompileEngine::Pdflatex => CompileCommand {
            program: "pdflatex",
            args: vec![
                "-interaction=nonstopmode".to_owned(),
                "-halt-on-error".to_owned(),
                tex.to_owned(),
            ],
        },
    }
}

/// Derives a kmux session name from the document stem and this process's pid.
/// tmux session names may not contain `.` or `:`, so every other character is
/// coerced to `-`; the pid keeps concurrent `kmux latex` runs from colliding.
fn session_name_for(stem: &str, pid: u32) -> String {
    let mut sanitized = String::with_capacity(stem.len());
    for character in stem.chars() {
        if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
            sanitized.push(character);
        } else {
            sanitized.push('-');
        }
    }
    if sanitized.is_empty() {
        sanitized.push_str("doc");
    }
    format!("kmux-latex-{sanitized}-{pid}")
}

/// The placeholder shown in the viewer pane when `kopitiam view` is unavailable.
fn placeholder_message(pdf: &Path) -> String {
    format!(
        "kopitiam view is not available on PATH.\n\n\
         The compiled PDF is at:\n  {}\n\n\
         Install the viewer with:\n  cargo install kopitiam\n\n\
         This pane refreshes automatically once the viewer is installed and the\n\
         .tex file is saved again.",
        pdf.display()
    )
}

/// Debounces file-save events sampled from a polled modification time.
///
/// The caller feeds the latest observed mtime and a monotonically increasing
/// "now" tick (milliseconds) on every poll. [`SaveDebouncer::observe`] returns
/// `true` exactly once per burst of saves: after the mtime has stopped changing
/// for the debounce window. The first observation only establishes a baseline —
/// the initial compile is performed separately by the workflow before the
/// watcher starts — so the watcher never recompiles an unedited document.
#[derive(Debug)]
struct SaveDebouncer {
    debounce_ms: u64,
    last_mtime: Option<SystemTime>,
    /// The tick at which the mtime last changed, while a compile is still owed.
    pending_since: Option<u64>,
}

impl SaveDebouncer {
    fn new(debounce_ms: u64) -> Self {
        Self {
            debounce_ms,
            last_mtime: None,
            pending_since: None,
        }
    }

    /// Records the current mtime observed at `now_tick_ms` and reports whether a
    /// compile should run now.
    fn observe(&mut self, mtime: Option<SystemTime>, now_tick_ms: u64) -> bool {
        let Some(mtime) = mtime else {
            // The file vanished (mid-save on some editors). Drop any pending
            // compile and wait for it to reappear.
            self.pending_since = None;
            return false;
        };

        match self.last_mtime {
            None => {
                // First sighting: establish the baseline without compiling.
                self.last_mtime = Some(mtime);
                false
            }
            Some(previous) if previous != mtime => {
                // The file changed. (Re)start the quiet window; a rapid burst of
                // saves keeps resetting this, collapsing into one compile.
                self.last_mtime = Some(mtime);
                self.pending_since = Some(now_tick_ms);
                false
            }
            Some(_) => match self.pending_since {
                Some(since) if now_tick_ms.saturating_sub(since) >= self.debounce_ms => {
                    self.pending_since = None;
                    true
                }
                _ => false,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Workflow orchestration (interactive; exercised manually, not in unit tests)
// ---------------------------------------------------------------------------

fn run_workflow(tex_arg: PathBuf) -> Result<i32, ExitFailure> {
    let tex = absolute_path(&tex_arg);
    if !tex.is_file() {
        return Err(ExitFailure::new(
            1,
            format!("kmux latex: file not found: {}", tex.display()),
        ));
    }
    let work_dir = tex
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let tex_name = tex
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| tex.to_string_lossy().into_owned());
    let stem = tex
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "doc".to_owned());
    let pdf = derived_pdf_path(&tex);

    let binary = std::env::current_exe().map_err(|error| {
        ExitFailure::new(
            1,
            format!("kmux latex: failed to resolve current kmux binary: {error}"),
        )
    })?;
    let session = session_name_for(&stem, process::id());

    // Best-effort initial compile so the viewer has a PDF to show immediately.
    run_compile(&work_dir, &tex_name);

    // Left pane: the editor.
    create_session(&binary, &session, &work_dir, &[os("kvim"), tex.clone().into()])?;

    // Right pane: the viewer wrapper. Capture its stable pane id for refreshes.
    let viewer_argv = [
        binary.clone().into_os_string(),
        os(INTERNAL_VIEWER_COMMAND),
        pdf.clone().into_os_string(),
    ];
    let pane_id = match split_viewer_pane(&binary, &session, &work_dir, &viewer_argv) {
        Ok(pane_id) => pane_id,
        Err(error) => {
            let _ = kill_session(&binary, &session);
            return Err(error);
        }
    };

    // Compile-on-save watcher.
    let running = Arc::new(AtomicBool::new(true));
    let watcher = spawn_watcher(WatcherContext {
        binary: binary.clone(),
        work_dir,
        tex_name,
        pdf,
        pane_id,
        running: Arc::clone(&running),
    });

    // Attach in the foreground; blocks until the user detaches or exits.
    let status = attach_session(&binary, &session);

    running.store(false, Ordering::Relaxed);
    let _ = watcher.join();
    let _ = kill_session(&binary, &session);

    status
}

/// Context handed to the watcher thread.
struct WatcherContext {
    binary: PathBuf,
    work_dir: PathBuf,
    tex_name: String,
    pdf: PathBuf,
    pane_id: String,
    running: Arc<AtomicBool>,
}

fn spawn_watcher(context: WatcherContext) -> thread::JoinHandle<()> {
    thread::spawn(move || run_watcher(context))
}

fn run_watcher(context: WatcherContext) {
    let tex_path = context.work_dir.join(&context.tex_name);
    let start = Instant::now();
    let mut debouncer = SaveDebouncer::new(DEBOUNCE.as_millis() as u64);

    while context.running.load(Ordering::Relaxed) {
        thread::sleep(POLL_INTERVAL);
        if !context.running.load(Ordering::Relaxed) {
            break;
        }
        let mtime = file_mtime(&tex_path);
        let now_tick = start.elapsed().as_millis() as u64;
        if debouncer.observe(mtime, now_tick) {
            run_compile(&context.work_dir, &context.tex_name);
            refresh_viewer_pane(&context.binary, &context.pane_id, &context.pdf);
        }
    }
}

fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
}

/// Runs one compile pass, trying each engine until one is found on `PATH`.
/// Output is discarded (the terminal is owned by the attached session; LaTeX
/// still writes its own `.log`). Missing tooling is silently tolerated.
fn run_compile(work_dir: &Path, tex: &str) {
    for engine in CompileEngine::ORDER {
        let command = compile_command(engine, tex);
        let result = ProcessCommand::new(command.program)
            .args(&command.args)
            .current_dir(work_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        match result {
            // The engine ran (whether or not the document compiled cleanly).
            Ok(_) => return,
            // This engine is not installed; try the next one.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            // Some other spawn failure; give up quietly rather than crash.
            Err(_) => return,
        }
    }
}

fn create_session(
    binary: &Path,
    session: &str,
    work_dir: &Path,
    editor_argv: &[OsString],
) -> Result<(), ExitFailure> {
    let mut command = kmux_command(binary);
    command
        .arg("new-session")
        .arg("-d")
        .arg("-s")
        .arg(session)
        .arg("-n")
        .arg("editor")
        .arg("-c")
        .arg(work_dir);
    append_pane_command(&mut command, editor_argv)?;
    let status = command.status().map_err(|error| {
        ExitFailure::new(
            1,
            format!("kmux latex: failed to create session: {error}"),
        )
    })?;
    if status.success() {
        Ok(())
    } else {
        Err(ExitFailure::new(
            status.code().unwrap_or(1),
            "kmux latex: failed to create session",
        ))
    }
}

fn split_viewer_pane(
    binary: &Path,
    session: &str,
    work_dir: &Path,
    viewer_argv: &[OsString],
) -> Result<String, ExitFailure> {
    let mut command = kmux_command(binary);
    command
        .arg("split-window")
        .arg("-h")
        .arg("-d")
        .arg("-t")
        .arg(session)
        .arg("-c")
        .arg(work_dir)
        .arg("-P")
        .arg("-F")
        .arg("#{pane_id}");
    append_pane_command(&mut command, viewer_argv)?;
    let output = command
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| {
            ExitFailure::new(
                1,
                format!("kmux latex: failed to split viewer pane: {error}"),
            )
        })?;
    if !output.status.success() {
        return Err(ExitFailure::new(
            output.status.code().unwrap_or(1),
            "kmux latex: failed to split viewer pane",
        ));
    }
    let pane_id = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            ExitFailure::new(1, "kmux latex: split-window did not report a pane id")
        })?;
    Ok(pane_id)
}

/// Restarts the viewer pane so it re-opens the freshly compiled PDF. Best
/// effort: a failed refresh must not tear the session down.
fn refresh_viewer_pane(binary: &Path, pane_id: &str, pdf: &Path) {
    let viewer_argv = [
        binary.as_os_str().to_os_string(),
        os(INTERNAL_VIEWER_COMMAND),
        pdf.as_os_str().to_os_string(),
    ];
    let mut command = kmux_command(binary);
    command
        .arg("respawn-pane")
        .arg("-k")
        .arg("-t")
        .arg(pane_id);
    if append_pane_command(&mut command, &viewer_argv).is_err() {
        return;
    }
    let _ = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn attach_session(binary: &Path, session: &str) -> Result<i32, ExitFailure> {
    let status = kmux_command(binary)
        .arg("attach-session")
        .arg("-t")
        .arg(session)
        .status()
        .map_err(|error| {
            ExitFailure::new(
                1,
                format!("kmux latex: failed to attach session: {error}"),
            )
        })?;
    Ok(status.code().unwrap_or(0))
}

fn kill_session(binary: &Path, session: &str) -> std::io::Result<()> {
    kmux_command(binary)
        .arg("kill-session")
        .arg("-t")
        .arg(session)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|_| ())
}

/// A child `kmux` invocation with the outer-multiplexer environment cleared, so
/// the nested session builds and attaches cleanly.
fn kmux_command(binary: &Path) -> ProcessCommand {
    let mut command = ProcessCommand::new(binary);
    for name in OUTER_MUX_ENV {
        command.env_remove(name);
    }
    command
}

/// Appends a pane command to a `kmux` invocation. On Unix, kmux (like tmux) runs
/// a pane's command through the shell, so the argv is quoted into a single
/// string. On other platforms there is no `/bin/sh`, so the argv is passed
/// directly. Mirrors `cli/claude_launcher.rs::append_runner_command`.
#[cfg(unix)]
fn append_pane_command(
    command: &mut ProcessCommand,
    argv: &[OsString],
) -> Result<(), ExitFailure> {
    command.arg("--").arg(shell_join(argv)?);
    Ok(())
}

#[cfg(not(unix))]
fn append_pane_command(
    command: &mut ProcessCommand,
    argv: &[OsString],
) -> Result<(), ExitFailure> {
    command.arg("--").args(argv);
    Ok(())
}

#[cfg(unix)]
fn shell_join(argv: &[OsString]) -> Result<String, ExitFailure> {
    let mut joined = String::new();
    for argument in argv {
        if !joined.is_empty() {
            joined.push(' ');
        }
        joined.push_str(&shell_quote(argument)?);
    }
    Ok(joined)
}

#[cfg(unix)]
fn shell_quote(value: &OsString) -> Result<String, ExitFailure> {
    let value = value.to_str().ok_or_else(|| {
        ExitFailure::new(1, "kmux latex: pane commands must be valid UTF-8 on this platform")
    })?;
    if value.is_empty() {
        return Ok("''".to_owned());
    }
    if value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'_' | b'-' | b':' | b'=' | b',' | b'%')
    }) {
        return Ok(value.to_owned());
    }
    Ok(format!("'{}'", value.replace('\'', "'\\''")))
}

fn absolute_path(path: &Path) -> PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .unwrap_or_else(|_| path.to_path_buf())
}

fn os(value: &str) -> OsString {
    OsString::from(value)
}

// ---------------------------------------------------------------------------
// Internal viewer wrapper (`kmux __latex-view <pdf>`)
// ---------------------------------------------------------------------------

fn run_viewer(pdf: &Path) -> Result<i32, ExitFailure> {
    match launch_kopitiam_view(pdf) {
        ViewerOutcome::Exited(code) => Ok(code),
        ViewerOutcome::Unavailable => Ok(hold_placeholder(pdf)),
    }
}

enum ViewerOutcome {
    /// `kopitiam view` ran and exited with this code. On Unix the wrapper
    /// `exec`s into `kopitiam`, so this variant is only ever constructed on
    /// non-Unix platforms that spawn-and-wait instead.
    #[cfg_attr(unix, allow(dead_code))]
    Exited(i32),
    /// `kopitiam` could not be launched; fall back to the placeholder.
    Unavailable,
}

#[cfg(unix)]
fn launch_kopitiam_view(pdf: &Path) -> ViewerOutcome {
    use std::os::unix::process::CommandExt;
    // exec replaces this wrapper with kopitiam so the pane's process *is* the
    // viewer; when kmux respawns/kills the pane it kills the viewer directly.
    let error = ProcessCommand::new("kopitiam")
        .arg("view")
        .arg(pdf)
        .exec();
    // exec only returns on failure.
    if error.kind() == std::io::ErrorKind::NotFound {
        ViewerOutcome::Unavailable
    } else {
        // Any other launch failure also degrades to the placeholder.
        ViewerOutcome::Unavailable
    }
}

#[cfg(not(unix))]
fn launch_kopitiam_view(pdf: &Path) -> ViewerOutcome {
    match ProcessCommand::new("kopitiam").arg("view").arg(pdf).status() {
        Ok(status) => ViewerOutcome::Exited(status.code().unwrap_or(0)),
        Err(_) => ViewerOutcome::Unavailable,
    }
}

/// Prints the placeholder and holds the pane open until kmux terminates it.
fn hold_placeholder(pdf: &Path) -> i32 {
    println!("{}", placeholder_message(pdf));
    loop {
        thread::sleep(Duration::from_secs(3600));
    }
}

// ---------------------------------------------------------------------------
// Shared top-level flag skipping (identical to the other cli extensions)
// ---------------------------------------------------------------------------

fn split_top_level_prefix(arguments: &[OsString]) -> Option<usize> {
    let mut index = 0;

    while let Some(argument) = arguments.get(index) {
        let value = argument.to_str()?;
        if value == "--" {
            return Some(index + 1);
        }
        if !value.starts_with('-') || value == "-" {
            return Some(index);
        }

        match value {
            "-2" | "-D" | "-N" | "-l" | "-u" => {}
            "-C" | "-v" => {}
            "-c" | "-f" | "-L" | "-S" | "-T" => {
                index += 1;
            }
            _ if value.starts_with("-L") && value.len() > 2 => {}
            _ if value.starts_with("-S") && value.len() > 2 => {}
            _ if value.starts_with("-f") && value.len() > 2 => {}
            _ if value.starts_with("-T") && value.len() > 2 => {}
            _ if is_short_flag_cluster(value, "2CDNluv") => {}
            _ => return Some(index),
        }

        index += 1;
    }

    None
}

fn is_short_flag_cluster(value: &str, allowed: &str) -> bool {
    value.len() > 2
        && value.starts_with('-')
        && !value.starts_with("--")
        && value.chars().skip(1).all(|flag| allowed.contains(flag))
}

#[cfg(test)]
mod tests {
    use super::{
        compile_command, derived_pdf_path, is_tex_file, parse_invocation, placeholder_message,
        session_name_for, CompileEngine, LatexInvocation, SaveDebouncer,
    };
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, SystemTime};

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    fn mtime(secs: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
    }

    #[test]
    fn derived_pdf_swaps_extension() {
        assert_eq!(derived_pdf_path(Path::new("paper.tex")), PathBuf::from("paper.pdf"));
        assert_eq!(
            derived_pdf_path(Path::new("chapters/intro.tex")),
            PathBuf::from("chapters/intro.pdf")
        );
        // Uppercase and extension-less inputs still yield a `.pdf`.
        assert_eq!(derived_pdf_path(Path::new("PAPER.TEX")), PathBuf::from("PAPER.pdf"));
        assert_eq!(derived_pdf_path(Path::new("paper")), PathBuf::from("paper.pdf"));
    }

    #[test]
    fn is_tex_file_is_case_insensitive() {
        assert!(is_tex_file(Path::new("paper.tex")));
        assert!(is_tex_file(Path::new("dir/PAPER.TEX")));
        assert!(!is_tex_file(Path::new("paper.pdf")));
        assert!(!is_tex_file(Path::new("paper")));
    }

    #[test]
    fn latexmk_command_is_non_interactive_and_halts_on_error() {
        let command = compile_command(CompileEngine::Latexmk, "paper.tex");
        assert_eq!(command.program, "latexmk");
        assert_eq!(
            command.args,
            vec![
                "-pdf".to_owned(),
                "-interaction=nonstopmode".to_owned(),
                "-halt-on-error".to_owned(),
                "paper.tex".to_owned(),
            ]
        );
    }

    #[test]
    fn pdflatex_fallback_command_is_non_interactive() {
        let command = compile_command(CompileEngine::Pdflatex, "paper.tex");
        assert_eq!(command.program, "pdflatex");
        assert_eq!(
            command.args,
            vec![
                "-interaction=nonstopmode".to_owned(),
                "-halt-on-error".to_owned(),
                "paper.tex".to_owned(),
            ]
        );
    }

    #[test]
    fn session_name_sanitizes_and_disambiguates() {
        assert_eq!(session_name_for("paper", 42), "kmux-latex-paper-42");
        // Dots and colons are illegal in tmux session names -> coerced to '-'.
        assert_eq!(session_name_for("my.paper:v2", 7), "kmux-latex-my-paper-v2-7");
        // Empty stems still produce a valid name.
        assert_eq!(session_name_for("", 9), "kmux-latex-doc-9");
    }

    #[test]
    fn placeholder_mentions_install_command() {
        let message = placeholder_message(Path::new("/tmp/paper.pdf"));
        assert!(message.contains("cargo install kopitiam"));
        assert!(message.contains("/tmp/paper.pdf"));
    }

    #[test]
    fn debouncer_first_observation_is_baseline_only() {
        let mut debouncer = SaveDebouncer::new(400);
        // First sighting establishes the baseline; no compile even after a long
        // quiet period.
        assert!(!debouncer.observe(Some(mtime(100)), 0));
        assert!(!debouncer.observe(Some(mtime(100)), 10_000));
    }

    #[test]
    fn debouncer_compiles_once_after_a_save_settles() {
        let mut debouncer = SaveDebouncer::new(400);
        assert!(!debouncer.observe(Some(mtime(100)), 0)); // baseline
        assert!(!debouncer.observe(Some(mtime(200)), 1_000)); // change seen -> pending
        assert!(!debouncer.observe(Some(mtime(200)), 1_200)); // 200ms quiet < 400ms
        assert!(debouncer.observe(Some(mtime(200)), 1_500)); // 500ms quiet >= 400ms -> compile
        // The compile only fires once until the next change.
        assert!(!debouncer.observe(Some(mtime(200)), 2_000));
    }

    #[test]
    fn debouncer_collapses_a_burst_of_saves_into_one_compile() {
        let mut debouncer = SaveDebouncer::new(400);
        assert!(!debouncer.observe(Some(mtime(100)), 0)); // baseline
        // A burst: each change restarts the quiet window even if 400ms elapsed
        // between individual saves.
        assert!(!debouncer.observe(Some(mtime(101)), 500));
        assert!(!debouncer.observe(Some(mtime(102)), 1_000));
        assert!(!debouncer.observe(Some(mtime(103)), 1_500));
        // Now the file goes quiet.
        assert!(!debouncer.observe(Some(mtime(103)), 1_700)); // 200ms < 400ms
        assert!(debouncer.observe(Some(mtime(103)), 1_950)); // 450ms >= 400ms -> single compile
    }

    #[test]
    fn debouncer_ignores_a_transiently_missing_file() {
        let mut debouncer = SaveDebouncer::new(400);
        assert!(!debouncer.observe(Some(mtime(100)), 0)); // baseline
        assert!(!debouncer.observe(Some(mtime(200)), 1_000)); // change -> pending
        // Editor briefly unlinks the file mid-save: the pending compile is
        // dropped rather than firing against a missing file.
        assert!(!debouncer.observe(None, 1_500));
        // File reappears unchanged; without a fresh change, no compile fires.
        assert!(!debouncer.observe(Some(mtime(200)), 2_000));
        assert!(!debouncer.observe(Some(mtime(200)), 3_000));
        // A genuine new save still triggers a compile.
        assert!(!debouncer.observe(Some(mtime(300)), 3_100)); // change -> pending
        assert!(debouncer.observe(Some(mtime(300)), 3_600)); // settled -> compile
    }

    #[test]
    fn parses_workflow_after_socket_flags() {
        let invocation = parse_invocation(&args(&["-Ldemo", "latex", "paper.tex"]))
            .expect("parse succeeds")
            .expect("latex invocation");
        assert_eq!(
            invocation,
            LatexInvocation::Workflow {
                tex: PathBuf::from("paper.tex")
            }
        );
    }

    #[test]
    fn parses_internal_viewer_command() {
        let invocation = parse_invocation(&args(&["__latex-view", "paper.pdf"]))
            .expect("parse succeeds")
            .expect("viewer invocation");
        assert_eq!(
            invocation,
            LatexInvocation::Viewer {
                pdf: PathBuf::from("paper.pdf")
            }
        );
    }

    #[test]
    fn rejects_non_tex_files() {
        let error = parse_invocation(&args(&["latex", "paper.pdf"]))
            .expect_err("non-tex file should fail");
        assert!(error.message().contains("expected a .tex file"));
    }

    #[test]
    fn rejects_missing_file_argument() {
        let error =
            parse_invocation(&args(&["latex"])).expect_err("missing file should fail");
        assert!(error.message().contains("expected a .tex file"));
    }

    #[test]
    fn rejects_extra_file_arguments() {
        let error = parse_invocation(&args(&["latex", "a.tex", "b.tex"]))
            .expect_err("two files should fail");
        assert!(error.message().contains("exactly one"));
    }

    #[test]
    fn ignores_other_commands() {
        assert!(parse_invocation(&args(&["list-sessions"]))
            .expect("parse succeeds")
            .is_none());
    }

    #[test]
    fn help_flag_exits_zero_with_usage() {
        let error = parse_invocation(&args(&["latex", "--help"]))
            .expect_err("help should short-circuit");
        assert_eq!(error.exit_code(), 0);
        assert!(error.message().contains("usage: kmux latex"));
    }
}
