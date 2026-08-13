use std::ffi::OsString;
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use super::uds::UnixStream;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, SystemTime};

use crate::api::QueryResult;

use super::spawn_sanitizer::{SpawnedDaemon, spawn_daemon_process};
use crate::surface::ipc::IpcError;
use crate::surface::ipc::{IPC_PROTOCOL_VERSION, Request, Response, ResponsePayload};

// =============================================================================
// Socket path
// =============================================================================

static RUNTIME_DIR_OVERRIDE: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static RUNTIME_DIR_OVERRIDE_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Set or clear the runtime directory override. Can be called multiple times; latest value wins.
pub fn set_runtime_dir_override(dir: Option<PathBuf>) {
    let lock = RUNTIME_DIR_OVERRIDE.get_or_init(|| Mutex::new(None));
    *lock.lock().expect("runtime dir lock poisoned") = dir;
}

#[doc(hidden)]
pub struct RuntimeDirOverrideGuard {
    prev: Option<PathBuf>,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for RuntimeDirOverrideGuard {
    fn drop(&mut self) {
        set_runtime_dir_override(self.prev.take());
    }
}

#[doc(hidden)]
pub fn override_runtime_dir_for_tests(dir: Option<PathBuf>) -> RuntimeDirOverrideGuard {
    let lock = RUNTIME_DIR_OVERRIDE_TEST_LOCK.get_or_init(|| Mutex::new(()));
    let test_lock = lock.lock().expect("runtime dir test lock poisoned");
    let prev = runtime_dir_override();
    set_runtime_dir_override(dir);
    RuntimeDirOverrideGuard {
        prev,
        _lock: test_lock,
    }
}

fn runtime_dir_override() -> Option<PathBuf> {
    let lock = RUNTIME_DIR_OVERRIDE.get()?;
    lock.lock().expect("runtime dir lock poisoned").clone()
}

/// Get the directory that will contain the daemon socket.
pub fn socket_dir() -> PathBuf {
    socket_dir_candidates()
        .into_iter()
        .next()
        .unwrap_or_else(per_user_tmp_dir)
}

/// Ensure the socket directory exists and is user-private.
pub fn ensure_socket_dir() -> Result<PathBuf, IpcError> {
    let mut last_err: Option<std::io::Error> = None;
    for dir in socket_dir_candidates() {
        match ensure_private_socket_dir_path(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) => last_err = Some(e),
        }
    }

    Err(IpcError::Transport {
        source: last_err.unwrap_or_else(|| {
            std::io::Error::other("unable to create a writable socket directory")
        }),
    })
}

fn ensure_socket_dir_path(dir: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(dir)?;
    Ok(())
}

fn ensure_private_socket_dir_path(dir: &Path) -> Result<(), std::io::Error> {
    ensure_socket_dir_path(dir)?;
    #[cfg(unix)]
    {
        let mode = fs::metadata(dir)?.permissions().mode() & 0o777;
        if mode != 0o700 {
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
        }
    }
    Ok(())
}

fn socket_parent_should_be_private(socket: &Path, dir: &Path) -> bool {
    socket.file_name() == Some(std::ffi::OsStr::new("daemon.sock"))
        && socket_dir_candidates()
            .iter()
            .any(|candidate| candidate == dir)
}

fn ensure_autostart_socket_parent(socket: &Path, dir: &Path) -> Result<(), std::io::Error> {
    if socket_parent_should_be_private(socket, dir) {
        ensure_private_socket_dir_path(dir)
    } else {
        ensure_socket_dir_path(dir)
    }
}

/// Get the daemon socket path.
pub fn socket_path() -> PathBuf {
    ensure_socket_dir()
        .map(|dir| dir.join("daemon.sock"))
        .unwrap_or_else(|_| per_user_tmp_dir().join("daemon.sock"))
}

/// Build a daemon socket path for a specific runtime directory.
pub fn socket_path_for_runtime_dir(runtime_dir: &Path) -> PathBuf {
    runtime_dir.join("beads").join("daemon.sock")
}

/// Read daemon metadata from the meta file in the socket directory.
/// Returns None if file doesn't exist or is corrupt.
fn read_daemon_meta_at(socket: &Path) -> Option<crate::api::DaemonInfo> {
    let meta_path = socket.with_file_name("daemon.meta.json");
    let contents = fs::read_to_string(&meta_path).ok()?;
    serde_json::from_str(&contents).ok()
}

fn daemon_pid_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        use nix::sys::signal::kill;
        use nix::unistd::Pid;
        kill(Pid::from_raw(pid as i32), None).is_ok()
    }
    #[cfg(windows)]
    {
        super::proc_util::pid_alive(pid)
    }
}

fn per_user_tmp_dir() -> PathBuf {
    #[cfg(unix)]
    {
        let uid = nix::unistd::geteuid();
        PathBuf::from("/tmp").join(format!("beads-{}", uid))
    }
    #[cfg(windows)]
    {
        // Windows has neither `/tmp` nor uids. `std::env::temp_dir()` resolves
        // `%TEMP%`, which is already per-user (…\Users\<user>\AppData\Local\Temp),
        // so this is the natural per-user runtime location. Client and
        // autostarted daemon share the same env, so both resolve the same path.
        std::env::temp_dir().join("beads")
    }
}

fn socket_dir_candidates() -> Vec<PathBuf> {
    socket_dir_candidates_with(|key| std::env::var(key).ok())
}

fn socket_dir_candidates_with<F>(mut lookup: F) -> Vec<PathBuf>
where
    F: FnMut(&str) -> Option<String>,
{
    let mut dirs = Vec::new();

    if let Some(runtime_dir) = runtime_dir_override() {
        dirs.push(runtime_dir.join("beads"));
    }

    // Env override (works even without config init)
    if let Some(dir) = lookup("BD_RUNTIME_DIR") {
        let trimmed = dir.trim();
        if !trimmed.is_empty() {
            dirs.push(PathBuf::from(trimmed).join("beads"));
        }
    }

    if let Some(dir) = lookup("XDG_RUNTIME_DIR") {
        let trimmed = dir.trim();
        if !trimmed.is_empty() {
            dirs.push(PathBuf::from(trimmed).join("beads"));
        }
    }
    if let Some(home) = lookup("HOME") {
        let trimmed = home.trim();
        if !trimmed.is_empty() {
            dirs.push(PathBuf::from(trimmed).join(".beads"));
        }
    }
    dirs.push(per_user_tmp_dir());
    dirs
}

fn expected_daemon_version(expected_version: Option<&str>) -> &str {
    expected_version.unwrap_or(env!("CARGO_PKG_VERSION"))
}

// =============================================================================
// Client - Send requests to daemon
// =============================================================================

#[derive(Clone, Debug)]
pub struct IpcClient {
    socket: PathBuf,
    autostart: bool,
    autostart_program: Option<PathBuf>,
    autostart_args: Vec<OsString>,
    expected_version: Option<String>,
}

impl IpcClient {
    pub fn new() -> Self {
        Self {
            socket: socket_path(),
            autostart: true,
            autostart_program: None,
            autostart_args: Vec::new(),
            expected_version: None,
        }
    }

    pub fn for_socket_path(socket: PathBuf) -> Self {
        Self {
            socket,
            autostart: true,
            autostart_program: None,
            autostart_args: Vec::new(),
            expected_version: None,
        }
    }

    pub fn for_runtime_dir(runtime_dir: &Path) -> Self {
        Self::for_socket_path(socket_path_for_runtime_dir(runtime_dir))
    }

    pub fn with_autostart(mut self, autostart: bool) -> Self {
        self.autostart = autostart;
        self
    }

    /// Override the program/args used for autostart.
    /// Default spawns `bn daemon run`.
    ///
    /// Override programs are treated as launcher-compatible wrappers: the
    /// client waits for the socket instead of assuming the process itself must
    /// stay alive until the daemon is ready.
    pub fn with_autostart_program(mut self, program: PathBuf, args: Vec<OsString>) -> Self {
        self.autostart_program = Some(program);
        self.autostart_args = args;
        self
    }

    /// Allow callers to relax or pin the version check.
    /// Default uses env!("CARGO_PKG_VERSION").
    pub fn with_expected_daemon_version(mut self, version: Option<String>) -> Self {
        self.expected_version = version;
        self
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket
    }

    pub fn connect(&self) -> Result<IpcConnection, IpcError> {
        IpcConnection::connect_with_options(
            self.socket.clone(),
            self.autostart,
            self.autostart_program.as_deref(),
            &self.autostart_args,
            self.expected_version.as_deref(),
        )
    }

    pub fn send_request(&self, req: &Request) -> Result<Response, IpcError> {
        if self.autostart {
            send_request_at_with_options(
                &self.socket,
                req,
                self.autostart_program.as_deref(),
                &self.autostart_args,
                self.expected_version.as_deref(),
            )
        } else {
            send_request_no_autostart_at_with_expected_version(
                &self.socket,
                req,
                self.expected_version.as_deref(),
            )
        }
    }

    pub fn send_request_no_autostart(&self, req: &Request) -> Result<Response, IpcError> {
        send_request_no_autostart_at_with_expected_version(
            &self.socket,
            req,
            self.expected_version.as_deref(),
        )
    }

    pub fn subscribe_stream(&self, req: &Request) -> Result<SubscriptionStream, IpcError> {
        if self.autostart {
            subscribe_stream_at_with_options(
                &self.socket,
                req,
                self.autostart_program.as_deref(),
                &self.autostart_args,
                self.expected_version.as_deref(),
            )
        } else {
            subscribe_stream_no_autostart_at_with_expected_version(
                &self.socket,
                req,
                self.expected_version.as_deref(),
            )
        }
    }

    pub fn wait_for_daemon_ready(&self, expected_version: &str) -> Result<(), IpcError> {
        wait_for_daemon_ready_at(&self.socket, expected_version)
    }
}

impl Default for IpcClient {
    fn default() -> Self {
        Self::new()
    }
}

pub struct IpcConnection {
    writer: UnixStream,
    reader: BufReader<UnixStream>,
}

impl IpcConnection {
    pub fn connect(socket: PathBuf, autostart: bool) -> Result<Self, IpcError> {
        Self::connect_with_options(socket, autostart, None, &[], None)
    }

    pub fn connect_with_options(
        socket: PathBuf,
        autostart: bool,
        autostart_program: Option<&Path>,
        autostart_args: &[OsString],
        expected_version: Option<&str>,
    ) -> Result<Self, IpcError> {
        const MAX_ATTEMPTS: u32 = 3;

        for attempt in 1..=MAX_ATTEMPTS {
            let stream = if autostart {
                connect_with_autostart(&socket, autostart_program, autostart_args)
            } else {
                UnixStream::connect(&socket).map_err(|source| IpcError::Transport { source })
            };
            let stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    if attempt >= MAX_ATTEMPTS {
                        return Err(e);
                    }
                    let backoff = Duration::from_millis(100 * (1 << (attempt - 1)));
                    std::thread::sleep(backoff);
                    continue;
                }
            };

            let mut conn = IpcConnection::new(stream)?;
            match verify_daemon_version(
                &socket,
                &mut conn.writer,
                &mut conn.reader,
                expected_version,
            ) {
                Ok(()) => return Ok(conn),
                Err(IpcError::DaemonVersionMismatch { daemon, .. }) if attempt < MAX_ATTEMPTS => {
                    tracing::info!(
                        "daemon version mismatch, restarting (attempt {}/{})",
                        attempt,
                        MAX_ATTEMPTS
                    );
                    if let Some(info) = daemon {
                        let _ = kill_daemon_forcefully(info.pid, &socket);
                    } else {
                        let _ = try_restart_daemon_by_socket(&socket);
                    }
                    let backoff = Duration::from_millis(100 * (1 << (attempt - 1)));
                    std::thread::sleep(backoff);
                }
                Err(IpcError::DaemonUnavailable(ref msg)) if attempt < MAX_ATTEMPTS => {
                    tracing::debug!("daemon unavailable ({}), retrying", msg);
                    let _ = try_restart_daemon_by_socket(&socket);
                    let backoff = Duration::from_millis(100 * (1 << (attempt - 1)));
                    std::thread::sleep(backoff);
                }
                Err(e) => return Err(e),
            }
        }

        Err(IpcError::DaemonUnavailable(
            "max retry attempts exceeded".into(),
        ))
    }

    fn new(stream: UnixStream) -> Result<Self, IpcError> {
        let reader_stream = stream
            .try_clone()
            .map_err(|source| IpcError::Transport { source })?;
        Ok(Self {
            writer: stream,
            reader: BufReader::new(reader_stream),
        })
    }

    pub fn send_request(&mut self, req: &Request) -> Result<Response, IpcError> {
        write_req_line(&mut self.writer, req)?;
        read_resp_line(&mut self.reader)
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> Result<(), IpcError> {
        self.reader
            .get_ref()
            .set_read_timeout(timeout)
            .map_err(|source| IpcError::Transport { source })?;
        Ok(())
    }

    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> Result<(), IpcError> {
        self.writer
            .set_write_timeout(timeout)
            .map_err(|source| IpcError::Transport { source })?;
        Ok(())
    }
}

fn should_autostart(err: &std::io::Error) -> bool {
    matches!(
        err.kind(),
        std::io::ErrorKind::NotFound
            | std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
    )
}

const AUTOSTART_CONNECT_TIMEOUT_SECS: u64 = 30;
const AUTOSTART_CONNECT_TIMEOUT_TEST_FAST_SECS: u64 = 5;
const AUTOSTART_LOCK_STALE_GRACE_SECS: u64 = 5;

fn autostart_connect_timeout() -> Duration {
    if std::env::var_os("BD_TEST_FAST").is_some() {
        Duration::from_secs(AUTOSTART_CONNECT_TIMEOUT_TEST_FAST_SECS)
    } else {
        Duration::from_secs(AUTOSTART_CONNECT_TIMEOUT_SECS)
    }
}

fn autostart_lock_stale_age(connect_timeout: Duration) -> Duration {
    connect_timeout.saturating_add(Duration::from_secs(AUTOSTART_LOCK_STALE_GRACE_SECS))
}

fn maybe_remove_stale_lock(lock_path: &PathBuf, max_age: Duration) {
    if let Ok(meta) = fs::metadata(lock_path)
        && let Ok(modified) = meta.modified()
        && let Ok(age) = modified.elapsed()
        && age > max_age
    {
        let _ = fs::remove_file(lock_path);
    }
}

#[derive(Clone, Copy)]
enum AutostartExitPolicy {
    FailFastOnDirectExit,
    WaitForSocket,
}

#[derive(Clone)]
struct AutostartCommand {
    program: PathBuf,
    args: Vec<OsString>,
    exit_policy: AutostartExitPolicy,
}

impl AutostartCommand {
    fn direct(program: PathBuf, args: Vec<OsString>) -> Self {
        Self {
            program,
            args,
            exit_policy: AutostartExitPolicy::FailFastOnDirectExit,
        }
    }

    fn launcher_compatible(program: PathBuf, args: Vec<OsString>) -> Self {
        Self {
            program,
            args,
            exit_policy: AutostartExitPolicy::WaitForSocket,
        }
    }
}

fn daemon_command() -> AutostartCommand {
    if let Ok(exe) = std::env::current_exe() {
        return AutostartCommand::direct(
            exe,
            vec![OsString::from("daemon"), OsString::from("run")],
        );
    }

    AutostartCommand::direct(
        PathBuf::from("bn"),
        vec![OsString::from("daemon"), OsString::from("run")],
    )
}

fn daemon_command_override(program: Option<&Path>, args: &[OsString]) -> AutostartCommand {
    if let Some(program) = program {
        return AutostartCommand::launcher_compatible(program.to_path_buf(), args.to_vec());
    }

    daemon_command()
}

fn connect_with_autostart(
    socket: &PathBuf,
    autostart_program: Option<&Path>,
    autostart_args: &[OsString],
) -> Result<UnixStream, IpcError> {
    let autostart = daemon_command_override(autostart_program, autostart_args);
    connect_with_autostart_command_with_timeout(socket, &autostart, autostart_connect_timeout())
}

fn connect_with_autostart_command_with_timeout(
    socket: &PathBuf,
    autostart: &AutostartCommand,
    connect_timeout: Duration,
) -> Result<UnixStream, IpcError> {
    match UnixStream::connect(socket) {
        Ok(stream) => Ok(stream),
        Err(e) if should_autostart(&e) => {
            // Try to autostart daemon with a simple lock to avoid herds.
            let stale_lock_age = autostart_lock_stale_age(connect_timeout);
            let dir = socket.parent().ok_or_else(|| {
                IpcError::DaemonUnavailable(format!(
                    "socket path {} has no parent directory",
                    socket.display()
                ))
            })?;
            ensure_autostart_socket_parent(socket, dir)
                .map_err(|source| IpcError::Transport { source })?;
            let lock_path = dir.join("daemon.lock");
            maybe_remove_stale_lock(&lock_path, stale_lock_age);

            let mut we_spawned = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
                .is_ok();
            let mut spawned_child = None;

            if we_spawned {
                spawned_child = Some(spawn_autostart_command(autostart)?);
            }

            let deadline = SystemTime::now() + connect_timeout;
            let mut backoff = Duration::from_millis(50);

            loop {
                match UnixStream::connect(socket) {
                    Ok(stream) => {
                        if we_spawned {
                            let _ = fs::remove_file(&lock_path);
                        }
                        return Ok(stream);
                    }
                    Err(e) if should_autostart(&e) => {
                        if matches!(
                            autostart.exit_policy,
                            AutostartExitPolicy::FailFastOnDirectExit
                        ) && let Some(message) =
                            take_spawned_child_exit_message(spawned_child.as_mut(), socket)?
                        {
                            if we_spawned {
                                let _ = fs::remove_file(&lock_path);
                            }
                            return Err(IpcError::DaemonUnavailable(message));
                        }
                        if !we_spawned {
                            // If the lock disappeared (spawner died), try to take over.
                            maybe_remove_stale_lock(&lock_path, stale_lock_age);
                            if OpenOptions::new()
                                .write(true)
                                .create_new(true)
                                .open(&lock_path)
                                .is_ok()
                            {
                                we_spawned = true;
                                match spawn_autostart_command(autostart) {
                                    Ok(child) => {
                                        spawned_child = Some(child);
                                    }
                                    Err(e) => {
                                        let _ = fs::remove_file(&lock_path);
                                        return Err(IpcError::DaemonUnavailable(format!(
                                            "failed to spawn daemon: {}",
                                            e
                                        )));
                                    }
                                }
                            }
                        }
                        if SystemTime::now() >= deadline {
                            if we_spawned {
                                let _ = fs::remove_file(&lock_path);
                            }
                            return Err(IpcError::DaemonUnavailable(format!(
                                "timed out waiting for daemon socket after {}s",
                                connect_timeout.as_secs()
                            )));
                        }
                        std::thread::sleep(backoff);
                        backoff = std::cmp::min(backoff * 2, Duration::from_millis(200));
                    }
                    Err(e) => {
                        if we_spawned {
                            let _ = fs::remove_file(&lock_path);
                        }
                        return Err(IpcError::Transport { source: e });
                    }
                }
            }
        }
        Err(e) => Err(IpcError::Transport { source: e }),
    }
}

fn spawn_autostart_command(command: &AutostartCommand) -> Result<SpawnedDaemon, IpcError> {
    spawn_daemon_process(&command.program, &command.args)
        .map_err(|e| IpcError::DaemonUnavailable(format!("failed to spawn daemon: {e}")))
}

fn take_spawned_child_exit_message(
    child: Option<&mut SpawnedDaemon>,
    socket: &Path,
) -> Result<Option<String>, IpcError> {
    let Some(child) = child else {
        return Ok(None);
    };
    let Some(status) = child.try_wait().map_err(|err| {
        IpcError::DaemonUnavailable(format!("failed to monitor spawned daemon: {err}"))
    })?
    else {
        return Ok(None);
    };
    let _ = fs::remove_file(socket);
    let _ = fs::remove_file(socket.with_file_name("daemon.meta.json"));
    Ok(Some(format!(
        "daemon exited before socket ready at {} ({})",
        socket.display(),
        format_exit_status(status)
    )))
}

fn format_exit_status(status: std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;

        if let Some(code) = status.code() {
            return format!("exit status {code}");
        }
        if let Some(signal) = status.signal() {
            return format!("signal {signal}");
        }
    }

    match status.code() {
        Some(code) => format!("exit status {code}"),
        None => status.to_string(),
    }
}

#[cfg(test)]
fn connect_with_autostart_for_test(
    socket: &PathBuf,
    autostart_program: Option<&Path>,
    autostart_args: &[OsString],
    connect_timeout: Duration,
) -> Result<UnixStream, IpcError> {
    let autostart = daemon_command_override(autostart_program, autostart_args);
    connect_with_autostart_command_with_timeout(socket, &autostart, connect_timeout)
}

#[cfg(test)]
fn connect_with_direct_autostart_for_test(
    socket: &PathBuf,
    program: &Path,
    args: &[OsString],
    connect_timeout: Duration,
) -> Result<UnixStream, IpcError> {
    let autostart = AutostartCommand::direct(program.to_path_buf(), args.to_vec());
    connect_with_autostart_command_with_timeout(socket, &autostart, connect_timeout)
}

fn write_req_line(stream: &mut UnixStream, req: &Request) -> Result<(), IpcError> {
    let mut json =
        serde_json::to_string(req).map_err(|source| IpcError::PayloadEncode { source })?;
    json.push('\n');
    stream
        .write_all(json.as_bytes())
        .map_err(|source| IpcError::Transport { source })?;
    Ok(())
}

fn read_resp_line(reader: &mut BufReader<UnixStream>) -> Result<Response, IpcError> {
    let mut line = String::new();
    let bytes_read = reader
        .read_line(&mut line)
        .map_err(|source| IpcError::Transport { source })?;
    // EOF means daemon closed connection (likely just shut down)
    if bytes_read == 0 || line.trim().is_empty() {
        return Err(IpcError::DaemonUnavailable(
            "daemon not running (stale socket)".into(),
        ));
    }
    serde_json::from_str(&line).map_err(|source| IpcError::PayloadDecode { source })
}

/// Read response line, converting parse errors to version mismatch.
///
/// Used during version verification where a parse failure likely indicates
/// an incompatible daemon version.
fn read_resp_line_version_check(
    reader: &mut BufReader<UnixStream>,
    expected_version: Option<&str>,
) -> Result<Response, IpcError> {
    let mut line = String::new();
    let bytes_read = reader
        .read_line(&mut line)
        .map_err(|source| IpcError::Transport { source })?;
    if bytes_read == 0 || line.trim().is_empty() {
        return Err(IpcError::DaemonUnavailable(
            "daemon not running (stale socket)".into(),
        ));
    }
    let expected_version = expected_daemon_version(expected_version);
    serde_json::from_str(&line).map_err(|e| IpcError::DaemonVersionMismatch {
        daemon: None,
        client_version: expected_version.to_string(),
        protocol_version: IPC_PROTOCOL_VERSION,
        parse_error: Some(e.to_string()),
    })
}

fn verify_daemon_version(
    socket: &Path,
    writer: &mut UnixStream,
    reader: &mut BufReader<UnixStream>,
    expected_version: Option<&str>,
) -> Result<(), IpcError> {
    let expected_version = expected_daemon_version(expected_version);
    if let Some(meta) = read_daemon_meta_at(socket)
        && daemon_pid_alive(meta.pid)
    {
        if meta.protocol_version == IPC_PROTOCOL_VERSION && meta.version == expected_version {
            return Ok(());
        }
        return Err(IpcError::DaemonVersionMismatch {
            daemon: Some(meta),
            client_version: expected_version.to_string(),
            protocol_version: IPC_PROTOCOL_VERSION,
            parse_error: None,
        });
    }

    write_req_line(writer, &Request::Ping)?;
    // Use version-check variant that converts parse errors to version mismatch
    let resp = read_resp_line_version_check(reader, Some(expected_version))?;
    let Response::Ok { ok } = resp else {
        return Err(IpcError::DaemonVersionMismatch {
            daemon: None,
            client_version: expected_version.to_string(),
            protocol_version: IPC_PROTOCOL_VERSION,
            parse_error: None,
        });
    };

    let ResponsePayload::Query(QueryResult::DaemonInfo(info)) = ok else {
        return Err(IpcError::DaemonVersionMismatch {
            daemon: None,
            client_version: expected_version.to_string(),
            protocol_version: IPC_PROTOCOL_VERSION,
            parse_error: Some("unexpected response payload type".into()),
        });
    };

    if info.protocol_version != IPC_PROTOCOL_VERSION || info.version != expected_version {
        return Err(IpcError::DaemonVersionMismatch {
            daemon: Some(info),
            client_version: expected_version.to_string(),
            protocol_version: IPC_PROTOCOL_VERSION,
            parse_error: None,
        });
    }

    Ok(())
}

fn send_request_over_stream(
    stream: UnixStream,
    socket: &Path,
    req: &Request,
    expected_version: Option<&str>,
) -> Result<Response, IpcError> {
    let mut writer = stream;
    let reader_stream = writer
        .try_clone()
        .map_err(|source| IpcError::Transport { source })?;
    let mut reader = BufReader::new(reader_stream);

    // Verify daemon version/protocol once per connection.
    if !matches!(req, Request::Ping) {
        verify_daemon_version(socket, &mut writer, &mut reader, expected_version)?;
    }

    write_req_line(&mut writer, req)?;
    read_resp_line(&mut reader)
}

/// Send a request to the daemon and receive a response.
///
/// Retries up to 3 times on version mismatch or stale socket errors,
/// with exponential backoff between attempts.
pub fn send_request_at(socket: &PathBuf, req: &Request) -> Result<Response, IpcError> {
    send_request_at_with_options(socket, req, None, &[], None)
}

fn send_request_at_with_options(
    socket: &PathBuf,
    req: &Request,
    autostart_program: Option<&Path>,
    autostart_args: &[OsString],
    expected_version: Option<&str>,
) -> Result<Response, IpcError> {
    const MAX_ATTEMPTS: u32 = 3;

    for attempt in 1..=MAX_ATTEMPTS {
        let stream = match connect_with_autostart(socket, autostart_program, autostart_args) {
            Ok(s) => s,
            Err(e) => {
                if attempt >= MAX_ATTEMPTS {
                    return Err(e);
                }
                let backoff = Duration::from_millis(100 * (1 << (attempt - 1)));
                std::thread::sleep(backoff);
                continue;
            }
        };

        match send_request_over_stream(stream, socket, req, expected_version) {
            Ok(resp) => return Ok(resp),
            Err(IpcError::DaemonVersionMismatch { daemon, .. }) if attempt < MAX_ATTEMPTS => {
                tracing::info!(
                    "daemon version mismatch, restarting (attempt {}/{})",
                    attempt,
                    MAX_ATTEMPTS
                );

                // Try to restart the daemon
                if let Some(info) = daemon {
                    let _ = kill_daemon_forcefully(info.pid, socket);
                } else {
                    let _ = try_restart_daemon_by_socket(socket);
                }

                // Exponential backoff: 100ms, 200ms, 400ms
                let backoff = Duration::from_millis(100 * (1 << (attempt - 1)));
                std::thread::sleep(backoff);
            }
            Err(IpcError::DaemonUnavailable(ref msg)) if attempt < MAX_ATTEMPTS => {
                tracing::debug!("daemon unavailable ({}), retrying", msg);
                // Socket might have gone stale mid-request
                let _ = try_restart_daemon_by_socket(socket);
                let backoff = Duration::from_millis(100 * (1 << (attempt - 1)));
                std::thread::sleep(backoff);
            }
            Err(e) => return Err(e),
        }
    }

    Err(IpcError::DaemonUnavailable(
        "max retry attempts exceeded".into(),
    ))
}

pub fn send_request(req: &Request) -> Result<Response, IpcError> {
    let socket = socket_path();
    send_request_at(&socket, req)
}

/// Send a request without auto-starting the daemon.
///
/// Returns `DaemonUnavailable` if daemon is not running.
pub fn send_request_no_autostart_at(socket: &PathBuf, req: &Request) -> Result<Response, IpcError> {
    send_request_no_autostart_at_with_expected_version(socket, req, None)
}

fn send_request_no_autostart_at_with_expected_version(
    socket: &PathBuf,
    req: &Request,
    expected_version: Option<&str>,
) -> Result<Response, IpcError> {
    let stream = UnixStream::connect(socket)
        .map_err(|e| IpcError::DaemonUnavailable(format!("daemon not running: {}", e)))?;
    send_request_over_stream(stream, socket, req, expected_version)
}

pub fn send_request_no_autostart(req: &Request) -> Result<Response, IpcError> {
    let socket = socket_path();
    send_request_no_autostart_at(&socket, req)
}

/// Stream responses for a subscribe request.
pub struct SubscriptionStream {
    _writer: UnixStream,
    reader: BufReader<UnixStream>,
}

impl SubscriptionStream {
    pub fn read_response(&mut self) -> Result<Option<Response>, IpcError> {
        let mut line = String::new();
        let bytes_read = self
            .reader
            .read_line(&mut line)
            .map_err(|source| IpcError::Transport { source })?;
        if bytes_read == 0 || line.trim().is_empty() {
            return Ok(None);
        }
        let response =
            serde_json::from_str(&line).map_err(|source| IpcError::PayloadDecode { source })?;
        Ok(Some(response))
    }

    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> Result<(), IpcError> {
        self.reader
            .get_ref()
            .set_read_timeout(timeout)
            .map_err(|source| IpcError::Transport { source })?;
        Ok(())
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> Result<(), IpcError> {
        self.reader
            .get_ref()
            .set_nonblocking(nonblocking)
            .map_err(|source| IpcError::Transport { source })?;
        Ok(())
    }
}

/// Send a subscribe request and return a stream of responses.
pub fn subscribe_stream_at(
    socket: &PathBuf,
    req: &Request,
) -> Result<SubscriptionStream, IpcError> {
    subscribe_stream_at_with_options(socket, req, None, &[], None)
}

fn subscribe_stream_at_with_options(
    socket: &PathBuf,
    req: &Request,
    autostart_program: Option<&Path>,
    autostart_args: &[OsString],
    expected_version: Option<&str>,
) -> Result<SubscriptionStream, IpcError> {
    if !matches!(req, Request::Subscribe { .. }) {
        return Err(IpcError::InvalidRequest {
            field: Some("op".into()),
            reason: "subscribe_stream expects subscribe request".into(),
        });
    }

    const MAX_ATTEMPTS: u32 = 3;
    for attempt in 1..=MAX_ATTEMPTS {
        let stream = match connect_with_autostart(socket, autostart_program, autostart_args) {
            Ok(s) => s,
            Err(e) => {
                if attempt >= MAX_ATTEMPTS {
                    return Err(e);
                }
                let backoff = Duration::from_millis(100 * (1 << (attempt - 1)));
                std::thread::sleep(backoff);
                continue;
            }
        };

        let mut writer = stream;
        let reader_stream = writer
            .try_clone()
            .map_err(|source| IpcError::Transport { source })?;
        let mut reader = BufReader::new(reader_stream);

        if let Err(err) = verify_daemon_version(socket, &mut writer, &mut reader, expected_version)
        {
            match err {
                IpcError::DaemonVersionMismatch { daemon, .. } if attempt < MAX_ATTEMPTS => {
                    tracing::info!(
                        "daemon version mismatch, restarting (attempt {}/{})",
                        attempt,
                        MAX_ATTEMPTS
                    );
                    if let Some(info) = daemon {
                        let _ = kill_daemon_forcefully(info.pid, socket);
                    } else {
                        let _ = try_restart_daemon_by_socket(socket);
                    }
                    let backoff = Duration::from_millis(100 * (1 << (attempt - 1)));
                    std::thread::sleep(backoff);
                    continue;
                }
                IpcError::PayloadDecode { source: _ }
                | IpcError::PayloadEncode { source: _ }
                | IpcError::Transport { .. }
                | IpcError::InvalidId(_)
                | IpcError::InvalidRequest { .. }
                | IpcError::Disconnected
                | IpcError::DaemonUnavailable(_)
                | IpcError::DaemonVersionMismatch { .. }
                | IpcError::FrameTooLarge { .. } => return Err(err),
            }
        }

        write_req_line(&mut writer, req)?;
        return Ok(SubscriptionStream {
            _writer: writer,
            reader,
        });
    }

    Err(IpcError::DaemonUnavailable(
        "max retry attempts exceeded".into(),
    ))
}

/// Send a subscribe request without auto-starting the daemon.
pub fn subscribe_stream_no_autostart_at(
    socket: &PathBuf,
    req: &Request,
) -> Result<SubscriptionStream, IpcError> {
    subscribe_stream_no_autostart_at_with_expected_version(socket, req, None)
}

fn subscribe_stream_no_autostart_at_with_expected_version(
    socket: &PathBuf,
    req: &Request,
    expected_version: Option<&str>,
) -> Result<SubscriptionStream, IpcError> {
    if !matches!(req, Request::Subscribe { .. }) {
        return Err(IpcError::InvalidRequest {
            field: Some("op".into()),
            reason: "subscribe_stream expects subscribe request".into(),
        });
    }

    let stream = UnixStream::connect(socket)
        .map_err(|e| IpcError::DaemonUnavailable(format!("daemon not running: {}", e)))?;
    let mut writer = stream;
    let reader_stream = writer
        .try_clone()
        .map_err(|source| IpcError::Transport { source })?;
    let mut reader = BufReader::new(reader_stream);
    verify_daemon_version(socket, &mut writer, &mut reader, expected_version)?;
    write_req_line(&mut writer, req)?;
    Ok(SubscriptionStream {
        _writer: writer,
        reader,
    })
}

/// Send a subscribe request and return a stream of responses.
pub fn subscribe_stream(req: &Request) -> Result<SubscriptionStream, IpcError> {
    let socket = socket_path();
    subscribe_stream_at(&socket, req)
}

/// Wait for daemon to be ready and responding with expected version.
///
/// Returns Ok if daemon is responsive with matching version, Err on timeout (30s).
pub fn wait_for_daemon_ready(expected_version: &str) -> Result<(), IpcError> {
    let socket = socket_path();
    wait_for_daemon_ready_at(&socket, expected_version)
}

/// Wait for daemon to be ready and responding with expected version.
pub fn wait_for_daemon_ready_at(socket: &PathBuf, expected_version: &str) -> Result<(), IpcError> {
    let deadline = SystemTime::now() + Duration::from_secs(30);
    let mut backoff = Duration::from_millis(50);

    while SystemTime::now() < deadline {
        match UnixStream::connect(socket) {
            Ok(stream) => {
                let mut writer = stream;
                let reader_stream = match writer.try_clone() {
                    Ok(s) => s,
                    Err(_) => {
                        std::thread::sleep(backoff);
                        backoff = std::cmp::min(backoff * 2, Duration::from_millis(500));
                        continue;
                    }
                };
                let mut reader = BufReader::new(reader_stream);

                if write_req_line(&mut writer, &Request::Ping).is_err() {
                    std::thread::sleep(backoff);
                    backoff = std::cmp::min(backoff * 2, Duration::from_millis(500));
                    continue;
                }

                if let Ok(Response::Ok {
                    ok: ResponsePayload::Query(QueryResult::DaemonInfo(info)),
                }) = read_resp_line(&mut reader)
                {
                    if info.version == expected_version {
                        tracing::info!("daemon ready with version {}", expected_version);
                        return Ok(());
                    }
                    // Wrong version - old daemon hasn't fully died yet
                    tracing::debug!(
                        "daemon has version {}, waiting for {}",
                        info.version,
                        expected_version
                    );
                }
                std::thread::sleep(backoff);
                backoff = std::cmp::min(backoff * 2, Duration::from_millis(500));
            }
            Err(_) => {
                std::thread::sleep(backoff);
                backoff = std::cmp::min(backoff * 2, Duration::from_millis(200));
            }
        }
    }

    Err(IpcError::DaemonUnavailable(format!(
        "timed out waiting for daemon version {}",
        expected_version
    )))
}

/// Kill daemon with SIGTERM, escalating to SIGKILL if needed.
#[cfg(unix)]
fn kill_daemon_forcefully(pid: u32, socket: &PathBuf) -> Result<(), IpcError> {
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    let nix_pid = Pid::from_raw(pid as i32);

    // First try SIGTERM (graceful)
    if let Err(e) = kill(nix_pid, Signal::SIGTERM) {
        // ESRCH = no such process - already dead, that's fine
        if e == nix::errno::Errno::ESRCH {
            let _ = fs::remove_file(socket);
            let _ = fs::remove_file(socket.with_file_name("daemon.meta.json"));
            return Ok(());
        }
        return Err(IpcError::DaemonUnavailable(format!(
            "failed to stop daemon pid {pid}: {e}"
        )));
    }

    // Wait for graceful shutdown (3 seconds)
    let deadline = SystemTime::now() + Duration::from_secs(3);
    while SystemTime::now() < deadline {
        if UnixStream::connect(socket).is_err() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    // Escalate to SIGKILL
    tracing::warn!(
        "daemon pid {} did not stop gracefully, sending SIGKILL",
        pid
    );
    if let Err(e) = kill(nix_pid, Signal::SIGKILL)
        && e != nix::errno::Errno::ESRCH
    {
        return Err(IpcError::DaemonUnavailable(format!(
            "failed to kill daemon pid {pid}: {e}"
        )));
    }

    // Wait for socket to become stale (2 more seconds)
    let deadline = SystemTime::now() + Duration::from_secs(2);
    while SystemTime::now() < deadline {
        if UnixStream::connect(socket).is_err() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    // Force remove socket and meta as last resort
    let _ = fs::remove_file(socket);
    let _ = fs::remove_file(socket.with_file_name("daemon.meta.json"));
    Ok(())
}

/// Windows has no SIGTERM; terminate the process directly, then clean up the
/// socket/meta files. Behaviour parity: best-effort stop, missing process is Ok.
#[cfg(windows)]
fn kill_daemon_forcefully(pid: u32, socket: &PathBuf) -> Result<(), IpcError> {
    if let Err(e) = super::proc_util::terminate(pid) {
        return Err(IpcError::DaemonUnavailable(format!(
            "failed to stop daemon pid {pid}: {e}"
        )));
    }

    // Wait for the socket to become stale (up to 5 seconds).
    let deadline = SystemTime::now() + Duration::from_secs(5);
    while SystemTime::now() < deadline {
        if UnixStream::connect(socket).is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    let _ = fs::remove_file(socket);
    let _ = fs::remove_file(socket.with_file_name("daemon.meta.json"));
    Ok(())
}

/// Try to restart daemon when we don't have the PID from response.
///
/// Uses daemon.meta.json to find PID if available.
fn try_restart_daemon_by_socket(socket: &PathBuf) -> Result<(), IpcError> {
    // Try to get PID from meta file first
    if let Some(meta) = read_daemon_meta_at(socket) {
        tracing::debug!("found daemon pid {} from meta file", meta.pid);
        return kill_daemon_forcefully(meta.pid, socket);
    }

    // No meta file (very old daemon or corrupt state)
    tracing::warn!("no daemon meta file found, removing stale socket");

    // Best effort: remove socket file. The orphaned daemon will eventually
    // exit when it has no clients and no work.
    if let Err(e) = fs::remove_file(socket)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        return Err(IpcError::DaemonUnavailable(format!(
            "failed to remove stale socket: {e}"
        )));
    }

    // Also remove meta file if present
    let _ = fs::remove_file(socket.with_file_name("daemon.meta.json"));

    Ok(())
}

// =============================================================================
// Daemon lifecycle — the supported way to bounce the daemon
// =============================================================================
//
// Everything below is a thin, PUBLIC wrapper over the kill/restart machinery
// that already sat privately in this file, reachable only from the IPC retry
// path. No new stopping mechanism was invented here — `kill_daemon_forcefully`
// and `try_restart_daemon_by_socket` still own the actual behaviour, so the
// retry path and `bn daemon stop` cannot drift apart.
//
// Why this exists at all: `bn sync` once hung and there was no supported way to
// bounce the daemon. The only recovery was `pkill -f "bn daemon run"` and hope
// the next command autostarts a clean one. For a user's own issue tracker, that
// is not an acceptable recovery path, so the mechanism got a front door.

/// What [`stop_daemon_at`] actually did, so the caller can tell the user straight.
///
/// All three outcomes are `Ok`. Stopping a daemon that is already gone is not a
/// failure — it is the end state you asked for, so it is success. This matters
/// more than it sounds: scripts and session-close hooks call stop blind, and if
/// "already stopped" were an error everybody would just append `|| true` and
/// throw away the real errors along with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaemonStopOutcome {
    /// Got a live one and now it's gone. `pid` is the process we signalled.
    Stopped { pid: u32 },
    /// Nothing alive, but got leftover rubbish — a socket file, a meta file, or
    /// both — from a daemon that died badly. We cleared it. `pid` is the dead
    /// pid the meta file named, `None` when no meta file was there to name one.
    StaleCleaned { pid: Option<u32> },
    /// Nothing running, nothing to clean. Already clean slate.
    NotRunning,
}

impl DaemonStopOutcome {
    /// The pid this outcome talks about, if any. `Stopped` always has one, a
    /// `StaleCleaned` may (it read the dead pid out of the meta file),
    /// `NotRunning` never does.
    pub fn pid(&self) -> Option<u32> {
        match self {
            DaemonStopOutcome::Stopped { pid } => Some(*pid),
            DaemonStopOutcome::StaleCleaned { pid } => *pid,
            DaemonStopOutcome::NotRunning => None,
        }
    }

    /// True only when a live daemon got stopped by this call. Use this, not
    /// `pid().is_some()` — a `StaleCleaned` also carries a pid, but that one
    /// was already dead before we showed up.
    pub fn stopped_a_live_daemon(&self) -> bool {
        matches!(self, DaemonStopOutcome::Stopped { .. })
    }
}

/// Stop whatever daemon is serving `socket`. Idempotent — see [`DaemonStopOutcome`].
///
/// # The contract: when this returns Ok, the process is GONE
///
/// Not "the socket stopped answering" — gone, reaped, holding nothing. See
/// [`wait_for_daemon_pid_gone`] for the bug that taught us the difference. Any
/// caller may immediately start a replacement and expect it to take the store
/// lock. If we cannot make that true, this returns `Err` rather than lie.
///
/// # Where the pid come from, and the one caveat
///
/// From `daemon.meta.json` sitting next to the socket, which is what the daemon
/// itself wrote at startup. Same source the IPC retry path has always trusted.
/// The caveat, stated plainly because it is a real (if small) hazard: if a
/// daemon died without cleaning up **and** the OS recycled its pid onto some
/// unrelated process, we would signal that innocent process. Cannot rule this
/// out portably — Linux `/proc/<pid>/cmdline` is no help on Windows — the
/// window is tiny, and the exposure is exactly what it already was before this
/// function existed. If it ever bites, the fix is for the daemon to write
/// something unforgeable into the meta (exe path + process start time) and for
/// this function to check it before signalling.
///
/// # Why we don't ping the socket first
///
/// Pinging would give us the pid straight from the horse's mouth, no reuse
/// hazard. But a *hung* daemon accepts the connection and then never answers —
/// and a hung daemon is precisely the state this command exists to rescue. A
/// ping-first design would therefore hang in the one case that matters most.
/// So: meta file, no round trip.
pub fn stop_daemon_at(socket: &PathBuf) -> Result<DaemonStopOutcome, IpcError> {
    let meta_path = socket.with_file_name("daemon.meta.json");

    let Some(meta) = read_daemon_meta_at(socket) else {
        // No meta, but a socket (or an unreadable meta) still lying around:
        // that's a corpse. Clear it so the next autostart gets a clean run.
        if socket.exists() || meta_path.exists() {
            try_restart_daemon_by_socket(socket)?;
            return Ok(DaemonStopOutcome::StaleCleaned { pid: None });
        }
        return Ok(DaemonStopOutcome::NotRunning);
    };

    let was_alive = daemon_pid_alive(meta.pid);
    kill_daemon_forcefully(meta.pid, socket)?;
    wait_for_daemon_pid_gone(meta.pid)?;

    // `kill_daemon_forcefully` can leave `daemon.meta.json` behind. Clear both
    // once the process is confirmed dead, otherwise every later `stop` keeps
    // reporting "cleaned stale state" forever instead of the honest
    // "nothing running".
    let _ = fs::remove_file(socket);
    let _ = fs::remove_file(&meta_path);

    if was_alive {
        Ok(DaemonStopOutcome::Stopped { pid: meta.pid })
    } else {
        Ok(DaemonStopOutcome::StaleCleaned {
            pid: Some(meta.pid),
        })
    }
}

/// Grace period for the daemon process to actually exit after
/// `kill_daemon_forcefully` reports its socket has gone.
///
/// Provenance for the number, because a magic constant with no source is a bug
/// that hasn't fired yet: measured on this workspace (55-bead store), a healthy
/// daemon exits **54-60 ms** after SIGTERM — three runs, 60/54/54 ms. Ten
/// seconds is ~170x that, headroom for a much bigger store flushing its WAL and
/// checkpointing on the way out. Waiting is strictly better than killing early
/// here: SIGKILL mid-flush costs recovery work on the next open.
const DAEMON_GRACEFUL_EXIT_WAIT: Duration = Duration::from_secs(10);

/// How long we give it after SIGKILL before admitting defeat. SIGKILL cannot be
/// caught, so anything past a second or two means the process is stuck in
/// uninterruptible I/O, which no amount of extra waiting fixes.
const DAEMON_FORCED_EXIT_WAIT: Duration = Duration::from_secs(3);

/// Block until process `pid` is really gone, escalating to SIGKILL if it drags.
///
/// # Why this exists — a bug the dogfood run actually caught
///
/// `kill_daemon_forcefully` waits for the **socket** to stop accepting, not for
/// the **process** to exit. Those are not the same instant, and the gap is
/// where the damage lives: `bn daemon stop` reported `daemon stopped (pid
/// 109982)` while 109982 was still winding down, the very next command
/// autostarted a fresh daemon, and that one died with
/// `store lock already held for f81c87f3-… at …/store.lock`. The old process
/// hadn't released the store lock yet, because it hadn't finished exiting.
///
/// So `stop` has to mean *stopped*. Socket-gone is not good enough, and neither
/// is a "usually fine" sleep — the whole point of this command is to be the one
/// recovery path you can trust when the daemon is misbehaving.
///
/// Escalation is not paranoia either: during the same session an orphaned
/// daemon (one whose socket file had been removed out from under it) sat
/// through a SIGTERM for over ten seconds and only died to SIGKILL.
fn wait_for_daemon_pid_gone(pid: u32) -> Result<(), IpcError> {
    if wait_for_pid_exit(pid, DAEMON_GRACEFUL_EXIT_WAIT) {
        return Ok(());
    }

    tracing::warn!(
        "daemon pid {} still alive {:?} after stop, sending SIGKILL",
        pid,
        DAEMON_GRACEFUL_EXIT_WAIT
    );
    force_kill_pid(pid);

    if wait_for_pid_exit(pid, DAEMON_FORCED_EXIT_WAIT) {
        return Ok(());
    }

    // Refuse to claim success. A caller told "stopped" here would go on to
    // start a replacement that then loses the store lock — exactly the failure
    // this function was written to stop.
    Err(IpcError::DaemonUnavailable(format!(
        "daemon pid {pid} is still running after SIGTERM and SIGKILL; \
         it may be stuck in uninterruptible I/O"
    )))
}

/// Poll until `pid` is gone or `timeout` runs out. `true` means gone.
///
/// 25 ms poll against a ~55 ms typical exit: fine-grained enough that the
/// common case costs one or two polls, coarse enough not to spin the CPU while
/// waiting out a slow shutdown.
fn wait_for_pid_exit(pid: u32, timeout: Duration) -> bool {
    let deadline = SystemTime::now() + timeout;
    while SystemTime::now() < deadline {
        if !daemon_pid_alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    !daemon_pid_alive(pid)
}

/// The strongest "die now" the platform gives us. Best effort: a process that
/// already exited is the outcome we wanted, so a failed signal is not an error
/// here — the caller re-checks liveness rather than trusting this return.
fn force_kill_pid(pid: u32) {
    #[cfg(unix)]
    {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;
        let _ = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);
    }
    #[cfg(windows)]
    {
        // No signals on Windows; `terminate` is the TerminateProcess path
        // `kill_daemon_forcefully` already uses, so behaviour stays matched.
        let _ = super::proc_util::terminate(pid);
    }
}

/// [`stop_daemon_at`] on the default socket, i.e. what `bn daemon stop` calls.
pub fn stop_daemon() -> Result<DaemonStopOutcome, IpcError> {
    let socket = socket_path();
    stop_daemon_at(&socket)
}

/// Ask a running daemon who it is, without autostarting one.
///
/// `Ping` skips the version handshake (see `send_request_over_stream`), so this
/// still gets an answer out of a daemon whose version we would otherwise
/// reject. Falls back to the meta file when the daemon cannot be reached — the
/// answer is then second-hand, so treat it as "who was here", not "who is
/// serving".
fn live_daemon_info_at(socket: &PathBuf) -> Option<crate::api::DaemonInfo> {
    match send_request_no_autostart_at(socket, &Request::Ping) {
        Ok(Response::Ok {
            ok: ResponsePayload::Query(QueryResult::DaemonInfo(info)),
        }) => Some(info),
        _ => read_daemon_meta_at(socket),
    }
}

/// Make sure a daemon of *this* build's version is up and serving `socket`,
/// starting one if there is none, and only come back once it really answers.
///
/// The order of the two steps is the whole point:
///
/// 1. `connect()` is what triggers autostart — it spawns `<current_exe> daemon
///    run` (see `daemon_command`). Nothing else here starts a process.
/// 2. `wait_for_daemon_ready` is what makes the promise true. Without it we
///    would return while the child is still opening its store, and the very
///    next command would race a daemon that isn't serving yet. Fire-and-hope is
///    not good enough for a restart the user is watching.
///
/// Returns the identity of the daemon now serving — pid included, which is how
/// `bn daemon restart` can show the caller it genuinely bounced.
pub fn ensure_daemon_at(socket: &PathBuf) -> Result<crate::api::DaemonInfo, IpcError> {
    let expected = env!("CARGO_PKG_VERSION");
    let client = IpcClient::for_socket_path(socket.clone()).with_autostart(true);
    // Only wanted the autostart side effect; close the connection straight away.
    drop(client.connect()?);
    client.wait_for_daemon_ready(expected)?;
    live_daemon_info_at(socket).ok_or_else(|| {
        IpcError::DaemonUnavailable(format!(
            "daemon on {} is serving but would not say who it is",
            socket.display()
        ))
    })
}

/// [`ensure_daemon_at`] on the default socket.
pub fn ensure_daemon() -> Result<crate::api::DaemonInfo, IpcError> {
    let socket = socket_path();
    ensure_daemon_at(&socket)
}

/// Both halves of a restart, so the caller can prove the bounce really happened.
#[derive(Debug, Clone)]
pub struct DaemonRestartOutcome {
    /// What the stop half found and did. `NotRunning` here is fine — restart
    /// with nothing running just means start.
    pub stopped: DaemonStopOutcome,
    /// The daemon now serving, confirmed ready.
    pub started: crate::api::DaemonInfo,
}

impl DaemonRestartOutcome {
    /// pid of the daemon we killed, if there was one to kill.
    pub fn old_pid(&self) -> Option<u32> {
        self.stopped.pid()
    }

    /// pid of the daemon now serving.
    pub fn new_pid(&self) -> u32 {
        self.started.pid
    }
}

/// Stop then start the daemon on `socket`, returning only once the new one is
/// serving. If nothing was running, this is just a start — not an error.
pub fn restart_daemon_at(socket: &PathBuf) -> Result<DaemonRestartOutcome, IpcError> {
    let stopped = stop_daemon_at(socket)?;
    let started = ensure_daemon_at(socket)?;
    Ok(DaemonRestartOutcome { stopped, started })
}

/// [`restart_daemon_at`] on the default socket, i.e. what `bn daemon restart` calls.
pub fn restart_daemon() -> Result<DaemonRestartOutcome, IpcError> {
    let socket = socket_path();
    restart_daemon_at(&socket)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    #[test]
    fn version_check_uses_meta_when_matching() {
        let temp = TempDir::new().expect("temp dir");
        let socket = temp.path().join("daemon.sock");
        let meta = crate::api::DaemonInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_version: IPC_PROTOCOL_VERSION,
            pid: std::process::id(),
            started_at_ms: None,
        };
        let meta_path = socket.with_file_name("daemon.meta.json");
        std::fs::write(&meta_path, serde_json::to_vec(&meta).unwrap()).unwrap();

        let (stream, _peer) = UnixStream::pair().expect("socket pair");
        let mut writer = stream;
        let reader_stream = writer.try_clone().expect("clone stream");
        let mut reader = BufReader::new(reader_stream);
        verify_daemon_version(&socket, &mut writer, &mut reader, None).expect("meta match");
    }

    #[test]
    fn version_check_rejects_meta_mismatch() {
        let temp = TempDir::new().expect("temp dir");
        let socket = temp.path().join("daemon.sock");
        let meta = crate::api::DaemonInfo {
            version: "0.0.0-fake".to_string(),
            protocol_version: IPC_PROTOCOL_VERSION,
            pid: std::process::id(),
            started_at_ms: None,
        };
        let meta_path = socket.with_file_name("daemon.meta.json");
        std::fs::write(&meta_path, serde_json::to_vec(&meta).unwrap()).unwrap();

        let (stream, _peer) = UnixStream::pair().expect("socket pair");
        let mut writer = stream;
        let reader_stream = writer.try_clone().expect("clone stream");
        let mut reader = BufReader::new(reader_stream);
        let err = verify_daemon_version(&socket, &mut writer, &mut reader, None).unwrap_err();
        assert!(matches!(
            err,
            IpcError::DaemonVersionMismatch {
                daemon: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn socket_dir_candidates_prefers_runtime_override() {
        let temp = TempDir::new().expect("temp dir");
        let _override = override_runtime_dir_for_tests(Some(temp.path().to_path_buf()));
        let expected = temp.path().join("beads");
        let dirs = socket_dir_candidates();
        assert_eq!(dirs.first(), Some(&expected));
    }

    #[test]
    fn socket_dir_candidates_prefers_runtime_env() {
        let temp = TempDir::new().expect("temp dir");
        let _override = override_runtime_dir_for_tests(None);
        let dirs = socket_dir_candidates_with(|key| match key {
            "BD_RUNTIME_DIR" => Some(
                temp.path()
                    .to_str()
                    .expect("runtime dir is valid utf-8")
                    .to_string(),
            ),
            _ => None,
        });
        let expected = temp.path().join("beads");
        assert_eq!(dirs.first(), Some(&expected));
    }

    #[test]
    fn connect_with_autostart_reports_spawned_child_exit_immediately() {
        let temp = TempDir::new().expect("temp dir");
        let runtime_dir = temp.path().join("runtime");
        let socket_dir = runtime_dir.join("beads");
        std::fs::create_dir_all(&socket_dir).expect("socket dir");
        let socket = socket_dir.join("daemon.sock");
        let args = vec![OsString::from("-c"), OsString::from("exit 23")];

        let started = Instant::now();
        let err = connect_with_direct_autostart_for_test(
            &socket,
            Path::new("/bin/sh"),
            &args,
            Duration::from_secs(2),
        )
        .expect_err("autostart should fail");
        let elapsed = started.elapsed();

        let IpcError::DaemonUnavailable(message) = err else {
            panic!("expected daemon unavailable, got {err:?}");
        };
        assert!(
            message.contains("exited before socket ready"),
            "expected actionable early-exit message, got {message}"
        );
        assert!(
            message.contains("23"),
            "expected exit status in message, got {message}"
        );
        assert!(
            elapsed < Duration::from_secs(1),
            "expected failure before socket timeout, took {elapsed:?}"
        );
    }

    #[test]
    fn connect_with_autostart_override_waits_for_socket_instead_of_child_exit() {
        let temp = TempDir::new().expect("temp dir");
        let runtime_dir = temp.path().join("runtime");
        let socket_dir = runtime_dir.join("beads");
        std::fs::create_dir_all(&socket_dir).expect("socket dir");
        let socket = socket_dir.join("daemon.sock");
        let args = vec![OsString::from("-c"), OsString::from("exit 23")];

        let started = Instant::now();
        let err = connect_with_autostart_for_test(
            &socket,
            Some(Path::new("/bin/sh")),
            &args,
            Duration::from_millis(250),
        )
        .expect_err("autostart should fail");
        let elapsed = started.elapsed();

        let IpcError::DaemonUnavailable(message) = err else {
            panic!("expected daemon unavailable, got {err:?}");
        };
        assert!(
            message.contains("timed out waiting for daemon socket"),
            "expected launcher-compatible timeout, got {message}"
        );
        assert!(
            !message.contains("exited before socket ready"),
            "override path should not treat launcher exit as direct daemon failure: {message}"
        );
        assert!(
            elapsed >= Duration::from_millis(200),
            "expected override path to keep waiting for the socket, took {elapsed:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn connect_with_autostart_does_not_chmod_custom_socket_parent() {
        let temp = TempDir::new().expect("temp dir");
        let custom_parent = temp.path().join("shared");
        fs::create_dir_all(&custom_parent).expect("custom socket parent");
        fs::set_permissions(&custom_parent, fs::Permissions::from_mode(0o755))
            .expect("set custom parent perms");
        let socket = custom_parent.join("daemon.sock");
        let args = vec![OsString::from("-c"), OsString::from("exit 23")];

        let err = connect_with_autostart_for_test(
            &socket,
            Some(Path::new("/bin/sh")),
            &args,
            Duration::from_millis(200),
        )
        .expect_err("autostart should fail");
        assert!(matches!(err, IpcError::DaemonUnavailable(_)));

        let mode = fs::metadata(&custom_parent)
            .expect("custom parent metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            mode, 0o755,
            "custom socket parent perms changed to {:o}",
            mode
        );
    }

    #[test]
    #[cfg(unix)]
    fn connect_with_autostart_restricts_default_socket_parent() {
        let temp = TempDir::new().expect("temp dir");
        let runtime_dir = temp.path().join("runtime");
        let _override = override_runtime_dir_for_tests(Some(runtime_dir.clone()));
        let socket = socket_path();
        let socket_parent = socket.parent().expect("socket parent");
        fs::set_permissions(socket_parent, fs::Permissions::from_mode(0o755))
            .expect("set permissive default socket parent");

        let args = vec![OsString::from("-c"), OsString::from("exit 23")];
        let err = connect_with_direct_autostart_for_test(
            &socket,
            Path::new("/bin/sh"),
            &args,
            Duration::from_millis(200),
        )
        .expect_err("autostart should fail");
        assert!(matches!(err, IpcError::DaemonUnavailable(_)));

        let mode = fs::metadata(socket_parent)
            .expect("socket parent metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            mode, 0o700,
            "default socket parent perms changed to {:o}",
            mode
        );
    }

    // -------------------------------------------------------------------
    // Daemon lifecycle
    //
    // Every one of these runs against a socket path inside a TempDir where no
    // daemon ever existed, so nothing here can touch the real daemon of
    // whoever is running the tests. That isolation is the point: the whole
    // contract being tested is "stop must be safe to call blind".
    // -------------------------------------------------------------------

    #[test]
    fn stop_daemon_on_empty_dir_reports_not_running() {
        let temp = TempDir::new().expect("temp dir");
        let socket = temp.path().join("daemon.sock");

        let outcome = stop_daemon_at(&socket).expect("stop must not error when nothing is running");
        assert_eq!(outcome, DaemonStopOutcome::NotRunning);
        assert_eq!(outcome.pid(), None);
        assert!(!outcome.stopped_a_live_daemon());
    }

    #[test]
    fn stop_daemon_is_idempotent() {
        // Scripts and session-close hooks call `stop` blind. Calling it twice
        // must not start being an error on the second go.
        let temp = TempDir::new().expect("temp dir");
        let socket = temp.path().join("daemon.sock");

        for _ in 0..3 {
            assert_eq!(
                stop_daemon_at(&socket).expect("repeat stop stays Ok"),
                DaemonStopOutcome::NotRunning
            );
        }
    }

    #[test]
    fn stop_daemon_clears_stale_socket_without_meta() {
        // A socket file with no `daemon.meta.json` next to it: a daemon died
        // badly and left a corpse. Nothing to signal, so just clear it — and
        // say so, rather than claiming we stopped something.
        let temp = TempDir::new().expect("temp dir");
        let socket = temp.path().join("daemon.sock");
        fs::write(&socket, b"").expect("write stale socket file");

        let outcome = stop_daemon_at(&socket).expect("stale socket cleanup is not an error");
        assert_eq!(outcome, DaemonStopOutcome::StaleCleaned { pid: None });
        assert!(!socket.exists(), "stale socket file should be gone");

        // Second call: nothing left at all.
        assert_eq!(
            stop_daemon_at(&socket).expect("second stop stays Ok"),
            DaemonStopOutcome::NotRunning
        );
    }

    #[test]
    fn stop_daemon_clears_meta_naming_a_dead_pid() {
        // Meta file naming a pid that is long gone. We must NOT report
        // `Stopped` — we stopped nothing — but we must clear the leftovers and
        // still exit Ok.
        let temp = TempDir::new().expect("temp dir");
        let socket = temp.path().join("daemon.sock");
        let meta_path = socket.with_file_name("daemon.meta.json");
        // pid 0 is never a normal user process on unix, and `kill(0, ...)`
        // means "the whole process group", so use a pid we can be confident is
        // both invalid to signal and not alive.
        let dead_pid = dead_pid_for_test();
        let meta = crate::api::DaemonInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_version: IPC_PROTOCOL_VERSION,
            pid: dead_pid,
            started_at_ms: None,
        };
        fs::write(&meta_path, serde_json::to_vec(&meta).unwrap()).expect("write meta");

        let outcome = stop_daemon_at(&socket).expect("dead-pid cleanup is not an error");
        assert_eq!(
            outcome,
            DaemonStopOutcome::StaleCleaned {
                pid: Some(dead_pid)
            }
        );
        assert!(!outcome.stopped_a_live_daemon());
        assert!(!meta_path.exists(), "stale meta file should be gone");

        assert_eq!(
            stop_daemon_at(&socket).expect("second stop stays Ok"),
            DaemonStopOutcome::NotRunning
        );
    }

    /// A pid that is reliably not a live process on this machine.
    ///
    /// Walking down from the max pid is the trick: fresh pids are handed out
    /// from the low end and wrap, so the very top of the range is almost never
    /// occupied. We still check liveness and keep walking, so the test cannot
    /// go flaky if some process really is sitting up there.
    fn dead_pid_for_test() -> u32 {
        let mut candidate = 4_194_303u32; // Linux PID_MAX_LIMIT
        while candidate > 1_000 {
            if !daemon_pid_alive(candidate) {
                return candidate;
            }
            candidate -= 1;
        }
        panic!("could not find a dead pid to test with");
    }

    #[test]
    fn wait_for_pid_exit_returns_immediately_for_a_dead_pid() {
        let dead = dead_pid_for_test();
        let started = Instant::now();
        assert!(wait_for_pid_exit(dead, Duration::from_secs(5)));
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "a dead pid must not cost us the whole timeout, took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn wait_for_pid_exit_gives_up_on_a_live_pid() {
        // Ourselves: guaranteed alive, guaranteed we won't kill it.
        let started = Instant::now();
        assert!(!wait_for_pid_exit(
            std::process::id(),
            Duration::from_millis(120)
        ));
        assert!(
            started.elapsed() >= Duration::from_millis(100),
            "must actually wait out the timeout before declaring the pid alive, took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn stop_outcome_pid_accessors_are_honest() {
        assert_eq!(DaemonStopOutcome::Stopped { pid: 42 }.pid(), Some(42));
        assert!(DaemonStopOutcome::Stopped { pid: 42 }.stopped_a_live_daemon());
        assert_eq!(
            DaemonStopOutcome::StaleCleaned { pid: Some(42) }.pid(),
            Some(42)
        );
        // A StaleCleaned carries a pid but stopped nothing — callers that want
        // "did we actually kill something" must not read `pid().is_some()`.
        assert!(!DaemonStopOutcome::StaleCleaned { pid: Some(42) }.stopped_a_live_daemon());
        assert_eq!(DaemonStopOutcome::StaleCleaned { pid: None }.pid(), None);
        assert_eq!(DaemonStopOutcome::NotRunning.pid(), None);
    }

    #[test]
    fn restart_outcome_reports_both_pids() {
        let outcome = DaemonRestartOutcome {
            stopped: DaemonStopOutcome::Stopped { pid: 4242 },
            started: crate::api::DaemonInfo {
                version: "0.1.4".to_string(),
                protocol_version: IPC_PROTOCOL_VERSION,
                pid: 9999,
                started_at_ms: None,
            },
        };
        assert_eq!(outcome.old_pid(), Some(4242));
        assert_eq!(outcome.new_pid(), 9999);
    }
}
