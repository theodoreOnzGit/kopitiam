//! Host identity helpers.

/// Returns the native local hostname when the platform exposes one.
#[cfg(windows)]
pub fn local_hostname() -> Option<String> {
    windows::local_hostname()
}

/// Returns the native local hostname when the platform exposes one.
#[cfg(unix)]
pub fn local_hostname() -> Option<String> {
    unix::local_hostname()
}

/// Returns the native local hostname when the platform exposes one.
#[cfg(all(not(unix), not(windows)))]
pub fn local_hostname() -> Option<String> {
    None
}

fn sanitize_hostname(value: String) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('\0').trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

#[cfg(unix)]
mod unix {
    pub(super) fn local_hostname() -> Option<String> {
        let mut buffer = [0_u8; 256];
        // SAFETY: `buffer` is valid for `buffer.len()` bytes and is only
        // written by `gethostname` for the duration of the call.
        let result =
            unsafe { libc::gethostname(buffer.as_mut_ptr().cast::<libc::c_char>(), buffer.len()) };
        if result != 0 {
            return None;
        }

        let len = buffer
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(buffer.len());
        super::sanitize_hostname(String::from_utf8_lossy(&buffer[..len]).into_owned())
    }
}

#[cfg(windows)]
mod windows {
    use windows_sys::Win32::Foundation::{GetLastError, ERROR_MORE_DATA};
    use windows_sys::Win32::System::SystemInformation::{
        ComputerNameDnsHostname, ComputerNamePhysicalDnsHostname, GetComputerNameExW,
    };

    pub(super) fn local_hostname() -> Option<String> {
        read_computer_name(ComputerNameDnsHostname)
            .or_else(|| read_computer_name(ComputerNamePhysicalDnsHostname))
    }

    fn read_computer_name(format: i32) -> Option<String> {
        let mut required = 0;

        // SAFETY: Passing a null output buffer with a zero length is the
        // documented size-discovery call. The function only writes to
        // `required`.
        let ok = unsafe { GetComputerNameExW(format, std::ptr::null_mut(), &mut required) };
        if ok != 0 {
            return None;
        }

        // SAFETY: `GetLastError` reads the thread-local Win32 error set by the
        // immediately preceding Win32 call.
        if unsafe { GetLastError() } != ERROR_MORE_DATA || required == 0 {
            return None;
        }

        let mut buffer = vec![0u16; required as usize];

        // SAFETY: `buffer` is valid for `required` UTF-16 code units. Windows
        // writes at most that many units and updates `required` with the count
        // excluding the trailing NUL.
        let ok = unsafe { GetComputerNameExW(format, buffer.as_mut_ptr(), &mut required) };
        if ok == 0 || required == 0 {
            return None;
        }

        buffer.truncate(required as usize);
        super::sanitize_hostname(String::from_utf16_lossy(&buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::sanitize_hostname;

    #[test]
    fn sanitize_hostname_trims_whitespace_and_nul() {
        assert_eq!(
            sanitize_hostname(" RMUXHOST\0 ".to_owned()),
            Some("RMUXHOST".to_owned())
        );
    }

    #[test]
    fn sanitize_hostname_rejects_empty_values() {
        assert_eq!(sanitize_hostname(" \0 ".to_owned()), None);
    }
}

/// The name of the public client executable KOPITIAM ships.
///
/// # Why this is a constant (KOPITIAM fork note)
///
/// Upstream rmux spelled its binary name as a bare `"rmux"` string literal in
/// roughly fifteen places across four crates: the tmux-shim generator, the SDK's
/// daemon discovery, the Claude launcher's helper resolution, the tiny-CLI
/// helper re-exec, and several user-facing suggestions. That is fine while the
/// name never changes — and it silently rots the moment it does.
///
/// Renaming the binary to `kmux` for this fork broke four of those sites at
/// once, in ways the type system could not catch: they are *file names* looked
/// up at runtime, so a stale literal is not a compile error, it is a feature
/// that quietly stops working (the tmux `$TMUX_PROGRAM` shim silently failed to
/// materialise, and the SDK could no longer find the client on `PATH`).
///
/// So the name lives here, once. The next rename is a one-line change, and any
/// site that forgets to use it is now the obvious anomaly rather than the norm.
pub const PUBLIC_BINARY_NAME: &str = "kmux";

/// The name of the internal daemon executable.
///
/// The client discovers the daemon as a *sibling file* of itself, by this exact
/// name — see `rmux-client`'s `auto_start` and `rmux-sdk`'s `connect`. It must
/// therefore agree with the `[[bin]]` name in `kopitiam-mux`'s manifest, which
/// the `compiled_binary_name_is_kmux` test guards.
pub const DAEMON_BINARY_NAME: &str = "kmux-daemon";

/// [`PUBLIC_BINARY_NAME`] with the platform's executable suffix (`.exe` on
/// Windows, empty elsewhere).
#[must_use]
pub fn public_binary_file_name() -> std::ffi::OsString {
    binary_file_name(PUBLIC_BINARY_NAME)
}

/// [`DAEMON_BINARY_NAME`] with the platform's executable suffix.
#[must_use]
pub fn daemon_binary_file_name() -> std::ffi::OsString {
    binary_file_name(DAEMON_BINARY_NAME)
}

fn binary_file_name(stem: &str) -> std::ffi::OsString {
    let mut name = std::ffi::OsString::from(stem);
    if !std::env::consts::EXE_SUFFIX.is_empty() {
        name.push(std::env::consts::EXE_SUFFIX);
    }
    name
}
