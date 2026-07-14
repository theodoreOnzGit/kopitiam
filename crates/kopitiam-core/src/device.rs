/// Where a tensor's storage lives and where kernels run on it.
///
/// Today there is exactly one variant. That is deliberate, not an oversight:
/// the Kopitiam Runtime is CPU-only by design (GPU support is explicitly out
/// of scope — see `docs/ai-decisions/` and the parent epic), and this project
/// does not pay for abstraction it has no use for.
///
/// So why does the type exist at all? Because it is the one place where a
/// future non-CPU backend would have to be admitted, and having it named
/// makes the CPU-only promise *checkable*: every signature that takes a
/// `Device` is a signature that would need review if that promise ever
/// changed. A bare `()` would hide that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    #[default]
    Cpu,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
        }
    }
}
