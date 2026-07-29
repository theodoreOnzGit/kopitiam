//! The Modelfile -- a model plus its params, prompt and framing, as one file.
//!
//! **Upstream:** `parser/parser.go` (ollama, MIT).
//!
//! ## What a Modelfile buys you
//!
//! A `.gguf` on disk is only weights. It does not say what temperature to run
//! at, what system prompt the model was tuned for, or how a turn is framed. A
//! Modelfile carries all of that **next to the weights**, in plain text, so the
//! full recipe travels with the model instead of living in somebody's shell
//! history:
//!
//! ```text
//! FROM ./qwen3-0.6b-q4_k_m.gguf
//! PARAMETER temperature 0.7
//! PARAMETER stop <|im_end|>
//! SYSTEM """You are a careful Rust reviewer."""
//! TEMPLATE """{{ .System }}{{ .Prompt }}"""
//! MESSAGE user Hello
//! MESSAGE assistant Hi -- what are we reviewing?
//! ```
//!
//! For KOPITIAM this is more than convenience, it is the **Knowledge endures**
//! rule applied to models: the reasoning about how to drive a given model
//! becomes a durable artifact instead of evaporating into a chat log.
//!
//! ## Why a rune state machine and not a line-based parser
//!
//! Because values can span lines. `SYSTEM """...multi-line..."""` has to keep
//! its newlines, and a naive `for line in text.lines()` cannot see that it is
//! inside a quote. Upstream therefore walks **one character at a time** through
//! six states ([`State`]), and the trick that makes it work is in [`Modelfile::parse`]:
//! when a value's quotes are still unterminated, the state machine **refuses the
//! transition** and keeps accumulating. Whitespace works the same way, which is
//! how `FROM my model.gguf` keeps its space.
//!
//! ## Scope of this port
//!
//! Ported: the state machine, the command grammar, quoting, and structured
//! access to the parsed commands.
//!
//! **Not ported (yet):** upstream's `CreateRequest`, `fileDigestMap`,
//! `filesForModel` and `expandPath`. Those walk the filesystem, sha256 every
//! shard, and content-type-sniff safetensors/pytorch/gguf globs -- they belong
//! with the manifest/blob store in a later stage of the port, and dragging them
//! in now would give this crate a filesystem dependency it does not otherwise
//! need. [`Modelfile`]'s accessors cover what a caller actually reads today.
//!
//! ## Deliberate divergences
//!
//! * **Input is `&str`, not an `io.Reader`.** Upstream wraps the reader in
//!   `unicode.BOMOverride(unicode.UTF8.NewDecoder())`, which also transcodes
//!   UTF-16 when it sees a UTF-16 BOM. Rust `&str` is already valid UTF-8, so
//!   we strip a UTF-8 BOM and stop there. A UTF-16 Modelfile must be decoded by
//!   the caller. Called out because it is a real capability gap, not an
//!   oversight -- see the bead on the port epic.
//! * Errors are a typed enum rather than upstream's `*ParserError` +
//!   `io.ErrUnexpectedEOF` sentinel mix. Same information, same line numbers.

use std::fmt;

/// One `NAME value` line. **Upstream:** `type Command struct`.
///
/// Watch the naming, it trips people: a `FROM` line is stored with
/// `name == "model"`, not `"from"`. That is upstream's normalisation and we keep
/// it, because the manifest and the create-request both speak "model".
///
/// `PARAMETER temperature 0.7` is stored flattened, as `name == "temperature"`,
/// `args == "0.7"` -- the word `PARAMETER` itself never survives parsing. So a
/// command name that is not one of the known keywords **is** a parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    /// Lowercased keyword (`model`, `template`, `system`, `message`, ...) or,
    /// for a `PARAMETER` line, the parameter's own name.
    ///
    /// **Not lowercased for parameters.** Upstream sets `cmd.Name = b.String()`
    /// straight out of the buffer in the parameter state, so
    /// `PARAMETER Temperature 0.7` yields `"Temperature"`. Faithful, and worth
    /// knowing before you match on it -- compare case-insensitively.
    pub name: String,
    /// Everything after the keyword, unquoted and trimmed.
    ///
    /// For a `MESSAGE` line this is `"{role}: {content}"` -- upstream packs the
    /// role into the args rather than adding a field. [`Command::as_message`]
    /// unpacks it.
    pub args: String,
}

impl Command {
    /// Split a `message` command's args back into `(role, content)`.
    ///
    /// **Upstream:** `strings.Cut(c.Args, ": ")` -- done inline in both
    /// `CreateRequest` and `Command.String()`. Returns `None` for any other
    /// command, so a caller cannot accidentally treat a `SYSTEM` line as a
    /// message.
    pub fn as_message(&self) -> Option<(&str, &str)> {
        if self.name != "message" {
            return None;
        }
        // Upstream's `Cut` with no separator found yields ("", whole, false) and
        // it ignores the ok flag -- role becomes the whole string, content
        // empty. Parsing guarantees the separator is there, so this is
        // defensive only.
        Some(self.args.split_once(": ").unwrap_or((self.args.as_str(), "")))
    }
}

impl fmt::Display for Command {
    /// **Upstream:** `Command.String()`. Round-trips through [`Modelfile::parse`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name.as_str() {
            // FROM's argument is never quoted upstream -- a path or a model name
            // has no newlines, and quoting it would change what `FROM` means to
            // the puller.
            "model" => write!(f, "FROM {}", self.args),
            "license" | "template" | "system" | "adapter" | "renderer" | "parser" | "requires"
            | "draft" => {
                write!(f, "{} {}", self.name.to_uppercase(), quote(&self.args))
            }
            "message" => {
                let (role, message) = self.args.split_once(": ").unwrap_or((&self.args, ""));
                write!(f, "MESSAGE {} {}", role, quote(message))
            }
            // Anything unrecognised is a parameter -- see the note on `name`.
            _ => write!(f, "PARAMETER {} {}", self.name, quote(&self.args)),
        }
    }
}

/// A parsed Modelfile: an ordered list of commands.
///
/// **Upstream:** `type Modelfile struct`.
///
/// Order is preserved and it **matters** -- repeated `PARAMETER stop` lines
/// accumulate, and `MESSAGE` lines form a conversation in the order written.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Modelfile {
    pub commands: Vec<Command>,
}

/// Keywords a Modelfile line may start with. **Upstream:** `isValidCommand`.
const VALID_COMMANDS: [&str; 11] = [
    "from",
    "license",
    "template",
    "system",
    "adapter",
    "draft",
    "renderer",
    "parser",
    "parameter",
    "message",
    "requires",
];

/// Roles a `MESSAGE` line may carry. **Upstream:** `isValidMessageRole`.
const VALID_MESSAGE_ROLES: [&str; 3] = ["system", "user", "assistant"];

/// Parameters upstream still parses but warns about.
///
/// **Upstream:** `deprecatedParameters`. Mirostat and friends were dropped when
/// the sampler was rewritten; `low_vram` / `f16_kv` / `use_mlock` are llama.cpp
/// knobs that no longer map onto anything. Kept as data (not silently rejected)
/// so an old Modelfile still parses and the caller can warn, exactly like
/// upstream's `"warning: parameter %s is deprecated"`.
pub const DEPRECATED_PARAMETERS: [&str; 9] = [
    "penalize_newline",
    "low_vram",
    "f16_kv",
    "logits_all",
    "vocab_only",
    "use_mlock",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
];

/// What went wrong parsing a Modelfile. **Upstream:** `ParserError`,
/// `errMissingFrom`, `errInvalidCommand`, `errInvalidMessageRole`,
/// `io.ErrUnexpectedEOF`.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ModelfileError {
    /// **Upstream:** `errInvalidCommand`, wrapped in a `ParserError`.
    #[error("(line {line}): command must be one of \"from\", \"license\", \"template\", \"system\", \"adapter\", \"draft\", \"renderer\", \"parser\", \"parameter\", \"message\", or \"requires\"")]
    InvalidCommand { line: usize },

    /// **Upstream:** `errInvalidMessageRole`, wrapped in a `ParserError`.
    #[error("(line {line}): message role must be one of \"system\", \"user\", or \"assistant\"")]
    InvalidMessageRole { line: usize },

    /// Input ended mid-command -- an unterminated `"""`, or a keyword with no
    /// value. **Upstream:** `io.ErrUnexpectedEOF` (with the partial buffer
    /// attached when it comes out of the state machine).
    #[error("unexpected EOF: {partial}")]
    UnexpectedEof { partial: String },

    /// **Upstream:** `errMissingFrom`. A Modelfile without a `FROM` names no
    /// model, so there is nothing to create.
    #[error("no FROM line")]
    MissingFrom,
}

/// Parser states. **Upstream:** `type state int`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Between commands: eating whitespace, waiting for a keyword or `#`.
    Nil,
    /// Reading the keyword.
    Name,
    /// Reading the value -- **the only state that can span spaces and newlines**.
    Value,
    /// Reading a parameter's name, after the `PARAMETER` keyword.
    Parameter,
    /// Reading a message's role, after the `MESSAGE` keyword.
    Message,
    /// Inside a `#` comment, discarding to end of line.
    Comment,
}

impl Modelfile {
    /// Parse a Modelfile.
    ///
    /// **Upstream:** `ParseFile(r io.Reader)`.
    ///
    /// A character-at-a-time state machine. Two mechanics carry the whole thing
    /// and both are easy to break by "tidying":
    ///
    /// **1. The refused transition.** When a value's quotes are unterminated, or
    /// the character that ended it was a plain space, the machine writes that
    /// character into the buffer and *skips the state change entirely*
    /// (upstream's `continue`). That is what lets `SYSTEM """line one\nline
    /// two"""` and `FROM my model.gguf` both work. Remove it and every quoted
    /// value truncates at its first space.
    ///
    /// **2. Only printable characters get buffered.** The final
    /// `if strconv.IsPrint(r)` drops `\n`, `\r` and `\t`, so newlines can only
    /// ever enter a value through mechanic 1 -- i.e. while a quote is open. That
    /// is precisely why a multi-line value keeps its newlines and an ordinary
    /// value cannot accidentally swallow one.
    ///
    /// Faithfully copied wart: `\r\n` increments the line counter **twice**,
    /// because upstream counts every `\r` and every `\n`. It only ever affects
    /// the line number in an error message on a CRLF file, so matching upstream
    /// beats being right.
    pub fn parse(input: &str) -> Result<Self, ModelfileError> {
        // Upstream gets BOM handling from `unicode.BOMOverride`. Ours is a
        // UTF-8-only strip -- see the module docs on that divergence.
        let input = input.strip_prefix('\u{feff}').unwrap_or(input);

        let mut cmd = Command {
            name: String::new(),
            args: String::new(),
        };
        let mut curr = State::Nil;
        let mut curr_line: usize = 1;
        let mut b = String::new();
        let mut role = String::new();
        let mut f = Modelfile::default();

        for ch in input.chars() {
            if is_newline(ch) {
                curr_line += 1;
            }

            let (mut next, r) = parse_rune_for_state(ch, curr, curr_line, &b)?;

            // A state change is the signal to run the CURRENT state's exit
            // action -- that is where a command actually gets committed.
            if next != curr {
                match curr {
                    State::Name => {
                        if !is_valid_command(&b) {
                            return Err(ModelfileError::InvalidCommand { line: curr_line });
                        }
                        let s = b.to_lowercase();
                        match s.as_str() {
                            // FROM is stored as "model" -- see `Command::name`.
                            "from" => cmd.name = "model".to_string(),
                            // The name comes later, from the parameter state; do
                            // NOT set cmd.name here (upstream has no fallthrough
                            // on this arm).
                            "parameter" => next = State::Parameter,
                            // Upstream `fallthrough`s from "message" into the
                            // default arm, so the name IS set here as well as
                            // the state being redirected.
                            "message" => {
                                next = State::Message;
                                cmd.name = s;
                            }
                            _ => cmd.name = s,
                        }
                    }
                    // Not lowercased, matching upstream -- see `Command::name`.
                    State::Parameter => cmd.name = b.clone(),
                    State::Message => {
                        if !is_valid_message_role(&b) {
                            return Err(ModelfileError::InvalidMessageRole { line: curr_line });
                        }
                        role = b.clone();
                    }
                    State::Comment | State::Nil => {}
                    State::Value => {
                        match unquote(b.trim()) {
                            // Mechanic 1: quotes still open, or the terminator
                            // was a mere space -> absorb it and stay put.
                            None => {
                                b.push(r);
                                continue;
                            }
                            Some(_) if is_space(r) => {
                                b.push(r);
                                continue;
                            }
                            Some(s) => {
                                let mut s = s.to_string();
                                // A MESSAGE's role was captured one state ago;
                                // upstream packs it into the args here.
                                if !role.is_empty() {
                                    s = format!("{role}: {s}");
                                    role.clear();
                                }
                                cmd.args = s;
                                f.commands.push(cmd.clone());
                            }
                        }
                    }
                }

                b.clear();
                curr = next;
            }

            // Mechanic 2: newlines and tabs never land in the buffer here.
            if is_print(r) {
                b.push(r);
            }
        }

        // Flush whatever the last line left open.
        match curr {
            State::Comment | State::Nil => {}
            State::Value => {
                let Some(s) = unquote(b.trim()) else {
                    return Err(ModelfileError::UnexpectedEof { partial: b.clone() });
                };
                let mut s = s.to_string();
                if !role.is_empty() {
                    s = format!("{role}: {s}");
                }
                cmd.args = s;
                f.commands.push(cmd);
            }
            // A keyword with no value at all.
            _ => return Err(ModelfileError::UnexpectedEof { partial: b.clone() }),
        }

        // **Upstream:** the trailing scan for a "model" command. A Modelfile
        // with no FROM names nothing, so it is rejected even though every
        // individual line parsed fine.
        if !f.commands.iter().any(|c| c.name == "model") {
            return Err(ModelfileError::MissingFrom);
        }

        Ok(f)
    }

    /// The `FROM` argument -- a model name or a local path.
    pub fn from(&self) -> Option<&str> {
        self.find("model")
    }

    /// The `TEMPLATE` block, if the Modelfile overrides the model's own.
    pub fn template(&self) -> Option<&str> {
        self.find("template")
    }

    /// The `SYSTEM` prompt, if set.
    pub fn system(&self) -> Option<&str> {
        self.find("system")
    }

    /// The `RENDERER` name -- a model-family-specific chat renderer that
    /// replaces the generic template path.
    pub fn renderer(&self) -> Option<&str> {
        self.find("renderer")
    }

    /// The `PARSER` name -- a model-family-specific response parser (tool calls,
    /// thinking blocks).
    pub fn parser(&self) -> Option<&str> {
        self.find("parser")
    }

    /// The `REQUIRES` semver, the minimum runtime version this model needs.
    pub fn requires(&self) -> Option<&str> {
        self.find("requires")
    }

    /// Every `LICENSE` block, in order. Upstream accumulates rather than
    /// overwrites -- a model can carry more than one licence.
    pub fn licenses(&self) -> Vec<&str> {
        self.collect("license")
    }

    /// The `MESSAGE` lines as `(role, content)`, in order -- a seeded
    /// conversation the model starts every chat with.
    pub fn messages(&self) -> Vec<(&str, &str)> {
        self.commands.iter().filter_map(Command::as_message).collect()
    }

    /// Every `PARAMETER` line as `(name, value)`, in order.
    ///
    /// Anything that is not a known keyword is a parameter -- see
    /// [`Command::name`]. Duplicates are **kept**, not deduped: repeated `stop`
    /// lines are how a Modelfile declares several stop strings, so collapsing
    /// them would silently drop terminators and let a model run past its own
    /// end-of-turn marker.
    pub fn parameters(&self) -> Vec<(&str, &str)> {
        self.commands
            .iter()
            .filter(|c| c.name != "message" && !is_reserved_command_name(&c.name))
            .map(|c| (c.name.as_str(), c.args.as_str()))
            .collect()
    }

    /// Parameters upstream would print a deprecation warning for. Returns the
    /// offending names so the caller can warn once, in its own voice.
    pub fn deprecated_parameters(&self) -> Vec<&str> {
        self.parameters()
            .into_iter()
            .map(|(k, _)| k)
            .filter(|k| DEPRECATED_PARAMETERS.contains(&k.to_lowercase().as_str()))
            .collect()
    }

    fn find(&self, name: &str) -> Option<&str> {
        // Last one wins for single-valued commands, matching upstream's
        // `req.X = c.Args` assignment inside the command loop.
        self.commands
            .iter()
            .rev()
            .find(|c| c.name == name)
            .map(|c| c.args.as_str())
    }

    fn collect(&self, name: &str) -> Vec<&str> {
        self.commands
            .iter()
            .filter(|c| c.name == name)
            .map(|c| c.args.as_str())
            .collect()
    }
}

impl fmt::Display for Modelfile {
    /// **Upstream:** `Modelfile.String()` -- one command per line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for cmd in &self.commands {
            writeln!(f, "{cmd}")?;
        }
        Ok(())
    }
}

/// The transition table. **Upstream:** `parseRuneForState`.
///
/// Returns the next state and the character to (maybe) buffer. Upstream returns
/// `0` for "buffer nothing"; we return `'\0'`, which [`is_print`] rejects, so the
/// effect is identical without a second return value.
///
/// `line` and `partial` are threaded in only to build the error values --
/// upstream's caller adds them afterwards.
fn parse_rune_for_state(
    r: char,
    cs: State,
    line: usize,
    partial: &str,
) -> Result<(State, char), ModelfileError> {
    Ok(match cs {
        State::Nil => {
            if r == '#' {
                (State::Comment, '\0')
            } else if is_space(r) || is_newline(r) {
                (State::Nil, '\0')
            } else {
                (State::Name, r)
            }
        }
        State::Name => {
            if is_alpha(r) {
                (State::Name, r)
            } else if is_space(r) {
                (State::Value, '\0')
            } else {
                return Err(ModelfileError::InvalidCommand { line });
            }
        }
        State::Value => {
            // Note this returns `r` itself, NOT '\0', for space and newline --
            // the exit handler in `parse` needs the actual character to decide
            // whether to absorb it (mechanic 1).
            (if is_newline(r) || is_space(r) { State::Nil } else { State::Value }, r)
        }
        State::Parameter => {
            if is_alpha(r) || is_number(r) || r == '_' {
                (State::Parameter, r)
            } else if is_space(r) {
                (State::Value, '\0')
            } else {
                return Err(ModelfileError::UnexpectedEof {
                    partial: partial.to_string(),
                });
            }
        }
        State::Message => {
            if is_alpha(r) {
                (State::Message, r)
            } else if is_space(r) {
                (State::Value, '\0')
            } else {
                return Err(ModelfileError::UnexpectedEof {
                    partial: partial.to_string(),
                });
            }
        }
        State::Comment => {
            if is_newline(r) {
                (State::Nil, '\0')
            } else {
                (State::Comment, '\0')
            }
        }
    })
}

/// Add quotes only when the value would not survive without them.
///
/// **Upstream:** `quote(s)`. Needs quoting when it contains a newline, or has
/// leading/trailing space -- exactly the cases the parser cannot recover
/// unaided. A value that itself contains `"` gets the triple-quote form so the
/// inner quotes need no escaping (there is no escape syntax at all; `"""` is
/// how upstream sidesteps needing one).
pub fn quote(s: &str) -> String {
    if s.contains('\n') || s.starts_with(' ') || s.ends_with(' ') {
        if s.contains('"') {
            return format!("\"\"\"{s}\"\"\"");
        }
        return format!("\"{s}\"");
    }
    s.to_string()
}

/// Strip surrounding quotes. **Upstream:** `unquote(s) (string, bool)`.
///
/// `None` is upstream's `ok == false`, and it means **"the quotes are still
/// open"**, not "malformed" -- the parser reads it as "keep accumulating". That
/// distinction is the whole multi-line mechanism, so do not be tempted to turn
/// `None` into an error here; it only becomes one at EOF.
///
/// Single quotes are not handled, matching upstream's own `// TODO: single quotes`.
fn unquote(s: &str) -> Option<&str> {
    if s.len() >= 3 && &s[..3] == "\"\"\"" {
        if s.len() >= 6 && &s[s.len() - 3..] == "\"\"\"" {
            return Some(&s[3..s.len() - 3]);
        }
        return None;
    }
    if s.starts_with('"') {
        if s.len() >= 2 && s.ends_with('"') {
            return Some(&s[1..s.len() - 1]);
        }
        return None;
    }
    Some(s)
}

fn is_alpha(r: char) -> bool {
    r.is_ascii_alphabetic()
}
fn is_number(r: char) -> bool {
    r.is_ascii_digit()
}
fn is_space(r: char) -> bool {
    r == ' ' || r == '\t'
}
fn is_newline(r: char) -> bool {
    r == '\r' || r == '\n'
}

/// **Upstream:** `strconv.IsPrint(r)` -- letters, marks, numbers, punctuation,
/// symbols, and the ASCII space. Notably NOT `\n`, `\r` or `\t`.
///
/// Our formula (ASCII space, or neither whitespace nor control) is exact over
/// ASCII, which is what the grammar's structural characters live in. It also
/// agrees with Go over non-ASCII whitespace like U+00A0, which Go's `IsPrint`
/// likewise excludes. It is not a bit-for-bit reimplementation of Go's Unicode
/// tables, and it does not need to be: the only decision riding on it is whether
/// a character can join a value, and the disagreements would be confined to
/// exotic format/control codepoints that have no business in a Modelfile either
/// way.
fn is_print(r: char) -> bool {
    r == ' ' || (!r.is_whitespace() && !r.is_control())
}

fn is_valid_message_role(role: &str) -> bool {
    VALID_MESSAGE_ROLES.contains(&role)
}

fn is_valid_command(cmd: &str) -> bool {
    VALID_COMMANDS.contains(&cmd.to_lowercase().as_str())
}

/// Is this parsed command name a **keyword**, as opposed to a parameter name?
///
/// **DERIVED from [`VALID_COMMANDS`] on purpose.** This used to be a second
/// hand-written list of nine strings sitting inside `parameters()`, which is a
/// silent-drift trap: add a keyword upstream, update one list and not the other,
/// and the forgotten one starts quietly reclassifying that keyword as a
/// *sampling parameter*. It would then be handed to the sampler as an unknown
/// option instead of being acted on. An audit flagged it; deriving removes the
/// possibility rather than documenting it.
///
/// Two names need translating on the way through, and both are noted where the
/// parser does them:
///
/// * **`from` is stored as `model`** -- see [`Command::name`].
/// * **`parameter` never survives parsing at all** -- a `PARAMETER x y` line is
///   flattened to `name = "x"`, so the keyword itself is gone by now. That is
///   exactly why anything left unmatched here *is* a parameter.
///
/// `message` is excluded by the caller rather than here, because it is a real
/// stored command name and the caller already filters it explicitly.
fn is_reserved_command_name(name: &str) -> bool {
    VALID_COMMANDS.iter().any(|kw| match *kw {
        "from" => name == "model",
        // Flattened away during parsing; never appears as a command name.
        "parameter" => false,
        // The caller filters this one itself.
        "message" => false,
        other => name == other,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(name: &str, args: &str) -> Command {
        Command {
            name: name.into(),
            args: args.into(),
        }
    }

    #[test]
    fn a_minimal_modelfile_parses() {
        let f = Modelfile::parse("FROM qwen3:0.6b\n").unwrap();
        assert_eq!(f.commands, vec![cmd("model", "qwen3:0.6b")]);
        assert_eq!(f.from(), Some("qwen3:0.6b"));
    }

    /// FROM is stored as `model`, and PARAMETER lines are flattened so the
    /// keyword itself disappears. Both are upstream normalisations that a
    /// reader will otherwise trip over.
    #[test]
    fn from_becomes_model_and_parameter_flattens_away() {
        let f = Modelfile::parse("FROM x.gguf\nPARAMETER temperature 0.7\n").unwrap();
        assert_eq!(
            f.commands,
            vec![cmd("model", "x.gguf"), cmd("temperature", "0.7")]
        );
        assert_eq!(f.parameters(), vec![("temperature", "0.7")]);
    }

    /// Mechanic 1, the refused transition: an unterminated `"""` keeps
    /// swallowing characters, newlines included.
    #[test]
    fn a_triple_quoted_value_keeps_its_newlines() {
        let src = "FROM x.gguf\nSYSTEM \"\"\"line one\nline two\"\"\"\n";
        let f = Modelfile::parse(src).unwrap();
        assert_eq!(f.system(), Some("line one\nline two"));
    }

    /// Mechanic 2's consequence: outside quotes, a value stops at the newline
    /// and never absorbs it.
    #[test]
    fn an_unquoted_value_stops_at_the_newline() {
        let f = Modelfile::parse("FROM x.gguf\nSYSTEM be terse\nPARAMETER top_k 20\n").unwrap();
        assert_eq!(f.system(), Some("be terse"), "spaces kept, newline not");
        assert_eq!(f.parameters(), vec![("top_k", "20")]);
    }

    /// The other half of mechanic 1: a plain space inside a value must not end
    /// it. This is what makes a path with spaces work.
    #[test]
    fn a_space_does_not_terminate_a_value() {
        let f = Modelfile::parse("FROM my model.gguf\n").unwrap();
        assert_eq!(f.from(), Some("my model.gguf"));
    }

    #[test]
    fn double_quotes_are_stripped() {
        let f = Modelfile::parse("FROM x.gguf\nSYSTEM \"  padded  \"\n").unwrap();
        assert_eq!(f.system(), Some("  padded  "));
    }

    #[test]
    fn messages_pack_their_role_into_the_args() {
        let src = "FROM x.gguf\nMESSAGE user Hello\nMESSAGE assistant Hi there\n";
        let f = Modelfile::parse(src).unwrap();
        assert_eq!(
            f.messages(),
            vec![("user", "Hello"), ("assistant", "Hi there")]
        );
        assert_eq!(f.commands[1], cmd("message", "user: Hello"));
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let src = "# a comment\n\nFROM x.gguf\n   # indented comment\n\nSYSTEM hi\n";
        let f = Modelfile::parse(src).unwrap();
        assert_eq!(f.commands, vec![cmd("model", "x.gguf"), cmd("system", "hi")]);
    }

    /// Repeated stop lines must ALL survive -- collapsing them would let a model
    /// run straight past its own end-of-turn marker.
    #[test]
    fn repeated_parameters_accumulate_rather_than_overwrite() {
        let src = "FROM x.gguf\nPARAMETER stop <|im_end|>\nPARAMETER stop <|endoftext|>\n";
        let f = Modelfile::parse(src).unwrap();
        assert_eq!(
            f.parameters(),
            vec![("stop", "<|im_end|>"), ("stop", "<|endoftext|>")]
        );
    }

    #[test]
    fn a_missing_from_is_rejected_even_when_every_line_parsed() {
        assert_eq!(
            Modelfile::parse("SYSTEM hi\nPARAMETER temperature 0.5\n"),
            Err(ModelfileError::MissingFrom)
        );
    }

    #[test]
    fn an_unknown_keyword_is_rejected_with_its_line_number() {
        assert_eq!(
            Modelfile::parse("FROM x.gguf\nBOGUS hi\n"),
            Err(ModelfileError::InvalidCommand { line: 2 })
        );
    }

    #[test]
    fn an_invalid_message_role_is_rejected() {
        assert_eq!(
            Modelfile::parse("FROM x.gguf\nMESSAGE robot hi\n"),
            Err(ModelfileError::InvalidMessageRole { line: 2 })
        );
        for role in ["system", "user", "assistant"] {
            let src = format!("FROM x.gguf\nMESSAGE {role} hi\n");
            assert!(Modelfile::parse(&src).is_ok(), "role {role:?} must be valid");
        }
    }

    /// An unterminated triple quote is only an error once the input runs out --
    /// mid-parse it is the signal to keep reading.
    #[test]
    fn an_unterminated_quote_is_an_eof_error_not_a_parse_error() {
        let err = Modelfile::parse("FROM x.gguf\nSYSTEM \"\"\"never closed\n").unwrap_err();
        assert!(
            matches!(err, ModelfileError::UnexpectedEof { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn a_keyword_with_no_value_is_an_eof_error() {
        let err = Modelfile::parse("FROM x.gguf\nSYSTEM").unwrap_err();
        assert!(matches!(err, ModelfileError::UnexpectedEof { .. }), "got {err:?}");
    }

    #[test]
    fn a_utf8_bom_is_stripped() {
        let f = Modelfile::parse("\u{feff}FROM x.gguf\n").unwrap();
        assert_eq!(f.from(), Some("x.gguf"));
    }

    /// Keywords are case-insensitive and normalise to lowercase; a PARAMETER's
    /// own name is NOT normalised, matching upstream.
    #[test]
    fn keywords_fold_case_but_parameter_names_do_not() {
        let f = Modelfile::parse("from x.gguf\nSyStEm hi\nPARAMETER Temperature 0.7\n").unwrap();
        assert_eq!(f.from(), Some("x.gguf"));
        assert_eq!(f.system(), Some("hi"));
        assert_eq!(f.parameters(), vec![("Temperature", "0.7")]);
    }

    /// The keyword set `parameters()` filters by must stay derived from
    /// [`VALID_COMMANDS`], not hand-maintained beside it. This pins the exact
    /// translation so a future upstream keyword cannot silently start being
    /// treated as a sampling parameter.
    #[test]
    fn the_reserved_name_set_is_derived_from_the_command_grammar() {
        // Every keyword that survives parsing under its own name.
        for kw in [
            "license", "template", "system", "adapter", "draft", "renderer", "parser", "requires",
        ] {
            assert!(is_reserved_command_name(kw), "{kw} must be reserved");
        }

        // The two that get translated on the way through.
        assert!(is_reserved_command_name("model"), "FROM is stored as `model`");
        assert!(!is_reserved_command_name("from"), "`from` never survives parsing");
        assert!(
            !is_reserved_command_name("parameter"),
            "PARAMETER is flattened away; the keyword itself never appears"
        );

        // And a real parameter name is not reserved.
        for p in ["temperature", "top_k", "stop", "mirostat"] {
            assert!(!is_reserved_command_name(p), "{p} is a parameter");
        }

        // The derived set must have exactly the arity the old hand-written list
        // had -- 9 = the 11 grammar keywords, minus `parameter` and `message`.
        let derived = VALID_COMMANDS
            .iter()
            .filter(|kw| !matches!(**kw, "parameter" | "message"))
            .count();
        assert_eq!(derived, 9, "grammar changed -- re-check is_reserved_command_name");
    }

    #[test]
    fn deprecated_parameters_are_parsed_and_reported_not_rejected() {
        let f = Modelfile::parse("FROM x.gguf\nPARAMETER mirostat 2\n").unwrap();
        assert_eq!(f.parameters(), vec![("mirostat", "2")]);
        assert_eq!(f.deprecated_parameters(), vec!["mirostat"]);
    }

    /// The full round trip: parse -> Display -> parse must reach a fixed point.
    /// This is what makes `ollama show --modelfile` trustworthy.
    #[test]
    fn display_round_trips_back_through_the_parser() {
        let src = concat!(
            "FROM ./qwen3-0.6b-q4_k_m.gguf\n",
            "PARAMETER temperature 0.7\n",
            "PARAMETER stop <|im_end|>\n",
            "SYSTEM \"\"\"You are a careful\nRust reviewer.\"\"\"\n",
            "MESSAGE user Hello\n",
            "MESSAGE assistant Hi there\n",
            "LICENSE MIT\n",
        );
        let a = Modelfile::parse(src).unwrap();
        let b = Modelfile::parse(&a.to_string()).unwrap();
        assert_eq!(a, b, "rendered:\n{a}");

        assert_eq!(a.system(), Some("You are a careful\nRust reviewer."));
        assert_eq!(a.licenses(), vec!["MIT"]);
    }

    /// A value containing a `"` AND a newline needs the triple-quote form, or
    /// the round trip loses the inner quote. There is no escape syntax, so this
    /// is the only way it can work.
    #[test]
    fn a_value_with_inner_quotes_round_trips_via_triple_quotes() {
        let src = "FROM x.gguf\nSYSTEM \"\"\"say \"hi\"\nthen stop\"\"\"\n";
        let a = Modelfile::parse(src).unwrap();
        assert_eq!(a.system(), Some("say \"hi\"\nthen stop"));
        let b = Modelfile::parse(&a.to_string()).unwrap();
        assert_eq!(a, b, "rendered:\n{a}");
    }

    #[test]
    fn quote_only_quotes_what_needs_it() {
        assert_eq!(quote("plain"), "plain");
        assert_eq!(quote("two words"), "two words", "an inner space is safe");
        assert_eq!(quote(" leading"), "\" leading\"");
        assert_eq!(quote("trailing "), "\"trailing \"");
        assert_eq!(quote("a\nb"), "\"a\nb\"");
        assert_eq!(quote("a\"b\nc"), "\"\"\"a\"b\nc\"\"\"");
    }

    #[test]
    fn unquote_signals_open_quotes_with_none() {
        assert_eq!(unquote("plain"), Some("plain"));
        assert_eq!(unquote("\"quoted\""), Some("quoted"));
        assert_eq!(unquote("\"\"\"triple\"\"\""), Some("triple"));
        assert_eq!(unquote("\"open"), None, "still accumulating");
        assert_eq!(unquote("\"\"\"open"), None, "still accumulating");
    }

    /// CRLF input must parse identically to LF -- KOPITIAM runs on Windows and
    /// Termux, and a Modelfile written on one gets read on the other.
    #[test]
    fn crlf_input_parses_the_same_as_lf() {
        let lf = Modelfile::parse("FROM x.gguf\nSYSTEM hi\n").unwrap();
        let crlf = Modelfile::parse("FROM x.gguf\r\nSYSTEM hi\r\n").unwrap();
        assert_eq!(lf, crlf);
    }

    /// The renderer/parser/requires trio, added upstream for model-family
    /// specific chat handling -- these are what a later stage of the port hangs
    /// per-model renderers off.
    #[test]
    fn renderer_parser_and_requires_are_parsed() {
        let src = "FROM x.gguf\nRENDERER qwen3-coder\nPARSER qwen3-coder\nREQUIRES 0.14.0\n";
        let f = Modelfile::parse(src).unwrap();
        assert_eq!(f.renderer(), Some("qwen3-coder"));
        assert_eq!(f.parser(), Some("qwen3-coder"));
        assert_eq!(f.requires(), Some("0.14.0"));
        // ...and none of them leak into the parameter list.
        assert!(f.parameters().is_empty(), "got {:?}", f.parameters());
    }
}
