//! The Anthropic **Messages API** compatibility layer -- `/v1/messages` in,
//! ollama's native chat types out, and back again.
//!
//! **Upstream:** `anthropic/anthropic.go` (ollama, MIT). Ported against
//! `4713800b08b2ddf5e14acf8398953cf7b12f169b`. The request/response *plumbing*
//! that sits in front of these conversions lives in [`crate::middleware`],
//! same as upstream splits `anthropic/` from `middleware/`.
//!
//! ## What this module is for
//!
//! Anthropic's Messages API is the shape Claude Code and every `anthropic-sdk-*`
//! client speak. Ollama serve it so those clients can point at a local model
//! without knowing anything changed. This module is the whole translation: it
//! is pure data-shuffling, no I/O, no model, no sockets -- which is exactly why
//! it is testable to the byte.
//!
//! ## The one thing you must understand: two different content models
//!
//! An Anthropic message carry an **array of content blocks** -- `text`,
//! `image`, `tool_use`, `tool_result`, `thinking`, `server_tool_use`,
//! `web_search_tool_result` -- in whatever order the client sent them.
//! Ollama's [`crate::api::Message`] carry **flat fields**: one `content`
//! string, one `thinking` string, a list of images, a list of tool calls.
//!
//! One is a sequence. The other is a record. **You cannot map a sequence onto
//! a record without dropping something**, so this translation is lossy in both
//! directions, and every place it loses is listed below. Nothing here is a bug
//! to be "fixed" -- it is upstream's behaviour, ported deliberately. But you
//! must know which way the information bleeds before you build anything on top.
//!
//! ### Anthropic -> ollama (request direction), what get dropped
//!
//! | What come in | What happen | What is gone |
//! |---|---|---|
//! | Several `text` blocks | Concatenated into one `content` string, **no separator** | Block boundaries. `["a","b"]` and `["ab"]` become identical |
//! | Block **order** inside one message | Text / images / tool_use / thinking are bucketed separately and re-emitted in a **fixed** order | `[text, tool_use, text]` becomes one `content` string plus one tool call. The interleaving is unrecoverable |
//! | Several `thinking` blocks | **Last one wins** (each assignment overwrite) | Every earlier thinking block |
//! | `thinking.signature` | Never read | The signature -- so a re-submitted thinking block cannot be verified |
//! | `text.citations` | Never read | All citations |
//! | `image.source.media_type` | Only the decoded bytes are kept | The MIME type |
//! | `image.source.url` | **Rejected** -- only `type:"base64"` is accepted | n/a (hard error, not silent) |
//! | `tool_result.is_error` | Never read | A failed tool result look exactly like a successful one |
//! | `tool_result` blocks | Lifted **out** of the message into separate `role:"tool"` messages | Their position inside the original message |
//! | `tool_result.content` that is neither string nor array | Becomes `""` | Everything, silently |
//! | `web_search_tool_result.content` | Flattened to `"- title: url\n"` lines | `encrypted_content`, `page_age`, the result body |
//! | Unknown block `type` | Counted, then discarded | The whole block |
//! | `system` as an array | Text blocks concatenated, **no separator** | Non-text system blocks |
//! | `tool_choice` | **Never read at all** | "You must call tool X" silently becomes "auto" |
//! | `metadata.user_id` | Never read | The user id |
//! | `thinking.budget_tokens` | Never read | The budget -- only enabled/disabled survive |
//! | `tools[].max_uses` | Never read | The cap on a server tool |
//!
//! The `tool_choice` row is the one that bites hardest in practice: a client
//! that forces a specific tool will not get its tool forced. Upstream know;
//! ollama's chat API has no equivalent knob to map it onto.
//!
//! ### ollama -> Anthropic (response direction), what get dropped
//!
//! | What come in | What happen | What is gone |
//! |---|---|---|
//! | `message.content` + `.thinking` + `.tool_calls` | Re-blocked in the **fixed** order thinking, text, `tool_use`* | Any interleaving the model actually produced |
//! | `done_reason` | `"stop"`->`end_turn`, `"length"`->`max_tokens`, **anything else non-empty** -> `stop_sequence` | Which reason it really was (`"load"`, `"unload"`, ... all read as `stop_sequence`) |
//! | `done_reason` **when tool calls are present** | Forced to `tool_use` | A `length` truncation that also carried a tool call reports as `tool_use` -- the truncation signal is lost |
//! | `stop_sequence` (which string matched) | Never filled in -- always `""` | The matched stop string |
//! | `Metrics` durations, `logprobs`, `_debug_info`, `remote_model`, `remote_host`, `created_at` | No Anthropic field exists | All of it. Only `prompt_eval_count` / `eval_count` survive, as `usage` |
//! | `message.images` on an assistant reply | Never read | Any image the model returned |
//!
//! ## Divergences from Go, stated out loud
//!
//! * **`request_id` and message ids take their entropy from the caller.**
//!   Upstream reach straight for `crypto/rand` inside `generateID`. A library
//!   crate got no business owning a CSPRNG (and this crate's only randomness
//!   dependency is optional, behind the `net` feature), so [`generate_id`]
//!   is a pure function of 12 bytes you supply. Same seam as
//!   `registry::Ambient`. Upstream's time-based fallback is dropped -- there is
//!   nothing to fall back *from* when the caller already hold the bytes.
//! * **`input_schema` is parsed JSON, not raw bytes.** Go hold it as
//!   `json.RawMessage`. Two consequences, both spelled out at
//!   [`Tool::input_schema`].
//! * **`content: []` where Go emit `content: null`.** See
//!   [`MessagesResponse::content`].
//! * **The "tool arguments would not marshal" branch is unreachable here.** See
//!   [`StreamConverter::process`].
//! * **`WebSearch` itself is not ported** -- it is HTTP + registry signing. See
//!   [`WEB_SEARCH_ENDPOINT`].

use serde::{Deserialize, Serialize};

use crate::api::{
    Message, PropertyType, ThinkLevel, ThinkValue, Tool as ApiTool, ToolCall, ToolCallArguments,
    ToolCallFunction, ToolFunction, ToolFunctionParameters, ToolProperty,
};
use crate::routes::{ChatRequest, ChatResponse};

// ===========================================================================
// Errors
// ===========================================================================

/// The `error` object inside an Anthropic error envelope.
/// **Upstream:** `anthropic.Error`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Error {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// The whole error body an Anthropic client expect.
/// **Upstream:** `anthropic.ErrorResponse`.
///
/// `type` is **always** the literal `"error"` -- it is the envelope's
/// discriminator, not the kind of error. The kind live in `error.type`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Always `"error"`.
    #[serde(rename = "type")]
    pub response_type: String,
    pub error: Error,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub request_id: String,
}

impl ErrorResponse {
    /// Build an error envelope for an HTTP status.
    ///
    /// **Upstream:** `anthropic.NewError`. **Divergence:** upstream mint the
    /// `request_id` internally with `crypto/rand`; here you pass it in (see the
    /// module docs on why). Pass `""` if you genuinely have none -- the field
    /// is `omitempty` on the wire, so it just disappears.
    pub fn new(code: u16, message: impl Into<String>, request_id: impl Into<String>) -> Self {
        ErrorResponse {
            response_type: "error".to_string(),
            error: Error {
                error_type: error_type_for_status(code).to_string(),
                message: message.into(),
            },
            request_id: request_id.into(),
        }
    }
}

/// HTTP status -> Anthropic `error.type`. **Upstream:** the `switch code` in
/// `NewError`.
///
/// Note **529** is in there next to 503. It is not a real IANA status -- it is
/// Anthropic's own "overloaded", and their SDKs retry on it. Dropping it would
/// turn a retryable overload into a generic `api_error` that clients give up on.
pub fn error_type_for_status(code: u16) -> &'static str {
    match code {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    }
}

/// Everything the Anthropic -> ollama conversion can refuse.
///
/// **Upstream:** the bare `errors.New` / `fmt.Errorf` values inside
/// `convertMessage`, `convertTool` and `resolveImageSource`. The wording is
/// reproduced **verbatim** -- upstream's own tests assert on these exact
/// strings, and so do ours.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConvertError {
    /// An `image` block with no `source` at all.
    #[error("invalid image source")]
    InvalidImageSource,
    /// **Upstream keeps the full stop and the capital O.** Copied as-is.
    #[error("invalid image source type: {0}. Only base64 images are supported.")]
    UnsupportedImageSourceType(String),
    /// **Divergence:** Go quote `base64.CorruptInputError`, whose text is
    /// `"illegal base64 data at input byte N"`. Our own decoder produce the
    /// same shape of message but the byte offset can differ by a few, because
    /// Go check padding lazily and we check it per 4-byte quantum. The *fact*
    /// of the rejection is identical; only the offset in the text may not be.
    #[error("invalid base64 image data: {0}")]
    InvalidBase64ImageData(String),
    #[error("tool_use block missing required 'id' field")]
    ToolUseMissingId,
    #[error("tool_use block missing required 'name' field")]
    ToolUseMissingName,
    #[error("invalid tool_result image source")]
    InvalidToolResultImageSource,
    /// **Upstream:** `fmt.Errorf("invalid input_schema for tool %q: %w", ...)`.
    /// Go's `%q` and Rust's `{:?}` on a `&str` produce the same quoted text for
    /// every tool name a client actually send.
    #[error("invalid input_schema for tool {tool:?}: {reason}")]
    InvalidInputSchema { tool: String, reason: String },
}

// ===========================================================================
// Request types
// ===========================================================================

/// `POST /v1/messages`. **Upstream:** `anthropic.MessagesRequest`.
///
/// `max_tokens` is **required** by Anthropic and has no `omitempty` here for
/// that reason -- it become ollama's `num_predict`. A request without it is
/// rejected up in [`crate::middleware`], not here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub max_tokens: i64,
    #[serde(default)]
    pub messages: Vec<MessageParam>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    /// **Parsed but never used.** See the lossiness table in the module docs --
    /// ollama's chat API has no "force this tool" knob to map it onto, so a
    /// `tool_choice` of `{"type":"tool","name":"x"}` silently degrade to auto.
    /// Kept on the struct so the field round-trips rather than vanishing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    /// Parsed, never used. Same story as `tool_choice`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
}

/// **Upstream:** `anthropic.OutputConfig`. The newer, `thinking`-independent way
/// a client ask for more reasoning.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputConfig {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub effort: String,
}

/// `system` is either one string or an array of content blocks.
/// **Upstream:** `System any`, switched over as `case string` / `case []any`.
///
/// The array arm is held as raw [`serde_json::Value`]s on purpose: upstream
/// only ever look at `block["type"] == "text"` and `block["text"]`, ignoring
/// everything else including malformed entries, and a raw `Value` reproduce
/// that tolerance exactly. Typing it as `Vec<ContentBlock>` would start
/// **rejecting** system arrays that upstream happily accept and skip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<serde_json::Value>),
}

/// One message in the request. **Upstream:** `anthropic.MessageParam`.
///
/// `content` is **always** a block array in memory. A client may send a bare
/// string and Anthropic's own API accept it; the custom deserialiser below
/// normalise that into a single `text` block, exactly as upstream's
/// `UnmarshalJSON` do. Every downstream reader can therefore assume blocks.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct MessageParam {
    /// `"user"` or `"assistant"`. Lowercased later, in [`convert_message`] --
    /// **not** here, because upstream lowercase at conversion time and the
    /// round-trip must give back what the client sent.
    pub role: String,
    pub content: Vec<ContentBlock>,
}

impl<'de> Deserialize<'de> for MessageParam {
    /// **Upstream:** `(*MessageParam).UnmarshalJSON` -- try `string` first, fall
    /// back to the block array.
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            #[serde(default)]
            role: String,
            #[serde(default)]
            content: serde_json::Value,
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Content {
            Text(String),
            Blocks(Vec<ContentBlock>),
        }

        let raw = Raw::deserialize(d)?;
        let content = match serde_json::from_value::<Content>(raw.content) {
            Ok(Content::Text(s)) => vec![ContentBlock::text(s)],
            Ok(Content::Blocks(b)) => b,
            Err(e) => return Err(serde::de::Error::custom(e)),
        };
        Ok(MessageParam { role: raw.role, content })
    }
}

/// One content block. **Upstream:** `anthropic.ContentBlock`.
///
/// This is a **union flattened into one struct**, the way the Anthropic wire
/// format actually is: `type` says which fields are meaningful and the rest are
/// absent. Field order below matches Go's declaration order on purpose --
/// [`estimate_tokens`] measure the serialised length of a block, so a
/// re-ordering would silently change the token estimate.
///
/// ## Why `text` and `thinking` are `Option<String>` and not `String`
///
/// Upstream use `*string` for exactly these two, with this comment: *"pointers
/// so the field only appears when set (SDK requires it for accumulation)"*.
/// A streaming client build up a block by watching for its key: a
/// `content_block_start` for a text block **must** carry `"text": ""`, or the
/// SDK has nothing to append the deltas to and reports "content block not
/// found". So `Some("")` -> `"text":""` (present, empty) and `None` -> key
/// absent are **two different things on the wire**. Collapsing them to `String`
/// would break every streaming SDK.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContentBlock {
    /// `text` · `image` · `tool_use` · `tool_result` · `thinking` ·
    /// `server_tool_use` · `web_search_tool_result`. Anything else is an
    /// unknown block and is **discarded** by [`convert_message`].
    #[serde(rename = "type")]
    pub block_type: String,

    /// Text blocks. See the struct docs for why this is an `Option`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Citations attached to a text block. **Parsed, never converted** -- see
    /// the lossiness table.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<Citation>,

    /// Image blocks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ImageSource>,

    /// `tool_use` / `server_tool_use`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// The tool's arguments, **insertion-ordered** (see
    /// [`crate::api::ToolCallArguments`] for why that matter).
    ///
    /// `Option` reproduce Go's `omitzero` tag, and the three states are all
    /// distinct on the wire: `None` -> key absent (a non-tool block),
    /// `Some(empty)` -> `"input":{}` (a tool block whose arguments will arrive
    /// as deltas), `Some(populated)` -> the arguments. A streaming
    /// `content_block_start` for `tool_use` **must** send `{}`, same reason as
    /// `text` above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<ToolCallArguments>,

    /// `tool_result` / `web_search_tool_result`: which call this answer.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_use_id: String,
    /// A string, an array of blocks, an array of [`WebSearchResult`], or a
    /// [`WebSearchToolResultError`] -- held raw because upstream's `any`
    /// accept all four and switch on the runtime type. See
    /// [`format_web_search_tool_result_content`] for how the arms collapse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    /// **Parsed, never converted.** A failed tool result reach the model
    /// looking exactly like a successful one.
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,

    /// Thinking blocks. See the struct docs for why this is an `Option`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// **Parsed, never converted.** Without it a re-submitted thinking block
    /// cannot be verified by anyone downstream.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub signature: String,
}

impl ContentBlock {
    /// A plain text block. The commonest construction by far.
    pub fn text(s: impl Into<String>) -> Self {
        ContentBlock {
            block_type: "text".to_string(),
            text: Some(s.into()),
            ..Default::default()
        }
    }

    /// A thinking block.
    pub fn thinking(s: impl Into<String>) -> Self {
        ContentBlock {
            block_type: "thinking".to_string(),
            thinking: Some(s.into()),
            ..Default::default()
        }
    }
}

/// A citation on a text block. **Upstream:** `anthropic.Citation`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    /// In practice always `"web_search_result_location"`.
    #[serde(rename = "type")]
    pub citation_type: String,
    /// No `omitempty` upstream -- always emitted, even empty.
    #[serde(default)]
    pub url: String,
    /// No `omitempty` upstream.
    #[serde(default)]
    pub title: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub encrypted_index: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub cited_text: String,
}

/// One web search hit, Anthropic-shaped.
/// **Upstream:** `anthropic.WebSearchResult`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchResult {
    /// Always `"web_search_result"`.
    #[serde(rename = "type")]
    pub result_type: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub title: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub encrypted_content: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub page_age: String,
}

/// A failed web search, as a `web_search_tool_result` body.
/// **Upstream:** `anthropic.WebSearchToolResultError`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSearchToolResultError {
    /// Always `"web_search_tool_result_error"`.
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(default)]
    pub error_code: String,
}

/// Where an image block's bytes come from.
/// **Upstream:** `anthropic.ImageSource`.
///
/// Only `type: "base64"` is ever accepted -- `url` is parsed and then refused,
/// because ollama would have to fetch it, and a compatibility shim that make
/// outbound requests on a client's say-so is an SSRF hole.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub media_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub data: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub url: String,
}

/// A tool offered in the request. **Upstream:** `anthropic.Tool`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    /// `"custom"` for a user-defined tool, or a dated server-tool identifier
    /// like `"web_search_20250305"`. The **prefix** `web_search` is what
    /// selects the built-in -- Anthropic version their server tools by date, so
    /// matching the whole string would break on every new revision.
    #[serde(rename = "type", default, skip_serializing_if = "String::is_empty")]
    pub tool_type: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// The tool's JSON Schema.
    ///
    /// **Divergence:** Go hold this as `json.RawMessage` -- the exact source
    /// bytes. We hold parsed JSON, which change two things and nothing else:
    ///
    /// 1. **Syntactically** invalid JSON (`{invalid json`) is now refused when
    ///    the *request* is parsed, not later in [`convert_tool`]. Same
    ///    rejection, earlier. **Structurally** invalid schema (valid JSON that
    ///    is not a parameters block, e.g. `{"required":"location"}`) still fail
    ///    in `convert_tool` with the identical message.
    /// 2. [`estimate_tokens`] measure the **compact re-serialisation**, where Go
    ///    measure the raw source bytes. A pretty-printed schema therefore
    ///    estimate slightly fewer tokens here than upstream. That heuristic is
    ///    `len/4` and carry upstream's own "replace me with real tokenisation"
    ///    TODO, so the drift is well inside its error bar.
    ///
    /// Holding raw bytes would need `serde_json`'s `raw_value` feature, which
    /// this crate does not enable, and enabling it is not this module's call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
    /// Server-tool only, and **never read**. See the lossiness table.
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub max_uses: i64,
}

/// **Upstream:** `anthropic.ToolChoice`. Parsed and then ignored -- see
/// [`MessagesRequest::tool_choice`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolChoice {
    /// `"auto"` · `"any"` · `"tool"` · `"none"`.
    #[serde(rename = "type")]
    pub choice_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub disable_parallel_tool_use: bool,
}

/// **Upstream:** `anthropic.ThinkingConfig`.
///
/// Only `"enabled"` and `"disabled"` are acted on. **Any other value --
/// including `"adaptive"` -- leave `think` unset**, which then let
/// `output_config.effort` decide. That is not an oversight; it is how an
/// adaptive request end up carrying an effort level.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub config_type: String,
    /// **Never read.** Ollama has no thinking token budget to map it onto.
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub budget_tokens: i64,
}

/// **Upstream:** `anthropic.Metadata`. Parsed, never read.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub user_id: String,
}

// ===========================================================================
// Response types
// ===========================================================================

/// The `/v1/messages` reply. **Upstream:** `anthropic.MessagesResponse`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    /// Always `"message"`.
    #[serde(rename = "type")]
    pub response_type: String,
    /// Always `"assistant"`.
    pub role: String,
    pub model: String,
    /// The re-blocked reply, in the fixed order thinking, text, `tool_use`*.
    ///
    /// **Divergence:** when there is nothing at all to say, we emit
    /// `"content":[]` and Go emit `"content":null` -- Go's field is a nil slice
    /// with no `omitempty`. `[]` is the shape Anthropic's own API document, and
    /// a strictly-typed client is likelier to choke on `null` than on `[]`, so
    /// this is a deliberate improvement rather than an oversight. It is,
    /// however, the one byte-level place this port does not match ollama, so if
    /// you are diffing wire captures, look here first.
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub stop_reason: String,
    /// **Never populated by this port, same as upstream.** Anthropic define it
    /// as "which stop sequence matched", and ollama's `done_reason` does not
    /// carry that, so it stays empty even when `stop_reason` is `stop_sequence`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub stop_sequence: String,
    #[serde(default)]
    pub usage: Usage,
}

/// Token accounting. **Upstream:** `anthropic.Usage`. Neither field is
/// `omitempty`, so both are always on the wire even at zero -- a client that
/// sum usage across a stream depend on that.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: i32,
    #[serde(default)]
    pub output_tokens: i32,
}

// ===========================================================================
// Streaming events
// ===========================================================================

/// **Upstream:** `anthropic.MessageStartEvent`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MessageStartEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub message: MessagesResponse,
}

/// **Upstream:** `anthropic.ContentBlockStartEvent`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContentBlockStartEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: usize,
    pub content_block: ContentBlock,
}

/// **Upstream:** `anthropic.ContentBlockDeltaEvent`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: usize,
    pub delta: Delta,
}

/// One incremental update. **Upstream:** `anthropic.Delta`.
///
/// `delta.type` picks which of the four payload fields is meaningful:
/// `text_delta` -> `text`, `input_json_delta` -> `partial_json`,
/// `thinking_delta` -> `thinking`, `signature_delta` -> `signature`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delta {
    #[serde(rename = "type")]
    pub delta_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub partial_json: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub thinking: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub signature: String,
}

/// **Upstream:** `anthropic.ContentBlockStopEvent`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentBlockStopEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub index: usize,
}

/// **Upstream:** `anthropic.MessageDeltaEvent`. Carries the final
/// `stop_reason` and the **cumulative** usage.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub delta: MessageDelta,
    pub usage: DeltaUsage,
}

/// **Upstream:** `anthropic.MessageDelta`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageDelta {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub stop_reason: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub stop_sequence: String,
}

/// **Upstream:** `anthropic.DeltaUsage`. Same two counters as [`Usage`], kept
/// as its own type because upstream keep it as its own type -- they are free to
/// diverge and one day probably will.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeltaUsage {
    #[serde(default)]
    pub input_tokens: i32,
    #[serde(default)]
    pub output_tokens: i32,
}

/// **Upstream:** `anthropic.MessageStopEvent`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageStopEvent {
    #[serde(rename = "type")]
    pub event_type: String,
}

/// A keepalive. **Upstream:** `anthropic.PingEvent`. Nothing in this port emit
/// one -- it is here so a client-side decoder built on these types is complete.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PingEvent {
    #[serde(rename = "type")]
    pub event_type: String,
}

/// **Upstream:** `anthropic.StreamErrorEvent`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamErrorEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub error: Error,
}

/// The payload of a [`StreamEvent`], as a real sum type.
///
/// **Upstream:** `StreamEvent.Data any`, switched over with a Go type switch.
/// Rust has enums, so we use one -- and it buys something real: a caller
/// tracking block indices (which [`crate::middleware`] must) cannot forget an
/// arm, where the Go version silently fall through `default`.
///
/// Serialises **untagged**, i.e. the variant contribute nothing and you get the
/// inner struct's JSON verbatim. That is required: the SSE `event:` line
/// already name the type, and the `data:` line must be the bare object.
///
/// The variants are lopsided (a `content_block_start` carry a whole
/// [`ContentBlock`], ~432 bytes; a `message_stop` is one string). Boxing the
/// big ones would shrink the enum and is **not** worth it here: an event is
/// built once, matched once, framed once and dropped, so the copy never
/// happens in a loop, and every `match` arm in [`crate::middleware`] would
/// grow a deref for nothing. Hence the `allow`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum StreamEventData {
    MessageStart(MessageStartEvent),
    ContentBlockStart(ContentBlockStartEvent),
    ContentBlockDelta(ContentBlockDeltaEvent),
    ContentBlockStop(ContentBlockStopEvent),
    MessageDelta(MessageDeltaEvent),
    MessageStop(MessageStopEvent),
    Ping(PingEvent),
    StreamError(StreamErrorEvent),
}

/// One SSE event: the `event:` name plus the `data:` payload.
/// **Upstream:** `anthropic.StreamEvent`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StreamEvent {
    pub event: String,
    pub data: StreamEventData,
}

impl StreamEvent {
    pub(crate) fn new(event: &str, data: StreamEventData) -> Self {
        StreamEvent { event: event.to_string(), data }
    }
}

// ===========================================================================
// Anthropic -> ollama
// ===========================================================================

/// Turn a Messages API request into an ollama chat request.
///
/// **Upstream:** `anthropic.FromMessagesRequest`.
///
/// Order of the produced messages, which is load-bearing:
///
/// 1. the system message, if any (one message, whatever shape `system` had);
/// 2. then, per input message, whatever [`convert_message`] produce -- which
///    for a `user` message put its `tool_result`s **before** its own text, and
///    for every other role put them **after**.
///
/// Options are built as a plain JSON map, not [`crate::options::Options`],
/// because that is what `api.ChatRequest.Options` is and the downstream
/// defaulting live in `Options::apply_map`. `num_predict` is **always** set
/// (from the required `max_tokens`); the rest appear only when the client sent
/// them, so an absent `temperature` keep the model's own default instead of
/// being pinned to zero.
pub fn from_messages_request(r: &MessagesRequest) -> Result<ChatRequest, ConvertError> {
    let mut messages: Vec<Message> = Vec::new();

    match &r.system {
        Some(SystemPrompt::Text(s)) if !s.is_empty() => {
            messages.push(Message::new("system", s));
        }
        Some(SystemPrompt::Blocks(blocks)) => {
            // Upstream concatenate with NO separator. Two adjacent text blocks
            // "You are helpful." and " Be concise." rely on the client's own
            // leading space -- we must not add one, or every existing prompt
            // shift by a character.
            let mut content = String::new();
            for block in blocks {
                if block.get("type").and_then(serde_json::Value::as_str) == Some("text")
                    && let Some(text) = block.get("text").and_then(serde_json::Value::as_str)
                {
                    content.push_str(text);
                }
            }
            if !content.is_empty() {
                messages.push(Message::new("system", &content));
            }
        }
        // An empty-string system prompt add no message at all -- upstream guard
        // on `sys != ""`. Same for a blocks array that yield nothing.
        _ => {}
    }

    for msg in &r.messages {
        messages.extend(convert_message(msg)?);
    }

    let mut options = serde_json::Map::new();
    options.insert("num_predict".to_string(), serde_json::json!(r.max_tokens));
    if let Some(t) = r.temperature {
        options.insert("temperature".to_string(), serde_json::json!(t));
    }
    if let Some(p) = r.top_p {
        options.insert("top_p".to_string(), serde_json::json!(p));
    }
    if let Some(k) = r.top_k {
        options.insert("top_k".to_string(), serde_json::json!(k));
    }
    if !r.stop_sequences.is_empty() {
        options.insert("stop".to_string(), serde_json::json!(r.stop_sequences));
    }

    // Anthropic's built-in web_search become an ollama function literally named
    // "web_search". If the client ALSO define its own tool by that name, the
    // model's call would be ambiguous -- which one do we route it to? Upstream
    // resolve it by dropping the user-defined one, and only when the built-in
    // is actually present. Note the guard is on the TYPE prefix, not the name:
    // a custom tool named web_search with no built-in in the request is kept.
    let has_builtin_web_search = r.tools.iter().any(|t| t.tool_type.starts_with("web_search"));

    let mut tools: Vec<ApiTool> = Vec::new();
    for t in &r.tools {
        if has_builtin_web_search && !t.tool_type.starts_with("web_search") && t.name == "web_search"
        {
            continue;
        }
        let (tool, _is_server_tool) = convert_tool(t)?;
        tools.push(tool);
    }

    let normalized_effort = r
        .output_config
        .as_ref()
        .map(|c| {
            let e = c.effort.trim().to_lowercase();
            // Anthropic's "xhigh" has no ollama equivalent; upstream flatten it
            // onto "high" rather than dropping the request's intent entirely.
            if e == "xhigh" { "high".to_string() } else { e }
        })
        .unwrap_or_default();

    // Precedence: an explicit thinking.type win; only when it says neither
    // "enabled" nor "disabled" (e.g. "adaptive", or absent) does effort get a
    // say. So `thinking:{type:"disabled"} + effort:"high"` really is disabled.
    let mut think: Option<ThinkValue> = None;
    if let Some(cfg) = &r.thinking {
        match cfg.config_type.as_str() {
            "enabled" => think = Some(ThinkValue::Bool(true)),
            "disabled" => think = Some(ThinkValue::Bool(false)),
            _ => {}
        }
    }
    if think.is_none() && r.output_config.is_some() {
        think = match normalized_effort.as_str() {
            "high" => Some(ThinkValue::Level(ThinkLevel::High)),
            "medium" => Some(ThinkValue::Level(ThinkLevel::Medium)),
            "low" => Some(ThinkValue::Level(ThinkLevel::Low)),
            "max" => Some(ThinkValue::Level(ThinkLevel::Max)),
            _ => None,
        };
    }

    Ok(ChatRequest {
        model: r.model.clone(),
        messages,
        // Always Some -- an explicit `false` is what tell the chat handler to
        // buffer. Leaving it None would mean "not stated", which streams.
        stream: Some(r.stream),
        tools,
        options: Some(options),
        think,
        ..Default::default()
    })
}

/// One Anthropic message -> zero or more ollama messages.
///
/// **Upstream:** `anthropic.convertMessage`.
///
/// The shape of the answer is the whole lossy story in one function:
///
/// * every `text` block is **appended to one string**, no separator;
/// * every `image` block is decoded and appended to one image list;
/// * `tool_use` **and** `server_tool_use` both become [`ToolCall`]s, in a
///   single list -- so a server tool call and a client tool call are
///   indistinguishable downstream;
/// * `thinking` blocks **overwrite** each other, last one wins;
/// * `tool_result` and `web_search_tool_result` become **separate messages**
///   with `role:"tool"`, lifted out of this one.
///
/// Then placement, which is the subtle bit: for a `user` message the tool
/// results go **first** and the user's own text second (so "here is the tool
/// output" precedes "now describe it"), and for every other role they go
/// **after**. The message itself is emitted only if it has *something* --
/// text, images, tool calls or thinking -- so a user message that is nothing
/// but tool results produce exactly the tool messages and no empty user turn.
pub fn convert_message(msg: &MessageParam) -> Result<Vec<Message>, ConvertError> {
    let mut messages: Vec<Message> = Vec::new();
    let role = msg.role.to_lowercase();

    let mut text_content = String::new();
    let mut images: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut thinking = String::new();
    let mut tool_results: Vec<Message> = Vec::new();

    for block in &msg.content {
        match block.block_type.as_str() {
            "text" => {
                if let Some(t) = &block.text {
                    text_content.push_str(t);
                }
            }
            "image" => {
                let source = block.source.as_ref().ok_or(ConvertError::InvalidImageSource)?;
                images.push(resolve_image_source(source)?);
            }
            "tool_use" => {
                if block.id.is_empty() {
                    return Err(ConvertError::ToolUseMissingId);
                }
                if block.name.is_empty() {
                    return Err(ConvertError::ToolUseMissingName);
                }
                tool_calls.push(ToolCall {
                    id: block.id.clone(),
                    function: ToolCallFunction {
                        index: 0,
                        name: block.name.clone(),
                        arguments: block.input.clone().unwrap_or_default(),
                    },
                });
            }
            "tool_result" => {
                let (result_content, result_images) =
                    convert_tool_result_content(block.content.as_ref())?;
                tool_results.push(Message {
                    role: "tool".to_string(),
                    content: result_content,
                    images: result_images,
                    tool_call_id: block.tool_use_id.clone(),
                    ..Default::default()
                });
            }
            "thinking" => {
                if let Some(t) = &block.thinking {
                    // Assignment, not append: upstream overwrite. Several
                    // thinking blocks in one message therefore keep only the
                    // last -- documented in the module's lossiness table.
                    thinking = t.clone();
                }
            }
            "server_tool_use" => {
                // Note: NO id/name validation here, unlike "tool_use". A server
                // tool call is minted by ollama itself, not by the client, so
                // upstream trust it. Copied as-is.
                tool_calls.push(ToolCall {
                    id: block.id.clone(),
                    function: ToolCallFunction {
                        index: 0,
                        name: block.name.clone(),
                        arguments: block.input.clone().unwrap_or_default(),
                    },
                });
            }
            "web_search_tool_result" => {
                tool_results.push(Message {
                    role: "tool".to_string(),
                    content: format_web_search_tool_result_content(block.content.as_ref()),
                    tool_call_id: block.tool_use_id.clone(),
                    ..Default::default()
                });
            }
            // Unknown block type: counted upstream (for a trace log) then
            // dropped. We drop it the same way.
            _ => {}
        }
    }

    if role == "user" && !tool_results.is_empty() {
        messages.extend(tool_results.iter().cloned());
    }

    if !text_content.is_empty()
        || !images.is_empty()
        || !tool_calls.is_empty()
        || !thinking.is_empty()
    {
        messages.push(Message {
            role: role.clone(),
            content: text_content,
            images,
            tool_calls,
            thinking,
            ..Default::default()
        });
    }

    if role != "user" || tool_results.is_empty() {
        messages.extend(tool_results);
    }

    Ok(messages)
}

/// Flatten a `web_search_tool_result` body into text a model can read.
///
/// **Upstream:** `anthropic.formatWebSearchToolResultContent`.
///
/// **Divergence, and it is benign:** Go switch over five runtime types --
/// `string`, `[]WebSearchResult`, `[]any`, `map[string]any`,
/// `WebSearchToolResultError`. Holding the body as [`serde_json::Value`]
/// collapse the typed arms into the untyped ones (a `[]WebSearchResult`
/// serialises to the same array of objects a `[]any` would), and every arm
/// produce **the identical string**, so nothing observable changes.
///
/// Output shapes:
/// * a string -> itself;
/// * an array -> one `"- {title}: {url}\n"` line per `web_search_result`
///   entry, and an array containing a `web_search_tool_result_error` **return
///   early** with just the error line, discarding any results before it;
/// * a lone error object -> `"web_search_tool_result_error: {code}"`, or the
///   bare `"web_search_tool_result_error"` when the code is empty;
/// * anything else -> its JSON, or `""` if even that fails.
pub fn format_web_search_tool_result_content(content: Option<&serde_json::Value>) -> String {
    let Some(content) = content else { return String::new() };

    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => {
            let mut out = String::new();
            for item in items {
                let Some(map) = item.as_object() else { continue };
                match map.get("type").and_then(serde_json::Value::as_str) {
                    Some("web_search_result") => {
                        let title =
                            map.get("title").and_then(serde_json::Value::as_str).unwrap_or("");
                        let url = map.get("url").and_then(serde_json::Value::as_str).unwrap_or("");
                        out.push_str(&format!("- {title}: {url}\n"));
                    }
                    Some("web_search_tool_result_error") => {
                        let code =
                            map.get("error_code").and_then(serde_json::Value::as_str).unwrap_or("");
                        if code.is_empty() {
                            return "web_search_tool_result_error".to_string();
                        }
                        return format!("web_search_tool_result_error: {code}");
                    }
                    _ => {}
                }
            }
            out
        }
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(serde_json::Value::as_str)
                == Some("web_search_tool_result_error")
            {
                let code = map.get("error_code").and_then(serde_json::Value::as_str).unwrap_or("");
                return if code.is_empty() {
                    "web_search_tool_result_error".to_string()
                } else {
                    format!("web_search_tool_result_error: {code}")
                };
            }
            serde_json::to_string(map).unwrap_or_default()
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// The description ollama attach to the synthesised built-in web search tool.
/// **Upstream:** the literal in `convertTool`. It is prompt text the model
/// actually see, so it is reproduced verbatim.
pub const WEB_SEARCH_DESCRIPTION: &str =
    "Search the web for current information. Use this to find up-to-date information about any topic.";

/// The description of its single `query` parameter. Also verbatim prompt text.
pub const WEB_SEARCH_QUERY_DESCRIPTION: &str = "The search query to look up on the web";

/// One Anthropic tool -> one ollama tool. The `bool` is *"this is a server
/// tool"*. **Upstream:** `anthropic.convertTool`.
///
/// A tool whose `type` **starts with** `web_search` is not translated at all --
/// it is **replaced** by a hand-written function tool with one required
/// `query: string` parameter. That is deliberate: Anthropic's built-in web
/// search has no client-visible schema, so ollama invent one the model can
/// actually call, and the middleware then executes it server-side.
pub fn convert_tool(t: &Tool) -> Result<(ApiTool, bool), ConvertError> {
    if t.tool_type.starts_with("web_search") {
        let mut props = indexmap::IndexMap::new();
        props.insert(
            "query".to_string(),
            ToolProperty {
                prop_type: PropertyType(vec!["string".to_string()]),
                description: WEB_SEARCH_QUERY_DESCRIPTION.to_string(),
                ..Default::default()
            },
        );
        return Ok((
            ApiTool {
                tool_type: "function".to_string(),
                items: None,
                function: ToolFunction {
                    name: "web_search".to_string(),
                    description: WEB_SEARCH_DESCRIPTION.to_string(),
                    parameters: ToolFunctionParameters {
                        param_type: "object".to_string(),
                        required: vec!["query".to_string()],
                        properties: Some(props),
                        ..Default::default()
                    },
                },
            },
            true,
        ));
    }

    let params = match &t.input_schema {
        Some(schema) => serde_json::from_value::<ToolFunctionParameters>(schema.clone()).map_err(
            |e| ConvertError::InvalidInputSchema { tool: t.name.clone(), reason: e.to_string() },
        )?,
        None => ToolFunctionParameters::default(),
    };

    Ok((
        ApiTool {
            tool_type: "function".to_string(),
            items: None,
            function: ToolFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: params,
            },
        },
        false,
    ))
}

/// Decode an image block's payload into the base64 string
/// [`crate::api::Message::images`] carry.
///
/// **Upstream:** `anthropic.resolveImageSource`, which decode to `[]byte`.
/// Ours decode **and re-encode**, which look like a no-op and is not: it
/// reproduce exactly what Go does, because a Go `[]byte` re-marshals as
/// **canonical** `StdEncoding`. So a client sending non-canonical-but-valid
/// base64 get it normalised, same as upstream, and invalid base64 is still
/// rejected here rather than reaching the model as garbage.
///
/// `media_type` is dropped on purpose (ollama's message has nowhere to put it)
/// and `url` sources are refused -- see [`ImageSource`].
fn resolve_image_source(source: &ImageSource) -> Result<String, ConvertError> {
    if source.source_type != "base64" {
        return Err(ConvertError::UnsupportedImageSourceType(source.source_type.clone()));
    }
    let decoded = decode_base64_std(&source.data).map_err(ConvertError::InvalidBase64ImageData)?;
    Ok(crate::registry::base64_std(&decoded))
}

/// Split a `tool_result` body into its text and its images.
///
/// **Upstream:** `anthropic.convertToolResultContent`. Note the last arm: a
/// body that is neither `null`, a string, nor an array -- a bare number, say --
/// yield `("", [])` with **no error**. Silent, and upstream's choice.
fn convert_tool_result_content(
    content: Option<&serde_json::Value>,
) -> Result<(String, Vec<String>), ConvertError> {
    let Some(content) = content else { return Ok((String::new(), Vec::new())) };

    match content {
        serde_json::Value::Null => Ok((String::new(), Vec::new())),
        serde_json::Value::String(s) => Ok((s.clone(), Vec::new())),
        serde_json::Value::Array(items) => {
            let mut text = String::new();
            let mut images = Vec::new();
            for cb in items {
                let Some(map) = cb.as_object() else { continue };
                match map.get("type").and_then(serde_json::Value::as_str) {
                    Some("text") => {
                        if let Some(t) = map.get("text").and_then(serde_json::Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    Some("image") => {
                        let raw_source = map
                            .get("source")
                            .and_then(serde_json::Value::as_object)
                            .ok_or(ConvertError::InvalidToolResultImageSource)?;
                        // Field by field, not serde: upstream pick out exactly
                        // three keys and ignore anything else, including a
                        // wrong-typed one. A `from_value` here would start
                        // rejecting bodies upstream accept.
                        let source = ImageSource {
                            source_type: raw_source
                                .get("type")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            media_type: raw_source
                                .get("media_type")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            data: raw_source
                                .get("data")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            url: String::new(),
                        };
                        images.push(resolve_image_source(&source)?);
                    }
                    _ => {}
                }
            }
            Ok((text, images))
        }
        _ => Ok((String::new(), Vec::new())),
    }
}

// ===========================================================================
// ollama -> Anthropic
// ===========================================================================

/// Turn a buffered ollama chat response into a Messages API response.
///
/// **Upstream:** `anthropic.ToMessagesResponse`.
///
/// Blocks come out in a **fixed** order -- thinking, then text, then one
/// `tool_use` per call. Whatever order the model actually produced them in is
/// not recoverable from a [`ChatResponse`], so this is the best that can be
/// done, not a choice.
pub fn to_messages_response(id: &str, r: &ChatResponse) -> MessagesResponse {
    let mut content: Vec<ContentBlock> = Vec::new();

    if !r.message.thinking.is_empty() {
        content.push(ContentBlock::thinking(&r.message.thinking));
    }
    if !r.message.content.is_empty() {
        content.push(ContentBlock::text(&r.message.content));
    }
    for tc in &r.message.tool_calls {
        content.push(ContentBlock {
            block_type: "tool_use".to_string(),
            id: tc.id.clone(),
            name: tc.function.name.clone(),
            input: Some(tc.function.arguments.clone()),
            ..Default::default()
        });
    }

    MessagesResponse {
        id: id.to_string(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        model: r.model.clone(),
        stop_reason: map_stop_reason(&r.done_reason, !r.message.tool_calls.is_empty()),
        stop_sequence: String::new(),
        content,
        usage: Usage {
            input_tokens: r.metrics.prompt_eval_count,
            output_tokens: r.metrics.eval_count,
        },
    }
}

/// ollama `done_reason` -> Anthropic `stop_reason`.
/// **Upstream:** `anthropic.mapStopReason`.
///
/// | in | tool calls? | out |
/// |---|---|---|
/// | anything | yes | `tool_use` |
/// | `stop` | no | `end_turn` |
/// | `length` | no | `max_tokens` |
/// | any other non-empty | no | `stop_sequence` |
/// | `""` | no | `""` |
///
/// Two lossy corners worth naming. The `tool_use` override come **first**, so
/// a generation that hit the token limit *and* emitted a tool call reports
/// `tool_use` and the truncation is invisible. And the catch-all fold
/// `"load"`, `"unload"` and every future reason into `stop_sequence`, which is
/// a lie in the pedantic sense -- no stop sequence matched -- but it is the
/// closest Anthropic value and it is what clients see today.
pub fn map_stop_reason(reason: &str, has_tool_calls: bool) -> String {
    if has_tool_calls {
        return "tool_use".to_string();
    }
    match reason {
        "stop" => "end_turn".to_string(),
        "length" => "max_tokens".to_string(),
        "" => String::new(),
        _ => "stop_sequence".to_string(),
    }
}

// ===========================================================================
// Streaming
// ===========================================================================

/// Turns a stream of ollama [`ChatResponse`] chunks into Anthropic SSE events.
///
/// **Upstream:** `anthropic.StreamConverter`.
///
/// ## Why this needs state at all
///
/// Ollama stream **flat deltas**: each chunk is a `ChatResponse` carrying a bit
/// more content, a bit more thinking, or a finished tool call. Anthropic stream
/// **framed blocks**: `content_block_start` / `..._delta`* / `..._stop`, each
/// carrying an `index`, and a client match deltas to blocks by that index. So
/// somebody has to remember which block is currently open and what index we are
/// up to. That somebody is this struct.
///
/// **Get the index wrong and clients fail hard** with *"content block not
/// found"* -- that is upstream issue #14816, which is exactly the
/// thinking-straight-into-tool_use case the tests below pin down.
///
/// ## The transitions, in one place
///
/// * thinking arrives while **text** is open -> close text, `index++`, open
///   thinking;
/// * text arrives while **thinking** is open -> close thinking, `index++`,
///   mark thinking done (**it never reopens**), open text;
/// * a tool call arrives -> close whichever of thinking/text is open,
///   `index++` per close, then emit start + one whole-arguments delta + stop,
///   `index++`;
/// * `done` -> close whatever is still open, then `message_delta` +
///   `message_stop`.
///
/// Note the asymmetry: **text can reopen after thinking, thinking cannot reopen
/// after text.** Once `thinking_done` is set nothing clears it, so a model that
/// interleave thinking and text more than once will have the later thinking
/// silently dropped from the stream.
#[derive(Debug, Clone)]
pub struct StreamConverter {
    /// The `msg_...` id every event carry.
    pub id: String,
    pub model: String,
    first_write: bool,
    content_index: usize,
    input_tokens: i32,
    output_tokens: i32,
    /// Used for `message_start` only, and only when the first chunk's real
    /// `prompt_eval_count` is still 0 -- which it usually is, because ollama
    /// only fill the metrics in on the final chunk.
    estimated_input_tokens: i32,
    thinking_started: bool,
    thinking_done: bool,
    text_started: bool,
    /// Tool call ids already emitted. Ollama may repeat a completed tool call
    /// on later chunks; without this the client would see the same block twice.
    tool_calls_sent: Vec<String>,
}

impl StreamConverter {
    /// **Upstream:** `anthropic.NewStreamConverter`.
    pub fn new(
        id: impl Into<String>,
        model: impl Into<String>,
        estimated_input_tokens: i32,
    ) -> Self {
        StreamConverter {
            id: id.into(),
            model: model.into(),
            first_write: true,
            content_index: 0,
            input_tokens: 0,
            output_tokens: 0,
            estimated_input_tokens,
            thinking_started: false,
            thinking_done: false,
            text_started: false,
            tool_calls_sent: Vec::new(),
        }
    }

    /// Feed one ollama chunk, get back the Anthropic events it produce.
    ///
    /// **Upstream:** `(*StreamConverter).Process`.
    ///
    /// **Divergence:** upstream have a branch that `slog.Error`s and skips a
    /// tool call whose arguments will not marshal -- possible in Go because
    /// `ToolCallFunctionArguments` hold `any`, so a client value could be a
    /// channel. Our arguments are [`serde_json::Value`], which is
    /// **serialisable by construction**, so that branch is unreachable and is
    /// not ported. Nothing is lost: there is no input that can reach it.
    pub fn process(&mut self, r: &ChatResponse) -> Vec<StreamEvent> {
        let mut events = Vec::new();

        if self.first_write {
            self.first_write = false;
            self.input_tokens = r.metrics.prompt_eval_count;
            if self.input_tokens == 0 && self.estimated_input_tokens > 0 {
                self.input_tokens = self.estimated_input_tokens;
            }
            events.push(StreamEvent::new(
                "message_start",
                StreamEventData::MessageStart(MessageStartEvent {
                    event_type: "message_start".to_string(),
                    message: MessagesResponse {
                        id: self.id.clone(),
                        response_type: "message".to_string(),
                        role: "assistant".to_string(),
                        model: self.model.clone(),
                        content: Vec::new(),
                        usage: Usage { input_tokens: self.input_tokens, output_tokens: 0 },
                        ..Default::default()
                    },
                }),
            ));
        }

        if !r.message.thinking.is_empty() && !self.thinking_done {
            if self.text_started {
                events.push(self.stop_current_block());
                self.content_index += 1;
                self.text_started = false;
            }
            if !self.thinking_started {
                self.thinking_started = true;
                events.push(self.start_block(ContentBlock::thinking("")));
            }
            events.push(self.delta(Delta {
                delta_type: "thinking_delta".to_string(),
                thinking: r.message.thinking.clone(),
                ..Default::default()
            }));
        }

        if !r.message.content.is_empty() {
            if self.thinking_started && !self.thinking_done {
                self.thinking_done = true;
                events.push(self.stop_current_block());
                self.content_index += 1;
            }
            if !self.text_started {
                self.text_started = true;
                events.push(self.start_block(ContentBlock::text("")));
            }
            events.push(self.delta(Delta {
                delta_type: "text_delta".to_string(),
                text: r.message.content.clone(),
                ..Default::default()
            }));
        }

        for tc in &r.message.tool_calls {
            if self.tool_calls_sent.iter().any(|id| id == &tc.id) {
                continue;
            }

            // thinking -> tool_use with no text between. Upstream issue #14816.
            if self.thinking_started && !self.thinking_done {
                self.thinking_done = true;
                events.push(self.stop_current_block());
                self.content_index += 1;
            }
            if self.text_started {
                events.push(self.stop_current_block());
                self.content_index += 1;
                self.text_started = false;
            }

            // The whole argument object goes out as ONE input_json_delta. Real
            // Anthropic dribble it out token by token; ollama only learn the
            // arguments once the call is complete, so there is nothing to
            // dribble. Clients concatenate partial_json, so one whole chunk
            // parse exactly the same.
            let args_json = tc.function.arguments.to_json_string();

            events.push(self.start_block(ContentBlock {
                block_type: "tool_use".to_string(),
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                // Some(empty), NOT None: the start event must carry `"input":{}`
                // or an SDK has nothing to accumulate the deltas onto.
                input: Some(ToolCallArguments::new()),
                ..Default::default()
            }));
            events.push(self.delta(Delta {
                delta_type: "input_json_delta".to_string(),
                partial_json: args_json,
                ..Default::default()
            }));
            events.push(self.stop_current_block());

            self.tool_calls_sent.push(tc.id.clone());
            self.content_index += 1;
        }

        if r.done {
            // Upstream write this as `if textStarted {...} else if thinking...
            // {...}` with the SAME event in both branches, so `||` is exactly
            // equivalent. (The two conditions cannot both hold anyway --
            // opening text clears text_started before thinking can reopen, and
            // opening thinking sets thinking_done -- but the collapse does not
            // rely on that.)
            if self.text_started || (self.thinking_started && !self.thinking_done) {
                events.push(self.stop_current_block());
            }

            // Overwrite with the real counts. Note this can REPLACE the
            // estimate with a genuine 0 if the backend reported no metrics --
            // upstream do the same, and a client that trusted message_start's
            // estimate will see it revised down here.
            self.input_tokens = r.metrics.prompt_eval_count;
            self.output_tokens = r.metrics.eval_count;
            let stop_reason = map_stop_reason(&r.done_reason, !self.tool_calls_sent.is_empty());

            events.push(StreamEvent::new(
                "message_delta",
                StreamEventData::MessageDelta(MessageDeltaEvent {
                    event_type: "message_delta".to_string(),
                    delta: MessageDelta { stop_reason, stop_sequence: String::new() },
                    usage: DeltaUsage {
                        input_tokens: self.input_tokens,
                        output_tokens: self.output_tokens,
                    },
                }),
            ));
            events.push(StreamEvent::new(
                "message_stop",
                StreamEventData::MessageStop(MessageStopEvent {
                    event_type: "message_stop".to_string(),
                }),
            ));
        }

        events
    }

    fn start_block(&self, block: ContentBlock) -> StreamEvent {
        StreamEvent::new(
            "content_block_start",
            StreamEventData::ContentBlockStart(ContentBlockStartEvent {
                event_type: "content_block_start".to_string(),
                index: self.content_index,
                content_block: block,
            }),
        )
    }

    fn delta(&self, delta: Delta) -> StreamEvent {
        StreamEvent::new(
            "content_block_delta",
            StreamEventData::ContentBlockDelta(ContentBlockDeltaEvent {
                event_type: "content_block_delta".to_string(),
                index: self.content_index,
                delta,
            }),
        )
    }

    fn stop_current_block(&self) -> StreamEvent {
        StreamEvent::new(
            "content_block_stop",
            StreamEventData::ContentBlockStop(ContentBlockStopEvent {
                event_type: "content_block_stop".to_string(),
                index: self.content_index,
            }),
        )
    }
}

// ===========================================================================
// Ids
// ===========================================================================

/// `{prefix}_{24 lowercase hex chars}`.
///
/// **Upstream:** `anthropic.generateID`, which read 12 bytes from `crypto/rand`
/// and format them with `%x`.
///
/// **Divergence:** the bytes come from you. A library crate has no business
/// owning a CSPRNG -- this crate's only randomness dependency is optional and
/// lives behind the `net` feature, and pulling one in unconditionally for an
/// id would break the "builds offline with no sockets" promise. Same seam as
/// `registry::Ambient`. Upstream's time-based fallback (for when `rand.Read`
/// fail) is dropped, because there is nothing to fall back from once the caller
/// already hold the bytes.
///
/// **What would make this wrong:** feeding it anything less than 12 bytes of
/// real entropy. These ids appear in `request_id` and in `msg_...`; a
/// predictable one let a client correlate or forge requests it should not see.
/// A counter is not acceptable here.
pub fn generate_id(prefix: &str, entropy: &[u8; 12]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(prefix.len() + 1 + 24);
    s.push_str(prefix);
    s.push('_');
    for b in entropy {
        // Writing into a String cannot fail; the Result is discarded on
        // purpose rather than unwrapped.
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// A `msg_...` id. **Upstream:** `anthropic.GenerateMessageID`.
/// See [`generate_id`] for where the entropy come from.
pub fn generate_message_id(entropy: &[u8; 12]) -> String {
    generate_id("msg", entropy)
}

// ===========================================================================
// Token estimation
// ===========================================================================

/// `POST /v1/messages/count_tokens`. **Upstream:** `anthropic.CountTokensRequest`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CountTokensRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub messages: Vec<MessageParam>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
}

/// **Upstream:** `anthropic.CountTokensResponse`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CountTokensResponse {
    pub input_tokens: i32,
}

/// Guess the input tokens of a `/v1/messages` request.
/// **Upstream:** `anthropic.EstimateInputTokens`.
///
/// Only used to fill `message_start.usage.input_tokens` while streaming,
/// because the real count is not known until generation finish. The final
/// `message_delta` overwrite it with the truth.
pub fn estimate_input_tokens(req: &MessagesRequest) -> i32 {
    estimate_tokens(&CountTokensRequest {
        model: req.model.clone(),
        messages: req.messages.clone(),
        system: req.system.clone(),
        tools: req.tools.clone(),
        thinking: req.thinking.clone(),
    })
}

/// The `len/4` heuristic. **Upstream:** `anthropic.estimateTokens`, TODO and all.
///
/// **This is not tokenisation.** It is characters divided by four -- upstream's
/// own words, *"rough approximation (~4 chars/token average)"*, with a standing
/// TODO to route it through the real Tokenize API. Do not build billing,
/// truncation, or context-budget logic on it: it is off by a lot for non-Latin
/// scripts, where one character can be one whole token.
///
/// What get counted: the system prompt, then per message the **role string
/// itself** (yes, four characters for `"user"`) plus its content, then per tool
/// its name + description + schema. Tool-use and tool-result blocks are counted
/// by their whole serialised length, not just their text.
///
/// The floor: if the arithmetic gives 0 but there **was** a message or a system
/// prompt, it return 1. Zero is reserved for "genuinely nothing was sent".
pub fn estimate_tokens(req: &CountTokensRequest) -> i32 {
    let mut total: usize = count_system(req.system.as_ref());

    for msg in &req.messages {
        total += msg.role.len();
        for block in &msg.content {
            total += count_content_block(block);
        }
    }

    for tool in &req.tools {
        total += tool.name.len() + tool.description.len() + schema_len(tool.input_schema.as_ref());
    }

    let tokens = total / 4;
    if tokens == 0 && (!req.messages.is_empty() || req.system.is_some()) {
        return 1;
    }
    tokens as i32
}

/// See [`Tool::input_schema`] -- this is the compact re-serialisation, where Go
/// measure the raw source bytes.
fn schema_len(schema: Option<&serde_json::Value>) -> usize {
    schema.and_then(|s| serde_json::to_string(s).ok()).map(|s| s.len()).unwrap_or(0)
}

/// **Upstream:** `anthropic.countAnyContent`, restricted to the `System` field
/// (the messages arm is inlined above, because ours is already typed).
fn count_system(system: Option<&SystemPrompt>) -> usize {
    match system {
        None => 0,
        Some(SystemPrompt::Text(s)) => s.len(),
        Some(SystemPrompt::Blocks(items)) => items
            .iter()
            .filter_map(|item| serde_json::from_value::<ContentBlock>(item.clone()).ok())
            .map(|b| count_content_block(&b))
            .sum(),
    }
}

/// **Upstream:** `anthropic.countContentBlock`.
///
/// Note a `tool_use` / `tool_result` block get counted **twice over** for its
/// text: once via `text`/`thinking`, and again inside the whole-block JSON.
/// That is upstream's arithmetic, kept as-is -- "fixing" it would silently
/// shift every existing estimate.
fn count_content_block(block: &ContentBlock) -> usize {
    let mut total = 0;
    if let Some(t) = &block.text {
        total += t.len();
    }
    if let Some(t) = &block.thinking {
        total += t.len();
    }
    if (block.block_type == "tool_use" || block.block_type == "tool_result")
        && let Ok(data) = serde_json::to_string(block)
    {
        total += data.len();
    }
    total
}

// ===========================================================================
// Ollama's own web search API
// ===========================================================================

/// **Upstream:** `anthropic.OllamaWebSearchRequest`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OllamaWebSearchRequest {
    pub query: String,
    #[serde(default, skip_serializing_if = "is_zero_i32")]
    pub max_results: i32,
}

/// **Upstream:** `anthropic.OllamaWebSearchResult`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OllamaWebSearchResult {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub content: String,
}

/// **Upstream:** `anthropic.OllamaWebSearchResponse`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OllamaWebSearchResponse {
    #[serde(default)]
    pub results: Vec<OllamaWebSearchResult>,
}

/// Where ollama's hosted web search live. **Upstream:** the
/// `WebSearchEndpoint` package var.
///
/// **Not ported: the `WebSearch` function itself.** It build an HTTP request,
/// sign the challenge with the registry key, and POST it -- all of which is
/// [`crate::registry`]'s territory and none of which belong in a pure
/// conversion module. `crate::middleware::WebSearchBackend` is the seam that
/// stand in for it, so the whole search loop stay testable with no socket.
pub const WEB_SEARCH_ENDPOINT: &str = "https://ollama.com/api/web_search";

/// The default `max_results` when a caller ask for none.
/// **Upstream:** the `if maxResults <= 0 { 5 }` clamp at the top of `WebSearch`.
pub const WEB_SEARCH_DEFAULT_MAX_RESULTS: i32 = 5;
/// The ceiling. **Upstream:** the `if maxResults > 10 { 10 }` clamp.
pub const WEB_SEARCH_MAX_MAX_RESULTS: i32 = 10;

/// Clamp a requested result count into ollama's accepted range.
/// **Upstream:** the two `if`s at the top of `anthropic.WebSearch`. Kept here
/// so a [`crate::middleware::WebSearchBackend`] implementation clamp the same
/// way ollama's own server would.
pub fn clamp_web_search_max_results(max_results: i32) -> i32 {
    if max_results <= 0 {
        WEB_SEARCH_DEFAULT_MAX_RESULTS
    } else if max_results > WEB_SEARCH_MAX_MAX_RESULTS {
        WEB_SEARCH_MAX_MAX_RESULTS
    } else {
        max_results
    }
}

/// ollama search hits -> Anthropic `web_search_result` blocks.
/// **Upstream:** `anthropic.ConvertOllamaToAnthropicResults`.
///
/// **Lossy on purpose:** ollama's result carry a `content` body and Anthropic's
/// `WebSearchResult` has no field for it, so the page text is dropped and only
/// title + url reach the client. The content is not wasted -- it reach the
/// *model* by a different route, via
/// [`crate::middleware::format_web_search_results_for_tool_message`].
pub fn convert_ollama_to_anthropic_results(
    ollama_results: &OllamaWebSearchResponse,
) -> Vec<WebSearchResult> {
    ollama_results
        .results
        .iter()
        .map(|r| WebSearchResult {
            result_type: "web_search_result".to_string(),
            url: r.url.clone(),
            title: r.title.clone(),
            encrypted_content: String::new(),
            page_age: String::new(),
        })
        .collect()
}

// ===========================================================================
// Small helpers
// ===========================================================================

fn is_false(b: &bool) -> bool {
    !*b
}

fn is_zero_i64(n: &i64) -> bool {
    *n == 0
}

fn is_zero_i32(n: &i32) -> bool {
    *n == 0
}

/// Go's `base64.StdEncoding.DecodeString` -- `+/` alphabet, `=` padded, strict.
///
/// [`crate::registry`] already own the **encoder**; nobody had needed a decoder
/// until image blocks arrived, and it lives here rather than there because
/// twenty lines of decoder is not worth a dependency. Kept strict on purpose,
/// matching Go: length must be a multiple of four, `=` only in the final
/// quantum's last two slots, no whitespace, no URL alphabet. A lenient decoder
/// would let a corrupt image through to a vision model as silent garbage
/// instead of a 400.
fn decode_base64_std(s: &str) -> Result<Vec<u8>, String> {
    fn value(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }

    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("illegal base64 data at input byte {}", bytes.len()));
    }

    let quanta = bytes.len() / 4;
    let mut out = Vec::with_capacity(quanta * 3);

    for (q, chunk) in bytes.chunks(4).enumerate() {
        let last = q + 1 == quanta;
        let mut vals = [0u8; 4];
        let mut pad = 0usize;

        for (i, &c) in chunk.iter().enumerate() {
            let at = q * 4 + i;
            if c == b'=' {
                // Padding is only ever legal in the last quantum, and only in
                // its last two slots.
                if !last || i < 2 {
                    return Err(format!("illegal base64 data at input byte {at}"));
                }
                pad += 1;
            } else {
                if pad > 0 {
                    // A real character after padding. `AB=C` is not base64.
                    return Err(format!("illegal base64 data at input byte {at}"));
                }
                vals[i] =
                    value(c).ok_or_else(|| format!("illegal base64 data at input byte {at}"))?;
            }
        }

        let n = ((vals[0] as u32) << 18)
            | ((vals[1] as u32) << 12)
            | ((vals[2] as u32) << 6)
            | (vals[3] as u32);
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }

    Ok(out)
}

// ===========================================================================
// Tests -- ported from `anthropic/anthropic_test.go`
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The 1x1 PNG upstream's tests use, byte for byte.
    const TEST_IMAGE: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

    /// **Upstream:** the `textContent` test helper.
    fn text_content(s: &str) -> Vec<ContentBlock> {
        vec![ContentBlock::text(s)]
    }

    /// **Upstream:** the `makeArgs` test helper.
    fn args(kvs: &[(&str, serde_json::Value)]) -> ToolCallArguments {
        let mut a = ToolCallArguments::new();
        for (k, v) in kvs {
            a.set(*k, v.clone());
        }
        a
    }

    fn user(content: Vec<ContentBlock>) -> MessageParam {
        MessageParam { role: "user".to_string(), content }
    }

    fn assistant(content: Vec<ContentBlock>) -> MessageParam {
        MessageParam { role: "assistant".to_string(), content }
    }

    fn basic_request(messages: Vec<MessageParam>) -> MessagesRequest {
        MessagesRequest {
            model: "test-model".to_string(),
            max_tokens: 1024,
            messages,
            ..Default::default()
        }
    }

    fn chat_response(message: Message, done_reason: &str) -> ChatResponse {
        ChatResponse {
            model: "test-model".to_string(),
            message,
            done: true,
            done_reason: done_reason.to_string(),
            ..Default::default()
        }
    }

    // -- from_messages_request -----------------------------------------------

    #[test]
    fn a_basic_request_carries_model_message_and_num_predict() {
        let req = basic_request(vec![user(text_content("Hello"))]);
        let got = from_messages_request(&req).expect("convert");

        assert_eq!(got.model, "test-model");
        assert_eq!(got.messages.len(), 1);
        assert_eq!(got.messages[0].role, "user");
        assert_eq!(got.messages[0].content, "Hello");
        assert_eq!(got.options.as_ref().and_then(|o| o.get("num_predict")), Some(&json!(1024)));
        // Not stated in the request means buffered, and we say so explicitly.
        assert_eq!(got.stream, Some(false));
    }

    #[test]
    fn a_string_system_prompt_becomes_the_first_message() {
        let mut req = basic_request(vec![user(text_content("Hello"))]);
        req.system = Some(SystemPrompt::Text("You are a helpful assistant.".to_string()));

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages.len(), 2);
        assert_eq!(got.messages[0].role, "system");
        assert_eq!(got.messages[0].content, "You are a helpful assistant.");
    }

    #[test]
    fn a_system_prompt_array_is_concatenated_with_no_separator() {
        let mut req = basic_request(vec![user(text_content("Hello"))]);
        req.system = Some(SystemPrompt::Blocks(vec![
            json!({"type": "text", "text": "You are helpful."}),
            json!({"type": "text", "text": " Be concise."}),
        ]));

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages.len(), 2);
        assert_eq!(got.messages[0].content, "You are helpful. Be concise.");
    }

    #[test]
    fn an_empty_system_prompt_adds_no_message_at_all() {
        let mut req = basic_request(vec![user(text_content("Hello"))]);
        req.system = Some(SystemPrompt::Text(String::new()));
        assert_eq!(from_messages_request(&req).expect("convert").messages.len(), 1);

        req.system = Some(SystemPrompt::Blocks(vec![json!({"type": "image"})]));
        assert_eq!(from_messages_request(&req).expect("convert").messages.len(), 1);
    }

    #[test]
    fn sampling_knobs_land_in_the_options_map() {
        let mut req = basic_request(vec![user(text_content("Hello"))]);
        req.max_tokens = 2048;
        req.temperature = Some(0.7);
        req.top_p = Some(0.9);
        req.top_k = Some(40);
        req.stop_sequences = vec!["\n".to_string(), "END".to_string()];

        let got = from_messages_request(&req).expect("convert");
        let opts = got.options.expect("options");
        assert_eq!(opts.get("num_predict"), Some(&json!(2048)));
        assert_eq!(opts.get("temperature"), Some(&json!(0.7)));
        assert_eq!(opts.get("top_p"), Some(&json!(0.9)));
        assert_eq!(opts.get("top_k"), Some(&json!(40)));
        assert_eq!(opts.get("stop"), Some(&json!(["\n", "END"])));
    }

    #[test]
    fn an_unset_sampling_knob_is_left_out_of_the_options_map() {
        let req = basic_request(vec![user(text_content("Hello"))]);
        let opts = from_messages_request(&req).expect("convert").options.expect("options");
        assert!(!opts.contains_key("temperature"));
        assert!(!opts.contains_key("top_p"));
        assert!(!opts.contains_key("top_k"));
        assert!(!opts.contains_key("stop"));
    }

    #[test]
    fn an_image_block_is_decoded_and_hangs_off_the_message() {
        let req = basic_request(vec![user(vec![
            ContentBlock::text("What's in this image?"),
            ContentBlock {
                block_type: "image".to_string(),
                source: Some(ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/png".to_string(),
                    data: TEST_IMAGE.to_string(),
                    url: String::new(),
                }),
                ..Default::default()
            },
        ])]);

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages.len(), 1);
        assert_eq!(got.messages[0].content, "What's in this image?");
        assert_eq!(got.messages[0].images.len(), 1);
        // Round-trips to canonical base64, exactly as a Go []byte would.
        assert_eq!(got.messages[0].images[0], TEST_IMAGE);
    }

    #[test]
    fn an_image_block_with_no_source_is_refused() {
        let req = basic_request(vec![user(vec![ContentBlock {
            block_type: "image".to_string(),
            ..Default::default()
        }])]);
        assert_eq!(from_messages_request(&req), Err(ConvertError::InvalidImageSource));
    }

    #[test]
    fn a_url_image_source_is_refused_because_only_base64_is_supported() {
        let req = basic_request(vec![user(vec![ContentBlock {
            block_type: "image".to_string(),
            source: Some(ImageSource {
                source_type: "url".to_string(),
                url: "https://example.com/cat.png".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        }])]);
        assert_eq!(
            from_messages_request(&req).unwrap_err().to_string(),
            "invalid image source type: url. Only base64 images are supported."
        );
    }

    #[test]
    fn a_tool_use_block_becomes_a_tool_call_on_the_assistant_message() {
        let req = basic_request(vec![
            user(text_content("What's the weather in Paris?")),
            assistant(vec![ContentBlock {
                block_type: "tool_use".to_string(),
                id: "call_123".to_string(),
                name: "get_weather".to_string(),
                input: Some(args(&[("location", json!("Paris"))])),
                ..Default::default()
            }]),
        ]);

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages.len(), 2);
        assert_eq!(got.messages[1].tool_calls.len(), 1);
        assert_eq!(got.messages[1].tool_calls[0].id, "call_123");
        assert_eq!(got.messages[1].tool_calls[0].function.name, "get_weather");
        assert_eq!(
            got.messages[1].tool_calls[0].function.arguments.get("location"),
            Some(&json!("Paris"))
        );
    }

    #[test]
    fn a_tool_result_block_is_lifted_out_into_its_own_tool_message() {
        let req = basic_request(vec![user(vec![ContentBlock {
            block_type: "tool_result".to_string(),
            tool_use_id: "call_123".to_string(),
            content: Some(json!("The weather in Paris is sunny, 22\u{b0}C")),
            ..Default::default()
        }])]);

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages.len(), 1);
        assert_eq!(got.messages[0].role, "tool");
        assert_eq!(got.messages[0].tool_call_id, "call_123");
        assert_eq!(got.messages[0].content, "The weather in Paris is sunny, 22\u{b0}C");
    }

    #[test]
    fn a_tool_result_can_carry_an_image_alongside_its_text() {
        let req = basic_request(vec![user(vec![ContentBlock {
            block_type: "tool_result".to_string(),
            tool_use_id: "call_img".to_string(),
            content: Some(json!([
                {"type": "text", "text": "Attached image"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": TEST_IMAGE
                }}
            ])),
            ..Default::default()
        }])]);

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages.len(), 1);
        assert_eq!(got.messages[0].role, "tool");
        assert_eq!(got.messages[0].tool_call_id, "call_img");
        assert_eq!(got.messages[0].content, "Attached image");
        assert_eq!(got.messages[0].images, vec![TEST_IMAGE.to_string()]);
    }

    #[test]
    fn a_user_messages_tool_results_come_before_its_own_text() {
        let req = basic_request(vec![
            assistant(vec![ContentBlock {
                block_type: "tool_use".to_string(),
                id: "call_read".to_string(),
                name: "Read".to_string(),
                input: Some(args(&[("file_path", json!("/Users/hoyyeva/Desktop/aaa.png"))])),
                ..Default::default()
            }]),
            user(vec![
                ContentBlock {
                    block_type: "tool_result".to_string(),
                    tool_use_id: "call_read".to_string(),
                    content: Some(json!("Read image (311.5KB)")),
                    ..Default::default()
                },
                ContentBlock::text("Please describe it."),
            ]),
        ]);

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages.len(), 3);
        assert_eq!(got.messages[1].role, "tool");
        assert_eq!(got.messages[1].tool_call_id, "call_read");
        assert_eq!(got.messages[2].role, "user");
        assert_eq!(got.messages[2].content, "Please describe it.");
    }

    #[test]
    fn a_non_user_messages_tool_results_come_after_its_own_text() {
        // The mirror of the test above -- same blocks, assistant role, and the
        // order flips. This asymmetry is upstream's, not ours.
        let req = basic_request(vec![assistant(vec![
            ContentBlock {
                block_type: "tool_result".to_string(),
                tool_use_id: "call_x".to_string(),
                content: Some(json!("done")),
                ..Default::default()
            },
            ContentBlock::text("Here you go."),
        ])]);

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages.len(), 2);
        assert_eq!(got.messages[0].role, "assistant");
        assert_eq!(got.messages[1].role, "tool");
    }

    #[test]
    fn output_config_effort_sets_the_think_level() {
        let mut req = basic_request(vec![user(text_content("Describe the image."))]);
        req.output_config = Some(OutputConfig { effort: "high".to_string() });

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.think, Some(ThinkValue::Level(ThinkLevel::High)));
    }

    #[test]
    fn output_config_effort_xhigh_flattens_onto_high() {
        let mut req = basic_request(vec![user(text_content("Describe the image."))]);
        req.output_config = Some(OutputConfig { effort: "xhigh".to_string() });

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.think.as_ref().map(ThinkValue::level), Some("high"));
    }

    #[test]
    fn thinking_disabled_beats_an_output_config_effort() {
        let mut req = basic_request(vec![user(text_content("Describe the image."))]);
        req.thinking =
            Some(ThinkingConfig { config_type: "disabled".to_string(), budget_tokens: 0 });
        req.output_config = Some(OutputConfig { effort: "high".to_string() });

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.think, Some(ThinkValue::Bool(false)));
    }

    #[test]
    fn thinking_adaptive_falls_through_to_the_output_config_effort() {
        // "adaptive" is neither enabled nor disabled, so it leaves think unset
        // and effort gets the say. Not a fallthrough bug -- the mechanism.
        let mut req = basic_request(vec![user(text_content("Describe the image."))]);
        req.thinking =
            Some(ThinkingConfig { config_type: "adaptive".to_string(), budget_tokens: 0 });
        req.output_config = Some(OutputConfig { effort: "high".to_string() });

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.think.as_ref().map(ThinkValue::level), Some("high"));
    }

    #[test]
    fn an_unrecognised_effort_leaves_think_unset() {
        let mut req = basic_request(vec![user(text_content("Hi"))]);
        req.output_config = Some(OutputConfig { effort: "turbo".to_string() });
        assert_eq!(from_messages_request(&req).expect("convert").think, None);
    }

    #[test]
    fn a_tool_definition_becomes_an_ollama_function_tool() {
        let mut req = basic_request(vec![user(text_content("Hello"))]);
        req.tools = vec![Tool {
            name: "get_weather".to_string(),
            description: "Get current weather".to_string(),
            input_schema: Some(json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            })),
            ..Default::default()
        }];

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.tools.len(), 1);
        assert_eq!(got.tools[0].tool_type, "function");
        assert_eq!(got.tools[0].function.name, "get_weather");
        assert_eq!(got.tools[0].function.description, "Get current weather");
        assert_eq!(got.tools[0].function.parameters.required, vec!["location".to_string()]);
    }

    #[test]
    fn a_custom_web_search_tool_is_dropped_when_the_builtin_is_present() {
        let mut req = basic_request(vec![user(text_content("Hello"))]);
        req.tools = vec![
            Tool {
                tool_type: "web_search_20250305".to_string(),
                name: "web_search".to_string(),
                ..Default::default()
            },
            Tool {
                tool_type: "custom".to_string(),
                name: "web_search".to_string(),
                description: "User-defined web search that should be dropped".to_string(),
                ..Default::default()
            },
            Tool {
                tool_type: "custom".to_string(),
                name: "get_weather".to_string(),
                description: "Get current weather".to_string(),
                input_schema: Some(json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                })),
                ..Default::default()
            },
        ];

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.tools.len(), 2);
        assert_eq!(got.tools[0].function.name, "web_search");
        assert_eq!(got.tools[0].function.description, WEB_SEARCH_DESCRIPTION);
        assert_eq!(got.tools[1].function.name, "get_weather");
    }

    #[test]
    fn a_custom_web_search_tool_survives_when_the_builtin_is_absent() {
        let mut req = basic_request(vec![user(text_content("Hello"))]);
        req.tools = vec![Tool {
            tool_type: "custom".to_string(),
            name: "web_search".to_string(),
            description: "User-defined web search".to_string(),
            input_schema: Some(json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            })),
            ..Default::default()
        }];

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.tools.len(), 1);
        assert_eq!(got.tools[0].function.name, "web_search");
        assert_eq!(got.tools[0].function.description, "User-defined web search");
    }

    #[test]
    fn thinking_enabled_becomes_think_true() {
        let mut req = basic_request(vec![user(text_content("Hello"))]);
        req.thinking =
            Some(ThinkingConfig { config_type: "enabled".to_string(), budget_tokens: 1000 });

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.think, Some(ThinkValue::Bool(true)));
    }

    #[test]
    fn a_message_of_nothing_but_a_thinking_block_still_produces_a_message() {
        let req = basic_request(vec![
            user(text_content("Hello")),
            assistant(vec![ContentBlock::thinking("Let me think about this...")]),
        ]);

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages.len(), 2);
        assert_eq!(got.messages[1].thinking, "Let me think about this...");
        assert_eq!(got.messages[1].content, "");
    }

    #[test]
    fn several_thinking_blocks_keep_only_the_last() {
        let req = basic_request(vec![assistant(vec![
            ContentBlock::thinking("first"),
            ContentBlock::thinking("second"),
        ])]);

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages[0].thinking, "second");
    }

    #[test]
    fn several_text_blocks_are_glued_together_with_no_separator() {
        let req =
            basic_request(vec![user(vec![ContentBlock::text("a"), ContentBlock::text("b")])]);

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages[0].content, "ab");
    }

    #[test]
    fn an_unknown_block_type_is_silently_discarded() {
        let req = basic_request(vec![user(vec![
            ContentBlock::text("keep"),
            ContentBlock { block_type: "redacted_thinking".to_string(), ..Default::default() },
        ])]);

        let got = from_messages_request(&req).expect("convert");
        assert_eq!(got.messages.len(), 1);
        assert_eq!(got.messages[0].content, "keep");
    }

    #[test]
    fn a_tool_use_block_without_an_id_is_refused() {
        let req = basic_request(vec![assistant(vec![ContentBlock {
            block_type: "tool_use".to_string(),
            name: "get_weather".to_string(),
            ..Default::default()
        }])]);
        assert_eq!(
            from_messages_request(&req).unwrap_err().to_string(),
            "tool_use block missing required 'id' field"
        );
    }

    #[test]
    fn a_tool_use_block_without_a_name_is_refused() {
        let req = basic_request(vec![assistant(vec![ContentBlock {
            block_type: "tool_use".to_string(),
            id: "call_123".to_string(),
            ..Default::default()
        }])]);
        assert_eq!(
            from_messages_request(&req).unwrap_err().to_string(),
            "tool_use block missing required 'name' field"
        );
    }

    #[test]
    fn a_structurally_invalid_input_schema_is_refused_with_the_tool_name_quoted() {
        // Valid JSON, invalid as a parameters block -- `required` must be an
        // array. This is the half of upstream's "invalid tool schema" test that
        // still reaches convert_tool; the syntactically-broken half is the test
        // below. See Tool::input_schema for why they split.
        let mut req = basic_request(vec![user(text_content("Hello"))]);
        req.tools = vec![Tool {
            name: "bad_tool".to_string(),
            input_schema: Some(json!({"required": "location"})),
            ..Default::default()
        }];

        let err = from_messages_request(&req).unwrap_err().to_string();
        assert!(err.starts_with(r#"invalid input_schema for tool "bad_tool": "#), "{err}");
    }

    #[test]
    fn a_syntactically_invalid_input_schema_is_refused_when_the_request_is_parsed() {
        let parsed = serde_json::from_str::<MessagesRequest>(
            r#"{"model":"m","max_tokens":1,"messages":[],"tools":[{"name":"bad","input_schema":{invalid json}}]}"#,
        );
        assert!(parsed.is_err(), "expected the request parse to refuse a broken schema");
    }

    // -- to_messages_response ------------------------------------------------

    #[test]
    fn a_plain_reply_becomes_one_text_block_with_end_turn() {
        let mut r = chat_response(
            Message {
                role: "assistant".into(),
                content: "Hello there!".into(),
                ..Default::default()
            },
            "stop",
        );
        r.metrics.prompt_eval_count = 10;
        r.metrics.eval_count = 5;

        let got = to_messages_response("msg_123", &r);
        assert_eq!(got.id, "msg_123");
        assert_eq!(got.response_type, "message");
        assert_eq!(got.role, "assistant");
        assert_eq!(got.content.len(), 1);
        assert_eq!(got.content[0].block_type, "text");
        assert_eq!(got.content[0].text.as_deref(), Some("Hello there!"));
        assert_eq!(got.stop_reason, "end_turn");
        assert_eq!(got.usage, Usage { input_tokens: 10, output_tokens: 5 });
    }

    #[test]
    fn a_reply_with_tool_calls_becomes_tool_use_blocks_and_stop_reason_tool_use() {
        let r = chat_response(
            Message {
                role: "assistant".into(),
                tool_calls: vec![ToolCall {
                    id: "call_123".into(),
                    function: ToolCallFunction {
                        index: 0,
                        name: "get_weather".into(),
                        arguments: args(&[("location", json!("Paris"))]),
                    },
                }],
                ..Default::default()
            },
            "stop",
        );

        let got = to_messages_response("msg_123", &r);
        assert_eq!(got.content.len(), 1);
        assert_eq!(got.content[0].block_type, "tool_use");
        assert_eq!(got.content[0].id, "call_123");
        assert_eq!(got.content[0].name, "get_weather");
        assert_eq!(got.stop_reason, "tool_use");
    }

    #[test]
    fn thinking_comes_out_as_the_first_block_before_text() {
        let r = chat_response(
            Message {
                role: "assistant".into(),
                content: "The answer is 42.".into(),
                thinking: "Let me think about this...".into(),
                ..Default::default()
            },
            "stop",
        );

        let got = to_messages_response("msg_123", &r);
        assert_eq!(got.content.len(), 2);
        assert_eq!(got.content[0].block_type, "thinking");
        assert_eq!(got.content[0].thinking.as_deref(), Some("Let me think about this..."));
        assert_eq!(got.content[1].block_type, "text");
    }

    #[test]
    fn map_stop_reason_covers_every_upstream_case() {
        for (reason, has_tools, want) in [
            ("stop", false, "end_turn"),
            ("length", false, "max_tokens"),
            ("stop", true, "tool_use"),
            ("other", false, "stop_sequence"),
            ("", false, ""),
            // The lossy corner: truncation plus a tool call reads as tool_use.
            ("length", true, "tool_use"),
        ] {
            assert_eq!(map_stop_reason(reason, has_tools), want, "{reason:?}/{has_tools}");
        }
    }

    // -- errors and ids ------------------------------------------------------

    #[test]
    fn every_http_status_maps_to_its_anthropic_error_type() {
        for (code, want) in [
            (400u16, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
            (429, "rate_limit_error"),
            (500, "api_error"),
            (503, "overloaded_error"),
            (529, "overloaded_error"),
        ] {
            let got = ErrorResponse::new(code, "test message", "req_x");
            assert_eq!(got.response_type, "error");
            assert_eq!(got.error.error_type, want, "status {code}");
            assert_eq!(got.error.message, "test message");
            assert_eq!(got.request_id, "req_x");
        }
    }

    #[test]
    fn an_error_response_serialises_to_the_anthropic_envelope() {
        let got = ErrorResponse::new(404, "model 'x' not found", "req_01");
        assert_eq!(
            serde_json::to_value(&got).expect("serialise"),
            json!({
                "type": "error",
                "error": {"type": "not_found_error", "message": "model 'x' not found"},
                "request_id": "req_01"
            })
        );
    }

    #[test]
    fn an_empty_request_id_disappears_from_the_wire() {
        let got = ErrorResponse::new(500, "boom", "");
        let v = serde_json::to_value(&got).expect("serialise");
        assert!(v.get("request_id").is_none(), "{v}");
    }

    #[test]
    fn a_generated_id_is_the_prefix_plus_twenty_four_hex_characters() {
        let id = generate_message_id(&[
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xff,
        ]);
        assert_eq!(id, "msg_00112233445566778899aaff");
        assert_eq!(generate_id("req", &[0; 12]), "req_000000000000000000000000");
    }

    // -- StreamConverter -----------------------------------------------------

    /// Squash the events into `event:type:index` strings, the way upstream's
    /// `TestStreamConverter_TextBeforeThinking` do -- it is far easier to read
    /// a failure in this shape than in a struct dump.
    fn trace(events: &[StreamEvent]) -> Vec<String> {
        events
            .iter()
            .map(|e| match &e.data {
                StreamEventData::ContentBlockStart(d) => {
                    format!("{}:{}:{}", e.event, d.content_block.block_type, d.index)
                }
                StreamEventData::ContentBlockDelta(d) => {
                    format!("{}:{}:{}", e.event, d.delta.delta_type, d.index)
                }
                StreamEventData::ContentBlockStop(d) => format!("{}:{}", e.event, d.index),
                _ => e.event.clone(),
            })
            .collect()
    }

    fn chunk(content: &str, thinking: &str) -> ChatResponse {
        ChatResponse {
            model: "test-model".to_string(),
            message: Message {
                role: "assistant".into(),
                content: content.into(),
                thinking: thinking.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn a_simple_text_stream_opens_delta_closes_and_reports_usage() {
        let mut conv = StreamConverter::new("msg_123", "test-model", 0);

        let mut first = chunk("Hello", "");
        first.metrics.prompt_eval_count = 10;
        let events1 = conv.process(&first);
        assert_eq!(
            trace(&events1),
            vec![
                "message_start",
                "content_block_start:text:0",
                "content_block_delta:text_delta:0"
            ]
        );

        let mut last = chunk(" world!", "");
        last.done = true;
        last.done_reason = "stop".to_string();
        last.metrics.prompt_eval_count = 10;
        last.metrics.eval_count = 5;
        let events2 = conv.process(&last);
        assert_eq!(
            trace(&events2),
            vec![
                "content_block_delta:text_delta:0",
                "content_block_stop:0",
                "message_delta",
                "message_stop"
            ]
        );

        let StreamEventData::MessageDelta(md) = &events2[2].data else {
            panic!("expected a message_delta, got {:?}", events2[2]);
        };
        assert_eq!(md.delta.stop_reason, "end_turn");
        assert_eq!(md.usage, DeltaUsage { input_tokens: 10, output_tokens: 5 });
    }

    #[test]
    fn a_tool_call_streams_as_start_one_whole_json_delta_and_stop() {
        let mut conv = StreamConverter::new("msg_123", "test-model", 0);
        let mut r = chat_response(
            Message {
                role: "assistant".into(),
                tool_calls: vec![ToolCall {
                    id: "call_123".into(),
                    function: ToolCallFunction {
                        index: 0,
                        name: "get_weather".into(),
                        arguments: args(&[("location", json!("Paris"))]),
                    },
                }],
                ..Default::default()
            },
            "stop",
        );
        r.metrics.prompt_eval_count = 10;
        r.metrics.eval_count = 5;

        let events = conv.process(&r);
        assert_eq!(
            trace(&events),
            vec![
                "message_start",
                "content_block_start:tool_use:0",
                "content_block_delta:input_json_delta:0",
                "content_block_stop:0",
                "message_delta",
                "message_stop"
            ]
        );

        let StreamEventData::ContentBlockDelta(d) = &events[2].data else {
            panic!("expected a content_block_delta");
        };
        assert_eq!(d.delta.partial_json, r#"{"location":"Paris"}"#);
    }

    #[test]
    fn the_same_tool_call_repeated_on_a_later_chunk_is_only_streamed_once() {
        let mut conv = StreamConverter::new("msg_123", "test-model", 0);
        let call = ToolCall {
            id: "call_123".into(),
            function: ToolCallFunction {
                index: 0,
                name: "get_weather".into(),
                arguments: args(&[("location", json!("Paris"))]),
            },
        };
        let mut r = chat_response(
            Message { role: "assistant".into(), tool_calls: vec![call], ..Default::default() },
            "stop",
        );
        r.done = false;
        r.done_reason = String::new();

        conv.process(&r);
        let again = conv.process(&r);
        assert!(trace(&again).is_empty(), "{:?}", trace(&again));
    }

    #[test]
    fn thinking_straight_into_a_tool_call_closes_the_thinking_block_first() {
        // Upstream #14816: reusing index 0 for the tool_use block made clients
        // report "content block not found".
        let mut conv = StreamConverter::new("msg_123", "test-model", 0);

        let events1 = conv.process(&chunk("", "I should call the tool."));
        assert_eq!(
            trace(&events1),
            vec![
                "message_start",
                "content_block_start:thinking:0",
                "content_block_delta:thinking_delta:0"
            ]
        );

        let mut r = chat_response(
            Message {
                role: "assistant".into(),
                tool_calls: vec![ToolCall {
                    id: "call_abc".into(),
                    function: ToolCallFunction {
                        index: 0,
                        name: "ask_user".into(),
                        arguments: args(&[("question", json!("cats or dogs?"))]),
                    },
                }],
                ..Default::default()
            },
            "stop",
        );
        r.metrics.prompt_eval_count = 10;
        r.metrics.eval_count = 5;

        assert_eq!(
            trace(&conv.process(&r)),
            vec![
                "content_block_stop:0",
                "content_block_start:tool_use:1",
                "content_block_delta:input_json_delta:1",
                "content_block_stop:1",
                "message_delta",
                "message_stop"
            ]
        );
    }

    #[test]
    fn text_then_thinking_then_text_walks_the_index_forward_three_blocks() {
        let mut conv = StreamConverter::new("msg_123", "test-model", 0);
        let mut got = Vec::new();

        got.extend(trace(&conv.process(&chunk("---\n", ""))));
        got.extend(trace(&conv.process(&chunk("", "Let me think."))));

        let mut last = chunk("The answer.", "");
        last.done = true;
        last.done_reason = "stop".to_string();
        last.metrics.prompt_eval_count = 10;
        last.metrics.eval_count = 5;
        got.extend(trace(&conv.process(&last)));

        assert_eq!(
            got,
            vec![
                "message_start",
                "content_block_start:text:0",
                "content_block_delta:text_delta:0",
                "content_block_stop:0",
                "content_block_start:thinking:1",
                "content_block_delta:thinking_delta:1",
                "content_block_stop:1",
                "content_block_start:text:2",
                "content_block_delta:text_delta:2",
                "content_block_stop:2",
                "message_delta",
                "message_stop"
            ]
        );
    }

    #[test]
    fn thinking_after_text_has_already_reopened_is_dropped_from_the_stream() {
        // Documents the asymmetry called out in StreamConverter's docs:
        // thinking_done is never cleared, so a second round of thinking after
        // text has started is silently lost. Upstream behaviour, pinned so a
        // future refactor cannot change it by accident.
        let mut conv = StreamConverter::new("msg_123", "test-model", 0);
        conv.process(&chunk("", "first thoughts"));
        conv.process(&chunk("some text", ""));
        assert!(trace(&conv.process(&chunk("", "second thoughts"))).is_empty());
    }

    #[test]
    fn the_estimated_input_token_count_fills_message_start_until_the_real_one_lands() {
        let mut conv = StreamConverter::new("msg_123", "test-model", 42);
        let events = conv.process(&chunk("hi", ""));
        let StreamEventData::MessageStart(ms) = &events[0].data else {
            panic!("expected a message_start");
        };
        assert_eq!(ms.message.usage, Usage { input_tokens: 42, output_tokens: 0 });

        let mut last = chunk("", "");
        last.done = true;
        last.done_reason = "stop".to_string();
        last.metrics.prompt_eval_count = 7;
        last.metrics.eval_count = 3;
        let events = conv.process(&last);
        let StreamEventData::MessageDelta(md) = &events[1].data else {
            panic!("expected a message_delta, got {:?}", events[1]);
        };
        assert_eq!(md.usage, DeltaUsage { input_tokens: 7, output_tokens: 3 });
    }

    #[test]
    fn a_real_prompt_eval_count_on_the_first_chunk_beats_the_estimate() {
        let mut conv = StreamConverter::new("msg_123", "test-model", 42);
        let mut first = chunk("hi", "");
        first.metrics.prompt_eval_count = 11;
        let events = conv.process(&first);
        let StreamEventData::MessageStart(ms) = &events[0].data else {
            panic!("expected a message_start");
        };
        assert_eq!(ms.message.usage.input_tokens, 11);
    }

    // -- JSON wire shape -----------------------------------------------------

    #[test]
    fn an_empty_text_block_still_carries_its_text_key() {
        // Not cosmetic: a streaming SDK needs the key present on the start
        // event or it has nothing to accumulate deltas onto.
        assert_eq!(
            serde_json::to_value(ContentBlock::text("")).expect("serialise"),
            json!({"type": "text", "text": ""})
        );
        assert_eq!(
            serde_json::to_value(ContentBlock::thinking("")).expect("serialise"),
            json!({"type": "thinking", "thinking": ""})
        );
        assert_eq!(
            serde_json::to_value(ContentBlock::text("hello")).expect("serialise"),
            json!({"type": "text", "text": "hello"})
        );
    }

    #[test]
    fn a_non_tool_block_never_carries_an_input_key() {
        for block in [
            ContentBlock::text("hello"),
            ContentBlock::thinking("let me think"),
            ContentBlock {
                block_type: "image".to_string(),
                source: Some(ImageSource {
                    source_type: "base64".to_string(),
                    media_type: "image/png".to_string(),
                    data: TEST_IMAGE.to_string(),
                    url: String::new(),
                }),
                ..Default::default()
            },
        ] {
            let v = serde_json::to_value(&block).expect("serialise");
            assert!(v.get("input").is_none(), "{v}");
        }
    }

    #[test]
    fn a_tool_use_content_block_start_carries_an_empty_input_object() {
        let mut conv = StreamConverter::new("msg_123", "test-model", 0);
        let events = conv.process(&chat_response(
            Message {
                role: "assistant".into(),
                tool_calls: vec![ToolCall {
                    id: "call_123".into(),
                    function: ToolCallFunction {
                        index: 0,
                        name: "get_weather".into(),
                        arguments: args(&[("location", json!("Paris"))]),
                    },
                }],
                ..Default::default()
            },
            "stop",
        ));

        let start = events
            .iter()
            .find_map(|e| match &e.data {
                StreamEventData::ContentBlockStart(d)
                    if d.content_block.block_type == "tool_use" =>
                {
                    Some(d)
                }
                _ => None,
            })
            .expect("a tool_use content_block_start");
        assert_eq!(start.content_block.input.as_ref().map(ToolCallArguments::len), Some(0));

        let v = serde_json::to_value(start).expect("serialise");
        assert_eq!(v["content_block"]["input"], json!({}));
    }

    #[test]
    fn a_message_start_event_serialises_with_an_empty_content_array() {
        let mut conv = StreamConverter::new("msg_123", "test-model", 0);
        let events = conv.process(&chunk("hi", ""));
        let v = serde_json::to_value(&events[0].data).expect("serialise");
        assert_eq!(v["type"], json!("message_start"));
        assert_eq!(v["message"]["content"], json!([]));
        assert_eq!(v["message"]["role"], json!("assistant"));
        assert_eq!(v["message"]["type"], json!("message"));
    }

    #[test]
    fn a_message_param_accepts_a_bare_string_content_and_normalises_it() {
        let m: MessageParam =
            serde_json::from_str(r#"{"role":"user","content":"Hello"}"#).expect("parse");
        assert_eq!(m.role, "user");
        assert_eq!(m.content, vec![ContentBlock::text("Hello")]);
    }

    #[test]
    fn a_message_param_accepts_a_block_array_content() {
        let m: MessageParam = serde_json::from_str(
            r#"{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}"#,
        )
        .expect("parse");
        assert_eq!(m.content.len(), 2);
    }

    #[test]
    fn a_request_round_trips_through_literal_wire_json() {
        let wire = json!({
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_123", "name": "get_weather",
                     "input": {"location": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_123", "content": "Sunny"}
                ]}
            ],
            "system": "Be brief.",
            "stream": true,
            "thinking": {"type": "enabled", "budget_tokens": 1000}
        });

        let req: MessagesRequest = serde_json::from_value(wire).expect("parse");
        assert_eq!(req.messages.len(), 3);
        assert!(req.stream);

        let chat = from_messages_request(&req).expect("convert");
        assert_eq!(chat.stream, Some(true));
        assert_eq!(chat.think, Some(ThinkValue::Bool(true)));
        // system, user, assistant(tool_call), tool
        assert_eq!(
            chat.messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            vec!["system", "user", "assistant", "tool"]
        );
    }

    #[test]
    fn a_messages_response_round_trips_through_literal_wire_json() {
        let wire = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 4}
        });

        let parsed: MessagesResponse = serde_json::from_value(wire.clone()).expect("parse");
        assert_eq!(serde_json::to_value(&parsed).expect("serialise"), wire);
    }

    // -- token estimation ----------------------------------------------------

    #[test]
    fn a_short_message_estimates_a_handful_of_tokens() {
        let req = CountTokensRequest {
            model: "test-model".to_string(),
            messages: vec![user(text_content("Hello, world!"))],
            ..Default::default()
        };
        // "user" (4) + "Hello, world!" (13) = 17 / 4 = 4
        assert_eq!(estimate_tokens(&req), 4);
    }

    #[test]
    fn a_system_prompt_adds_to_the_estimate() {
        let req = CountTokensRequest {
            model: "test-model".to_string(),
            system: Some(SystemPrompt::Text("You are a helpful assistant.".to_string())),
            messages: vec![user(text_content("Hello"))],
            ..Default::default()
        };
        assert!(estimate_tokens(&req) >= 5, "{}", estimate_tokens(&req));
    }

    #[test]
    fn tool_definitions_add_to_the_estimate() {
        let req = CountTokensRequest {
            model: "test-model".to_string(),
            messages: vec![user(text_content("What's the weather?"))],
            tools: vec![Tool {
                name: "get_weather".to_string(),
                description: "Get the current weather for a location".to_string(),
                input_schema: Some(json!({
                    "type": "object", "properties": {"location": {"type": "string"}}
                })),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(estimate_tokens(&req) >= 10, "{}", estimate_tokens(&req));
    }

    #[test]
    fn thinking_content_is_counted_in_the_estimate() {
        let req = CountTokensRequest {
            model: "test-model".to_string(),
            messages: vec![
                user(text_content("Hello")),
                assistant(vec![
                    ContentBlock::thinking("Let me think about this carefully..."),
                    ContentBlock::text("Here is my response."),
                ]),
            ],
            ..Default::default()
        };
        assert!(estimate_tokens(&req) >= 10, "{}", estimate_tokens(&req));
    }

    #[test]
    fn nothing_at_all_estimates_zero_tokens() {
        let req = CountTokensRequest { model: "test-model".to_string(), ..Default::default() };
        assert_eq!(estimate_tokens(&req), 0);
    }

    #[test]
    fn a_message_that_rounds_down_to_zero_still_estimates_one_token() {
        // The floor: zero is reserved for "nothing was sent at all".
        let req = CountTokensRequest {
            model: "m".to_string(),
            messages: vec![MessageParam { role: String::new(), content: vec![] }],
            ..Default::default()
        };
        assert_eq!(estimate_tokens(&req), 1);
    }

    #[test]
    fn estimate_input_tokens_reads_the_same_fields_as_count_tokens() {
        let req = basic_request(vec![user(text_content("Hello, world!"))]);
        assert_eq!(
            estimate_input_tokens(&req),
            estimate_tokens(&CountTokensRequest {
                model: req.model.clone(),
                messages: req.messages.clone(),
                ..Default::default()
            })
        );
    }

    // -- tools and web search ------------------------------------------------

    #[test]
    fn a_web_search_tool_is_replaced_by_a_synthesised_query_function() {
        let (tool, is_server_tool) = convert_tool(&Tool {
            tool_type: "web_search_20250305".to_string(),
            name: "web_search".to_string(),
            max_uses: 5,
            ..Default::default()
        })
        .expect("convert");

        assert!(is_server_tool);
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "web_search");
        assert_eq!(tool.function.description, WEB_SEARCH_DESCRIPTION);
        assert_eq!(tool.function.parameters.param_type, "object");
        assert_eq!(tool.function.parameters.required, vec!["query".to_string()]);
        let q = tool.function.parameters.property("query").expect("a query property");
        assert_eq!(q.prop_type.0, vec!["string".to_string()]);
        assert_eq!(q.description, WEB_SEARCH_QUERY_DESCRIPTION);
    }

    #[test]
    fn a_regular_tool_is_not_a_server_tool() {
        let (tool, is_server_tool) = convert_tool(&Tool {
            tool_type: "custom".to_string(),
            name: "get_weather".to_string(),
            description: "Get the weather".to_string(),
            input_schema: Some(json!({
                "type": "object", "properties": {"location": {"type": "string"}}
            })),
            ..Default::default()
        })
        .expect("convert");

        assert!(!is_server_tool);
        assert_eq!(tool.function.name, "get_weather");
    }

    #[test]
    fn a_server_tool_use_block_becomes_a_tool_call_without_id_validation() {
        // Deliberately no id/name check on this arm -- ollama mint these ids
        // itself, so upstream trust them. An empty-id server_tool_use is
        // therefore accepted where a tool_use would be refused.
        let msgs = convert_message(&assistant(vec![ContentBlock {
            block_type: "server_tool_use".to_string(),
            id: "srvtoolu_123".to_string(),
            name: "web_search".to_string(),
            input: Some(args(&[("query", json!("test query"))])),
            ..Default::default()
        }]))
        .expect("convert");

        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].tool_calls.len(), 1);
        assert_eq!(msgs[0].tool_calls[0].id, "srvtoolu_123");
        assert_eq!(msgs[0].tool_calls[0].function.name, "web_search");
    }

    #[test]
    fn a_web_search_tool_result_becomes_a_tool_message_of_title_and_url_lines() {
        let msgs = convert_message(&user(vec![ContentBlock {
            block_type: "web_search_tool_result".to_string(),
            tool_use_id: "srvtoolu_123".to_string(),
            content: Some(json!([{
                "type": "web_search_result",
                "title": "Test Result",
                "url": "https://example.com"
            }])),
            ..Default::default()
        }]))
        .expect("convert");

        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "tool");
        assert_eq!(msgs[0].tool_call_id, "srvtoolu_123");
        assert_eq!(msgs[0].content, "- Test Result: https://example.com\n");
    }

    #[test]
    fn an_empty_web_search_tool_result_still_produces_a_tool_message() {
        let msgs = convert_message(&user(vec![ContentBlock {
            block_type: "web_search_tool_result".to_string(),
            tool_use_id: "srvtoolu_empty".to_string(),
            content: Some(json!([])),
            ..Default::default()
        }]))
        .expect("convert");

        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "tool");
        assert_eq!(msgs[0].tool_call_id, "srvtoolu_empty");
        assert_eq!(msgs[0].content, "");
    }

    #[test]
    fn a_failed_web_search_tool_result_carries_its_error_code_into_the_tool_message() {
        let msgs = convert_message(&user(vec![ContentBlock {
            block_type: "web_search_tool_result".to_string(),
            tool_use_id: "srvtoolu_error".to_string(),
            content: Some(json!({
                "type": "web_search_tool_result_error",
                "error_code": "max_uses_exceeded"
            })),
            ..Default::default()
        }]))
        .expect("convert");

        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "web_search_tool_result_error: max_uses_exceeded");
    }

    #[test]
    fn a_web_search_error_with_no_code_gives_the_bare_error_word() {
        assert_eq!(
            format_web_search_tool_result_content(Some(&json!({
                "type": "web_search_tool_result_error"
            }))),
            "web_search_tool_result_error"
        );
        assert_eq!(
            format_web_search_tool_result_content(Some(&json!([
                {"type": "web_search_result", "title": "a", "url": "u"},
                {"type": "web_search_tool_result_error", "error_code": ""}
            ]))),
            "web_search_tool_result_error"
        );
    }

    #[test]
    fn a_web_search_result_body_that_is_a_plain_string_passes_straight_through() {
        assert_eq!(
            format_web_search_tool_result_content(Some(&json!("already text"))),
            "already text"
        );
        assert_eq!(format_web_search_tool_result_content(None), "");
    }

    #[test]
    fn ollama_search_hits_become_anthropic_result_blocks_losing_the_page_body() {
        let results = convert_ollama_to_anthropic_results(&OllamaWebSearchResponse {
            results: vec![
                OllamaWebSearchResult {
                    title: "Test Title".to_string(),
                    url: "https://example.com".to_string(),
                    content: "Test content".to_string(),
                },
                OllamaWebSearchResult {
                    title: "Another Result".to_string(),
                    url: "https://example.org".to_string(),
                    content: "More content".to_string(),
                },
            ],
        });

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].result_type, "web_search_result");
        assert_eq!(results[0].title, "Test Title");
        assert_eq!(results[0].url, "https://example.com");
        // The page body has nowhere to go in Anthropic's shape.
        assert_eq!(results[0].encrypted_content, "");
    }

    #[test]
    fn the_web_search_result_count_is_clamped_into_ollamas_range() {
        assert_eq!(clamp_web_search_max_results(0), 5);
        assert_eq!(clamp_web_search_max_results(-3), 5);
        assert_eq!(clamp_web_search_max_results(3), 3);
        assert_eq!(clamp_web_search_max_results(50), 10);
    }

    #[test]
    fn a_web_search_result_round_trips_through_json() {
        let result = WebSearchResult {
            result_type: "web_search_result".to_string(),
            url: "https://example.com".to_string(),
            title: "Test".to_string(),
            encrypted_content: "abc123".to_string(),
            page_age: "2025-01-01".to_string(),
        };
        let data = serde_json::to_string(&result).expect("serialise");
        assert_eq!(serde_json::from_str::<WebSearchResult>(&data).expect("parse"), result);

        let err = WebSearchToolResultError {
            error_type: "web_search_tool_result_error".to_string(),
            error_code: "max_uses_exceeded".to_string(),
        };
        let data = serde_json::to_string(&err).expect("serialise");
        assert_eq!(serde_json::from_str::<WebSearchToolResultError>(&data).expect("parse"), err);
    }

    #[test]
    fn a_citation_round_trips_through_json() {
        let citation = Citation {
            citation_type: "web_search_result_location".to_string(),
            url: "https://example.com".to_string(),
            title: "Example".to_string(),
            encrypted_index: "enc123".to_string(),
            cited_text: "Some cited text...".to_string(),
        };
        let data = serde_json::to_string(&citation).expect("serialise");
        assert_eq!(serde_json::from_str::<Citation>(&data).expect("parse"), citation);
    }

    // -- base64 --------------------------------------------------------------

    #[test]
    fn the_base64_decoder_matches_the_rfc4648_vectors() {
        for (encoded, want) in [
            ("", ""),
            ("Zg==", "f"),
            ("Zm8=", "fo"),
            ("Zm9v", "foo"),
            ("Zm9vYg==", "foob"),
            ("Zm9vYmE=", "fooba"),
            ("Zm9vYmFy", "foobar"),
        ] {
            assert_eq!(
                decode_base64_std(encoded).expect(encoded),
                want.as_bytes(),
                "decode({encoded:?})"
            );
        }
        assert_eq!(decode_base64_std("+/8=").expect("+/8="), vec![0xfb, 0xff]);
    }

    #[test]
    fn the_base64_decoder_refuses_bad_input() {
        // Wrong length, illegal character, URL alphabet, padding in the middle.
        for bad in ["Zg=", "Zm9v!!!!", "-_8=", "Z=g=", "Zm9\n"] {
            assert!(decode_base64_std(bad).is_err(), "decode({bad:?}) should have failed");
        }
    }

    #[test]
    fn base64_decode_then_encode_is_the_identity_for_canonical_input() {
        let decoded = decode_base64_std(TEST_IMAGE).expect("decode");
        assert_eq!(crate::registry::base64_std(&decoded), TEST_IMAGE);
    }
}
