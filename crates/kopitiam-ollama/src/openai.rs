//! The OpenAI-compatible wire types, and the two-way translation to ollama's
//! native ones.
//!
//! **Upstream:** `openai/openai.go` (ollama, MIT, pinned at
//! `4713800b08b2ddf5e14acf8398953cf7b12f169b`). Every public item below names
//! the exact Go symbol it came from.
//!
//! ## What this module is for
//!
//! `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`
//! and `/v1/audio/transcriptions` are the OpenAI surface ollama speak, so that
//! every OpenAI SDK on earth can point at a local server and just work. None of
//! those endpoints got their own engine -- they are a **translation layer**: an
//! OpenAI request goes in, an ollama-native [`crate::routes`] request comes out,
//! the native response come back, and an OpenAI-shaped response goes out.
//!
//! So the translation IS the substance here. The types boring on purpose; the
//! field mapping is where correctness live.
//!
//! ```text
//!   ChatCompletionRequest  --from_chat_request-->  routes::ChatRequest
//!   routes::ChatResponse   --to_chat_completion->  ChatCompletion         (buffered)
//!   routes::ChatResponse   --to_chunks----------->  [ChatCompletionChunk]  (streamed)
//! ```
//!
//! ## No server in here, same as `routes`
//!
//! Upstream already split this cleanly: `openai/openai.go` is **pure** -- no
//! gin, no `http.ResponseWriter`, no goroutines. The HTTP plumbing (the
//! `ChatWriter` / `CompleteWriter` / `ListWriter` gin middleware that call into
//! these functions, buffer the body, and write `data: ...\n\n` SSE frames) live
//! one package over in `middleware/openai.go`, and that one is
//! [`crate::middleware`]'s job, not this module's. Nothing here opens a socket
//! or write a byte to a client.
//!
//! **One deliberate divergence from that purity, said once here and again at
//! each site:** upstream's [`to_chunk`] / [`to_chunks`] / [`to_complete_chunk`]
//! call `time.Now().Unix()` inside the conversion. We take the timestamp as a
//! `created` **parameter** instead. Reaching for an ambient clock inside a pure
//! conversion makes the output untestable (cannot assert on a value that change
//! every call) and would smuggle a side effect into a module that otherwise got
//! none. [`now_unix`] is provided so the adapter got one obvious place to get
//! the number from.
//!
//! ## The fiddly bits, all in one place
//!
//! Anybody touching this module should know these first, hor:
//!
//! * **`stop` is a string OR an array.** OpenAI accept both; ollama's
//!   `options["stop"]` is always `[]string`. See [`from_chat_request`] and
//!   [`from_complete_request`] -- and note the two disagree on a bad element:
//!   chat **silently skips** a non-string, completions **errors**. That
//!   asymmetry is upstream's, and it is preserved.
//! * **`max_tokens` becomes `num_predict`.** There is no `max_completion_tokens`
//!   in this pinned revision -- the newer OpenAI spelling simply is not accepted
//!   yet, and pretending otherwise here would silently ignore the field. If you
//!   reading this after upstream add it, that is the moment to add it, not
//!   before.
//! * **`temperature` and `top_p` default to `1.0`, NOT to ollama's defaults.**
//!   An OpenAI client that send neither expect OpenAI's defaults, so the
//!   conversion write them in explicitly. Drop them and the request falls
//!   through to [`crate::options`]'s defaults (0.8 / 0.9), which quietly change
//!   behaviour for every OpenAI client.
//! * **`finish_reason`.** ollama's `done_reason` (`"stop"`, `"length"`,
//!   `"load"`, `"unload"`, ...) pass through **unchanged**, with exactly one
//!   override: if the turn produced tool calls, the reason becomes
//!   `"tool_calls"`. Empty `done_reason` -> `null`, never `""`.
//! * **`system_fingerprint` is the literal `"fp_ollama"`**, on every response
//!   shape. Clients key cache behaviour off it.
//! * **`stream_options.include_usage`** is read by the middleware, not here --
//!   this module only define the type plus [`to_usage`] / [`to_usage_generate`].
//!
//! ## Streaming: OpenAI send deltas, ollama send whole messages
//!
//! An OpenAI chunk carry a `delta`, and [`to_chunk`] always stamp
//! `delta.role = "assistant"` (upstream do the same on **every** chunk, not
//! only the first -- see the note on [`to_chunk`]). The genuinely subtle part is
//! [`to_chunks`]: when one ollama chunk carry **both** thinking and
//! content/tool-calls, it gets split into **two** OpenAI chunks -- reasoning
//! first, then content -- because an OpenAI delta got no way to say "here is
//! reasoning and content in the same breath" that clients reliably handle. Both
//! halves share one `created`, the finish reason ride on the second, and the
//! logprobs ride on the **first only**.
//!
//! ## Errors
//!
//! OpenAI's error envelope is `{"error":{"message":..,"type":..,"param":..,
//! "code":..}}`; ollama's is a bare `{"error":"..."}` (or
//! [`crate::routes::StatusError`]). [`new_error`] build the OpenAI side from an
//! HTTP status. Unwrapping the ollama side and picking the status is the
//! middleware's seam, not ours.
//!
//! ## nil-vs-empty, same divergence as `routes`
//!
//! Go can tell `null` (nil slice) from `[]` (empty slice); Rust's `Vec` cannot.
//! `ListCompletion::data`, `EmbeddingList::data` and
//! `ChatCompletionRequest::messages` / `::tools` are the fields where upstream
//! may emit `null` and we always emit `[]`. We accept both inbound, so no client
//! break. See [`crate::routes`]'s module header for the full reasoning.

use serde::{Deserialize, Serialize};

use crate::api::{self, ThinkLevel, ThinkValue, Tool, ToolCallArguments};
use crate::registry::base64_std;
use crate::routes::{
    ChatRequest, ChatResponse, EmbedResponse, GenerateRequest, GenerateResponse, ListResponse,
    Logprob, ShowResponse, Timestamp,
};

/// **Upstream:** `var finishReasonToolCalls = "tool_calls"`.
pub const FINISH_REASON_TOOL_CALLS: &str = "tool_calls";

/// **Upstream:** the `SystemFingerprint: "fp_ollama"` literal, repeated at every
/// construction site. Hoisted into a constant here because four separate
/// response shapes carry it, and a typo in one of them stay invisible until
/// somebody's cache key go wrong.
pub const SYSTEM_FINGERPRINT: &str = "fp_ollama";

// ===========================================================================
// Errors
// ===========================================================================

/// Everything [`from_chat_request`], [`from_complete_request`] and
/// [`from_completion_tool_call`] can refuse.
///
/// The `#[error(..)]` strings are **byte-for-byte upstream's**, because the
/// middleware put them straight into `{"error":{"message": ...}}` and clients
/// match on them. Reword one and you break somebody's error handling, so don't.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OpenAiError {
    /// **Upstream:** `errors.New("invalid message format")` -- a content part
    /// that is not an object, or got a `type` we don't know, or whose `text` /
    /// `image_url` is the wrong shape.
    #[error("invalid message format")]
    InvalidMessageFormat,

    /// **Upstream:** `fmt.Errorf("invalid message content type: %T", content)`.
    /// The `%T` rendering follow Go's names for a value decoded into `any` --
    /// see [`go_type_name`].
    #[error("invalid message content type: {0}")]
    InvalidMessageContentType(String),

    /// **Upstream:** `errors.New("invalid input_audio format")`.
    #[error("invalid input_audio format")]
    InvalidInputAudioFormat,

    /// **Upstream:** `errors.New("invalid input_audio format: missing data")`.
    #[error("invalid input_audio format: missing data")]
    InvalidInputAudioMissingData,

    /// **Upstream:** `fmt.Errorf("invalid input_audio base64 data: %w", err)`.
    /// The wrapped text is Go's own base64 error, e.g.
    /// `illegal base64 data at input byte 3` -- reproduced in
    /// [`base64_std_decode_error`].
    #[error("invalid input_audio base64 data: {0}")]
    InvalidInputAudioBase64(String),

    /// **Upstream:** `errors.New("image URLs are not currently supported, please
    /// use base64 encoded data instead")`. An `http://` / `https://` image URL
    /// would need the server to go fetch it, and KOPITIAM is offline-first, so
    /// this refusal suit us fine.
    #[error("image URLs are not currently supported, please use base64 encoded data instead")]
    ImageUrlUnsupported,

    /// **Upstream:** `errors.New("invalid image input")` -- neither a known
    /// `data:image/<jpeg|jpg|png|webp>;base64,` prefix nor the bare
    /// `data:;base64,` form, or the payload is not valid base64.
    #[error("invalid image input")]
    InvalidImageInput,

    /// **Upstream:** `errors.New("invalid tool call arguments")` -- the
    /// `function.arguments` string is not a JSON object.
    #[error("invalid tool call arguments")]
    InvalidToolCallArguments,

    /// **Upstream:** `fmt.Errorf("invalid type for 'stop' field: %T", s)`. Only
    /// [`from_complete_request`] raise this one; the chat path skip silently.
    #[error("invalid type for 'stop' field: {0}")]
    InvalidStopFieldType(String),

    /// **Upstream:** `fmt.Errorf("invalid reasoning value: '%s' (must be
    /// \"high\", \"medium\", \"low\", \"max\", or \"none\")", effort)`.
    #[error(
        "invalid reasoning value: '{0}' (must be \"high\", \"medium\", \"low\", \"max\", or \"none\")"
    )]
    InvalidReasoningValue(String),
}

/// Go's `%T` for a value that came out of `encoding/json` into an `any`.
///
/// `encoding/json` only ever produce six dynamic types for `any`, and this is
/// the exact set: `nil`, `bool`, `float64` (**every** JSON number, integer or
/// not), `string`, `[]interface {}`, `map[string]interface {}`. Note the space
/// inside `interface {}` -- that is how Go print it, and the error string is a
/// contract, so keep it.
fn go_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "<nil>",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "float64",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "[]interface {}",
        serde_json::Value::Object(_) => "map[string]interface {}",
    }
}

// ===========================================================================
// Wire types
// ===========================================================================

/// The inner half of OpenAI's error envelope. **Upstream:** `openai.Error`.
///
/// `param` and `code` carry **no** `omitempty`, so both always emitted --
/// `null` when unset. An SDK doing `err.code == null` must still see the key.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Error {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    /// Go `any`, no `omitempty`. Never populated by [`new_error`]; upstream keep
    /// the field because the OpenAI schema got it.
    #[serde(default)]
    pub param: Option<serde_json::Value>,
    /// Go `*string`, no `omitempty`.
    #[serde(default)]
    pub code: Option<String>,
}

/// The whole error body. **Upstream:** `openai.ErrorResponse`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: Error,
}

/// One OpenAI message -- serve as a request message, a response `message`, and
/// a streaming `delta`, all three. **Upstream:** `openai.Message`.
///
/// ## Why `content` is a raw [`serde_json::Value`]
///
/// Upstream type it `any` with **no** `omitempty`, and that is load-bearing in
/// three separate ways:
///
/// 1. It can be a plain **string** (the common case),
/// 2. or an **array of parts** (`{"type":"text",...}`, `{"type":"image_url",...}`,
///    `{"type":"input_audio",...}`) for multimodal input,
/// 3. or **absent / `null`**, which is legal *only* when `tool_calls` is there.
///
/// A typed enum would reject case 3's cousins (a number, a bool) at
/// **deserialise** time with a serde error, but upstream reject them later, at
/// **conversion** time, with `invalid message content type: float64` -- and only
/// when there are no tool calls. Keeping the raw `Value` is what preserve that,
/// and it make the Go type switch in [`from_chat_request`] a one-to-one match.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// **Not** lowercased on the way in, unlike [`crate::api::Message::role`].
    /// Upstream's `openai.Message` got no custom unmarshaller, so `"User"`
    /// survive as `"User"` and is handed to ollama as-is.
    pub role: String,

    /// No `omitempty` -- always emitted, `null` when unset. See the type docs.
    #[serde(default)]
    pub content: serde_json::Value,

    /// OpenAI's name for what ollama call `thinking`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning: String,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,

    /// On a `role: "tool"` message this is the **tool's** name. Used to recover
    /// `api.Message.ToolName`, falling back to [`name_from_tool_call_id`].
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,

    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub tool_call_id: String,
}

/// **Upstream:** `openai.ChoiceLogprobs`. `content` got no `omitempty`, but the
/// whole struct only ever hangs off a pointer, so in practice you never see it
/// null.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChoiceLogprobs {
    #[serde(default)]
    pub content: Vec<Logprob>,
}

/// One buffered choice. **Upstream:** `openai.Choice`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub index: i64,
    pub message: Message,
    /// No `omitempty` -- `null` while the turn not finished yet, never `""`.
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// One streamed choice. **Upstream:** `openai.ChunkChoice`. Same as [`Choice`]
/// except the message is called `delta`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: i64,
    pub delta: Message,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// A `/v1/completions` choice, buffered or streamed. **Upstream:**
/// `openai.CompleteChunkChoice`. Legacy completions got no message, just `text`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompleteChunkChoice {
    pub text: String,
    pub index: i64,
    #[serde(default)]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Token accounting. **Upstream:** `openai.Usage`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

/// `response_format`. **Upstream:** `openai.ResponseFormat`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchema>,
}

/// **Upstream:** `openai.JsonSchema`, whose `Schema` is a `json.RawMessage` with
/// no `omitempty` -- so an absent schema serialise as `"schema":null`, and that
/// null is exactly what make [`from_chat_request`] leave `format` unset.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JsonSchema {
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
}

/// `POST /v1/embeddings`. **Upstream:** `openai.EmbedRequest`.
///
/// There is **no** `from_embed_request` here, and that is not an omission:
/// upstream's middleware build the native `api.EmbedRequest` inline (including
/// the `input == "" -> [""]` fixup and the `encoding_format` validation), so
/// that logic belong to [`crate::middleware`]. This module own the type and the
/// response side ([`to_embedding_list`]) only.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbedRequest {
    /// Go `any`, no `omitempty`: a string, or an array of strings.
    #[serde(default)]
    pub input: serde_json::Value,
    pub model: String,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    pub dimensions: i64,
    /// `"float"` or `"base64"`. Empty means float. Anything else is a 400 from
    /// the middleware, never from here.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub encoding_format: String,
}

/// **Upstream:** `openai.StreamOptions`. Only field that exist, and the
/// middleware read it to decide whether to emit the final usage-only chunk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

/// **Upstream:** `openai.Reasoning` -- the nested
/// `{"reasoning":{"effort":".."}}` spelling. The flat `reasoning_effort` is the
/// other one; see [`ChatCompletionRequest::reasoning_effort`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reasoning {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub effort: String,
}

/// `POST /v1/chat/completions`. **Upstream:** `openai.ChatCompletionRequest`.
///
/// Almost nothing here carry `omitempty`, so a serialised request keep every key
/// -- `"max_tokens":null`, `"stop":null`, `"logprobs":null` and friends. That is
/// upstream's shape; adding `skip_serializing_if` would be changing a wire
/// format, not tidying one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    /// No `omitempty` upstream; nil would be `null` there, `[]` here -- see the
    /// module header's nil-vs-empty note.
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Becomes `options["num_predict"]`. There is deliberately no
    /// `max_completion_tokens` -- see the module header.
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default)]
    pub seed: Option<i64>,
    /// Go `any`: a string, or an array of strings. See [`from_chat_request`].
    #[serde(default)]
    pub stop: serde_json::Value,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Reused straight from [`crate::api`] -- OpenAI's tool schema and ollama's
    /// are the same shape, which is why upstream type this `[]api.Tool` and not
    /// an openai-local struct.
    #[serde(default)]
    pub tools: Vec<Tool>,
    /// The nested spelling. Win over [`Self::reasoning_effort`] whenever it is
    /// present at all -- upstream check this one first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// The flat spelling. Same accepted values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// `i32` (not Go's `int`) so it drop straight into
    /// [`crate::routes::ChatRequest::top_logprobs`], where
    /// `validate_top_logprobs` enforce the 0..=20 range.
    #[serde(default)]
    pub top_logprobs: i32,
    /// Render the prompt and return it instead of calling the model. The leading
    /// underscore is upstream's marker for "debug, unstable".
    #[serde(rename = "_debug_render_only", default)]
    pub debug_render_only: bool,
}

/// A buffered `/v1/chat/completions` reply. **Upstream:**
/// `openai.ChatCompletion`.
///
/// `usage,omitempty` on a **struct** is a Go no-op -- it is always emitted. Same
/// trap as `ShowResponse.details`; don't "fix" it into a `skip_serializing_if`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletion {
    pub id: String,
    /// Always `"chat.completion"`.
    pub object: String,
    /// Unix **seconds**, taken from the ollama response's `created_at`.
    pub created: i64,
    pub model: String,
    /// Always [`SYSTEM_FINGERPRINT`].
    pub system_fingerprint: String,
    #[serde(default)]
    pub choices: Vec<Choice>,
    /// `omitempty` on a struct -- always emitted. See the type docs.
    #[serde(default)]
    pub usage: Usage,
    #[serde(rename = "_debug_info", default, skip_serializing_if = "Option::is_none")]
    pub debug_info: Option<crate::routes::DebugInfo>,
}

/// One streamed SSE frame. **Upstream:** `openai.ChatCompletionChunk`.
///
/// `usage` here is a **pointer** with `omitempty`, unlike [`ChatCompletion`]'s
/// -- so it really do disappear from ordinary chunks, and only show up on the
/// final usage-only frame the middleware emit when
/// `stream_options.include_usage` is set.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: String,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: String,
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// `POST /v1/completions` -- the legacy text-completion endpoint. **Upstream:**
/// `openai.CompletionRequest`.
///
/// Upstream carry a TODO (ollama issue #5259): `prompt` is a plain `string`
/// only, so `[]string`, `[]int` and `[][]int` prompts are **not** supported. We
/// port the limitation as-is rather than invent a wider type nothing else
/// understand.
///
/// Note the penalties and `top_p` are **non-pointer** here where the chat
/// request use pointers -- so "unset" and "zero" cannot be told apart, and
/// [`from_complete_request`] treat them accordingly. Upstream's asymmetry, kept.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    /// Non-pointer: always written into `options`, even at 0.
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub max_tokens: Option<i64>,
    /// Non-pointer: always written into `options`, even at 0.
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stop: serde_json::Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// `f32` here where the chat request use `f64` -- upstream's asymmetry.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Non-pointer, so `0.0` mean "unset" and becomes `1.0`.
    #[serde(default)]
    pub top_p: f32,
    /// Fill-in-the-middle text that come **after** the prompt.
    #[serde(default)]
    pub suffix: String,
    /// OpenAI's legacy completions spell this as a **count**, not a bool: `n`
    /// alternatives per token. Non-zero turn logprobs on and set
    /// `top_logprobs = n`.
    #[serde(default)]
    pub logprobs: Option<i32>,
    #[serde(rename = "_debug_render_only", default)]
    pub debug_render_only: bool,
}

/// A buffered `/v1/completions` reply. **Upstream:** `openai.Completion`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Completion {
    pub id: String,
    /// Always `"text_completion"`.
    pub object: String,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: String,
    #[serde(default)]
    pub choices: Vec<CompleteChunkChoice>,
    /// `omitempty` on a struct -- always emitted.
    #[serde(default)]
    pub usage: Usage,
}

/// One streamed `/v1/completions` frame. **Upstream:** `openai.CompletionChunk`.
///
/// Note the field **order** differ from [`Completion`] upstream (`choices`
/// before `model`). Go's `encoding/json` emit struct fields in declaration
/// order, so the two shapes really do serialise with keys in a different order;
/// cosmetic for any JSON parser, and we keep the declaration order anyway so a
/// byte-diff against upstream stay readable.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionChunk {
    pub id: String,
    /// Always `"text_completion"` -- yes, same as the buffered object, not a
    /// `.chunk` variant. That is upstream's, and clients rely on it.
    pub object: String,
    pub created: i64,
    #[serde(default)]
    pub choices: Vec<CompleteChunkChoice>,
    pub model: String,
    pub system_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A tool call in OpenAI's shape. **Upstream:** `openai.ToolCall`.
///
/// The differences from [`crate::api::ToolCall`] are the whole reason this type
/// exist, so know them:
///
/// * `arguments` is a **JSON string**, not an object -- `"{\"city\":\"SG\"}"`.
///   That is OpenAI's format and it is why [`to_tool_calls`] marshal and
///   [`from_completion_tool_call`] unmarshal.
/// * `index` sit on the **call**, where ollama keep it on the *function*.
/// * `type` is always the literal `"function"`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub index: i64,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

/// The inner half of [`ToolCall`]. **Upstream:** an *anonymous* struct inside
/// `openai.ToolCall`; named here because Rust got no anonymous structs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// A JSON **object serialised to a string**. An argumentless call come out
    /// as `"{}"`, never `""` and never `"null"`.
    pub arguments: String,
}

/// One row of `/v1/models`. **Upstream:** `openai.Model`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    /// Unix seconds of the model's `modified_at`.
    pub created: i64,
    /// The model name's **namespace** -- `"library"` for an unqualified name,
    /// otherwise whatever the name say. Not a vendor string: OpenAI's
    /// `owned_by` is repurposed as ollama's namespace. See
    /// [`to_list_completion`].
    pub owned_by: String,
}

/// One embedding. **Upstream:** `openai.Embedding`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// Always `"embedding"`.
    pub object: String,
    /// Go `any`: either a `[]float32` (the `"float"` encoding) or a base64
    /// **string** (the `"base64"` encoding). See [`to_embedding_list`].
    pub embedding: serde_json::Value,
    pub index: i64,
}

/// `GET /v1/models`. **Upstream:** `openai.ListCompletion`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListCompletion {
    /// Always `"list"`.
    pub object: String,
    #[serde(default)]
    pub data: Vec<Model>,
}

/// `POST /v1/embeddings` reply. **Upstream:** `openai.EmbeddingList`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingList {
    /// Always `"list"` -- except on the empty-input path, where upstream return
    /// a zero value and this come out `""`. See [`to_embedding_list`].
    pub object: String,
    #[serde(default)]
    pub data: Vec<Embedding>,
    pub model: String,
    /// `omitempty` on a struct -- always emitted.
    #[serde(default)]
    pub usage: EmbeddingUsage,
}

/// **Upstream:** `openai.EmbeddingUsage`. Embeddings got no completion tokens,
/// so `total_tokens == prompt_tokens` always.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: i64,
    pub total_tokens: i64,
}

/// `POST /v1/audio/transcriptions` reply. **Upstream:**
/// `openai.TranscriptionResponse`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

/// The parsed multipart form for `/v1/audio/transcriptions`. **Upstream:**
/// `openai.TranscriptionRequest` -- note it carry **no** json tags at all,
/// because it never come off the wire as JSON; the middleware fill it in from a
/// multipart form. So no `Serialize`/`Deserialize` here either.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TranscriptionRequest {
    pub model: String,
    /// Raw audio bytes, straight out of the form file part. Base64-encoded on
    /// the way into [`crate::api::Message::images`] -- see
    /// [`from_transcription_request`].
    pub audio_data: Vec<u8>,
    /// `"json"`, `"text"` or `"verbose_json"`. Read by the middleware only.
    pub response_format: String,
    pub language: String,
    pub prompt: String,
}

// ===========================================================================
// ollama -> OpenAI
// ===========================================================================

/// Build the OpenAI error envelope for an HTTP status. **Upstream:** `NewError`.
///
/// Only three buckets exist, and everything that is not 400 or 404 is
/// `"api_error"` -- 500s and 503s included. `param` and `code` stay unset.
///
/// ```
/// # use kopitiam_ollama::openai::new_error;
/// assert_eq!(new_error(400, "bad").error.error_type, "invalid_request_error");
/// assert_eq!(new_error(404, "gone").error.error_type, "not_found_error");
/// assert_eq!(new_error(503, "busy").error.error_type, "api_error");
/// ```
pub fn new_error(code: u16, message: &str) -> ErrorResponse {
    let etype = match code {
        400 => "invalid_request_error",
        404 => "not_found_error",
        _ => "api_error",
    };
    ErrorResponse {
        error: Error {
            message: message.to_string(),
            error_type: etype.to_string(),
            param: None,
            code: None,
        },
    }
}

/// **Upstream:** `ToUsage`. Chat token accounting, straight off the metrics.
pub fn to_usage(r: &ChatResponse) -> Usage {
    let prompt = i64::from(r.metrics.prompt_eval_count);
    let completion = i64::from(r.metrics.eval_count);
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

/// **Upstream:** `ToUsageGenerate`. Same arithmetic, `/v1/completions` side.
pub fn to_usage_generate(r: &GenerateResponse) -> Usage {
    let prompt = i64::from(r.metrics.prompt_eval_count);
    let completion = i64::from(r.metrics.eval_count);
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

/// ollama tool calls -> OpenAI tool calls. **Upstream:** `ToToolCalls`.
///
/// Three things move: `type` becomes the literal `"function"`, the *function's*
/// `index` gets hoisted onto the **call**, and the argument map is serialised
/// into a JSON **string**.
///
/// **Divergence, and it is a shrinking one:** upstream log and `continue` if
/// `json.Marshal` of the arguments fail, leaving that entry's `arguments` as
/// `""`. Our argument map is an `IndexMap<String, Value>`, whose serialisation
/// cannot fail, so [`ToolCallArguments::to_json_string`] give `"{}"` for the
/// empty case and a real object otherwise -- the `continue` branch is
/// unreachable here, so it is not ported. Upstream's own tests never exercise it
/// either.
pub fn to_tool_calls(tc: &[api::ToolCall]) -> Vec<ToolCall> {
    tc.iter()
        .map(|c| ToolCall {
            id: c.id.clone(),
            index: c.function.index as i64,
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: c.function.name.clone(),
                arguments: c.function.arguments.to_json_string(),
            },
        })
        .collect()
}

/// A buffered ollama chat response -> an OpenAI `chat.completion`.
/// **Upstream:** `ToChatCompletion`.
///
/// The mapping, field by field:
///
/// | ollama | OpenAI |
/// |---|---|
/// | `created_at` (RFC 3339) | `created` (unix seconds, via [`timestamp_unix`]) |
/// | `message.content` | `choices[0].message.content` |
/// | `message.thinking` | `choices[0].message.reasoning` |
/// | `message.tool_calls` | `choices[0].message.tool_calls` (see [`to_tool_calls`]) |
/// | `done_reason` | `choices[0].finish_reason`, **unless** tool calls -> `"tool_calls"` |
/// | `logprobs` | `choices[0].logprobs.content`, or absent when empty |
/// | `metrics` | `usage` |
///
/// There is always **exactly one** choice, index 0 -- ollama got no `n`
/// parameter, so multiple choices can never happen.
///
/// **What would make this wrong:** emitting `finish_reason: ""` instead of
/// `null` while the turn still going (SDKs read `""` as a finished turn), or
/// forgetting the tool-call override (then the client never dispatch the tool).
pub fn to_chat_completion(id: &str, r: &ChatResponse) -> ChatCompletion {
    let tool_calls = to_tool_calls(&r.message.tool_calls);
    let logprobs = (!r.logprobs.is_empty()).then(|| ChoiceLogprobs {
        content: r.logprobs.clone(),
    });

    let finish_reason = if !tool_calls.is_empty() {
        Some(FINISH_REASON_TOOL_CALLS.to_string())
    } else if !r.done_reason.is_empty() {
        Some(r.done_reason.clone())
    } else {
        None
    };

    ChatCompletion {
        id: id.to_string(),
        object: "chat.completion".to_string(),
        created: timestamp_unix(&r.created_at).unwrap_or(0),
        model: r.model.clone(),
        system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: r.message.role.clone(),
                content: serde_json::Value::String(r.message.content.clone()),
                reasoning: r.message.thinking.clone(),
                tool_calls,
                ..Default::default()
            },
            finish_reason,
            logprobs,
        }],
        usage: to_usage(r),
        debug_info: r.debug_info.clone(),
    }
}

/// One ollama chat chunk -> one OpenAI `chat.completion.chunk`. **Upstream:**
/// the unexported `toChunk`; also exported upstream as `ToChunk`, which carry
/// *Deprecated: use ToChunks for streaming conversion*.
///
/// Prefer [`to_chunks`]. This one stay because the middleware still call it on
/// one path (building the final usage-only frame when `to_chunks` somehow gave
/// back nothing).
///
/// Two things to know:
///
/// * **`delta.role` is `"assistant"` on every chunk**, not only the first.
///   OpenAI's own servers send the role once then omit it; ollama repeat it, and
///   every SDK tolerate that because a repeated identical role is a no-op merge.
///   Don't "fix" it to first-chunk-only -- that change bytes clients already
///   parse, for no gain.
/// * **`finish_reason` take the tool-call override from `tool_call_sent` too**,
///   not only from this chunk's calls. The middleware set that flag once it has
///   streamed any tool call, so the *final* chunk -- which carry the reason but
///   no calls -- still report `"tool_calls"`.
///
/// `created` is a parameter, not `time.Now()`. See the module header.
pub fn to_chunk(
    id: &str,
    r: &ChatResponse,
    tool_call_sent: bool,
    created: i64,
) -> ChatCompletionChunk {
    let tool_calls = to_tool_calls(&r.message.tool_calls);
    let logprobs = (!r.logprobs.is_empty()).then(|| ChoiceLogprobs {
        content: r.logprobs.clone(),
    });

    let finish_reason = if r.done_reason.is_empty() {
        None
    } else if tool_call_sent || !tool_calls.is_empty() {
        Some(FINISH_REASON_TOOL_CALLS.to_string())
    } else {
        Some(r.done_reason.clone())
    };

    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: r.model.clone(),
        system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Message {
                role: "assistant".to_string(),
                content: serde_json::Value::String(r.message.content.clone()),
                reasoning: r.message.thinking.clone(),
                tool_calls,
                ..Default::default()
            },
            finish_reason,
            logprobs,
        }],
        usage: None,
    }
}

/// One ollama chat chunk -> **one or two** OpenAI chunks. **Upstream:**
/// `ToChunks`. This is the one the streaming path should use.
///
/// ## Why it can return two
///
/// ollama stream whole messages, so a single chunk can carry thinking **and**
/// content, or thinking **and** tool calls, in the same object. OpenAI's `delta`
/// got no agreed way to express that pairing, so a mixed chunk gets split:
///
/// 1. **reasoning chunk** -- `delta.reasoning` only. Content cleared, tool calls
///    cleared, `finish_reason` forced to `null` (turn not over yet).
/// 2. **content / tool-call chunk** -- `delta.reasoning` cleared, everything
///    else as normal, and it carry the real `finish_reason`.
///
/// Two details that look arbitrary but are not:
///
/// * Both chunks share **one** `created`. They are one logical emission, and a
///   client ordering by timestamp must not see them a second apart.
/// * The **logprobs ride on the first chunk only**. Upstream's own comment is
///   honest that this is approximate -- those logprobs may cover tokens that
///   ended up in the *second* chunk, because the split is by field, not by
///   token. Duplicating them onto both would double-count, so first-only is the
///   lesser evil.
///
/// A non-mixed chunk (thinking only, content only, tool calls only, or empty)
/// come back as exactly one chunk, unsplit.
pub fn to_chunks(
    id: &str,
    r: &ChatResponse,
    tool_call_sent: bool,
    created: i64,
) -> Vec<ChatCompletionChunk> {
    let has_mixed_response = !r.message.thinking.is_empty()
        && (!r.message.content.is_empty() || !r.message.tool_calls.is_empty());

    if !has_mixed_response {
        return vec![to_chunk(id, r, tool_call_sent, created)];
    }

    let mut reasoning_chunk = to_chunk(id, r, tool_call_sent, created);
    if let Some(choice) = reasoning_chunk.choices.first_mut() {
        choice.delta.content = serde_json::Value::String(String::new());
        choice.delta.tool_calls.clear();
        choice.finish_reason = None;
    }

    let mut content_or_tool_calls_chunk = to_chunk(id, r, tool_call_sent, created);
    if let Some(choice) = content_or_tool_calls_chunk.choices.first_mut() {
        choice.delta.reasoning.clear();
        choice.logprobs = None;
    }

    vec![reasoning_chunk, content_or_tool_calls_chunk]
}

/// A buffered ollama generate response -> an OpenAI `text_completion`.
/// **Upstream:** `ToCompletion`.
///
/// Note there is **no** tool-call override on `finish_reason` here -- legacy
/// completions got no tools, so `done_reason` pass through untouched.
pub fn to_completion(id: &str, r: &GenerateResponse) -> Completion {
    Completion {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created: timestamp_unix(&r.created_at).unwrap_or(0),
        model: r.model.clone(),
        system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
        choices: vec![CompleteChunkChoice {
            text: r.response.clone(),
            index: 0,
            finish_reason: (!r.done_reason.is_empty()).then(|| r.done_reason.clone()),
            logprobs: None,
        }],
        usage: to_usage_generate(r),
    }
}

/// One streamed generate chunk -> one OpenAI completion frame. **Upstream:**
/// `ToCompleteChunk`.
///
/// Unlike [`to_completion`] this carry **no** `usage` -- the middleware attach
/// one only on the final frame, and only when `stream_options.include_usage` was
/// asked for.
///
/// `created` is a parameter, not `time.Now()`. See the module header.
pub fn to_complete_chunk(id: &str, r: &GenerateResponse, created: i64) -> CompletionChunk {
    CompletionChunk {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: r.model.clone(),
        system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
        choices: vec![CompleteChunkChoice {
            text: r.response.clone(),
            index: 0,
            finish_reason: (!r.done_reason.is_empty()).then(|| r.done_reason.clone()),
            logprobs: None,
        }],
        usage: None,
    }
}

/// `/api/tags` -> `/v1/models`. **Upstream:** `ToListCompletion`.
///
/// Two decisions worth naming:
///
/// * **`id` prefer `model` over `name`.** Both fields exist on a tags row and
///   they can differ (`name` is the legacy spelling); the `model` field is the
///   identity a client should send back, so it win, with `name` as fallback when
///   `model` is empty.
/// * **`owned_by` is the name's namespace**, via [`crate::name::Name::parse`] --
///   so `namespace/exposed-model:latest` give `"namespace"`, and a bare
///   `fallback-name:latest` give `"library"`, because `ParseName` fill the
///   default namespace in. It is **not** a vendor or organisation string;
///   OpenAI's field is simply repurposed.
pub fn to_list_completion(r: &ListResponse) -> ListCompletion {
    let data = r
        .models
        .iter()
        .map(|m| {
            let id = if m.model.is_empty() { &m.name } else { &m.model };
            Model {
                id: id.clone(),
                object: "model".to_string(),
                created: timestamp_unix(&m.modified_at).unwrap_or(0),
                owned_by: crate::name::Name::parse(id).namespace,
            }
        })
        .collect();

    ListCompletion {
        object: "list".to_string(),
        data,
    }
}

/// `/api/embed` -> `/v1/embeddings`. **Upstream:** `ToEmbeddingList`.
///
/// `encoding_format` is `"float"`, `"base64"`, or empty. **Only a
/// case-insensitive `"base64"` select base64**; everything else -- nonsense like
/// `"protobuf"` included -- silently fall back to float. That leniency is
/// upstream's; the middleware is what reject a bad value with a 400, so by the
/// time a request reach here it is already validated.
///
/// **The empty case is not the obvious one.** When there are no embeddings at
/// all, upstream return a **zero** `EmbeddingList` -- so `object` is `""`, not
/// `"list"`, and `model` is `""` even though a model was passed in. Look like an
/// oversight, is what clients get, so it is what we emit.
pub fn to_embedding_list(model: &str, r: &EmbedResponse, encoding_format: &str) -> EmbeddingList {
    if r.embeddings.is_empty() {
        return EmbeddingList::default();
    }

    let base64 = encoding_format.eq_ignore_ascii_case("base64");
    let data = r
        .embeddings
        .iter()
        .enumerate()
        .map(|(i, e)| Embedding {
            object: "embedding".to_string(),
            embedding: if base64 {
                serde_json::Value::String(floats_to_base64(e))
            } else {
                serde_json::Value::Array(
                    e.iter()
                        .map(|f| json_f64(f64::from(*f)))
                        .collect(),
                )
            },
            index: i as i64,
        })
        .collect();

    let prompt_tokens = i64::from(r.prompt_eval_count);
    EmbeddingList {
        object: "list".to_string(),
        data,
        model: model.to_string(),
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }
}

/// Pack floats as **little-endian IEEE-754 binary32**, then standard base64.
/// **Upstream:** `floatsToBase64`, which is `binary.Write(&buf,
/// binary.LittleEndian, floats)` followed by `base64.StdEncoding`.
///
/// Little-endian is not a taste thing, it is the contract: OpenAI's own base64
/// embedding format is little-endian float32, and every client (numpy's
/// `frombuffer(..., dtype='<f4')`, the openai-python SDK) unpack it that way.
/// Emit big-endian and the numbers come back as garbage of the right length --
/// the worst kind of wrong, because it look like it worked.
///
/// ```
/// # use kopitiam_ollama::openai::floats_to_base64;
/// // 4 bytes per float, so 3 floats -> 12 bytes -> 16 base64 chars, no padding.
/// assert_eq!(floats_to_base64(&[0.1, -0.2, 0.3]), "zczMPc3MTL6amZk+");
/// assert_eq!(floats_to_base64(&[]), "");
/// ```
pub fn floats_to_base64(floats: &[f32]) -> String {
    let mut buf = Vec::with_capacity(floats.len() * 4);
    for f in floats {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    base64_std(&buf)
}

/// `/api/show` -> one `/v1/models/{id}` row. **Upstream:** `ToModel`.
///
/// The id is the string the **caller asked for**, not anything out of the show
/// response -- so `/v1/models/qwen3` echo `qwen3` back, and `owned_by` is that
/// same string's namespace.
pub fn to_model(r: &ShowResponse, m: &str) -> Model {
    Model {
        id: m.to_string(),
        object: "model".to_string(),
        created: timestamp_unix(&r.modified_at).unwrap_or(0),
        owned_by: crate::name::Name::parse(m).namespace,
    }
}

// ===========================================================================
// OpenAI -> ollama
// ===========================================================================

/// An OpenAI chat request -> ollama's native one. **Upstream:**
/// `FromChatRequest`.
///
/// This is the big one. What moves where:
///
/// | OpenAI | ollama |
/// |---|---|
/// | `messages[].content` (string) | `messages[].content` |
/// | `messages[].content` (parts array) | **one message per part** -- see below |
/// | `messages[].reasoning` | `messages[].thinking` |
/// | `messages[].tool_calls` | `messages[].tool_calls` (arguments string -> object) |
/// | `messages[].name` / `tool_call_id` | `tool_name` / `tool_call_id` |
/// | `stop` (string or array) | `options["stop"]` (always an array) |
/// | `max_tokens` | `options["num_predict"]` |
/// | `temperature` / `top_p` | same names, **defaulting to 1.0** |
/// | `seed`, `frequency_penalty`, `presence_penalty` | same names, only when set |
/// | `response_format` | `format` |
/// | `reasoning.effort` / `reasoning_effort` | `think` |
/// | `logprobs` / `top_logprobs` | same names |
///
/// ## The parts array fan out into several messages
///
/// A multimodal message becomes **one ollama message per part**: a `text` part
/// become a text message, an `image_url` or `input_audio` part become a message
/// carrying one image. Any `tool_calls` on the original then hang off the
/// **last** message produced -- along with `tool_name`, `tool_call_id` and
/// `thinking`.
///
/// ## Content that is neither string nor array
///
/// `null`, a number, a bool -- all fall to the last branch, which is legal
/// **only if `tool_calls` is present** (an assistant turn that is nothing but a
/// tool call got no content). Otherwise it is
/// [`OpenAiError::InvalidMessageContentType`]. Note that branch does **not**
/// carry `tool_name` across, unlike the string branch -- an upstream
/// inconsistency, ported as-is instead of quietly "fixed", because a tool-result
/// message always got string content in practice and diverging here would be an
/// untested guess.
///
/// ## `temperature` / `top_p` default to 1.0
///
/// Read the module header. Not defaulting them hand OpenAI clients ollama's
/// defaults instead of OpenAI's.
pub fn from_chat_request(r: &ChatCompletionRequest) -> Result<ChatRequest, OpenAiError> {
    let mut messages: Vec<api::Message> = Vec::new();

    for msg in &r.messages {
        let mut tool_name = String::new();
        if msg.role.to_lowercase() == "tool" {
            tool_name = msg.name.clone();
            if tool_name.is_empty() && !msg.tool_call_id.is_empty() {
                tool_name = name_from_tool_call_id(&r.messages, &msg.tool_call_id);
            }
        }

        match &msg.content {
            serde_json::Value::String(content) => {
                messages.push(api::Message {
                    role: msg.role.clone(),
                    content: content.clone(),
                    thinking: msg.reasoning.clone(),
                    tool_calls: from_completion_tool_call(&msg.tool_calls)?,
                    tool_name,
                    tool_call_id: msg.tool_call_id.clone(),
                    ..Default::default()
                });
            }
            serde_json::Value::Array(parts) => {
                for c in parts {
                    let data = c.as_object().ok_or(OpenAiError::InvalidMessageFormat)?;
                    match data.get("type").and_then(serde_json::Value::as_str) {
                        Some("text") => {
                            let text = data
                                .get("text")
                                .and_then(serde_json::Value::as_str)
                                .ok_or(OpenAiError::InvalidMessageFormat)?;
                            messages.push(api::Message {
                                role: msg.role.clone(),
                                content: text.to_string(),
                                ..Default::default()
                            });
                        }
                        Some("image_url") => {
                            // Two accepted shapes: `{"url": "..."}` (the spec)
                            // and a bare string (seen in the wild, tolerated).
                            let url = match data.get("image_url") {
                                Some(serde_json::Value::Object(m)) => m
                                    .get("url")
                                    .and_then(serde_json::Value::as_str)
                                    .ok_or(OpenAiError::InvalidMessageFormat)?,
                                Some(serde_json::Value::String(s)) => s.as_str(),
                                _ => return Err(OpenAiError::InvalidMessageFormat),
                            };
                            messages.push(api::Message {
                                role: msg.role.clone(),
                                images: vec![decode_image_url(url)?],
                                ..Default::default()
                            });
                        }
                        Some("input_audio") => {
                            let audio_map = data
                                .get("input_audio")
                                .and_then(serde_json::Value::as_object)
                                .ok_or(OpenAiError::InvalidInputAudioFormat)?;
                            let b64_data = audio_map
                                .get("data")
                                .and_then(serde_json::Value::as_str)
                                .ok_or(OpenAiError::InvalidInputAudioMissingData)?;
                            let bytes = base64_std_decode(b64_data)
                                .map_err(OpenAiError::InvalidInputAudioBase64)?;
                            messages.push(api::Message {
                                role: msg.role.clone(),
                                images: vec![base64_std(&bytes)],
                                ..Default::default()
                            });
                        }
                        _ => return Err(OpenAiError::InvalidMessageFormat),
                    }
                }

                // We may have pushed several messages above, so the tool calls
                // hang off the last one.
                if !messages.is_empty() && !msg.tool_calls.is_empty() {
                    let tool_calls = from_completion_tool_call(&msg.tool_calls)?;
                    if let Some(last) = messages.last_mut() {
                        last.tool_calls = tool_calls;
                        last.tool_name = tool_name;
                        last.tool_call_id = msg.tool_call_id.clone();
                        last.thinking = msg.reasoning.clone();
                    }
                }
            }
            other => {
                // Content is optional ONLY when tool calls are present.
                if msg.tool_calls.is_empty() {
                    return Err(OpenAiError::InvalidMessageContentType(
                        go_type_name(other).to_string(),
                    ));
                }
                messages.push(api::Message {
                    role: msg.role.clone(),
                    thinking: msg.reasoning.clone(),
                    tool_calls: from_completion_tool_call(&msg.tool_calls)?,
                    tool_call_id: msg.tool_call_id.clone(),
                    ..Default::default()
                });
            }
        }
    }

    let mut options = serde_json::Map::new();

    // `stop` is a string OR an array of strings. A non-string element gets
    // silently DROPPED here -- `from_complete_request` error instead. Upstream
    // asymmetry, kept.
    match &r.stop {
        serde_json::Value::String(s) => {
            options.insert(
                "stop".to_string(),
                serde_json::Value::Array(vec![serde_json::Value::String(s.clone())]),
            );
        }
        serde_json::Value::Array(list) => {
            let stops: Vec<serde_json::Value> =
                list.iter().filter(|v| v.is_string()).cloned().collect();
            options.insert("stop".to_string(), serde_json::Value::Array(stops));
        }
        _ => {}
    }

    if let Some(max_tokens) = r.max_tokens {
        options.insert("num_predict".to_string(), max_tokens.into());
    }

    options.insert(
        "temperature".to_string(),
        json_f64(r.temperature.unwrap_or(1.0)),
    );

    if let Some(seed) = r.seed {
        options.insert("seed".to_string(), seed.into());
    }
    if let Some(v) = r.frequency_penalty {
        options.insert("frequency_penalty".to_string(), json_f64(v));
    }
    if let Some(v) = r.presence_penalty {
        options.insert("presence_penalty".to_string(), json_f64(v));
    }

    options.insert("top_p".to_string(), json_f64(r.top_p.unwrap_or(1.0)));

    // `json_object` is the old OpenAI spelling and map to ollama's bare
    // `"json"` format; `json_schema` pass the schema straight through. Any other
    // type (the ordinary `"text"` included) leave `format` unset.
    let format = r.response_format.as_ref().and_then(|rf| {
        match rf.format_type.trim().to_lowercase().as_str() {
            "json_object" => Some(serde_json::Value::String("json".to_string())),
            "json_schema" => rf.json_schema.as_ref().and_then(|js| js.schema.clone()),
            _ => None,
        }
    });

    // The nested `reasoning.effort` is checked FIRST; the flat
    // `reasoning_effort` only apply when `reasoning` is absent entirely.
    let effort = match (&r.reasoning, &r.reasoning_effort) {
        (Some(reasoning), _) => reasoning.effort.clone(),
        (None, Some(e)) => e.clone(),
        (None, None) => String::new(),
    };

    let think = if effort.is_empty() {
        None
    } else {
        match effort.as_str() {
            "none" => Some(ThinkValue::Bool(false)),
            "low" => Some(ThinkValue::Level(ThinkLevel::Low)),
            "medium" => Some(ThinkValue::Level(ThinkLevel::Medium)),
            "high" => Some(ThinkValue::Level(ThinkLevel::High)),
            "max" => Some(ThinkValue::Level(ThinkLevel::Max)),
            _ => return Err(OpenAiError::InvalidReasoningValue(effort)),
        }
    };

    Ok(ChatRequest {
        model: r.model.clone(),
        messages,
        format,
        options: Some(options),
        stream: Some(r.stream),
        tools: r.tools.clone(),
        think,
        logprobs: r.logprobs.unwrap_or(false),
        top_logprobs: r.top_logprobs,
        debug_render_only: r.debug_render_only,
        ..Default::default()
    })
}

/// Recover a tool's name from an earlier assistant message that called it.
/// **Upstream:** `nameFromToolCallID`.
///
/// A `role: "tool"` message often carry only `tool_call_id`, no `name`, but
/// ollama's templates want the tool's name. So we walk the conversation for the
/// call with that id.
///
/// **Iterated backwards on purpose**: duplicate tool-call ids do happen in long
/// conversations, and last-one-wins is the resilient answer -- the most recent
/// call with that id is the one this result is answering.
///
/// Give back `""` when nothing match, same as upstream.
pub fn name_from_tool_call_id(messages: &[Message], tool_call_id: &str) -> String {
    for msg in messages.iter().rev() {
        for tc in &msg.tool_calls {
            if tc.id == tool_call_id {
                return tc.function.name.clone();
            }
        }
    }
    String::new()
}

/// Decode a base64 **data URI** into the base64 payload ollama want.
/// **Upstream:** `decodeImageURL`.
///
/// Accepted prefixes, and nothing else:
///
/// * `data:image/jpeg;base64,`, `.../jpg`, `.../png`, `.../webp`
/// * `data:;base64,` -- a **blank** mime type, accepted so this endpoint match
///   `/api/chat`, which take unadorned base64.
///
/// `http://` and `https://` are refused flat out: fetching a remote image would
/// need a network, and KOPITIAM is offline-first, so that refusal suit us fine.
///
/// **Divergence forced by the type, and it is wire-equivalent:** upstream return
/// raw `[]byte`, but [`crate::api::Message::images`] is `Vec<String>` holding
/// **base64 text** (because Go marshal `[][]byte` as base64 strings, so the
/// string IS the wire form). So we decode -- to validate, exactly like upstream
/// do -- then **re-encode canonically**. The re-encode is not busywork: Go's
/// base64 decoder is lenient about non-zero trailing bits, so a
/// sloppy-but-accepted input would otherwise be stored verbatim and serialise
/// differently from what upstream would emit for the same bytes.
fn decode_image_url(url: &str) -> Result<String, OpenAiError> {
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(OpenAiError::ImageUrlUnsupported);
    }

    const TYPES: [&str; 4] = ["jpeg", "jpg", "png", "webp"];

    let payload = if let Some(rest) = url.strip_prefix("data:;base64,") {
        rest
    } else {
        let mut found = None;
        for t in TYPES {
            let prefix = format!("data:image/{t};base64,");
            if let Some(rest) = url.strip_prefix(&prefix) {
                found = Some(rest);
                break;
            }
        }
        found.ok_or(OpenAiError::InvalidImageInput)?
    };

    let bytes = base64_std_decode(payload).map_err(|_| OpenAiError::InvalidImageInput)?;
    Ok(base64_std(&bytes))
}

/// OpenAI tool calls -> ollama's. **Upstream:** `FromCompletionToolCall`.
///
/// The one real job is parsing `function.arguments`, which OpenAI ship as a
/// **JSON string**, back into an object -- and insertion order survive the trip,
/// because [`ToolCallArguments`] is an `IndexMap`. See [`crate::api`]'s module
/// docs for why that order is load-bearing.
///
/// Note what is **not** carried across: `index` and `type`. Upstream set only
/// `ID`, `Function.Name` and `Function.Arguments`, so the ollama call's
/// `function.index` stay 0 even when the OpenAI call said otherwise. Faithful,
/// and it matter because a template printing `.Function.Index` would otherwise
/// disagree with upstream.
///
/// An empty or non-object `arguments` string is
/// [`OpenAiError::InvalidToolCallArguments`].
pub fn from_completion_tool_call(
    tool_calls: &[ToolCall],
) -> Result<Vec<api::ToolCall>, OpenAiError> {
    tool_calls
        .iter()
        .map(|tc| {
            let arguments: ToolCallArguments = serde_json::from_str(&tc.function.arguments)
                .map_err(|_| OpenAiError::InvalidToolCallArguments)?;
            Ok(api::ToolCall {
                id: tc.id.clone(),
                function: api::ToolCallFunction {
                    index: 0,
                    name: tc.function.name.clone(),
                    arguments,
                },
            })
        })
        .collect()
}

/// An OpenAI legacy-completion request -> ollama's generate request.
/// **Upstream:** `FromCompleteRequest`.
///
/// Differ from [`from_chat_request`] in three ways, all coming from the Go types
/// being non-pointers here:
///
/// * `frequency_penalty` and `presence_penalty` are **always** written into
///   options, even at 0 -- no way to tell "unset" from "zero".
/// * `top_p` at exactly `0.0` counts as unset and becomes `1.0`.
/// * A non-string element in a `stop` array is a hard **error** here, where the
///   chat path drop it silently.
///
/// `logprobs` is OpenAI's legacy **count** form: `Some(n)` with `n > 0` turn
/// logprobs on and set `top_logprobs = n`. `Some(0)` and `None` both leave it
/// off.
pub fn from_complete_request(r: &CompletionRequest) -> Result<GenerateRequest, OpenAiError> {
    let mut options = serde_json::Map::new();

    match &r.stop {
        serde_json::Value::String(s) => {
            options.insert(
                "stop".to_string(),
                serde_json::Value::Array(vec![serde_json::Value::String(s.clone())]),
            );
        }
        serde_json::Value::Array(list) => {
            let mut stops = Vec::with_capacity(list.len());
            for s in list {
                match s {
                    serde_json::Value::String(_) => stops.push(s.clone()),
                    other => {
                        return Err(OpenAiError::InvalidStopFieldType(
                            go_type_name(other).to_string(),
                        ))
                    }
                }
            }
            options.insert("stop".to_string(), serde_json::Value::Array(stops));
        }
        _ => {}
    }

    if let Some(max_tokens) = r.max_tokens {
        options.insert("num_predict".to_string(), max_tokens.into());
    }

    // f32 -> f64 widening, deliberately NOT re-rounded to the shortest decimal:
    // Go's map hold the float32 value itself, so `0.8f32` really is
    // 0.800000011920929 the moment anything read it as a float64. Matching Go's
    // in-memory value is what keep sampling identical.
    options.insert(
        "temperature".to_string(),
        json_f64(f64::from(r.temperature.unwrap_or(1.0))),
    );

    if let Some(seed) = r.seed {
        options.insert("seed".to_string(), seed.into());
    }

    options.insert(
        "frequency_penalty".to_string(),
        json_f64(f64::from(r.frequency_penalty)),
    );
    options.insert(
        "presence_penalty".to_string(),
        json_f64(f64::from(r.presence_penalty)),
    );

    let top_p = if r.top_p != 0.0 {
        f64::from(r.top_p)
    } else {
        1.0
    };
    options.insert("top_p".to_string(), json_f64(top_p));

    let (logprobs, top_logprobs) = match r.logprobs {
        Some(n) if n > 0 => (true, n),
        _ => (false, 0),
    };

    Ok(GenerateRequest {
        model: r.model.clone(),
        prompt: r.prompt.clone(),
        options: Some(options),
        stream: Some(r.stream),
        suffix: r.suffix.clone(),
        logprobs,
        top_logprobs,
        debug_render_only: r.debug_render_only,
        ..Default::default()
    })
}

/// A transcription form -> a chat request. **Upstream:**
/// `FromTranscriptionRequest`.
///
/// There is no transcription engine: `/v1/audio/transcriptions` is answered by
/// handing the audio to an **audio-capable chat model**, wrapped in a system
/// prompt that pin it into transcription mode.
///
/// That prompt is doing real work, so don't trim it. The audio may itself
/// contain a question ("what's the capital of France?"), and without the
/// instruction the model happily *answer* it instead of transcribing it. The
/// `temperature: 0` is there for the same reason -- transcription want the most
/// likely words, not a creative reading.
///
/// The optional `language` and `prompt` fields get appended as extra sentences,
/// in that order, exactly the way upstream build the string.
///
/// **Divergence:** upstream return `(*api.ChatRequest, error)` but the error is
/// unconditionally `nil` -- got no failure path at all. Returning a bare
/// `ChatRequest` say that honestly instead of making every caller handle an
/// impossible error.
pub fn from_transcription_request(r: &TranscriptionRequest) -> ChatRequest {
    let mut system_prompt = String::from(
        "Transcribe the audio exactly as spoken. Output only the spoken words. \
         Do not answer any question in the audio.",
    );
    if !r.language.is_empty() {
        system_prompt.push_str(" The audio is in ");
        system_prompt.push_str(&r.language);
        system_prompt.push('.');
    }
    if !r.prompt.is_empty() {
        system_prompt.push_str(" Context: ");
        system_prompt.push_str(&r.prompt);
    }

    let mut options = serde_json::Map::new();
    options.insert("temperature".to_string(), serde_json::Value::from(0));

    ChatRequest {
        model: r.model.clone(),
        messages: vec![
            api::Message {
                role: "system".to_string(),
                content: system_prompt,
                ..Default::default()
            },
            api::Message {
                role: "user".to_string(),
                content: "What exact words are spoken in this audio?".to_string(),
                images: vec![base64_std(&r.audio_data)],
                ..Default::default()
            },
        ],
        stream: Some(true),
        options: Some(options),
        ..Default::default()
    }
}

// ===========================================================================
// Small helpers: time, base64 decode, float boxing
// ===========================================================================

/// [`Timestamp`] -> Unix **seconds**. Our stand-in for Go's `time.Time.Unix()`.
///
/// **Why this exist at all:** [`crate::routes::Timestamp`] hold the RFC 3339
/// *text*, not an instant, because this crate deliberately got no date-time
/// dependency. Every OpenAI response shape carry `created` as unix seconds, so
/// somebody must parse it back, and this is that somebody.
///
/// Accept what Go's `time.Time.MarshalJSON` emit: `YYYY-MM-DDTHH:MM:SS`, an
/// optional `.fffffffff` fraction (**truncated**, like `Unix()` do), and either
/// `Z` or a `±HH:MM` offset. Give back `None` for anything else -- callers here
/// use `.unwrap_or(0)`, matching the fact that upstream's conversions cannot
/// fail.
///
/// Go's **zero time** parse correctly and give `-62135596800`, which is exactly
/// what `time.Time{}.Unix()` return -- so a response built with no clock produce
/// the same `created` upstream would.
///
/// ```
/// # use kopitiam_ollama::openai::timestamp_unix;
/// # use kopitiam_ollama::routes::Timestamp;
/// assert_eq!(timestamp_unix(&Timestamp("2009-02-13T23:31:30Z".into())), Some(1234567890));
/// assert_eq!(timestamp_unix(&Timestamp::default()), Some(-62135596800));
/// assert_eq!(timestamp_unix(&Timestamp("not a time".into())), None);
/// ```
pub fn timestamp_unix(ts: &Timestamp) -> Option<i64> {
    let s = ts.as_str().as_bytes();
    // Shortest legal form is `YYYY-MM-DDTHH:MM:SSZ` -- 20 bytes.
    if s.len() < 20 || s[4] != b'-' || s[7] != b'-' || s[13] != b':' || s[16] != b':' {
        return None;
    }
    if !(s[10] == b'T' || s[10] == b't') {
        return None;
    }

    let num = |from: usize, to: usize| -> Option<i64> {
        let mut acc: i64 = 0;
        for &b in &s[from..to] {
            if !b.is_ascii_digit() {
                return None;
            }
            acc = acc * 10 + i64::from(b - b'0');
        }
        Some(acc)
    };

    let year = num(0, 4)?;
    let month = num(5, 7)?;
    let day = num(8, 10)?;
    let hour = num(11, 13)?;
    let minute = num(14, 16)?;
    let second = num(17, 19)?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 60 {
        return None;
    }

    // Skip the optional fractional part; `Unix()` throw it away anyway.
    let mut i = 19;
    if s.get(i) == Some(&b'.') {
        i += 1;
        let start = i;
        while s.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == start {
            return None;
        }
    }

    let offset = match s.get(i) {
        Some(b'Z' | b'z') if i + 1 == s.len() => 0,
        Some(sign @ (b'+' | b'-')) if i + 6 == s.len() && s[i + 3] == b':' => {
            let oh = num(i + 1, i + 3)?;
            let om = num(i + 4, i + 6)?;
            let mag = oh * 3600 + om * 60;
            if *sign == b'-' {
                -mag
            } else {
                mag
            }
        }
        _ => return None,
    };

    let days = days_from_civil(year, month as u32, day as u32);
    Some(days * 86_400 + hour * 3600 + minute * 60 + second - offset)
}

/// `(year, month, day)` -> days since the Unix epoch, proleptic Gregorian.
///
/// **Upstream:** Howard Hinnant's `days_from_civil` (public domain,
/// <https://howardhinnant.github.io/date_algorithms.html>) -- the exact inverse
/// of the `civil_from_days` that [`crate::routes::Timestamp::from_unix_nanos`]
/// already use, written out here for the same reason: no date crate, and the
/// Pure Rust Core promise is worth more than a dependency.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = i64::from(if m > 2 { m - 3 } else { m + 9 }); // [0, 11]
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Unix seconds, right now. The **one** impure function in this module.
///
/// Exist so the streaming adapter got an obvious place to take the `created`
/// value that upstream get from `time.Now().Unix()` inside the conversion -- see
/// the module header for why the conversions themselves take it as a parameter.
///
/// Give back 0 if the system clock is somehow before the epoch, which is the
/// same "don't panic over a clock" posture the rest of the crate take.
pub fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Box an `f64` as a JSON number, falling back to `null` for NaN/infinity.
///
/// JSON got no NaN and no infinity, and neither do `serde_json::Number`. Go
/// would *error* out of `json.Marshal` on such a value; we cannot error here
/// (these options maps are built infallibly), so `null` it is -- and
/// [`crate::options::Options::apply_map`] will reject a null where it want a
/// float, which is the right place for that complaint to surface.
fn json_f64(v: f64) -> serde_json::Value {
    serde_json::Number::from_f64(v)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

fn is_zero_i64(v: &i64) -> bool {
    *v == 0
}

/// Go's `base64.StdEncoding.DecodeString`, error message and all.
///
/// The encode half live in [`crate::registry::base64_std`] and is reused rather
/// than copied. Only the decode half is new here, because nothing else in the
/// crate needed to decode until now.
///
/// Two behaviours copied from Go on purpose:
///
/// * **`\r` and `\n` are ignored**, anywhere in the input -- Go's decoder
///   document this, and MIME-wrapped base64 inside a data URI is a real thing.
/// * **Trailing bits are not checked.** Go accept `"zg=="`-style input whose
///   final group got non-zero unused bits; so do we. That is exactly why
///   [`decode_image_url`] re-encode instead of storing the input verbatim.
///
/// The `Err` payload is Go's own message text (`illegal base64 data at input
/// byte N`), because it gets wrapped into
/// [`OpenAiError::InvalidInputAudioBase64`] and clients read it.
fn base64_std_decode(input: &str) -> Result<Vec<u8>, String> {
    let mut sextets: Vec<u8> = Vec::with_capacity(input.len());
    let mut pad = 0usize;

    for (i, b) in input.bytes().enumerate() {
        match b {
            b'\r' | b'\n' => continue,
            b'=' => {
                pad += 1;
                sextets.push(0);
            }
            _ => {
                if pad > 0 {
                    // Data after padding: Go report the offending byte.
                    return Err(base64_std_decode_error(i));
                }
                match b64_value(b) {
                    Some(v) => sextets.push(v),
                    None => return Err(base64_std_decode_error(i)),
                }
            }
        }
    }

    if pad > 2 || !sextets.len().is_multiple_of(4) {
        // Go report a truncated / over-padded input at the final byte.
        return Err(base64_std_decode_error(input.len()));
    }

    let mut out = Vec::with_capacity(sextets.len() / 4 * 3);
    for chunk in sextets.chunks_exact(4) {
        let n = (u32::from(chunk[0]) << 18)
            | (u32::from(chunk[1]) << 12)
            | (u32::from(chunk[2]) << 6)
            | u32::from(chunk[3]);
        out.push((n >> 16) as u8);
        out.push((n >> 8) as u8);
        out.push(n as u8);
    }
    out.truncate(out.len().saturating_sub(pad));
    Ok(out)
}

/// Go's `base64.CorruptInputError(n).Error()`.
fn base64_std_decode_error(at: usize) -> String {
    format!("illegal base64 data at input byte {at}")
}

fn b64_value(b: u8) -> Option<u8> {
    match b {
        b'A'..=b'Z' => Some(b - b'A'),
        b'a'..=b'z' => Some(b - b'a' + 26),
        b'0'..=b'9' => Some(b - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ToolCallFunction as ApiToolCallFunction;
    use crate::routes::{ListModelResponse, Metrics, TokenLogprob};

    /// Upstream's `testArgs` helper -- build arguments from pairs, in order.
    fn test_args(pairs: &[(&str, serde_json::Value)]) -> ToolCallArguments {
        let mut args = ToolCallArguments::new();
        for (k, v) in pairs {
            args.set(*k, v.clone());
        }
        args
    }

    fn user(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: serde_json::Value::String(content.to_string()),
            ..Default::default()
        }
    }

    fn ts(secs: i64) -> Timestamp {
        Timestamp::from_unix_nanos(secs * 1_000_000_000)
    }

    const IMAGE_PREFIX: &str = "data:image/jpeg;base64,";
    const IMAGE: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

    // -- FromChatRequest ----------------------------------------------------

    #[test]
    fn a_basic_chat_request_carries_model_and_message_through() {
        let req = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![user("Hello")],
            ..Default::default()
        };

        let result = from_chat_request(&req).expect("conversion should succeed");

        assert_eq!(result.model, "test-model");
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
        assert_eq!(result.messages[0].content, "Hello");
    }

    #[test]
    fn reasoning_effort_maps_onto_think_and_none_disables_it() {
        // Upstream: TestFromChatRequest_ReasoningEffort.
        let cases: [(&str, Option<Option<ThinkValue>>); 7] = [
            ("", Some(None)),
            ("high", Some(Some(ThinkValue::Level(ThinkLevel::High)))),
            ("medium", Some(Some(ThinkValue::Level(ThinkLevel::Medium)))),
            ("low", Some(Some(ThinkValue::Level(ThinkLevel::Low)))),
            ("max", Some(Some(ThinkValue::Level(ThinkLevel::Max)))),
            ("none", Some(Some(ThinkValue::Bool(false)))),
            ("extreme", None), // None here means "expect an error"
        ];

        for (effort, want) in cases {
            let req = ChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![user("hi")],
                reasoning_effort: (!effort.is_empty()).then(|| effort.to_string()),
                ..Default::default()
            };
            match (from_chat_request(&req), want) {
                (Ok(result), Some(expected)) => {
                    assert_eq!(result.think, expected, "effort={effort}")
                }
                (Err(e), None) => assert_eq!(
                    e,
                    OpenAiError::InvalidReasoningValue("extreme".to_string()),
                    "effort={effort}"
                ),
                (got, _) => panic!("effort={effort}: unexpected {got:?}"),
            }
        }
    }

    #[test]
    fn the_invalid_reasoning_message_names_all_five_accepted_values() {
        let err = OpenAiError::InvalidReasoningValue("extreme".to_string());
        assert_eq!(
            err.to_string(),
            r#"invalid reasoning value: 'extreme' (must be "high", "medium", "low", "max", or "none")"#
        );
    }

    #[test]
    fn the_nested_reasoning_object_wins_over_the_flat_reasoning_effort() {
        // Upstream check `r.Reasoning != nil` first, so an empty nested object
        // shadow the flat field entirely -- and an empty effort means no think.
        let req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![user("hi")],
            reasoning: Some(Reasoning::default()),
            reasoning_effort: Some("high".to_string()),
            ..Default::default()
        };
        assert_eq!(from_chat_request(&req).expect("ok").think, None);

        let req = ChatCompletionRequest {
            reasoning: Some(Reasoning {
                effort: "low".to_string(),
            }),
            reasoning_effort: Some("high".to_string()),
            ..req
        };
        assert_eq!(
            from_chat_request(&req).expect("ok").think,
            Some(ThinkValue::Level(ThinkLevel::Low))
        );
    }

    #[test]
    fn a_content_parts_array_fans_out_into_one_message_per_part() {
        // Upstream: TestFromChatRequest_WithImage.
        let req = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": format!("{IMAGE_PREFIX}{IMAGE}")}},
                ]),
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = from_chat_request(&req).expect("conversion should succeed");

        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].content, "Hello");
        assert_eq!(result.messages[1].images.len(), 1);
        // Upstream compare raw bytes; our api::Message::images hold the base64
        // text (same wire form), so we compare against the canonical re-encode.
        assert_eq!(result.messages[1].images[0], IMAGE);
    }

    #[test]
    fn a_bare_string_image_url_is_tolerated_as_well_as_the_object_form() {
        let req = ChatCompletionRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "image_url", "image_url": format!("{IMAGE_PREFIX}{IMAGE}")},
                ]),
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = from_chat_request(&req).expect("ok");
        assert_eq!(result.messages[0].images, vec![IMAGE.to_string()]);
    }

    #[test]
    fn a_blank_mime_data_uri_is_accepted_so_it_matches_api_chat() {
        let url = format!("data:;base64,{IMAGE}");
        assert_eq!(decode_image_url(&url).expect("ok"), IMAGE);
    }

    #[test]
    fn http_image_urls_are_refused_because_we_never_fetch() {
        assert_eq!(
            decode_image_url("https://example.com/cat.png"),
            Err(OpenAiError::ImageUrlUnsupported)
        );
        assert_eq!(
            decode_image_url("http://example.com/cat.png"),
            Err(OpenAiError::ImageUrlUnsupported)
        );
    }

    #[test]
    fn an_unknown_data_uri_or_bad_payload_is_invalid_image_input() {
        assert_eq!(
            decode_image_url("data:image/gif;base64,AAAA"),
            Err(OpenAiError::InvalidImageInput)
        );
        assert_eq!(
            decode_image_url("data:;base64,not base64!"),
            Err(OpenAiError::InvalidImageInput)
        );
    }

    #[test]
    fn an_input_audio_part_becomes_an_image_payload() {
        let audio = base64_std(b"fake-wav-bytes");
        let req = ChatCompletionRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
                ]),
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = from_chat_request(&req).expect("ok");
        assert_eq!(result.messages[0].images, vec![audio]);
    }

    #[test]
    fn a_malformed_input_audio_part_names_which_way_it_is_malformed() {
        let missing_object = ChatCompletionRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([{"type": "input_audio", "input_audio": "nope"}]),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(
            from_chat_request(&missing_object),
            Err(OpenAiError::InvalidInputAudioFormat)
        );

        let missing_data = ChatCompletionRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([{"type": "input_audio", "input_audio": {}}]),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(
            from_chat_request(&missing_data),
            Err(OpenAiError::InvalidInputAudioMissingData)
        );

        let bad_base64 = ChatCompletionRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([
                    {"type": "input_audio", "input_audio": {"data": "!!!"}},
                ]),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(
            from_chat_request(&bad_base64),
            Err(OpenAiError::InvalidInputAudioBase64(
                "illegal base64 data at input byte 0".to_string()
            ))
        );
    }

    #[test]
    fn an_unknown_part_type_is_an_invalid_message_format() {
        let req = ChatCompletionRequest {
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([{"type": "video_url", "video_url": "x"}]),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(
            from_chat_request(&req),
            Err(OpenAiError::InvalidMessageFormat)
        );
    }

    #[test]
    fn content_that_is_neither_string_nor_array_needs_tool_calls_to_be_legal() {
        // A number with no tool calls: rejected, and the message name Go's type.
        let bad = ChatCompletionRequest {
            messages: vec![Message {
                role: "assistant".to_string(),
                content: serde_json::json!(42),
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(
            from_chat_request(&bad),
            Err(OpenAiError::InvalidMessageContentType("float64".to_string()))
        );
        assert_eq!(
            OpenAiError::InvalidMessageContentType("float64".to_string()).to_string(),
            "invalid message content type: float64"
        );

        // Null WITH tool calls: fine -- that is the tool-call-only assistant turn.
        let ok = ChatCompletionRequest {
            messages: vec![Message {
                role: "assistant".to_string(),
                content: serde_json::Value::Null,
                tool_calls: vec![ToolCall {
                    id: "call_1".to_string(),
                    index: 3,
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "get_weather".to_string(),
                        arguments: r#"{"location":"Seattle"}"#.to_string(),
                    },
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = from_chat_request(&ok).expect("ok");
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].content, "");
        assert_eq!(result.messages[0].tool_calls.len(), 1);
        // `index` deliberately NOT carried across -- see from_completion_tool_call.
        assert_eq!(result.messages[0].tool_calls[0].function.index, 0);
    }

    #[test]
    fn go_type_names_match_what_encoding_json_produces_for_any() {
        assert_eq!(go_type_name(&serde_json::Value::Null), "<nil>");
        assert_eq!(go_type_name(&serde_json::json!(true)), "bool");
        assert_eq!(go_type_name(&serde_json::json!(1)), "float64");
        assert_eq!(go_type_name(&serde_json::json!(1.5)), "float64");
        assert_eq!(go_type_name(&serde_json::json!("x")), "string");
        assert_eq!(go_type_name(&serde_json::json!([])), "[]interface {}");
        assert_eq!(
            go_type_name(&serde_json::json!({})),
            "map[string]interface {}"
        );
    }

    #[test]
    fn a_tool_message_recovers_its_tool_name_from_an_earlier_call_id() {
        let req = ChatCompletionRequest {
            messages: vec![
                Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(String::new()),
                    tool_calls: vec![ToolCall {
                        id: "call_abc".to_string(),
                        call_type: "function".to_string(),
                        function: ToolCallFunction {
                            name: "get_weather".to_string(),
                            arguments: "{}".to_string(),
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                Message {
                    role: "tool".to_string(),
                    content: serde_json::Value::String("31C".to_string()),
                    tool_call_id: "call_abc".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let result = from_chat_request(&req).expect("ok");
        assert_eq!(result.messages[1].tool_name, "get_weather");
        assert_eq!(result.messages[1].tool_call_id, "call_abc");
    }

    #[test]
    fn duplicate_tool_call_ids_resolve_last_one_wins() {
        let msgs = vec![
            Message {
                tool_calls: vec![ToolCall {
                    id: "dup".to_string(),
                    function: ToolCallFunction {
                        name: "first".to_string(),
                        arguments: "{}".to_string(),
                    },
                    ..Default::default()
                }],
                ..Default::default()
            },
            Message {
                tool_calls: vec![ToolCall {
                    id: "dup".to_string(),
                    function: ToolCallFunction {
                        name: "second".to_string(),
                        arguments: "{}".to_string(),
                    },
                    ..Default::default()
                }],
                ..Default::default()
            },
        ];
        assert_eq!(name_from_tool_call_id(&msgs, "dup"), "second");
        assert_eq!(name_from_tool_call_id(&msgs, "nope"), "");
    }

    #[test]
    fn an_explicit_name_on_a_tool_message_beats_the_call_id_lookup() {
        let req = ChatCompletionRequest {
            messages: vec![Message {
                role: "TOOL".to_string(), // role match is case-insensitive
                content: serde_json::Value::String("ok".to_string()),
                name: "explicit_name".to_string(),
                tool_call_id: "call_abc".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = from_chat_request(&req).expect("ok");
        assert_eq!(result.messages[0].tool_name, "explicit_name");
        // Role case survive, exactly like upstream -- only the *comparison* is
        // lowercased.
        assert_eq!(result.messages[0].role, "TOOL");
    }

    #[test]
    fn stop_accepts_a_bare_string_and_an_array_and_drops_junk_silently() {
        let one = ChatCompletionRequest {
            stop: serde_json::json!("STOP"),
            ..Default::default()
        };
        let opts = from_chat_request(&one)
            .expect("ok")
            .options
            .expect("options");
        assert_eq!(opts["stop"], serde_json::json!(["STOP"]));

        let many = ChatCompletionRequest {
            stop: serde_json::json!(["a", 7, "b"]),
            ..Default::default()
        };
        let opts = from_chat_request(&many)
            .expect("ok")
            .options
            .expect("options");
        // The 7 is DROPPED, not an error -- chat path only.
        assert_eq!(opts["stop"], serde_json::json!(["a", "b"]));

        let none = ChatCompletionRequest::default();
        let opts = from_chat_request(&none)
            .expect("ok")
            .options
            .expect("options");
        assert!(!opts.contains_key("stop"));
    }

    #[test]
    fn temperature_and_top_p_default_to_one_not_to_ollamas_defaults() {
        let req = ChatCompletionRequest::default();
        let opts = from_chat_request(&req)
            .expect("ok")
            .options
            .expect("options");
        assert_eq!(opts["temperature"], serde_json::json!(1.0));
        assert_eq!(opts["top_p"], serde_json::json!(1.0));
        // Nothing else got invented.
        assert!(!opts.contains_key("seed"));
        assert!(!opts.contains_key("num_predict"));
        assert!(!opts.contains_key("frequency_penalty"));
        assert!(!opts.contains_key("presence_penalty"));
    }

    #[test]
    fn max_tokens_becomes_num_predict_and_the_penalties_pass_through() {
        let req = ChatCompletionRequest {
            max_tokens: Some(128),
            seed: Some(42),
            temperature: Some(0.25),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(-0.5),
            top_p: Some(0.75),
            ..Default::default()
        };
        let opts = from_chat_request(&req)
            .expect("ok")
            .options
            .expect("options");
        assert_eq!(opts["num_predict"], serde_json::json!(128));
        assert_eq!(opts["seed"], serde_json::json!(42));
        assert_eq!(opts["temperature"], serde_json::json!(0.25));
        assert_eq!(opts["frequency_penalty"], serde_json::json!(0.5));
        assert_eq!(opts["presence_penalty"], serde_json::json!(-0.5));
        assert_eq!(opts["top_p"], serde_json::json!(0.75));
    }

    #[test]
    fn response_format_json_object_becomes_the_bare_json_format() {
        let req = ChatCompletionRequest {
            response_format: Some(ResponseFormat {
                format_type: "  JSON_Object ".to_string(), // trimmed + lowercased
                json_schema: None,
            }),
            ..Default::default()
        };
        assert_eq!(
            from_chat_request(&req).expect("ok").format,
            Some(serde_json::json!("json"))
        );
    }

    #[test]
    fn response_format_json_schema_passes_the_schema_straight_through() {
        let schema = serde_json::json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let req = ChatCompletionRequest {
            response_format: Some(ResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: Some(JsonSchema {
                    schema: Some(schema.clone()),
                }),
            }),
            ..Default::default()
        };
        assert_eq!(from_chat_request(&req).expect("ok").format, Some(schema));
    }

    #[test]
    fn an_unknown_or_schemaless_response_format_leaves_format_unset() {
        for rf in [
            ResponseFormat {
                format_type: "text".to_string(),
                json_schema: None,
            },
            ResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: None,
            },
            ResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: Some(JsonSchema { schema: None }),
            },
        ] {
            let req = ChatCompletionRequest {
                response_format: Some(rf),
                ..Default::default()
            };
            assert_eq!(from_chat_request(&req).expect("ok").format, None);
        }
    }

    #[test]
    fn logprobs_and_top_logprobs_ride_through_the_chat_conversion() {
        // Upstream: TestFromChatRequest_WithLogprobs + _TopLogprobsRange.
        for n in [0, 1, 10, 20] {
            let req = ChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![user("Hello")],
                logprobs: Some(true),
                top_logprobs: n,
                ..Default::default()
            };
            let result = from_chat_request(&req).expect("ok");
            assert!(result.logprobs);
            assert_eq!(result.top_logprobs, n);
        }
    }

    #[test]
    fn logprobs_default_to_off_when_the_request_says_nothing() {
        // Upstream: TestFromChatRequest_LogprobsDefault.
        let req = ChatCompletionRequest {
            model: "test-model".to_string(),
            messages: vec![user("Hello")],
            ..Default::default()
        };
        let result = from_chat_request(&req).expect("ok");
        assert!(!result.logprobs);
        assert_eq!(result.top_logprobs, 0);
    }

    #[test]
    fn stream_is_always_set_explicitly_even_when_false() {
        // ollama's `stream: None` mean "stream anyway", so a non-streaming
        // OpenAI request MUST come out Some(false) or it silently streams.
        let req = ChatCompletionRequest::default();
        assert_eq!(from_chat_request(&req).expect("ok").stream, Some(false));
        let req = ChatCompletionRequest {
            stream: true,
            ..Default::default()
        };
        assert_eq!(from_chat_request(&req).expect("ok").stream, Some(true));
    }

    // -- FromCompleteRequest ------------------------------------------------

    #[test]
    fn a_basic_completion_request_carries_model_prompt_and_temperature() {
        // Upstream: TestFromCompleteRequest_Basic.
        let req = CompletionRequest {
            model: "test-model".to_string(),
            prompt: "Hello".to_string(),
            temperature: Some(0.8),
            ..Default::default()
        };

        let result = from_complete_request(&req).expect("ok");

        assert_eq!(result.model, "test-model");
        assert_eq!(result.prompt, "Hello");
        // Upstream assert against a float32 0.8; widened to f64 that is
        // 0.800000011920929, which is Go's in-memory value too.
        let opts = result.options.expect("options");
        assert_eq!(opts["temperature"].as_f64(), Some(f64::from(0.8f32)));
    }

    #[test]
    fn completion_penalties_are_always_written_even_at_zero() {
        let req = CompletionRequest::default();
        let opts = from_complete_request(&req)
            .expect("ok")
            .options
            .expect("options");
        // Non-pointer upstream, so "unset" cannot be told from 0 -- both land
        // in the map.
        assert_eq!(opts["frequency_penalty"], serde_json::json!(0.0));
        assert_eq!(opts["presence_penalty"], serde_json::json!(0.0));
        // top_p == 0.0 counts as unset, so it becomes 1.0.
        assert_eq!(opts["top_p"], serde_json::json!(1.0));
        assert_eq!(opts["temperature"], serde_json::json!(1.0));
    }

    #[test]
    fn a_non_string_in_a_completion_stop_array_is_an_error_unlike_chat() {
        let req = CompletionRequest {
            stop: serde_json::json!(["a", 7]),
            ..Default::default()
        };
        assert_eq!(
            from_complete_request(&req),
            Err(OpenAiError::InvalidStopFieldType("float64".to_string()))
        );
        assert_eq!(
            OpenAiError::InvalidStopFieldType("float64".to_string()).to_string(),
            "invalid type for 'stop' field: float64"
        );
    }

    #[test]
    fn the_legacy_logprobs_count_turns_logprobs_on_and_sets_the_top_n() {
        // Upstream: TestFromCompleteRequest_WithLogprobs.
        let req = CompletionRequest {
            model: "test-model".to_string(),
            prompt: "Hello".to_string(),
            logprobs: Some(5),
            ..Default::default()
        };
        let result = from_complete_request(&req).expect("ok");
        assert!(result.logprobs);
        assert_eq!(result.top_logprobs, 5);

        // Zero and unset both mean off.
        for logprobs in [Some(0), None] {
            let req = CompletionRequest {
                logprobs,
                ..Default::default()
            };
            let result = from_complete_request(&req).expect("ok");
            assert!(!result.logprobs);
            assert_eq!(result.top_logprobs, 0);
        }
    }

    #[test]
    fn the_completion_suffix_reaches_the_generate_request() {
        let req = CompletionRequest {
            prompt: "fn main() {".to_string(),
            suffix: "}".to_string(),
            ..Default::default()
        };
        assert_eq!(from_complete_request(&req).expect("ok").suffix, "}");
    }

    // -- FromCompletionToolCall ---------------------------------------------

    #[test]
    fn tool_call_arguments_parse_back_into_an_insertion_ordered_object() {
        let calls = vec![ToolCall {
            id: "call_1".to_string(),
            index: 0,
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: "get_weather".to_string(),
                arguments: r#"{"zebra":1,"apple":2}"#.to_string(),
            },
        }];
        let got = from_completion_tool_call(&calls).expect("ok");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].id, "call_1");
        assert_eq!(got[0].function.name, "get_weather");
        // Insertion order survive -- NOT sorted.
        let keys: Vec<&String> = got[0].function.arguments.0.keys().collect();
        assert_eq!(keys, vec!["zebra", "apple"]);
    }

    #[test]
    fn unparseable_tool_call_arguments_are_rejected() {
        for arguments in ["", "not json", "[1,2]"] {
            let calls = vec![ToolCall {
                function: ToolCallFunction {
                    name: "f".to_string(),
                    arguments: arguments.to_string(),
                },
                ..Default::default()
            }];
            assert_eq!(
                from_completion_tool_call(&calls),
                Err(OpenAiError::InvalidToolCallArguments),
                "arguments={arguments:?}"
            );
        }
    }

    // -- ToUsage / NewError -------------------------------------------------

    #[test]
    fn usage_sums_the_prompt_and_eval_counts() {
        // Upstream: TestToUsage.
        let resp = ChatResponse {
            metrics: Metrics {
                prompt_eval_count: 10,
                eval_count: 20,
                ..Default::default()
            },
            ..Default::default()
        };
        let usage = to_usage(&resp);
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn generate_usage_sums_the_same_way() {
        let resp = GenerateResponse {
            metrics: Metrics {
                prompt_eval_count: 3,
                eval_count: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            to_usage_generate(&resp),
            Usage {
                prompt_tokens: 3,
                completion_tokens: 4,
                total_tokens: 7,
            }
        );
    }

    #[test]
    fn error_types_bucket_by_http_status() {
        // Upstream: TestNewError.
        for (code, want) in [
            (400, "invalid_request_error"),
            (404, "not_found_error"),
            (500, "api_error"),
        ] {
            let result = new_error(code, "test message");
            assert_eq!(result.error.error_type, want, "code={code}");
            assert_eq!(result.error.message, "test message");
        }
    }

    #[test]
    fn the_error_envelope_always_carries_param_and_code_as_null() {
        let json = serde_json::to_string(&new_error(400, "boom")).expect("serialise");
        assert_eq!(
            json,
            r#"{"error":{"message":"boom","type":"invalid_request_error","param":null,"code":null}}"#
        );
    }

    // -- ToToolCalls --------------------------------------------------------

    #[test]
    fn tool_calls_keep_their_ids_and_hoist_the_function_index() {
        // Upstream: TestToToolCallsPreservesIDs. The "input not mutated" half of
        // that test is unrepresentable here -- we take `&[api::ToolCall]`, so
        // the compiler already guarantee it.
        let original = vec![
            api::ToolCall {
                id: "call_abc123".to_string(),
                function: ApiToolCallFunction {
                    index: 2,
                    name: "get_weather".to_string(),
                    arguments: test_args(&[("location", serde_json::json!("Seattle"))]),
                },
            },
            api::ToolCall {
                id: "call_def456".to_string(),
                function: ApiToolCallFunction {
                    index: 7,
                    name: "get_time".to_string(),
                    arguments: test_args(&[("timezone", serde_json::json!("UTC"))]),
                },
            },
        ];

        let got = to_tool_calls(&original);

        assert_eq!(
            got,
            vec![
                ToolCall {
                    id: "call_abc123".to_string(),
                    index: 2,
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "get_weather".to_string(),
                        arguments: r#"{"location":"Seattle"}"#.to_string(),
                    },
                },
                ToolCall {
                    id: "call_def456".to_string(),
                    index: 7,
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: "get_time".to_string(),
                        arguments: r#"{"timezone":"UTC"}"#.to_string(),
                    },
                },
            ]
        );
    }

    #[test]
    fn an_argumentless_tool_call_serialises_as_an_empty_object_not_null() {
        let got = to_tool_calls(&[api::ToolCall {
            id: "c".to_string(),
            function: ApiToolCallFunction {
                index: 0,
                name: "ping".to_string(),
                arguments: ToolCallArguments::new(),
            },
        }]);
        assert_eq!(got[0].function.arguments, "{}");
    }

    // -- ToChatCompletion ---------------------------------------------------

    #[test]
    fn a_chat_completion_carries_logprobs_when_the_response_has_them() {
        // Upstream: TestToChatCompletion_WithLogprobs.
        let resp = ChatResponse {
            model: "test-model".to_string(),
            created_at: ts(1_234_567_890),
            message: api::Message::new("assistant", "Hello there"),
            logprobs: vec![
                Logprob {
                    token_logprob: TokenLogprob {
                        token: "Hello".to_string(),
                        logprob: -0.5,
                        bytes: Vec::new(),
                    },
                    top_logprobs: vec![
                        TokenLogprob {
                            token: "Hello".to_string(),
                            logprob: -0.5,
                            bytes: Vec::new(),
                        },
                        TokenLogprob {
                            token: "Hi".to_string(),
                            logprob: -1.2,
                            bytes: Vec::new(),
                        },
                    ],
                },
                Logprob {
                    token_logprob: TokenLogprob {
                        token: " there".to_string(),
                        logprob: -0.3,
                        bytes: Vec::new(),
                    },
                    top_logprobs: vec![
                        TokenLogprob {
                            token: " there".to_string(),
                            logprob: -0.3,
                            bytes: Vec::new(),
                        },
                        TokenLogprob {
                            token: " world".to_string(),
                            logprob: -1.5,
                            bytes: Vec::new(),
                        },
                    ],
                },
            ],
            done: true,
            metrics: Metrics {
                prompt_eval_count: 5,
                eval_count: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = to_chat_completion("test-id", &resp);

        assert_eq!(result.id, "test-id");
        assert_eq!(result.created, 1_234_567_890);
        assert_eq!(result.object, "chat.completion");
        assert_eq!(result.system_fingerprint, "fp_ollama");
        assert_eq!(result.choices.len(), 1);

        let choice = &result.choices[0];
        assert_eq!(choice.message.content, serde_json::json!("Hello there"));
        let logprobs = choice.logprobs.as_ref().expect("logprobs present");
        assert_eq!(logprobs.content.len(), 2);
        assert_eq!(logprobs.content[0].token_logprob.token, "Hello");
        assert_eq!(logprobs.content[0].token_logprob.logprob, -0.5);
        assert_eq!(logprobs.content[0].top_logprobs.len(), 2);
        assert_eq!(logprobs.content[1].token_logprob.token, " there");
    }

    #[test]
    fn a_chat_completion_omits_logprobs_when_none_were_requested() {
        // Upstream: TestToChatCompletion_WithoutLogprobs.
        let resp = ChatResponse {
            model: "test-model".to_string(),
            created_at: ts(1_234_567_890),
            message: api::Message::new("assistant", "Hello"),
            done: true,
            metrics: Metrics {
                prompt_eval_count: 5,
                eval_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = to_chat_completion("test-id", &resp);
        assert_eq!(result.choices.len(), 1);
        assert!(result.choices[0].logprobs.is_none());
    }

    #[test]
    fn the_finish_reason_is_null_mid_turn_and_tool_calls_when_tools_fired() {
        let mut resp = ChatResponse {
            model: "m".to_string(),
            message: api::Message::new("assistant", "hi"),
            ..Default::default()
        };
        assert_eq!(
            to_chat_completion("id", &resp).choices[0].finish_reason,
            None
        );

        resp.done_reason = "stop".to_string();
        assert_eq!(
            to_chat_completion("id", &resp).choices[0].finish_reason,
            Some("stop".to_string())
        );

        // ollama's own "length" reason pass through untouched -- there is no
        // separate mapping table.
        resp.done_reason = "length".to_string();
        assert_eq!(
            to_chat_completion("id", &resp).choices[0].finish_reason,
            Some("length".to_string())
        );

        // Tool calls override whatever the reason was.
        resp.message.tool_calls = vec![api::ToolCall {
            id: "c".to_string(),
            function: ApiToolCallFunction {
                index: 0,
                name: "f".to_string(),
                arguments: ToolCallArguments::new(),
            },
        }];
        assert_eq!(
            to_chat_completion("id", &resp).choices[0].finish_reason,
            Some("tool_calls".to_string())
        );
    }

    #[test]
    fn thinking_becomes_reasoning_on_the_buffered_message() {
        let resp = ChatResponse {
            message: api::Message {
                role: "assistant".to_string(),
                content: "answer".to_string(),
                thinking: "pondering".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let result = to_chat_completion("id", &resp);
        assert_eq!(result.choices[0].message.reasoning, "pondering");
        assert_eq!(result.choices[0].message.role, "assistant");
    }

    // -- ToChunks -----------------------------------------------------------

    #[test]
    fn a_mixed_thinking_and_content_chunk_splits_into_two() {
        // Upstream: TestToChunks_SplitsThinkingAndContent.
        let resp = ChatResponse {
            model: "test-model".to_string(),
            message: api::Message {
                thinking: "step-by-step".to_string(),
                content: "final answer".to_string(),
                ..Default::default()
            },
            done: true,
            done_reason: "stop".to_string(),
            ..Default::default()
        };

        let chunks = to_chunks("test-id", &resp, false, 42);
        assert_eq!(chunks.len(), 2);

        let reasoning = &chunks[0].choices[0];
        assert_eq!(reasoning.delta.reasoning, "step-by-step");
        assert_eq!(reasoning.delta.content, serde_json::json!(""));
        assert!(reasoning.delta.tool_calls.is_empty());
        assert_eq!(reasoning.finish_reason, None);

        let content = &chunks[1].choices[0];
        assert_eq!(content.delta.reasoning, "");
        assert_eq!(content.delta.content, serde_json::json!("final answer"));
        assert_eq!(content.finish_reason, Some("stop".to_string()));

        // Both halves are one logical emission, so one timestamp.
        assert_eq!(chunks[0].created, chunks[1].created);
    }

    #[test]
    fn a_mixed_thinking_and_tool_call_chunk_splits_and_finishes_as_tool_calls() {
        // Upstream: TestToChunks_SplitsThinkingAndToolCalls.
        let resp = ChatResponse {
            model: "test-model".to_string(),
            message: api::Message {
                thinking: "need a tool".to_string(),
                tool_calls: vec![api::ToolCall {
                    id: "call_123".to_string(),
                    function: ApiToolCallFunction {
                        index: 0,
                        name: "get_weather".to_string(),
                        arguments: test_args(&[("location", serde_json::json!("Seattle"))]),
                    },
                }],
                ..Default::default()
            },
            done: true,
            done_reason: "stop".to_string(),
            ..Default::default()
        };

        let chunks = to_chunks("test-id", &resp, false, 0);
        assert_eq!(chunks.len(), 2);

        let reasoning = &chunks[0].choices[0];
        assert_eq!(reasoning.delta.reasoning, "need a tool");
        assert!(reasoning.delta.tool_calls.is_empty());
        assert_eq!(reasoning.finish_reason, None);

        let tool_chunk = &chunks[1].choices[0];
        assert_eq!(tool_chunk.delta.reasoning, "");
        assert_eq!(tool_chunk.delta.tool_calls.len(), 1);
        assert_eq!(tool_chunk.delta.tool_calls[0].id, "call_123");
        assert_eq!(
            tool_chunk.finish_reason,
            Some(FINISH_REASON_TOOL_CALLS.to_string())
        );
    }

    #[test]
    fn an_unmixed_chunk_stays_a_single_chunk() {
        // Upstream: TestToChunks_SingleChunkForNonMixedResponses.
        let tool_calls = vec![api::ToolCall {
            id: "call_456".to_string(),
            function: ApiToolCallFunction {
                index: 0,
                name: "get_time".to_string(),
                arguments: test_args(&[("timezone", serde_json::json!("UTC"))]),
            },
        }];

        let cases = [
            (
                "thinking-only",
                api::Message {
                    thinking: "pondering".to_string(),
                    ..Default::default()
                },
            ),
            (
                "content-only",
                api::Message {
                    content: "hello".to_string(),
                    ..Default::default()
                },
            ),
            (
                "toolcalls-only",
                api::Message {
                    tool_calls,
                    ..Default::default()
                },
            ),
            ("empty", api::Message::default()),
        ];

        for (name, message) in cases {
            let resp = ChatResponse {
                model: "test-model".to_string(),
                message,
                ..Default::default()
            };
            assert_eq!(to_chunks("test-id", &resp, false, 0).len(), 1, "case={name}");
        }
    }

    #[test]
    fn a_split_chunk_that_is_not_done_carries_no_finish_reason_on_either_half() {
        // Upstream: TestToChunks_SplitsThinkingAndToolCallsWhenNotDone and
        // TestToChunks_SplitsThinkingAndContentWhenNotDone.
        let with_tools = ChatResponse {
            model: "test-model".to_string(),
            message: api::Message {
                thinking: "need a tool".to_string(),
                tool_calls: vec![api::ToolCall {
                    id: "call_789".to_string(),
                    function: ApiToolCallFunction {
                        index: 0,
                        name: "get_weather".to_string(),
                        arguments: test_args(&[("location", serde_json::json!("San Francisco"))]),
                    },
                }],
                ..Default::default()
            },
            done: false,
            ..Default::default()
        };
        let chunks = to_chunks("test-id", &with_tools, false, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].choices[0].delta.reasoning, "need a tool");
        assert_eq!(chunks[0].choices[0].finish_reason, None);
        assert_eq!(chunks[1].choices[0].delta.tool_calls.len(), 1);
        assert_eq!(chunks[1].choices[0].delta.tool_calls[0].id, "call_789");
        assert_eq!(chunks[1].choices[0].finish_reason, None);

        let with_content = ChatResponse {
            model: "test-model".to_string(),
            message: api::Message {
                thinking: "thinking".to_string(),
                content: "partial content".to_string(),
                ..Default::default()
            },
            done: false,
            ..Default::default()
        };
        let chunks = to_chunks("test-id", &with_content, false, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].choices[0].delta.reasoning, "thinking");
        assert_eq!(chunks[0].choices[0].finish_reason, None);
        assert_eq!(
            chunks[1].choices[0].delta.content,
            serde_json::json!("partial content")
        );
        assert_eq!(chunks[1].choices[0].finish_reason, None);
    }

    #[test]
    fn a_split_chunk_sends_logprobs_on_the_first_half_only() {
        // Upstream: TestToChunks_SplitSendsLogprobsOnlyOnFirstChunk.
        let resp = ChatResponse {
            model: "test-model".to_string(),
            message: api::Message {
                thinking: "thinking".to_string(),
                content: "content".to_string(),
                ..Default::default()
            },
            logprobs: vec![Logprob {
                token_logprob: TokenLogprob {
                    token: "tok".to_string(),
                    logprob: -0.25,
                    bytes: Vec::new(),
                },
                top_logprobs: Vec::new(),
            }],
            done: true,
            done_reason: "stop".to_string(),
            ..Default::default()
        };

        let chunks = to_chunks("test-id", &resp, false, 0);
        assert_eq!(chunks.len(), 2);

        let first = chunks[0].choices[0]
            .logprobs
            .as_ref()
            .expect("first has logprobs");
        assert_eq!(first.content.len(), 1);
        assert_eq!(first.content[0].token_logprob.token, "tok");
        assert!(chunks[1].choices[0].logprobs.is_none());
    }

    #[test]
    fn the_deprecated_single_chunk_path_keeps_thinking_and_content_together() {
        // Upstream: TestToChunk_LegacyMixedThinkingAndContentSingleChunk.
        let resp = ChatResponse {
            model: "test-model".to_string(),
            message: api::Message {
                thinking: "reasoning".to_string(),
                content: "answer".to_string(),
                ..Default::default()
            },
            done: true,
            done_reason: "stop".to_string(),
            ..Default::default()
        };

        let chunk = to_chunk("test-id", &resp, false, 7);
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.reasoning, "reasoning");
        assert_eq!(chunk.choices[0].delta.content, serde_json::json!("answer"));
        assert_eq!(chunk.created, 7);
    }

    #[test]
    fn a_delta_always_says_assistant_even_when_ollama_said_nothing() {
        let resp = ChatResponse {
            message: api::Message {
                role: String::new(),
                content: "hi".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            to_chunk("id", &resp, false, 0).choices[0].delta.role,
            "assistant"
        );
    }

    #[test]
    fn tool_call_sent_forces_the_final_chunk_to_finish_as_tool_calls() {
        // The last chunk carry the done_reason but NO tool calls, so without the
        // flag the client would see "stop" and never dispatch the tool.
        let resp = ChatResponse {
            model: "m".to_string(),
            done: true,
            done_reason: "stop".to_string(),
            ..Default::default()
        };
        assert_eq!(
            to_chunk("id", &resp, true, 0).choices[0].finish_reason,
            Some("tool_calls".to_string())
        );
        assert_eq!(
            to_chunk("id", &resp, false, 0).choices[0].finish_reason,
            Some("stop".to_string())
        );
    }

    // -- ToCompletion / ToCompleteChunk -------------------------------------

    #[test]
    fn a_buffered_completion_carries_text_usage_and_the_finish_reason() {
        let resp = GenerateResponse {
            model: "test-model".to_string(),
            created_at: ts(1_234_567_890),
            response: "the answer".to_string(),
            done: true,
            done_reason: "length".to_string(),
            metrics: Metrics {
                prompt_eval_count: 2,
                eval_count: 5,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = to_completion("cmpl-1", &resp);
        assert_eq!(result.object, "text_completion");
        assert_eq!(result.created, 1_234_567_890);
        assert_eq!(result.system_fingerprint, "fp_ollama");
        assert_eq!(result.choices.len(), 1);
        assert_eq!(result.choices[0].text, "the answer");
        assert_eq!(result.choices[0].index, 0);
        assert_eq!(result.choices[0].finish_reason, Some("length".to_string()));
        assert_eq!(result.usage.total_tokens, 7);
    }

    #[test]
    fn a_streamed_completion_chunk_carries_no_usage_and_takes_its_created() {
        let resp = GenerateResponse {
            model: "test-model".to_string(),
            response: "tok".to_string(),
            ..Default::default()
        };
        let chunk = to_complete_chunk("cmpl-1", &resp, 99);
        assert_eq!(chunk.object, "text_completion");
        assert_eq!(chunk.created, 99);
        assert_eq!(chunk.choices[0].text, "tok");
        assert_eq!(chunk.choices[0].finish_reason, None);
        assert!(chunk.usage.is_none());
    }

    // -- ToListCompletion / ToModel -----------------------------------------

    #[test]
    fn the_model_list_prefers_the_model_field_and_falls_back_to_name() {
        // Upstream: TestToListCompletionUsesModelIdentity.
        let result = to_list_completion(&ListResponse {
            models: vec![
                ListModelResponse {
                    name: "legacy-name:latest".to_string(),
                    model: "namespace/exposed-model:latest".to_string(),
                    modified_at: ts(1_234_567_890),
                    ..Default::default()
                },
                ListModelResponse {
                    name: "fallback-name:latest".to_string(),
                    modified_at: ts(1_234_567_891),
                    ..Default::default()
                },
            ],
        });

        assert_eq!(result.object, "list");
        assert_eq!(result.data.len(), 2);

        assert_eq!(result.data[0].id, "namespace/exposed-model:latest");
        assert_eq!(result.data[0].owned_by, "namespace");
        assert_eq!(result.data[0].created, 1_234_567_890);
        assert_eq!(result.data[0].object, "model");

        assert_eq!(result.data[1].id, "fallback-name:latest");
        // ParseName fill the default namespace in.
        assert_eq!(result.data[1].owned_by, "library");
        assert_eq!(result.data[1].created, 1_234_567_891);
    }

    #[test]
    fn an_empty_model_list_is_still_object_list() {
        let result = to_list_completion(&ListResponse::default());
        assert_eq!(result.object, "list");
        assert!(result.data.is_empty());
    }

    #[test]
    fn to_model_echoes_the_requested_id_and_takes_its_namespace() {
        let show = ShowResponse {
            modified_at: ts(1_700_000_000),
            ..Default::default()
        };
        let m = to_model(&show, "qwen3:0.6b");
        assert_eq!(m.id, "qwen3:0.6b");
        assert_eq!(m.object, "model");
        assert_eq!(m.created, 1_700_000_000);
        assert_eq!(m.owned_by, "library");

        assert_eq!(to_model(&show, "myns/mymodel").owned_by, "myns");
    }

    // -- ToEmbeddingList / floatsToBase64 -----------------------------------

    #[test]
    fn embeddings_render_as_floats_or_base64_by_encoding_format() {
        // Upstream: TestToEmbeddingList.
        struct Case {
            name: &'static str,
            embeddings: Vec<Vec<f32>>,
            format: &'static str,
            expect_base64: Option<Vec<&'static str>>,
            expect_count: usize,
            prompt_eval: i32,
        }

        let cases = [
            Case {
                name: "float format",
                embeddings: vec![vec![0.1, -0.2, 0.3]],
                format: "float",
                expect_base64: None,
                expect_count: 1,
                prompt_eval: 10,
            },
            Case {
                name: "base64 format",
                embeddings: vec![vec![0.1, -0.2, 0.3]],
                format: "base64",
                expect_base64: Some(vec!["zczMPc3MTL6amZk+"]),
                expect_count: 1,
                prompt_eval: 5,
            },
            Case {
                name: "default to float",
                embeddings: vec![vec![0.1, -0.2, 0.3]],
                format: "",
                expect_base64: None,
                expect_count: 1,
                prompt_eval: 0,
            },
            Case {
                name: "invalid defaults to float",
                embeddings: vec![vec![0.1, -0.2, 0.3]],
                format: "invalid",
                expect_base64: None,
                expect_count: 1,
                prompt_eval: 0,
            },
            Case {
                name: "multiple embeddings",
                embeddings: vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]],
                format: "base64",
                expect_base64: Some(vec!["zczMPc3MTD4=", "mpmZPs3MzD4=", "AAAAP5qZGT8="]),
                expect_count: 3,
                prompt_eval: 0,
            },
            Case {
                name: "empty embeddings",
                embeddings: Vec::new(),
                format: "float",
                expect_base64: None,
                expect_count: 0,
                prompt_eval: 0,
            },
        ];

        for case in cases {
            let resp = EmbedResponse {
                embeddings: case.embeddings,
                prompt_eval_count: case.prompt_eval,
                ..Default::default()
            };
            let result = to_embedding_list("test-model", &resp, case.format);

            if case.expect_count == 0 {
                assert!(result.data.is_empty(), "case={}", case.name);
                continue;
            }

            assert_eq!(result.data.len(), case.expect_count, "case={}", case.name);
            assert_eq!(result.model, "test-model", "case={}", case.name);

            match &case.expect_base64 {
                None => {
                    assert!(
                        result.data[0].embedding.is_array(),
                        "case={}: expected a float array",
                        case.name
                    );
                }
                Some(want) => {
                    for (i, data) in result.data.iter().enumerate() {
                        let got = data.embedding.as_str().unwrap_or_else(|| {
                            panic!("case={}: embedding {i} should be a string", case.name)
                        });
                        assert_eq!(got, want[i], "case={} embedding={i}", case.name);
                        assert!(base64_std_decode(got).is_ok(), "case={}", case.name);
                    }
                }
            }

            for (i, d) in result.data.iter().enumerate() {
                assert_eq!(d.index, i as i64, "case={}", case.name);
            }

            if case.prompt_eval > 0 {
                assert_eq!(
                    result.usage.prompt_tokens,
                    i64::from(case.prompt_eval),
                    "case={}",
                    case.name
                );
                assert_eq!(result.usage.total_tokens, i64::from(case.prompt_eval));
            }
        }
    }

    #[test]
    fn an_empty_embedding_response_returns_the_zero_value_not_object_list() {
        // Upstream return `EmbeddingList{}` -- so `object` is "" and `model` is
        // "" even though a model was passed in. Look wrong, is upstream's.
        let result = to_embedding_list("test-model", &EmbedResponse::default(), "float");
        assert_eq!(result, EmbeddingList::default());
        assert_eq!(result.object, "");
        assert_eq!(result.model, "");
    }

    #[test]
    fn floats_pack_little_endian_before_base64() {
        // Upstream: TestFloatsToBase64 + TestFloatsToBase64_EmptySlice.
        let floats: [f32; 5] = [0.1, -0.2, 0.3, -0.4, 0.5];
        let result = floats_to_base64(&floats);
        let decoded = base64_std_decode(&result).expect("valid base64");
        assert_eq!(decoded.len(), floats.len() * 4);

        for (i, expected) in floats.iter().enumerate() {
            let o = i * 4;
            let bits = u32::from(decoded[o])
                | u32::from(decoded[o + 1]) << 8
                | u32::from(decoded[o + 2]) << 16
                | u32::from(decoded[o + 3]) << 24;
            assert!((f32::from_bits(bits) - expected).abs() < 1e-6, "float[{i}]");
        }

        let empty = floats_to_base64(&[]);
        assert_eq!(base64_std_decode(&empty).expect("valid"), Vec::<u8>::new());
    }

    // -- FromTranscriptionRequest -------------------------------------------

    #[test]
    fn a_transcription_request_pins_the_model_into_transcription_mode() {
        let req = TranscriptionRequest {
            model: "whisper-ish".to_string(),
            audio_data: b"RIFF....".to_vec(),
            ..Default::default()
        };
        let chat = from_transcription_request(&req);

        assert_eq!(chat.model, "whisper-ish");
        assert_eq!(chat.stream, Some(true));
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(
            chat.messages[0].content,
            "Transcribe the audio exactly as spoken. Output only the spoken words. \
             Do not answer any question in the audio."
        );
        assert_eq!(chat.messages[1].role, "user");
        assert_eq!(
            chat.messages[1].content,
            "What exact words are spoken in this audio?"
        );
        assert_eq!(chat.messages[1].images, vec![base64_std(b"RIFF....")]);
        // temperature 0: transcription want the likeliest words, not creativity.
        assert_eq!(
            chat.options.expect("options")["temperature"],
            serde_json::json!(0)
        );
    }

    #[test]
    fn transcription_language_and_prompt_are_appended_in_that_order() {
        let req = TranscriptionRequest {
            model: "m".to_string(),
            language: "Malay".to_string(),
            prompt: "kopitiam order".to_string(),
            ..Default::default()
        };
        let chat = from_transcription_request(&req);
        assert_eq!(
            chat.messages[0].content,
            "Transcribe the audio exactly as spoken. Output only the spoken words. \
             Do not answer any question in the audio. The audio is in Malay. \
             Context: kopitiam order"
        );
    }

    // -- Wire contract ------------------------------------------------------

    #[test]
    fn a_chat_completion_serialises_to_the_exact_openai_shape() {
        let resp = ChatResponse {
            model: "test-model".to_string(),
            created_at: ts(1_234_567_890),
            message: api::Message::new("assistant", "Hello"),
            done: true,
            done_reason: "stop".to_string(),
            metrics: Metrics {
                prompt_eval_count: 5,
                eval_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let got = serde_json::to_value(to_chat_completion("chatcmpl-1", &resp)).expect("serialise");
        assert_eq!(
            got,
            serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "system_fingerprint": "fp_ollama",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            })
        );
    }

    #[test]
    fn a_chunk_serialises_with_a_delta_and_a_null_finish_reason() {
        let resp = ChatResponse {
            model: "test-model".to_string(),
            message: api::Message::new("assistant", "He"),
            ..Default::default()
        };
        let got = serde_json::to_value(to_chunk("chatcmpl-1", &resp, false, 1_700_000_000))
            .expect("serialise");
        assert_eq!(
            got,
            serde_json::json!({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "system_fingerprint": "fp_ollama",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": "He"},
                    "finish_reason": null
                }]
            })
        );
    }

    #[test]
    fn a_usage_only_final_frame_serialises_with_empty_choices() {
        // This is the shape the middleware build when
        // `stream_options.include_usage` is set: same chunk, choices emptied,
        // usage attached.
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1,
            model: "m".to_string(),
            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
            choices: Vec::new(),
            usage: Some(Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            }),
        };
        let got = serde_json::to_value(&chunk).expect("serialise");
        assert_eq!(got["choices"], serde_json::json!([]));
        assert_eq!(
            got["usage"],
            serde_json::json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
        );
    }

    #[test]
    fn a_chat_request_round_trips_through_literal_openai_json() {
        let raw = r#"{
            "model": "qwen3",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "index": 0, "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"SG\"}"}}
                ]},
                {"role": "tool", "content": "31C", "tool_call_id": "call_1"}
            ],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 256,
            "stop": ["\n\n"],
            "temperature": 0.5,
            "top_p": 0.9,
            "response_format": {"type": "json_object"},
            "logprobs": true,
            "top_logprobs": 3
        }"#;

        let req: ChatCompletionRequest = serde_json::from_str(raw).expect("parse");
        assert_eq!(
            req.stream_options,
            Some(StreamOptions {
                include_usage: true
            })
        );
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.top_logprobs, 3);

        let native = from_chat_request(&req).expect("convert");
        assert_eq!(native.model, "qwen3");
        assert_eq!(native.messages.len(), 4);
        assert_eq!(native.messages[2].tool_calls.len(), 1);
        assert_eq!(
            native.messages[2].tool_calls[0]
                .function
                .arguments
                .get("city")
                .and_then(serde_json::Value::as_str),
            Some("SG")
        );
        assert_eq!(native.messages[3].tool_name, "get_weather");
        assert_eq!(native.format, Some(serde_json::json!("json")));
        assert!(native.logprobs);
        assert_eq!(native.top_logprobs, 3);

        let opts = native.options.expect("options");
        assert_eq!(opts["stop"], serde_json::json!(["\n\n"]));
        assert_eq!(opts["num_predict"], serde_json::json!(256));
        assert_eq!(opts["temperature"], serde_json::json!(0.5));
        assert_eq!(opts["top_p"], serde_json::json!(0.9));
    }

    #[test]
    fn a_request_with_only_a_model_still_emits_every_unset_key_as_null() {
        // Nothing in ChatCompletionRequest carry omitempty except `reasoning`
        // and `reasoning_effort`, so the serialised form keep the keys.
        let req = ChatCompletionRequest {
            model: "m".to_string(),
            ..Default::default()
        };
        let got = serde_json::to_value(&req).expect("serialise");
        for key in [
            "stream_options",
            "max_tokens",
            "seed",
            "stop",
            "temperature",
            "frequency_penalty",
            "presence_penalty",
            "top_p",
            "response_format",
            "logprobs",
        ] {
            assert_eq!(got[key], serde_json::Value::Null, "key={key}");
        }
        assert_eq!(got["messages"], serde_json::json!([]));
        assert_eq!(got["tools"], serde_json::json!([]));
        assert_eq!(got["top_logprobs"], serde_json::json!(0));
        assert_eq!(got["_debug_render_only"], serde_json::json!(false));
        assert!(got.get("reasoning").is_none());
        assert!(got.get("reasoning_effort").is_none());
    }

    #[test]
    fn a_models_listing_serialises_to_the_openai_shape() {
        let list = to_list_completion(&ListResponse {
            models: vec![ListModelResponse {
                name: "qwen3:0.6b".to_string(),
                model: "qwen3:0.6b".to_string(),
                modified_at: ts(1_700_000_000),
                ..Default::default()
            }],
        });
        assert_eq!(
            serde_json::to_value(&list).expect("serialise"),
            serde_json::json!({
                "object": "list",
                "data": [{
                    "id": "qwen3:0.6b",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "library"
                }]
            })
        );
    }

    #[test]
    fn an_openai_tool_call_serialises_with_its_arguments_as_a_string() {
        let call = to_tool_calls(&[api::ToolCall {
            id: "call_1".to_string(),
            function: ApiToolCallFunction {
                index: 4,
                name: "get_weather".to_string(),
                arguments: test_args(&[("city", serde_json::json!("Singapore"))]),
            },
        }]);
        assert_eq!(
            serde_json::to_value(&call[0]).expect("serialise"),
            serde_json::json!({
                "id": "call_1",
                "index": 4,
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Singapore\"}"}
            })
        );
    }

    #[test]
    fn an_embed_request_parses_both_the_string_and_the_array_input_forms() {
        let one: EmbedRequest =
            serde_json::from_str(r#"{"model":"m","input":"hello"}"#).expect("parse");
        assert_eq!(one.input, serde_json::json!("hello"));
        assert_eq!(one.encoding_format, "");

        let many: EmbedRequest = serde_json::from_str(
            r#"{"model":"m","input":["a","b"],"encoding_format":"base64","dimensions":8}"#,
        )
        .expect("parse");
        assert_eq!(many.input, serde_json::json!(["a", "b"]));
        assert_eq!(many.encoding_format, "base64");
        assert_eq!(many.dimensions, 8);
    }

    // -- helpers ------------------------------------------------------------

    #[test]
    fn timestamps_parse_back_to_the_unix_seconds_go_would_report() {
        assert_eq!(
            timestamp_unix(&Timestamp("2009-02-13T23:31:30Z".to_string())),
            Some(1_234_567_890)
        );
        // Fractional seconds get truncated, exactly like time.Time.Unix().
        assert_eq!(
            timestamp_unix(&Timestamp("2009-02-13T23:31:30.999999999Z".to_string())),
            Some(1_234_567_890)
        );
        // Offsets get applied.
        assert_eq!(
            timestamp_unix(&Timestamp("2009-02-14T07:31:30+08:00".to_string())),
            Some(1_234_567_890)
        );
        assert_eq!(
            timestamp_unix(&Timestamp("2009-02-13T18:31:30-05:00".to_string())),
            Some(1_234_567_890)
        );
        // Go's zero time.
        assert_eq!(timestamp_unix(&Timestamp::default()), Some(-62_135_596_800));
        assert_eq!(
            timestamp_unix(&Timestamp("1970-01-01T00:00:00Z".to_string())),
            Some(0)
        );
    }

    #[test]
    fn a_malformed_timestamp_is_none_and_the_conversions_fall_back_to_zero() {
        for bad in [
            "",
            "not a time",
            "2009-02-13",
            "2009-02-13T23:31:30",      // no zone
            "2009-02-13 23:31:30Z",     // space instead of T
            "2009-13-13T23:31:30Z",     // month 13
            "2009-02-13T24:31:30Z",     // hour 24
            "2009-02-13T23:31:30.Z",    // empty fraction
            "2009-02-13T23:31:30+0800", // offset with no colon
        ] {
            assert_eq!(
                timestamp_unix(&Timestamp(bad.to_string())),
                None,
                "bad={bad:?}"
            );
        }

        let resp = ChatResponse {
            created_at: Timestamp("rubbish".to_string()),
            ..Default::default()
        };
        assert_eq!(to_chat_completion("id", &resp).created, 0);
    }

    #[test]
    fn the_timestamp_round_trip_agrees_with_routes_own_formatter() {
        // Range limited on purpose: `from_unix_nanos` take **nanoseconds** in an
        // i64, so it can only express roughly 1678..2262. Go's zero time
        // (-62135596800 s) is way outside that and would overflow the `* 1e9`
        // here -- it is covered by parsing the literal `Timestamp::default()`
        // string in the test above instead.
        for secs in [-2_000_000_000, -1, 0, 1, 1_234_567_890, 4_102_444_800] {
            let t = Timestamp::from_unix_nanos(secs * 1_000_000_000);
            assert_eq!(timestamp_unix(&t), Some(secs), "secs={secs}");
        }
    }

    #[test]
    fn base64_decode_matches_the_rfc4648_vectors_and_the_encoder() {
        for (input, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_std(input.as_bytes()), encoded, "input={input:?}");
            assert_eq!(
                base64_std_decode(encoded).expect("decodes"),
                input.as_bytes(),
                "encoded={encoded:?}"
            );
        }
        // The +/ alphabet, both directions.
        assert_eq!(base64_std_decode("+/8=").expect("decodes"), vec![0xfb, 0xff]);
    }

    #[test]
    fn base64_decode_ignores_newlines_and_reports_bad_bytes_like_go() {
        assert_eq!(
            base64_std_decode("Zm9v\r\nYmFy").expect("decodes"),
            b"foobar"
        );
        assert_eq!(
            base64_std_decode("Zm9v!"),
            Err("illegal base64 data at input byte 4".to_string())
        );
        // Data after padding.
        assert_eq!(
            base64_std_decode("Zg==A"),
            Err("illegal base64 data at input byte 4".to_string())
        );
        // Not a multiple of four.
        assert_eq!(
            base64_std_decode("Zm9"),
            Err("illegal base64 data at input byte 3".to_string())
        );
    }

    #[test]
    fn now_unix_is_a_sane_wall_clock() {
        // Not asserting an exact value -- just that the one impure helper is
        // returning seconds, not millis or nanos. 2020-01-01 .. 2100-01-01.
        let now = now_unix();
        assert!(now > 1_577_836_800, "now_unix() = {now}");
        assert!(now < 4_102_444_800, "now_unix() = {now}");
    }
}
