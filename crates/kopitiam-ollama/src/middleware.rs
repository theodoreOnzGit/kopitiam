//! The adaptation layer that sit in **front of** `/v1/messages`.
//!
//! **Upstream:** `middleware/anthropic.go` (ollama, MIT), ported against
//! `4713800b08b2ddf5e14acf8398953cf7b12f169b`. The conversions themselves live
//! in [`crate::anthropic`]; this module is everything *around* them --
//! validation, error-shape rewriting, the SSE framing, and the server-side
//! web-search loop.
//!
//! ## What is NOT ported, and why
//!
//! Upstream is a `gin.HandlerFunc` that swap `c.Writer` for a wrapper, then
//! call `c.Next()`. The wrapper's `Write([]byte)` is where every conversion
//! actually happen -- so the *logic* is entangled with an HTTP framework's
//! response-writer interface, a `c.Set("relax_thinking", true)` context bag,
//! and header juggling.
//!
//! **None of that plumbing is ported.** This crate has no web framework and no
//! async runtime, and per `routes.rs` the house style is: a handler is a
//! **pure function from a typed request to a typed decision**, and whoever owns
//! the socket does the socket bit. So here:
//!
//! | Upstream | Here |
//! |---|---|
//! | `AnthropicMessagesMiddleware()` gin handler | [`prepare_messages_request`] -> [`MessagesPlan`] |
//! | `c.Set("relax_thinking", true)` | [`MessagesPlan::relax_thinking`], a plain flag |
//! | `c.Writer.Header().Set(...)` for SSE | [`SSE_STREAM_HEADERS`], a table the caller applies |
//! | `(*AnthropicWriter).Write([]byte)` | [`AnthropicWriter::write`] -> [`WriterOutput`] |
//! | `writeSSE(w, ...)` writing to a socket | [`write_sse`] returning the frame as a `String` |
//! | `go func(){...}` + `chan` for the async loop | a **synchronous** [`run_web_search_loop`]; see its docs |
//! | `http.DefaultClient.Do(...)` for search + follow-up chat | the [`WebSearchBackend`] trait |
//! | `internal/modelref.HasExplicitCloudSource` | a `bool` you pass to [`cloud_web_search_gate`] |
//!
//! The payoff is that the whole web-search loop -- three iterations, usage
//! accounting, `max_uses_exceeded` -- is testable with **no socket, no server,
//! no runtime**, which is the same seam `registry::Transport` and
//! `routes::ModelCatalog` already use.
//!
//! ## The shape of a request through here
//!
//! ```text
//!   client JSON --> MessagesRequest
//!        |
//!        +-- prepare_messages_request  -> 400 ErrorResponse, or a MessagesPlan
//!        |                                (plan.chat_request goes to /api/chat)
//!        v
//!   ollama ChatResponse chunks
//!        |
//!        +-- AnthropicWriter::write    -> WriterOutput::Json | Events | Error
//!        |
//!        +-- (only if the request offered a web_search tool)
//!            WebSearchStreamState + run_web_search_loop
//! ```

use serde::Serialize;

use crate::anthropic::{
    self, ContentBlock, ContentBlockDeltaEvent, ContentBlockStartEvent, ContentBlockStopEvent,
    Delta, DeltaUsage, ErrorResponse, MessageDelta, MessageDeltaEvent, MessageStartEvent,
    MessageStopEvent, MessagesRequest, MessagesResponse, OllamaWebSearchResponse,
    OllamaWebSearchResult, StreamConverter, StreamEvent, StreamEventData, Usage,
    WebSearchToolResultError,
};
use crate::api::{Message, ToolCall, ToolCallArguments};
use crate::routes::{ChatRequest, ChatResponse, Metrics};

// ===========================================================================
// Errors the middleware itself raise
// ===========================================================================

/// A refusal from the middleware, before any model is touched.
///
/// **Upstream:** every `c.AbortWithStatusJSON(status, anthropic.NewError(...))`
/// in `AnthropicMessagesMiddleware`. Modelled exactly like
/// [`crate::routes::RouteError`] -- a status plus a message -- and turned into
/// the Anthropic envelope only at the edge, because the envelope needs a
/// `request_id` and the entropy for that belongs to the caller (see
/// [`crate::anthropic::generate_id`]).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct AnthropicError {
    pub status: u16,
    pub message: String,
}

impl AnthropicError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        AnthropicError { status, message: message.into() }
    }

    /// `400 Bad Request` -- everything this module refuse except the cloud gate.
    pub fn bad_request(message: impl Into<String>) -> Self {
        AnthropicError::new(400, message)
    }

    /// The body to send back. Pass the `request_id` you minted for this request.
    pub fn to_error_response(&self, request_id: impl Into<String>) -> ErrorResponse {
        ErrorResponse::new(self.status, self.message.clone(), request_id)
    }
}

/// **Upstream:** `internal/cloud.DisabledMessagePrefix`.
pub const CLOUD_DISABLED_MESSAGE_PREFIX: &str = "ollama cloud is disabled";

/// **Upstream:** `internal/cloud.DisabledError`. An empty operation give the
/// bare prefix, otherwise `"{prefix}: {operation}"`.
pub fn cloud_disabled_error(operation: &str) -> String {
    if operation.is_empty() {
        CLOUD_DISABLED_MESSAGE_PREFIX.to_string()
    } else {
        format!("{CLOUD_DISABLED_MESSAGE_PREFIX}: {operation}")
    }
}

/// Should a web-search request be refused outright?
///
/// **Upstream:** the `if isCloudModelName(req.Model) { if disabled ... 403 }`
/// block in `AnthropicMessagesMiddleware`.
///
/// **Seam:** upstream ask `modelref.HasExplicitCloudSource(name)` and
/// `internalcloud.Status()`. Neither is ported (one is a name-grammar detail
/// that belong in [`crate::name`], the other read process env), so both arrive
/// as booleans. That keeps the *policy* here -- which is the bit with teeth --
/// and leaves the *facts* to whoever can actually see them.
///
/// Note how narrow the gate is, and it is deliberate: only a **cloud** model is
/// refused up front. A **local** model may still be handed web_search tool
/// definitions, and execution is checked later, when the model actually emit a
/// `web_search` call. Upstream's own comment says exactly this.
pub fn cloud_web_search_gate(
    is_cloud_model: bool,
    cloud_disabled: bool,
) -> Result<(), AnthropicError> {
    if is_cloud_model && cloud_disabled {
        return Err(AnthropicError::new(403, cloud_disabled_error("web search is unavailable")));
    }
    Ok(())
}

// ===========================================================================
// Request preparation
// ===========================================================================

/// Headers a streaming `/v1/messages` reply must carry.
///
/// **Upstream:** the three `c.Writer.Header().Set(...)` calls in
/// `AnthropicMessagesMiddleware`. A table rather than side effects, because
/// this crate does not own anybody's socket -- the caller applies them.
///
/// `Cache-Control: no-cache` is not decoration: without it an intermediary
/// proxy will happily buffer the whole SSE stream and hand the client one
/// lump at the end, which look exactly like the model hanging.
pub const SSE_STREAM_HEADERS: [(&str, &str); 3] = [
    ("Content-Type", "text/event-stream"),
    ("Cache-Control", "no-cache"),
    ("Connection", "keep-alive"),
];

/// Everything the middleware worked out before the model is called.
///
/// **Upstream:** the local variables `AnthropicMessagesMiddleware` build up
/// before `c.Next()` -- the rewritten body, the estimated tokens, the
/// `relax_thinking` context flag, and which writer to install.
#[derive(Debug, Clone, PartialEq)]
pub struct MessagesPlan {
    /// The rewritten body. Upstream literally replace `c.Request.Body` with
    /// this, so the ordinary `/api/chat` handler downstream never learn it was
    /// an Anthropic request.
    pub chat_request: ChatRequest,
    /// Did the client ask to stream? Decides [`WriterOutput`]'s shape and
    /// whether [`SSE_STREAM_HEADERS`] apply.
    pub stream: bool,
    /// Fills `message_start.usage.input_tokens` until the real count lands.
    /// See [`crate::anthropic::estimate_tokens`] -- it is `len/4`, not real
    /// tokenisation.
    pub estimated_input_tokens: i32,
    /// **Upstream:** `c.Set("relax_thinking", true)`, always, unconditionally.
    ///
    /// Upstream's comment: *"Set think to nil when being used with Anthropic
    /// API to connect to tools like claude code"*. The reason is real -- Claude
    /// Code send `thinking` on models that do not advertise the `thinking`
    /// capability, and without this the request would 400 on a capability
    /// check instead of just quietly not thinking. Always `true`; it is a
    /// field rather than a constant so a caller cannot forget it exists.
    pub relax_thinking: bool,
    /// Did the request offer a built-in web_search tool? If so the caller must
    /// wrap the response path in [`WebSearchStreamState`] +
    /// [`run_web_search_loop`] instead of using [`AnthropicWriter`] alone.
    pub web_search: bool,
}

/// Validate a `/v1/messages` request and rewrite it into an ollama chat request.
///
/// **Upstream:** `AnthropicMessagesMiddleware`, down to (but not including)
/// `c.Next()`.
///
/// The three required-field checks are in **upstream's order**, and the order
/// is observable: a request missing both `model` and `max_tokens` reports
/// `"model is required"`, never the other one. Clients and test suites match on
/// these exact strings, so they are reproduced verbatim.
///
/// | refused when | status | message |
/// |---|---|---|
/// | `model` empty | 400 | `model is required` |
/// | `max_tokens` <= 0 | 400 | `max_tokens is required and must be positive` |
/// | `messages` empty | 400 | `messages is required` |
/// | conversion fail | 400 | whatever [`crate::anthropic::ConvertError`] say |
///
/// Note `max_tokens` is checked `<= 0`, not `== 0`: a client sending `-1`
/// gets the same refusal, rather than a `num_predict: -1` that would mean
/// "generate forever" to the runtime.
///
/// **Not ported:** the JSON-decode failure branch. Upstream's `ShouldBindJSON`
/// both parse and validate; in Rust the parse is `serde_json::from_slice`, so
/// its error is the caller's to catch -- turn it into
/// [`AnthropicError::bad_request`] with the serde message and you have the same
/// 400 `invalid_request_error`.
pub fn prepare_messages_request(req: &MessagesRequest) -> Result<MessagesPlan, AnthropicError> {
    if req.model.is_empty() {
        return Err(AnthropicError::bad_request("model is required"));
    }
    if req.max_tokens <= 0 {
        return Err(AnthropicError::bad_request("max_tokens is required and must be positive"));
    }
    if req.messages.is_empty() {
        return Err(AnthropicError::bad_request("messages is required"));
    }

    let chat_request =
        anthropic::from_messages_request(req).map_err(|e| AnthropicError::bad_request(e.to_string()))?;

    Ok(MessagesPlan {
        chat_request,
        stream: req.stream,
        estimated_input_tokens: anthropic::estimate_input_tokens(req),
        relax_thinking: true,
        web_search: has_web_search_tool(&req.tools),
    })
}

/// Does the request offer a built-in web search tool?
/// **Upstream:** `middleware.hasWebSearchTool`.
///
/// Matched on the **type prefix**, not the name -- Anthropic date-version their
/// server tools (`web_search_20250305`), so a full-string match would go blind
/// on the next revision.
pub fn has_web_search_tool(tools: &[anthropic::Tool]) -> bool {
    tools.iter().any(|t| t.tool_type.starts_with("web_search"))
}

// ===========================================================================
// SSE framing
// ===========================================================================

/// One Server-Sent Event frame: `event: <name>\ndata: <json>\n\n`.
///
/// **Upstream:** `middleware.writeSSE`, minus the `http.Flusher` call -- we
/// return the bytes and whoever own the socket decides when to flush.
///
/// **What would make this wrong:** dropping either newline. SSE frames are
/// delimited by a *blank* line, so `\n\n` at the end is the delimiter, not
/// padding; and a JSON payload containing a raw newline would split one event
/// into two. `serde_json` escape newlines inside strings, so a compact
/// serialisation is safe here -- a **pretty** one would not be.
pub fn write_sse<T: Serialize>(event: &str, data: &T) -> Result<String, serde_json::Error> {
    let json = serde_json::to_string(data)?;
    Ok(format!("event: {event}\ndata: {json}\n\n"))
}

/// [`write_sse`] for an already-built event.
pub fn sse_frame(event: &StreamEvent) -> Result<String, serde_json::Error> {
    write_sse(&event.event, &event.data)
}

/// Frame a whole batch, in order.
pub fn sse_frames(events: &[StreamEvent]) -> Result<String, serde_json::Error> {
    let mut out = String::new();
    for e in events {
        out.push_str(&sse_frame(e)?);
    }
    Ok(out)
}

// ===========================================================================
// The response writer
// ===========================================================================

/// What the caller should put on the wire.
///
/// **Upstream:** the side effects of `(*AnthropicWriter).Write` -- it set a
/// Content-Type and encode straight into the socket. Returning a value instead
/// is what make the whole thing testable without a server.
#[derive(Debug, Clone, PartialEq)]
pub enum WriterOutput {
    /// A buffered reply. `Content-Type: application/json`.
    Json(Box<MessagesResponse>),
    /// Streaming events, to be framed with [`sse_frames`].
    /// `Content-Type: text/event-stream`.
    Events(Vec<StreamEvent>),
    /// An upstream failure, re-shaped into Anthropic's envelope.
    /// `Content-Type: application/json`.
    Error(Box<ErrorResponse>),
}

impl WriterOutput {
    /// **Upstream:** the exact `Header().Set("Content-Type", ...)` each branch
    /// of `Write` perform.
    pub fn content_type(&self) -> &'static str {
        match self {
            WriterOutput::Events(_) => "text/event-stream",
            WriterOutput::Json(_) | WriterOutput::Error(_) => "application/json",
        }
    }
}

/// Pull an error message out of whatever ollama's handlers wrote.
///
/// **Upstream:** the anonymous `struct{ Error string }` unmarshal at the top of
/// `(*AnthropicWriter).writeError`.
///
/// Handles three real shapes, and that is the whole point:
///
/// * `{"error": "..."}` -- what nearly every `routes.go` handler write
///   (`gin.H{"error": msg}`), and it carries **no status**, which is why the
///   HTTP status has to be passed in separately;
/// * [`crate::routes::StatusError`] -- `{"StatusCode":404,"Status":"","error":"..."}`.
///   Note it also has an `error` key, so the same one-field unmarshal catch it;
/// * anything that is not JSON at all -- a panic dump, an HTML proxy page --
///   in which case the **raw bytes become the message**. Upstream's comment:
///   *"rather than surfacing a confusing JSON parse error"*, and they are
///   right: a client seeing "invalid character '<'" learn nothing.
pub fn error_message_from_body(body: &[u8]) -> String {
    #[derive(serde::Deserialize)]
    struct ErrData {
        #[serde(default)]
        error: String,
    }
    match serde_json::from_slice::<ErrData>(body) {
        Ok(d) => d.error,
        Err(_) => String::from_utf8_lossy(body).into_owned(),
    }
}

/// The ordinary (no web search) response path.
///
/// **Upstream:** `middleware.AnthropicWriter`. One per request, and it is
/// **stateful** because the streaming case is -- see [`StreamConverter`].
#[derive(Debug, Clone)]
pub struct AnthropicWriter {
    /// The `msg_...` id. Every event and the buffered reply carry it.
    pub id: String,
    stream: bool,
    converter: StreamConverter,
}

impl AnthropicWriter {
    /// `id` is the message id (see [`crate::anthropic::generate_message_id`]),
    /// `model` the **Anthropic request's** model name -- not whatever ollama
    /// resolved it to, because the client asked about the former.
    pub fn new(
        id: impl Into<String>,
        model: impl Into<String>,
        stream: bool,
        estimated_input_tokens: i32,
    ) -> Self {
        let id = id.into();
        AnthropicWriter {
            converter: StreamConverter::new(id.clone(), model, estimated_input_tokens),
            id,
            stream,
        }
    }

    /// Feed one chunk of whatever `/api/chat` wrote.
    ///
    /// **Upstream:** `(*AnthropicWriter).Write`. The status is the gate: a
    /// non-200 mean the body is an error and never a `ChatResponse`, so it is
    /// re-shaped and nothing else happens.
    ///
    /// A 200 body that will not parse as a [`ChatResponse`] is a genuine
    /// `Err` -- upstream return the unmarshal error too, and there is nothing
    /// sensible to send a client in that case.
    pub fn write(
        &mut self,
        status: u16,
        body: &[u8],
        request_id: &str,
    ) -> Result<WriterOutput, serde_json::Error> {
        if status != 200 {
            return Ok(WriterOutput::Error(Box::new(
                self.write_error(status, body, request_id),
            )));
        }
        let chat: ChatResponse = serde_json::from_slice(body)?;
        Ok(self.write_response(&chat))
    }

    /// **Upstream:** `(*AnthropicWriter).writeError`. See
    /// [`error_message_from_body`] for the shapes it swallow.
    pub fn write_error(&self, status: u16, body: &[u8], request_id: &str) -> ErrorResponse {
        ErrorResponse::new(status, error_message_from_body(body), request_id)
    }

    /// **Upstream:** `(*AnthropicWriter).writeResponse`.
    ///
    /// Streaming feed the chunk through the [`StreamConverter`] (which may
    /// return **no events at all** for a chunk with nothing new in it -- that
    /// is fine, write nothing). Buffered convert the whole response in one go.
    pub fn write_response(&mut self, chat: &ChatResponse) -> WriterOutput {
        if self.stream {
            WriterOutput::Events(self.converter.process(chat))
        } else {
            WriterOutput::Json(Box::new(anthropic::to_messages_response(&self.id, chat)))
        }
    }

    /// The converter, for a caller that need to drive it directly (the
    /// web-search passthrough path does).
    pub fn converter_mut(&mut self) -> &mut StreamConverter {
        &mut self.converter
    }

    pub fn is_streaming(&self) -> bool {
        self.stream
    }
}

// ===========================================================================
// Web search: small helpers
// ===========================================================================

/// How many search -> re-ask rounds before we give up.
/// **Upstream:** `middleware.maxWebSearchLoops`.
///
/// Three, and the cap matter: each loop is a real search **and** a full
/// re-generation, so an unbounded loop is a model that can spend somebody's
/// machine indefinitely. Hitting the cap is not an error -- see
/// [`run_web_search_loop`], it end the turn with a `max_uses_exceeded` result
/// block.
pub const MAX_WEB_SEARCH_LOOPS: u32 = 3;

/// How long upstream give the whole loop.
/// **Upstream:** `context.WithTimeout(requestCtx, 5*time.Minute)`.
///
/// Nothing in this module enforce it -- there is no runtime here to enforce it
/// with. It is exposed so a caller that own the I/O use ollama's number
/// instead of inventing one.
pub const WEB_SEARCH_LOOP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Derive the `srvtoolu_...` id from the `msg_...` id.
/// **Upstream:** `middleware.serverToolUseID`.
///
/// Just a prefix swap, and note it is `TrimPrefix`, not "strip everything up to
/// the first underscore": an id that does **not** start with `msg_` keeps its
/// whole self, so `nomsgprefix` become `srvtoolu_nomsgprefix`.
pub fn server_tool_use_id(message_id: &str) -> String {
    format!("srvtoolu_{}", message_id.strip_prefix("msg_").unwrap_or(message_id))
}

/// The same id, suffixed per loop iteration.
/// **Upstream:** `middleware.loopServerToolUseID`.
///
/// **The first loop get no suffix** -- `loop <= 1` return the bare id. That is
/// not an off-by-one: the overwhelmingly common case is a single search, and it
/// keeps that id identical to the non-loop one.
pub fn loop_server_tool_use_id(message_id: &str, loop_n: u32) -> String {
    let base = server_tool_use_id(message_id);
    if loop_n <= 1 {
        base
    } else {
        format!("{base}_{loop_n}")
    }
}

/// A one-key `{"query": ...}` argument map.
/// **Upstream:** `middleware.queryArgs`.
pub fn query_args(query: &str) -> ToolCallArguments {
    let mut args = ToolCallArguments::new();
    args.set("query", serde_json::Value::String(query.to_string()));
    args
}

/// Pull the search string out of a `web_search` tool call.
/// **Upstream:** `middleware.extractQueryFromToolCall`.
///
/// Returns `""` for every failure -- no `query` key, or a `query` that is not a
/// string (a model emitting `{"query": 5}` is not unheard of). The caller must
/// treat empty as a refusal; [`run_web_search_loop`] does, with the
/// `invalid_request` code.
pub fn extract_query_from_tool_call(tc: &ToolCall) -> String {
    tc.function
        .arguments
        .get("query")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_string()
}

/// Find the first `web_search` call, and say whether anything else was
/// alongside it. **Upstream:** `middleware.findWebSearchToolCall`.
///
/// Returns `(the call, there were other tool calls too)`.
///
/// **First** web_search wins; later ones in the same chunk are ignored. And
/// when both a server tool and client tools appear in one response, upstream
/// **prefer web_search** and drop the client calls (they only `slog.Debug` it).
/// That is a real behaviour, not a nicety: the client never sees those tool
/// calls, so a model that asked to search *and* call a client tool in one turn
/// will find its client tool silently unanswered.
pub fn find_web_search_tool_call(tool_calls: &[ToolCall]) -> (Option<ToolCall>, bool) {
    let mut web_search_call: Option<ToolCall> = None;
    let mut has_other_tools = false;

    for tc in tool_calls {
        if tc.function.name == "web_search" {
            if web_search_call.is_none() {
                web_search_call = Some(tc.clone());
            }
            continue;
        }
        has_other_tools = true;
    }

    (web_search_call, has_other_tools)
}

/// Rebuild the assistant turn that asked for the search, so the follow-up
/// request has a coherent history. **Upstream:**
/// `middleware.buildWebSearchAssistantMessage`.
///
/// Only the **web_search** call is carried over -- any client tool calls in the
/// same response are dropped here too, consistent with
/// [`find_web_search_tool_call`]'s preference.
pub fn build_web_search_assistant_message(
    response: &ChatResponse,
    web_search_call: &ToolCall,
) -> Message {
    Message {
        role: "assistant".to_string(),
        content: response.message.content.clone(),
        thinking: response.message.thinking.clone(),
        tool_calls: vec![web_search_call.clone()],
        ..Default::default()
    }
}

/// Render search hits as the `role:"tool"` message the model will read.
/// **Upstream:** `middleware.formatWebSearchResultsForToolMessage`.
///
/// `"Title: ...\nURL: ...\n"`, plus `"Content: ...\n"` when there is a body,
/// then a **blank line** between hits. This is where ollama's `content` field
/// finally reach the model -- the Anthropic-facing
/// [`crate::anthropic::convert_ollama_to_anthropic_results`] drop it, because
/// Anthropic's `WebSearchResult` has no field for it. Two audiences, two
/// renderings, on purpose.
pub fn format_web_search_results_for_tool_message(results: &[OllamaWebSearchResult]) -> String {
    let mut out = String::new();
    for r in results {
        out.push_str(&format!("Title: {}\nURL: {}\n", r.title, r.url));
        if !r.content.is_empty() {
            out.push_str(&format!("Content: {}\n", r.content));
        }
        out.push('\n');
    }
    out
}

/// The reply for "the search could not be done".
/// **Upstream:** `(*WebSearchAnthropicWriter).webSearchErrorResponse`.
///
/// Note it is **not** an error envelope -- it is a perfectly normal `200`
/// assistant message whose content is a `server_tool_use` + a
/// `web_search_tool_result` carrying the error code, and whose `stop_reason` is
/// `end_turn`. That is Anthropic's own convention for a failed server tool, and
/// it is why a client sees "I tried to search and could not" rather than a
/// dead request.
pub fn web_search_error_response(
    message_id: &str,
    model: &str,
    error_code: &str,
    query: &str,
    usage: Usage,
) -> MessagesResponse {
    let tool_use_id = server_tool_use_id(message_id);
    MessagesResponse {
        id: message_id.to_string(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        model: model.to_string(),
        content: vec![
            ContentBlock {
                block_type: "server_tool_use".to_string(),
                id: tool_use_id.clone(),
                name: "web_search".to_string(),
                input: Some(query_args(query)),
                ..Default::default()
            },
            ContentBlock {
                block_type: "web_search_tool_result".to_string(),
                tool_use_id,
                content: Some(
                    serde_json::to_value(WebSearchToolResultError {
                        error_type: "web_search_tool_result_error".to_string(),
                        error_code: error_code.to_string(),
                    })
                    .unwrap_or(serde_json::Value::Null),
                ),
                ..Default::default()
            },
        ],
        stop_reason: "end_turn".to_string(),
        stop_sequence: String::new(),
        usage,
    }
}

/// Glue the server-tool blocks in front of the model's final answer.
/// **Upstream:** `(*WebSearchAnthropicWriter).combineServerAndFinalContent`.
///
/// The `usage` is **not** the final response's -- it is the running total the
/// loop accumulated, so the client see what the whole multi-round turn cost,
/// not just the last generation.
pub fn combine_server_and_final_content(
    message_id: &str,
    model: &str,
    server_content: &[ContentBlock],
    final_response: &ChatResponse,
    usage: Usage,
) -> MessagesResponse {
    let converted = anthropic::to_messages_response(message_id, final_response);

    let mut content = Vec::with_capacity(server_content.len() + converted.content.len());
    content.extend_from_slice(server_content);
    content.extend(converted.content);

    MessagesResponse {
        id: message_id.to_string(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        model: model.to_string(),
        content,
        stop_reason: converted.stop_reason,
        stop_sequence: converted.stop_sequence,
        usage,
    }
}

// ===========================================================================
// Web search: the loop
// ===========================================================================

/// The two I/O calls the search loop need, behind one trait.
///
/// **Upstream:** `anthropic.WebSearch` (a signed POST to
/// [`crate::anthropic::WEB_SEARCH_ENDPOINT`]) and
/// `(*WebSearchAnthropicWriter).callFollowUpChat` (a POST to the server's own
/// `/api/chat`).
///
/// **Divergence, and it is the whole reason this module is testable:** neither
/// is ported. Both are HTTP, one needs registry request signing, and both
/// would drag a client into a crate that promises to build with no socket. So
/// they become two methods a caller implement -- exactly the seam
/// `registry::Transport` and `routes::ModelCatalog` already use.
///
/// Errors are `String` rather than a typed enum because the loop only ever does
/// one thing with them: attach them to a [`WebSearchLoopError`] as context.
/// A caller's real error type stay its own business.
pub trait WebSearchBackend {
    /// Run a web search. `max_results` arrive already clamped by
    /// [`crate::anthropic::clamp_web_search_max_results`].
    fn search(&mut self, query: &str, max_results: i32) -> Result<OllamaWebSearchResponse, String>;

    /// Re-ask the model, **buffered** -- `req.stream` is already `Some(false)`.
    /// Upstream call the server's own `/api/chat` over loopback HTTP; you may
    /// call the handler directly if you have it.
    fn chat(&mut self, req: &ChatRequest) -> Result<ChatResponse, String>;
}

/// Why the search loop gave up.
/// **Upstream:** `middleware.webSearchLoopError`.
///
/// `code` is one of `invalid_request` (the model asked to search with an empty
/// query), `unavailable` (the search itself failed) or `api_error` (the
/// follow-up generation failed). It goes straight into the
/// `web_search_tool_result_error` block that [`web_search_error_response`]
/// build, so it is client-visible text, not an internal tag.
///
/// `usage` is the running total **at the moment of failure** -- a failed turn
/// still consumed tokens and the client is still told about them.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{}", match cause { Some(c) => format!("{code}: {c}"), None => code.clone() })]
pub struct WebSearchLoopError {
    pub code: String,
    pub query: String,
    pub usage: Usage,
    /// The underlying failure, if there was one. **Upstream:** the `err` field,
    /// and its `Error()` render as `"{code}: {err}"` when set, bare `code`
    /// otherwise -- reproduced by the `#[error]` above.
    pub cause: Option<String>,
}

/// Run the search -> re-ask loop until the model stop asking to search.
///
/// **Upstream:** `(*WebSearchAnthropicWriter).runWebSearchLoop`.
///
/// Each iteration:
///
/// 1. pull the query out of the current `web_search` call -- **empty is fatal**
///    (`invalid_request`);
/// 2. search (5 results, [`crate::anthropic::WEB_SEARCH_DEFAULT_MAX_RESULTS`]);
///    a failure is fatal (`unavailable`);
/// 3. append a `server_tool_use` + `web_search_tool_result` pair to the
///    client-facing content, under a per-loop id
///    ([`loop_server_tool_use_id`]);
/// 4. append the assistant turn + the tool result to the **model-facing**
///    history and re-ask;
/// 5. add the follow-up's tokens to the running usage;
/// 6. if the follow-up asked to search again, go round; otherwise stitch the
///    server blocks in front of its answer and return.
///
/// Hitting [`MAX_WEB_SEARCH_LOOPS`] is **not an error**: the turn ends with one
/// last `server_tool_use` + a `max_uses_exceeded` result and `stop_reason:
/// end_turn`, so the client gets a coherent message saying the search budget
/// ran out. Note that last block use loop index `MAX+1`, so its id never
/// collide with a real round's.
///
/// **Divergence: this is synchronous.** Upstream fire the loop on a goroutine
/// (`startLoopWorker`) so the *original* generation can keep streaming to the
/// client while the search runs, then join on a channel when the original
/// finishes. There is no async runtime in this crate, so the concurrency is the
/// caller's to arrange if they want it -- run this on a thread and keep the
/// passthrough events flowing, or just call it inline and accept the latency.
/// The **logic** is identical either way; only the scheduling moved out.
pub fn run_web_search_loop<B: WebSearchBackend>(
    backend: &mut B,
    message_id: &str,
    model: &str,
    chat_req: &ChatRequest,
    initial_response: &ChatResponse,
    initial_tool_call: &ToolCall,
    initial_usage: Usage,
) -> Result<MessagesResponse, WebSearchLoopError> {
    let mut follow_up_messages: Vec<Message> = chat_req.messages.clone();
    let follow_up_tools = chat_req.tools.clone();
    let mut usage = initial_usage;

    let mut current_response = initial_response.clone();
    let mut current_tool_call = initial_tool_call.clone();
    let mut server_content: Vec<ContentBlock> = Vec::new();

    for loop_n in 1..=MAX_WEB_SEARCH_LOOPS {
        let query = extract_query_from_tool_call(&current_tool_call);
        if query.is_empty() {
            return Err(WebSearchLoopError {
                code: "invalid_request".to_string(),
                query: String::new(),
                usage,
                cause: None,
            });
        }

        let search_resp = backend
            .search(&query, anthropic::WEB_SEARCH_DEFAULT_MAX_RESULTS)
            .map_err(|e| WebSearchLoopError {
                code: "unavailable".to_string(),
                query: query.clone(),
                usage,
                cause: Some(e),
            })?;

        let tool_use_id = loop_server_tool_use_id(message_id, loop_n);
        let search_results = anthropic::convert_ollama_to_anthropic_results(&search_resp);
        server_content.push(ContentBlock {
            block_type: "server_tool_use".to_string(),
            id: tool_use_id.clone(),
            name: "web_search".to_string(),
            input: Some(query_args(&query)),
            ..Default::default()
        });
        server_content.push(ContentBlock {
            block_type: "web_search_tool_result".to_string(),
            tool_use_id,
            content: Some(
                serde_json::to_value(&search_results).unwrap_or(serde_json::Value::Null),
            ),
            ..Default::default()
        });

        follow_up_messages.push(build_web_search_assistant_message(
            &current_response,
            &current_tool_call,
        ));
        follow_up_messages.push(Message {
            role: "tool".to_string(),
            content: format_web_search_results_for_tool_message(&search_resp.results),
            tool_call_id: current_tool_call.id.clone(),
            ..Default::default()
        });

        let follow_up_request = ChatRequest {
            model: chat_req.model.clone(),
            messages: follow_up_messages.clone(),
            // Always buffered: the loop need the WHOLE answer to decide
            // whether to go round again.
            stream: Some(false),
            tools: follow_up_tools.clone(),
            options: chat_req.options.clone(),
            ..Default::default()
        };

        let follow_up_response = backend.chat(&follow_up_request).map_err(|e| WebSearchLoopError {
            code: "api_error".to_string(),
            query: query.clone(),
            usage,
            cause: Some(e),
        })?;

        usage.input_tokens += follow_up_response.metrics.prompt_eval_count;
        usage.output_tokens += follow_up_response.metrics.eval_count;

        let (next_tool_call, _has_other_tools) =
            find_web_search_tool_call(&follow_up_response.message.tool_calls);

        let Some(next_tool_call) = next_tool_call else {
            return Ok(combine_server_and_final_content(
                message_id,
                model,
                &server_content,
                &follow_up_response,
                usage,
            ));
        };

        current_response = follow_up_response;
        current_tool_call = next_tool_call;
    }

    // Budget exhausted. Not an error -- a coherent end_turn saying so.
    let max_loop_query = extract_query_from_tool_call(&current_tool_call);
    let max_loop_tool_use_id = loop_server_tool_use_id(message_id, MAX_WEB_SEARCH_LOOPS + 1);
    server_content.push(ContentBlock {
        block_type: "server_tool_use".to_string(),
        id: max_loop_tool_use_id.clone(),
        name: "web_search".to_string(),
        input: Some(query_args(&max_loop_query)),
        ..Default::default()
    });
    server_content.push(ContentBlock {
        block_type: "web_search_tool_result".to_string(),
        tool_use_id: max_loop_tool_use_id,
        content: Some(
            serde_json::to_value(WebSearchToolResultError {
                error_type: "web_search_tool_result_error".to_string(),
                error_code: "max_uses_exceeded".to_string(),
            })
            .unwrap_or(serde_json::Value::Null),
        ),
        ..Default::default()
    });

    Ok(MessagesResponse {
        id: message_id.to_string(),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        model: model.to_string(),
        content: server_content,
        stop_reason: "end_turn".to_string(),
        stop_sequence: String::new(),
        usage,
    })
}

// ===========================================================================
// Web search: streaming bookkeeping
// ===========================================================================

/// Tracks what a web-search-wrapped stream has already sent.
///
/// **Upstream:** the streaming fields of `WebSearchAnthropicWriter`
/// (`terminalSent`, `streamMessageStarted`, `streamHasOpenBlock`,
/// `streamOpenBlockIndex`, `streamNextIndex`) plus its usage counters.
///
/// ## Why it exists
///
/// The web-search path have **two** producers of events into one stream: the
/// original generation, whose events come out of a [`StreamConverter`] and
/// pass straight through, and the search loop's final answer, emitted later by
/// [`WebSearchStreamState::terminal`]. The second must carry on from where the
/// first stopped -- same message, no duplicate `message_start`, no block index
/// reused, no block left open. So this struct watch the passthrough events go
/// by and remember exactly that.
///
/// **What would make this wrong:** feeding it the passthrough events out of
/// order, or not feeding it some of them. It is a mirror of what actually
/// reached the client; if it drifts, the terminal events will reuse an index
/// and the client reports "content block not found".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WebSearchStreamState {
    message_started: bool,
    has_open_block: bool,
    open_block_index: usize,
    next_index: usize,
    terminal_sent: bool,
    observed_prompt_eval_count: i32,
    observed_eval_count: i32,
    loop_base_input_tokens: i32,
    loop_base_output_tokens: i32,
}

impl WebSearchStreamState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Has the terminal `message_stop` already gone out? Once true, everything
    /// else on this state is a no-op -- upstream's `Write` return early on it,
    /// because a second `message_stop` would corrupt the stream.
    pub fn terminal_sent(&self) -> bool {
        self.terminal_sent
    }

    /// **Upstream:** `(*WebSearchAnthropicWriter).recordObservedUsage`.
    ///
    /// **Monotonic maximum, not last-writer-wins.** Ollama report metrics only
    /// on some chunks (often only the last), so a later chunk carrying zeros
    /// must not wipe a real count seen earlier.
    pub fn record_observed_usage(&mut self, metrics: &Metrics) {
        if metrics.prompt_eval_count > self.observed_prompt_eval_count {
            self.observed_prompt_eval_count = metrics.prompt_eval_count;
        }
        if metrics.eval_count > self.observed_eval_count {
            self.observed_eval_count = metrics.eval_count;
        }
    }

    /// **Upstream:** `(*WebSearchAnthropicWriter).currentObservedUsage`.
    pub fn current_observed_usage(&self) -> Usage {
        Usage {
            input_tokens: self.observed_prompt_eval_count,
            output_tokens: self.observed_eval_count,
        }
    }

    /// The usage the loop should start from, and the baseline it will later be
    /// topped up against. **Upstream:** the `max(observed, chunk)` pair at the
    /// top of `startLoopWorker` / the sync branch of `Write`.
    ///
    /// `max` of the two because the triggering chunk and the running observed
    /// maximum can each be the fresher one, depending on where in the stream
    /// the tool call landed.
    pub fn begin_loop(&mut self, metrics: &Metrics) -> Usage {
        let usage = Usage {
            input_tokens: self.observed_prompt_eval_count.max(metrics.prompt_eval_count),
            output_tokens: self.observed_eval_count.max(metrics.eval_count),
        };
        self.loop_base_input_tokens = usage.input_tokens;
        self.loop_base_output_tokens = usage.output_tokens;
        usage
    }

    /// Add whatever the **original** generation kept spending while the loop
    /// ran. **Upstream:**
    /// `(*WebSearchAnthropicWriter).applyObservedUsageDeltaToUsage`.
    ///
    /// Only the growth **since** [`begin_loop`](Self::begin_loop) is added, and
    /// only if positive -- the loop's own totals already include the baseline,
    /// so adding the whole observed count again would double-bill the client.
    pub fn apply_observed_usage_delta(&self, usage: &mut Usage) {
        let delta_in = self.observed_prompt_eval_count - self.loop_base_input_tokens;
        if delta_in > 0 {
            usage.input_tokens += delta_in;
        }
        let delta_out = self.observed_eval_count - self.loop_base_output_tokens;
        if delta_out > 0 {
            usage.output_tokens += delta_out;
        }
    }

    /// Watch the passthrough events go out, and remember where the stream is.
    /// **Upstream:** the type switch in
    /// `(*WebSearchAnthropicWriter).writePassthroughStreamChunk`.
    ///
    /// The Rust enum make this exhaustive where Go's type switch had a silent
    /// `default` -- see [`StreamEventData`].
    pub fn observe_passthrough(&mut self, events: &[StreamEvent]) {
        for e in events {
            match &e.data {
                StreamEventData::MessageStart(_) => self.message_started = true,
                StreamEventData::ContentBlockStart(d) => {
                    self.has_open_block = true;
                    self.open_block_index = d.index;
                    self.next_index = self.next_index.max(d.index + 1);
                }
                StreamEventData::ContentBlockStop(d) => {
                    if self.has_open_block && self.open_block_index == d.index {
                        self.has_open_block = false;
                    }
                    self.next_index = self.next_index.max(d.index + 1);
                }
                StreamEventData::MessageStop(_) => self.terminal_sent = true,
                _ => {}
            }
        }
    }

    /// Emit the loop's final answer, carrying on from wherever the passthrough
    /// left the stream.
    ///
    /// **Upstream:** `(*WebSearchAnthropicWriter).writeTerminalResponse` (and
    /// `streamResponse`, which is just an alias for it).
    ///
    /// Returns `None` when the terminal has already been sent -- that is the
    /// idempotence upstream get from its `terminalSent` early return, and it
    /// matters because both the "original generation finished" path and the
    /// "loop finished" path can reach here.
    ///
    /// Buffered: one [`WriterOutput::Json`]. Streaming, in order:
    ///
    /// 1. `message_start`, **only if one has not already gone out**;
    /// 2. `content_block_stop` for whatever block the passthrough left open;
    /// 3. per content block -- a `text` block get start(empty) + delta + stop,
    ///    **everything else** get start(whole block) + stop with no delta. That
    ///    asymmetry is deliberate: only text has anything to stream, while a
    ///    `server_tool_use` or `web_search_tool_result` is already complete, so
    ///    it ride entirely in its start event;
    /// 4. `message_delta` with the stop reason and total usage, then
    ///    `message_stop`.
    pub fn terminal(
        &mut self,
        stream: bool,
        model: &str,
        estimated_input_tokens: i32,
        response: &MessagesResponse,
    ) -> Option<WriterOutput> {
        if self.terminal_sent {
            return None;
        }

        if !stream {
            self.terminal_sent = true;
            return Some(WriterOutput::Json(Box::new(response.clone())));
        }

        let mut events = Vec::new();
        events.extend(self.ensure_stream_message_start(
            &response.id,
            model,
            estimated_input_tokens,
            response.usage,
        ));
        events.extend(self.close_open_stream_block());
        events.extend(self.stream_content_blocks(&response.content));

        events.push(StreamEvent::new(
            "message_delta",
            StreamEventData::MessageDelta(MessageDeltaEvent {
                event_type: "message_delta".to_string(),
                delta: MessageDelta {
                    stop_reason: response.stop_reason.clone(),
                    stop_sequence: String::new(),
                },
                usage: DeltaUsage {
                    input_tokens: response.usage.input_tokens,
                    output_tokens: response.usage.output_tokens,
                },
            }),
        ));
        events.push(StreamEvent::new(
            "message_stop",
            StreamEventData::MessageStop(MessageStopEvent {
                event_type: "message_stop".to_string(),
            }),
        ));

        self.terminal_sent = true;
        Some(WriterOutput::Events(events))
    }

    /// **Upstream:** `(*WebSearchAnthropicWriter).ensureStreamMessageStart`.
    ///
    /// The estimate is only used when the real `input_tokens` is still 0 --
    /// which happen when the loop failed before any generation reported metrics.
    fn ensure_stream_message_start(
        &mut self,
        message_id: &str,
        model: &str,
        estimated_input_tokens: i32,
        usage: Usage,
    ) -> Vec<StreamEvent> {
        if self.message_started {
            return Vec::new();
        }
        let input_tokens =
            if usage.input_tokens == 0 { estimated_input_tokens } else { usage.input_tokens };

        self.message_started = true;
        vec![StreamEvent::new(
            "message_start",
            StreamEventData::MessageStart(MessageStartEvent {
                event_type: "message_start".to_string(),
                message: MessagesResponse {
                    id: message_id.to_string(),
                    response_type: "message".to_string(),
                    role: "assistant".to_string(),
                    model: model.to_string(),
                    content: Vec::new(),
                    // Note only input_tokens is set -- output is still 0 here,
                    // same as upstream's partially-filled Usage literal.
                    usage: Usage { input_tokens, output_tokens: 0 },
                    ..Default::default()
                },
            }),
        )]
    }

    /// **Upstream:** `(*WebSearchAnthropicWriter).closeOpenStreamBlock`.
    fn close_open_stream_block(&mut self) -> Vec<StreamEvent> {
        if !self.has_open_block {
            return Vec::new();
        }
        let index = self.open_block_index;
        self.next_index = self.next_index.max(index + 1);
        self.has_open_block = false;

        vec![StreamEvent::new(
            "content_block_stop",
            StreamEventData::ContentBlockStop(ContentBlockStopEvent {
                event_type: "content_block_stop".to_string(),
                index,
            }),
        )]
    }

    /// **Upstream:** `(*WebSearchAnthropicWriter).writeStreamContentBlocks`.
    fn stream_content_blocks(&mut self, content: &[ContentBlock]) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        for block in content {
            let index = self.next_index;

            if block.block_type == "text" {
                events.push(StreamEvent::new(
                    "content_block_start",
                    StreamEventData::ContentBlockStart(ContentBlockStartEvent {
                        event_type: "content_block_start".to_string(),
                        index,
                        // Empty text, then the real text as a delta -- the SDK
                        // accumulation contract. See ContentBlock's docs.
                        content_block: ContentBlock::text(""),
                    }),
                ));
                events.push(StreamEvent::new(
                    "content_block_delta",
                    StreamEventData::ContentBlockDelta(ContentBlockDeltaEvent {
                        event_type: "content_block_delta".to_string(),
                        index,
                        delta: Delta {
                            delta_type: "text_delta".to_string(),
                            text: block.text.clone().unwrap_or_default(),
                            ..Default::default()
                        },
                    }),
                ));
            } else {
                // Already complete -- the whole block rides in the start event
                // and there is nothing to delta.
                events.push(StreamEvent::new(
                    "content_block_start",
                    StreamEventData::ContentBlockStart(ContentBlockStartEvent {
                        event_type: "content_block_start".to_string(),
                        index,
                        content_block: block.clone(),
                    }),
                ));
            }

            events.push(StreamEvent::new(
                "content_block_stop",
                StreamEventData::ContentBlockStop(ContentBlockStopEvent {
                    event_type: "content_block_stop".to_string(),
                    index,
                }),
            ));

            self.next_index += 1;
        }
        events
    }
}

// ===========================================================================
// Tests -- ported from `middleware/anthropic_test.go`
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{ThinkValue, ToolCallFunction};
    use serde_json::json;

    fn request(body: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(body).expect("parse the request")
    }

    fn call(name: &str, args: &[(&str, serde_json::Value)]) -> ToolCall {
        let mut a = ToolCallArguments::new();
        for (k, v) in args {
            a.set(*k, v.clone());
        }
        ToolCall {
            id: format!("call_{name}"),
            function: ToolCallFunction { index: 0, name: name.to_string(), arguments: a },
        }
    }

    // -- prepare_messages_request --------------------------------------------

    #[test]
    fn a_basic_request_is_rewritten_into_a_buffered_chat_request() {
        let plan = prepare_messages_request(&request(json!({
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}]
        })))
        .expect("prepare");

        assert_eq!(plan.chat_request.model, "test-model");
        assert_eq!(plan.chat_request.messages.len(), 1);
        assert_eq!(plan.chat_request.messages[0].content, "Hello");
        assert_eq!(plan.chat_request.stream, Some(false));
        assert!(!plan.stream);
        assert!(plan.relax_thinking);
        assert!(!plan.web_search);
    }

    #[test]
    fn a_system_prompt_is_prepended_as_its_own_message() {
        let plan = prepare_messages_request(&request(json!({
            "model": "test-model",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hello"}]
        })))
        .expect("prepare");

        assert_eq!(
            plan.chat_request
                .messages
                .iter()
                .map(|m| (m.role.as_str(), m.content.as_str()))
                .collect::<Vec<_>>(),
            vec![("system", "You are helpful."), ("user", "Hello")]
        );
    }

    #[test]
    fn sampling_options_survive_the_rewrite() {
        let plan = prepare_messages_request(&request(json!({
            "model": "test-model",
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["\n", "END"],
            "messages": [{"role": "user", "content": "Hello"}]
        })))
        .expect("prepare");

        let opts = plan.chat_request.options.expect("options");
        assert_eq!(opts.get("num_predict"), Some(&json!(2048)));
        assert_eq!(opts.get("temperature"), Some(&json!(0.7)));
        assert_eq!(opts.get("top_p"), Some(&json!(0.9)));
        assert_eq!(opts.get("top_k"), Some(&json!(40)));
        assert_eq!(opts.get("stop"), Some(&json!(["\n", "END"])));
    }

    #[test]
    fn an_explicit_stream_true_is_carried_through_to_both_the_plan_and_the_chat_request() {
        let plan = prepare_messages_request(&request(json!({
            "model": "test-model",
            "max_tokens": 1024,
            "stream": true,
            "messages": [{"role": "user", "content": "Hello"}]
        })))
        .expect("prepare");

        assert!(plan.stream);
        assert_eq!(plan.chat_request.stream, Some(true));
    }

    #[test]
    fn a_tool_definition_survives_the_rewrite() {
        let plan = prepare_messages_request(&request(json!({
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }]
        })))
        .expect("prepare");

        assert_eq!(plan.chat_request.tools.len(), 1);
        let t = &plan.chat_request.tools[0];
        assert_eq!(t.tool_type, "function");
        assert_eq!(t.function.name, "get_weather");
        assert_eq!(t.function.description, "Get current weather");
        assert_eq!(t.function.parameters.param_type, "object");
        assert_eq!(t.function.parameters.required, vec!["location".to_string()]);
        assert!(t.function.parameters.property("location").is_some());
    }

    #[test]
    fn a_tool_result_turn_becomes_a_tool_role_message() {
        let plan = prepare_messages_request(&request(json!({
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call_123", "name": "get_weather",
                     "input": {"location": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_123", "content": "Sunny, 22C"}
                ]}
            ]
        })))
        .expect("prepare");

        let msgs = &plan.chat_request.messages;
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].tool_calls[0].id, "call_123");
        assert_eq!(msgs[2].role, "tool");
        assert_eq!(msgs[2].content, "Sunny, 22C");
        assert_eq!(msgs[2].tool_call_id, "call_123");
    }

    #[test]
    fn thinking_enabled_survives_the_rewrite() {
        let plan = prepare_messages_request(&request(json!({
            "model": "test-model",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 1000},
            "messages": [{"role": "user", "content": "Hello"}]
        })))
        .expect("prepare");

        assert_eq!(plan.chat_request.think, Some(ThinkValue::Bool(true)));
    }

    #[test]
    fn the_three_required_fields_are_refused_in_upstreams_order() {
        for (body, want) in [
            (
                json!({"max_tokens": 1024, "messages": [{"role": "user", "content": "Hi"}]}),
                "model is required",
            ),
            (
                json!({"model": "m", "messages": [{"role": "user", "content": "Hi"}]}),
                "max_tokens is required and must be positive",
            ),
            (json!({"model": "m", "max_tokens": 1024}), "messages is required"),
            // Both missing: model wins, because it is checked first.
            (json!({"messages": [{"role": "user", "content": "Hi"}]}), "model is required"),
        ] {
            let err = prepare_messages_request(&request(body)).unwrap_err();
            assert_eq!(err.status, 400);
            assert_eq!(err.message, want);
            assert_eq!(err.to_error_response("req_1").error.error_type, "invalid_request_error");
        }
    }

    #[test]
    fn a_negative_max_tokens_is_refused_not_passed_through_as_generate_forever() {
        let err = prepare_messages_request(&request(json!({
            "model": "m", "max_tokens": -1,
            "messages": [{"role": "user", "content": "Hi"}]
        })))
        .unwrap_err();
        assert_eq!(err.message, "max_tokens is required and must be positive");
    }

    #[test]
    fn a_conversion_failure_surfaces_as_a_four_hundred_with_the_converters_own_wording() {
        let err = prepare_messages_request(&request(json!({
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "assistant", "content": [{"type": "tool_use", "name": "test"}]}]
        })))
        .unwrap_err();

        assert_eq!(err.status, 400);
        assert_eq!(err.message, "tool_use block missing required 'id' field");
    }

    #[test]
    fn a_request_offering_a_web_search_tool_is_flagged_for_the_search_path() {
        let plan = prepare_messages_request(&request(json!({
            "model": "test-model",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}]
        })))
        .expect("prepare");
        assert!(plan.web_search);
    }

    #[test]
    fn has_web_search_tool_matches_on_the_type_prefix_only() {
        assert!(!has_web_search_tool(&[]));
        assert!(!has_web_search_tool(&[anthropic::Tool {
            tool_type: "custom".to_string(),
            name: "get_weather".to_string(),
            ..Default::default()
        }]));
        assert!(has_web_search_tool(&[anthropic::Tool {
            tool_type: "web_search_20250305".to_string(),
            name: "web_search".to_string(),
            ..Default::default()
        }]));
        // A custom tool merely NAMED web_search is not the built-in.
        assert!(!has_web_search_tool(&[anthropic::Tool {
            tool_type: "custom".to_string(),
            name: "web_search".to_string(),
            ..Default::default()
        }]));
        assert!(has_web_search_tool(&[
            anthropic::Tool {
                tool_type: "custom".to_string(),
                name: "get_weather".to_string(),
                ..Default::default()
            },
            anthropic::Tool {
                tool_type: "web_search_20250305".to_string(),
                name: "web_search".to_string(),
                ..Default::default()
            },
        ]));
    }

    // -- the cloud gate ------------------------------------------------------

    #[test]
    fn the_cloud_gate_only_refuses_a_cloud_model_while_cloud_is_disabled() {
        assert!(cloud_web_search_gate(false, false).is_ok());
        assert!(cloud_web_search_gate(false, true).is_ok(), "a local model is never gated");
        assert!(cloud_web_search_gate(true, false).is_ok());

        let err = cloud_web_search_gate(true, true).unwrap_err();
        assert_eq!(err.status, 403);
        assert_eq!(err.message, "ollama cloud is disabled: web search is unavailable");
        assert_eq!(err.to_error_response("req_1").error.error_type, "permission_error");
    }

    #[test]
    fn the_cloud_disabled_message_drops_the_colon_when_there_is_no_operation() {
        assert_eq!(cloud_disabled_error(""), "ollama cloud is disabled");
        assert_eq!(cloud_disabled_error("pull"), "ollama cloud is disabled: pull");
    }

    // -- AnthropicWriter -----------------------------------------------------

    fn chat_body(content: &str, prompt_eval: i32, eval: i32) -> Vec<u8> {
        serde_json::to_vec(&ChatResponse {
            model: "test-model".to_string(),
            message: Message {
                role: "assistant".into(),
                content: content.into(),
                ..Default::default()
            },
            done: true,
            done_reason: "stop".to_string(),
            metrics: Metrics {
                prompt_eval_count: prompt_eval,
                eval_count: eval,
                ..Default::default()
            },
            ..Default::default()
        })
        .expect("serialise")
    }

    #[test]
    fn a_buffered_two_hundred_becomes_one_anthropic_message() {
        let mut w = AnthropicWriter::new("msg_1", "test-model", false, 0);
        let out = w.write(200, &chat_body("Hello there!", 10, 5), "req_1").expect("write");

        assert_eq!(out.content_type(), "application/json");
        let WriterOutput::Json(resp) = out else { panic!("expected buffered JSON, got {out:?}") };
        assert_eq!(resp.response_type, "message");
        assert_eq!(resp.role, "assistant");
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.content[0].text.as_deref(), Some("Hello there!"));
        assert_eq!(resp.stop_reason, "end_turn");
        assert_eq!(resp.usage, Usage { input_tokens: 10, output_tokens: 5 });
    }

    #[test]
    fn a_streaming_two_hundred_becomes_sse_events() {
        let mut w = AnthropicWriter::new("msg_1", "test-model", true, 0);
        let out = w.write(200, &chat_body("Hi", 10, 5), "req_1").expect("write");

        assert_eq!(out.content_type(), "text/event-stream");
        let WriterOutput::Events(events) = out else { panic!("expected events, got {out:?}") };
        assert_eq!(
            events.iter().map(|e| e.event.as_str()).collect::<Vec<_>>(),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
    }

    #[test]
    fn a_non_two_hundred_is_reshaped_into_the_anthropic_error_envelope() {
        // The four shapes upstream's TestAnthropicWriter_ErrorFromRoutes cover.
        for (status, body, want_type, want_message) in [
            (
                404u16,
                json!({"error": "model 'nonexistent' not found"}),
                "not_found_error",
                "model 'nonexistent' not found",
            ),
            (400, json!({"error": "model is required"}), "invalid_request_error", "model is required"),
            (
                500,
                json!({"error": "something went wrong"}),
                "api_error",
                "something went wrong",
            ),
            (
                404,
                json!({"StatusCode": 404, "Status": "", "error": "model not found via StatusError"}),
                "not_found_error",
                "model not found via StatusError",
            ),
        ] {
            let mut w = AnthropicWriter::new("msg_1", "test-model", false, 0);
            let out = w
                .write(status, serde_json::to_vec(&body).expect("serialise").as_slice(), "req_9")
                .expect("write");

            assert_eq!(out.content_type(), "application/json");
            let WriterOutput::Error(err) = out else { panic!("expected an error, got {out:?}") };
            assert_eq!(err.response_type, "error");
            assert_eq!(err.error.error_type, want_type, "status {status}");
            assert_eq!(err.error.message, want_message);
            assert_eq!(err.request_id, "req_9");
        }
    }

    #[test]
    fn an_error_body_that_is_not_json_becomes_the_message_verbatim() {
        // Upstream's reason: "invalid character '<'" teaches a client nothing.
        assert_eq!(error_message_from_body(b"<html>502 Bad Gateway</html>"), "<html>502 Bad Gateway</html>");
        assert_eq!(error_message_from_body(b"{}"), "");
        assert_eq!(error_message_from_body(br#"{"error":"boom"}"#), "boom");
    }

    #[test]
    fn a_two_hundred_body_that_is_not_a_chat_response_is_a_hard_error() {
        let mut w = AnthropicWriter::new("msg_1", "test-model", false, 0);
        assert!(w.write(200, b"not json at all", "req_1").is_err());
    }

    // -- SSE framing ---------------------------------------------------------

    #[test]
    fn an_sse_frame_is_the_event_line_the_data_line_and_a_blank_line() {
        let frame = write_sse("message_stop", &json!({"type": "message_stop"})).expect("frame");
        assert_eq!(frame, "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    }

    #[test]
    fn a_newline_inside_the_payload_is_escaped_and_cannot_split_the_frame() {
        // If this ever stops holding, one event silently becomes two and every
        // client downstream desynchronises.
        let frame = write_sse("x", &json!({"text": "a\nb"})).expect("frame");
        assert_eq!(frame.matches("\n\n").count(), 1);
        assert!(frame.contains(r"a\nb"));
    }

    #[test]
    fn a_batch_of_events_frames_in_order() {
        let mut w = AnthropicWriter::new("msg_1", "test-model", true, 0);
        let WriterOutput::Events(events) =
            w.write(200, &chat_body("Hi", 1, 1), "req_1").expect("write")
        else {
            panic!("expected events");
        };
        let framed = sse_frames(&events).expect("frame");
        assert_eq!(framed.matches("event: ").count(), events.len());
        assert!(framed.starts_with("event: message_start\ndata: {"));
        assert!(framed.ends_with("\n\n"));
    }

    // -- web search helpers --------------------------------------------------

    #[test]
    fn a_server_tool_use_id_is_the_message_id_with_its_prefix_swapped() {
        assert_eq!(server_tool_use_id("msg_abc123"), "srvtoolu_abc123");
        assert_eq!(server_tool_use_id("msg_"), "srvtoolu_");
        // No msg_ prefix means nothing is trimmed -- TrimPrefix, not "cut".
        assert_eq!(server_tool_use_id("nomsgprefix"), "srvtoolu_nomsgprefix");
    }

    #[test]
    fn the_first_loop_gets_the_bare_server_tool_use_id_and_later_ones_a_suffix() {
        assert_eq!(loop_server_tool_use_id("msg_a", 0), "srvtoolu_a");
        assert_eq!(loop_server_tool_use_id("msg_a", 1), "srvtoolu_a");
        assert_eq!(loop_server_tool_use_id("msg_a", 2), "srvtoolu_a_2");
        assert_eq!(loop_server_tool_use_id("msg_a", 4), "srvtoolu_a_4");
    }

    #[test]
    fn a_query_is_pulled_out_of_a_tool_call_or_comes_back_empty() {
        assert_eq!(
            extract_query_from_tool_call(&call("web_search", &[("query", json!("test search"))])),
            "test search"
        );
        assert_eq!(extract_query_from_tool_call(&call("web_search", &[])), "");
        assert_eq!(
            extract_query_from_tool_call(&call("web_search", &[("other", json!("value"))])),
            ""
        );
        // A non-string query is a refusal, not a coercion.
        assert_eq!(
            extract_query_from_tool_call(&call("web_search", &[("query", json!(5))])),
            ""
        );
    }

    #[test]
    fn query_args_is_a_single_key_map() {
        let args = query_args("cats");
        assert_eq!(args.len(), 1);
        assert_eq!(args.get("query"), Some(&json!("cats")));
    }

    #[test]
    fn finding_the_web_search_call_prefers_the_first_and_reports_any_others() {
        let (found, others) = find_web_search_tool_call(&[]);
        assert!(found.is_none());
        assert!(!others);

        let ws = call("web_search", &[("query", json!("a"))]);
        let (found, others) = find_web_search_tool_call(std::slice::from_ref(&ws));
        assert_eq!(found.map(|c| c.id), Some(ws.id.clone()));
        assert!(!others);

        let other = call("get_weather", &[]);
        let (found, others) = find_web_search_tool_call(&[other.clone(), ws.clone()]);
        assert_eq!(found.map(|c| c.function.name), Some("web_search".to_string()));
        assert!(others, "the client tool call must be reported as present");

        // Two web_search calls: the FIRST wins, and it does not count as
        // "other tools".
        let mut second = call("web_search", &[("query", json!("b"))]);
        second.id = "call_second".to_string();
        let (found, others) = find_web_search_tool_call(&[ws, second]);
        assert_eq!(
            found.map(|c| extract_query_from_tool_call(&c)),
            Some("a".to_string())
        );
        assert!(!others);
    }

    #[test]
    fn the_rebuilt_assistant_turn_keeps_only_the_web_search_call() {
        let ws = call("web_search", &[("query", json!("a"))]);
        let response = ChatResponse {
            message: Message {
                role: "assistant".into(),
                content: "let me look".into(),
                thinking: "hmm".into(),
                tool_calls: vec![call("get_weather", &[]), ws.clone()],
                ..Default::default()
            },
            ..Default::default()
        };

        let msg = build_web_search_assistant_message(&response, &ws);
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, "let me look");
        assert_eq!(msg.thinking, "hmm");
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].function.name, "web_search");
    }

    #[test]
    fn search_hits_render_for_the_model_with_their_page_bodies_kept() {
        let text = format_web_search_results_for_tool_message(&[
            OllamaWebSearchResult {
                title: "A".to_string(),
                url: "http://a".to_string(),
                content: "body a".to_string(),
            },
            OllamaWebSearchResult {
                title: "B".to_string(),
                url: "http://b".to_string(),
                content: String::new(),
            },
        ]);
        assert_eq!(text, "Title: A\nURL: http://a\nContent: body a\n\nTitle: B\nURL: http://b\n\n");
        assert_eq!(format_web_search_results_for_tool_message(&[]), "");
    }

    #[test]
    fn a_web_search_error_response_is_a_normal_end_turn_message_not_an_error_envelope() {
        let resp = web_search_error_response(
            "msg_err001",
            "test-model",
            "unavailable",
            "test query",
            Usage { input_tokens: 3, output_tokens: 4 },
        );

        assert_eq!(resp.response_type, "message");
        assert_eq!(resp.role, "assistant");
        assert_eq!(resp.stop_reason, "end_turn");
        assert_eq!(resp.usage, Usage { input_tokens: 3, output_tokens: 4 });
        assert_eq!(resp.content.len(), 2);

        assert_eq!(resp.content[0].block_type, "server_tool_use");
        assert_eq!(resp.content[0].id, "srvtoolu_err001");
        assert_eq!(resp.content[0].name, "web_search");
        assert_eq!(
            resp.content[0].input.as_ref().and_then(|a| a.get("query")),
            Some(&json!("test query"))
        );

        assert_eq!(resp.content[1].block_type, "web_search_tool_result");
        assert_eq!(resp.content[1].tool_use_id, "srvtoolu_err001");
        assert_eq!(
            resp.content[1].content,
            Some(json!({"type": "web_search_tool_result_error", "error_code": "unavailable"}))
        );
    }

    // -- the search loop -----------------------------------------------------

    /// A scripted backend: a queue of search results and a queue of follow-up
    /// responses, each entry either a success or a failure.
    struct FakeBackend {
        searches: Vec<Result<OllamaWebSearchResponse, String>>,
        chats: Vec<Result<ChatResponse, String>>,
        seen_queries: Vec<String>,
        seen_chats: Vec<ChatRequest>,
    }

    impl FakeBackend {
        fn new() -> Self {
            FakeBackend {
                searches: Vec::new(),
                chats: Vec::new(),
                seen_queries: Vec::new(),
                seen_chats: Vec::new(),
            }
        }

        fn with_hit(mut self, title: &str, url: &str) -> Self {
            self.searches.push(Ok(OllamaWebSearchResponse {
                results: vec![OllamaWebSearchResult {
                    title: title.to_string(),
                    url: url.to_string(),
                    content: "body".to_string(),
                }],
            }));
            self
        }

        fn with_search_failure(mut self, msg: &str) -> Self {
            self.searches.push(Err(msg.to_string()));
            self
        }

        fn with_answer(mut self, text: &str, tool_calls: Vec<ToolCall>) -> Self {
            self.chats.push(Ok(ChatResponse {
                model: "test-model".to_string(),
                message: Message {
                    role: "assistant".into(),
                    content: text.into(),
                    tool_calls,
                    ..Default::default()
                },
                done: true,
                done_reason: "stop".to_string(),
                metrics: Metrics { prompt_eval_count: 1, eval_count: 2, ..Default::default() },
                ..Default::default()
            }));
            self
        }

        fn with_chat_failure(mut self, msg: &str) -> Self {
            self.chats.push(Err(msg.to_string()));
            self
        }
    }

    impl WebSearchBackend for FakeBackend {
        fn search(
            &mut self,
            query: &str,
            max_results: i32,
        ) -> Result<OllamaWebSearchResponse, String> {
            assert_eq!(max_results, anthropic::WEB_SEARCH_DEFAULT_MAX_RESULTS);
            self.seen_queries.push(query.to_string());
            if self.searches.is_empty() {
                return Err("the fake backend ran out of scripted searches".to_string());
            }
            self.searches.remove(0)
        }

        fn chat(&mut self, req: &ChatRequest) -> Result<ChatResponse, String> {
            assert_eq!(req.stream, Some(false), "the follow-up must be buffered");
            self.seen_chats.push(req.clone());
            if self.chats.is_empty() {
                return Err("the fake backend ran out of scripted answers".to_string());
            }
            self.chats.remove(0)
        }
    }

    fn loop_inputs() -> (ChatRequest, ChatResponse, ToolCall) {
        let ws = call("web_search", &[("query", json!("what is rust"))]);
        let chat_req = ChatRequest {
            model: "test-model".to_string(),
            messages: vec![Message::new("user", "what is rust")],
            stream: Some(true),
            ..Default::default()
        };
        let initial = ChatResponse {
            model: "test-model".to_string(),
            message: Message {
                role: "assistant".into(),
                tool_calls: vec![ws.clone()],
                ..Default::default()
            },
            done: true,
            metrics: Metrics { prompt_eval_count: 10, eval_count: 4, ..Default::default() },
            ..Default::default()
        };
        (chat_req, initial, ws)
    }

    #[test]
    fn one_search_round_stitches_the_server_blocks_in_front_of_the_answer() {
        let (chat_req, initial, ws) = loop_inputs();
        let mut backend =
            FakeBackend::new().with_hit("Rust", "https://rust-lang.org").with_answer("Rust is a language.", vec![]);

        let resp = run_web_search_loop(
            &mut backend,
            "msg_abc",
            "test-model",
            &chat_req,
            &initial,
            &ws,
            Usage { input_tokens: 10, output_tokens: 4 },
        )
        .expect("the loop");

        assert_eq!(
            resp.content.iter().map(|b| b.block_type.as_str()).collect::<Vec<_>>(),
            vec!["server_tool_use", "web_search_tool_result", "text"]
        );
        assert_eq!(resp.content[0].id, "srvtoolu_abc");
        assert_eq!(resp.content[1].tool_use_id, "srvtoolu_abc");
        assert_eq!(
            resp.content[1].content,
            Some(json!([{
                "type": "web_search_result",
                "url": "https://rust-lang.org",
                "title": "Rust"
            }]))
        );
        assert_eq!(resp.content[2].text.as_deref(), Some("Rust is a language."));
        assert_eq!(resp.stop_reason, "end_turn");
        // 10 + 1 and 4 + 2 -- the follow-up's tokens are ADDED, not replaced.
        assert_eq!(resp.usage, Usage { input_tokens: 11, output_tokens: 6 });

        // The follow-up history: the original user turn, the rebuilt assistant
        // turn, then the tool result.
        let sent = &backend.seen_chats[0];
        assert_eq!(
            sent.messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(),
            vec!["user", "assistant", "tool"]
        );
        assert_eq!(sent.messages[2].tool_call_id, ws.id);
        assert!(sent.messages[2].content.contains("Title: Rust"));
    }

    #[test]
    fn the_loop_goes_round_again_when_the_model_asks_to_search_again() {
        let (chat_req, initial, ws) = loop_inputs();
        let mut backend = FakeBackend::new()
            .with_hit("One", "https://one")
            .with_answer("", vec![call("web_search", &[("query", json!("second query"))])])
            .with_hit("Two", "https://two")
            .with_answer("Done.", vec![]);

        let resp = run_web_search_loop(
            &mut backend,
            "msg_abc",
            "test-model",
            &chat_req,
            &initial,
            &ws,
            Usage::default(),
        )
        .expect("the loop");

        assert_eq!(backend.seen_queries, vec!["what is rust", "second query"]);
        assert_eq!(
            resp.content.iter().map(|b| b.block_type.as_str()).collect::<Vec<_>>(),
            vec![
                "server_tool_use",
                "web_search_tool_result",
                "server_tool_use",
                "web_search_tool_result",
                "text"
            ]
        );
        // Round one is bare, round two is suffixed.
        assert_eq!(resp.content[0].id, "srvtoolu_abc");
        assert_eq!(resp.content[2].id, "srvtoolu_abc_2");
        assert_eq!(resp.usage, Usage { input_tokens: 2, output_tokens: 4 });
    }

    #[test]
    fn running_out_of_loops_ends_the_turn_with_max_uses_exceeded_rather_than_an_error() {
        let (chat_req, initial, ws) = loop_inputs();
        let mut backend = FakeBackend::new();
        for i in 0..MAX_WEB_SEARCH_LOOPS {
            backend = backend.with_hit("Hit", "https://hit").with_answer(
                "",
                vec![call("web_search", &[("query", json!(format!("q{i}")))])],
            );
        }

        let resp = run_web_search_loop(
            &mut backend,
            "msg_abc",
            "test-model",
            &chat_req,
            &initial,
            &ws,
            Usage::default(),
        )
        .expect("the loop must SUCCEED at the cap, not error");

        // Three real rounds, then the max_uses_exceeded pair.
        assert_eq!(resp.content.len(), (MAX_WEB_SEARCH_LOOPS as usize + 1) * 2);
        let last = resp.content.last().expect("a last block");
        assert_eq!(last.block_type, "web_search_tool_result");
        assert_eq!(
            last.content,
            Some(json!({
                "type": "web_search_tool_result_error",
                "error_code": "max_uses_exceeded"
            }))
        );
        // The final id uses MAX+1 so it cannot collide with a real round's.
        assert_eq!(resp.content[resp.content.len() - 2].id, "srvtoolu_abc_4");
        assert_eq!(resp.stop_reason, "end_turn");
    }

    #[test]
    fn an_empty_query_stops_the_loop_with_invalid_request() {
        let (chat_req, initial, _) = loop_inputs();
        let empty = call("web_search", &[]);
        let mut backend = FakeBackend::new();

        let err = run_web_search_loop(
            &mut backend,
            "msg_abc",
            "test-model",
            &chat_req,
            &initial,
            &empty,
            Usage { input_tokens: 7, output_tokens: 1 },
        )
        .unwrap_err();

        assert_eq!(err.code, "invalid_request");
        assert_eq!(err.query, "");
        // A failed turn still reports what it spent.
        assert_eq!(err.usage, Usage { input_tokens: 7, output_tokens: 1 });
        assert!(backend.seen_queries.is_empty(), "no search should have been attempted");
    }

    #[test]
    fn a_failed_search_stops_the_loop_with_unavailable_and_keeps_the_cause() {
        let (chat_req, initial, ws) = loop_inputs();
        let mut backend = FakeBackend::new().with_search_failure("connection refused");

        let err = run_web_search_loop(
            &mut backend,
            "msg_abc",
            "test-model",
            &chat_req,
            &initial,
            &ws,
            Usage::default(),
        )
        .unwrap_err();

        assert_eq!(err.code, "unavailable");
        assert_eq!(err.query, "what is rust");
        assert_eq!(err.cause.as_deref(), Some("connection refused"));
        assert_eq!(err.to_string(), "unavailable: connection refused");
    }

    #[test]
    fn a_failed_followup_generation_stops_the_loop_with_api_error() {
        let (chat_req, initial, ws) = loop_inputs();
        let mut backend =
            FakeBackend::new().with_hit("Hit", "https://hit").with_chat_failure("status 500");

        let err = run_web_search_loop(
            &mut backend,
            "msg_abc",
            "test-model",
            &chat_req,
            &initial,
            &ws,
            Usage::default(),
        )
        .unwrap_err();

        assert_eq!(err.code, "api_error");
        assert_eq!(err.cause.as_deref(), Some("status 500"));
    }

    #[test]
    fn a_followup_that_calls_a_client_tool_ends_the_loop_with_stop_reason_tool_use() {
        let (chat_req, initial, ws) = loop_inputs();
        let mut backend = FakeBackend::new()
            .with_hit("Hit", "https://hit")
            .with_answer("here", vec![call("get_weather", &[("city", json!("SG"))])]);

        let resp = run_web_search_loop(
            &mut backend,
            "msg_abc",
            "test-model",
            &chat_req,
            &initial,
            &ws,
            Usage::default(),
        )
        .expect("the loop");

        // A client tool call is NOT a web_search call, so the loop ends and the
        // stop reason comes from the ordinary conversion.
        assert_eq!(resp.stop_reason, "tool_use");
        assert_eq!(
            resp.content.iter().map(|b| b.block_type.as_str()).collect::<Vec<_>>(),
            vec!["server_tool_use", "web_search_tool_result", "text", "tool_use"]
        );
    }

    #[test]
    fn the_loop_error_renders_with_and_without_a_cause() {
        assert_eq!(
            WebSearchLoopError {
                code: "api_error".to_string(),
                query: String::new(),
                usage: Usage::default(),
                cause: None
            }
            .to_string(),
            "api_error"
        );
    }

    // -- WebSearchStreamState ------------------------------------------------

    fn final_response() -> MessagesResponse {
        MessagesResponse {
            id: "msg_test123".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            model: "test-model".to_string(),
            content: vec![
                ContentBlock {
                    block_type: "server_tool_use".to_string(),
                    id: "srvtoolu_test123".to_string(),
                    name: "web_search".to_string(),
                    input: Some(query_args("test query")),
                    ..Default::default()
                },
                ContentBlock {
                    block_type: "web_search_tool_result".to_string(),
                    tool_use_id: "srvtoolu_test123".to_string(),
                    content: Some(json!([{
                        "type": "web_search_result",
                        "url": "https://example.com",
                        "title": "Example"
                    }])),
                    ..Default::default()
                },
                ContentBlock::text("Here is the answer."),
            ],
            stop_reason: "end_turn".to_string(),
            stop_sequence: String::new(),
            usage: Usage { input_tokens: 20, output_tokens: 10 },
        }
    }

    fn names(events: &[StreamEvent]) -> Vec<&str> {
        events.iter().map(|e| e.event.as_str()).collect()
    }

    #[test]
    fn a_fresh_stream_terminal_emits_the_whole_message_from_message_start_to_stop() {
        // Upstream's TestWebSearchStreamResponse, event for event.
        let mut state = WebSearchStreamState::new();
        let Some(WriterOutput::Events(events)) =
            state.terminal(true, "test-model", 0, &final_response())
        else {
            panic!("expected streaming events");
        };

        assert_eq!(
            names(&events),
            vec![
                "message_start",
                "content_block_start", // server_tool_use, index 0
                "content_block_stop",  // index 0
                "content_block_start", // web_search_tool_result, index 1
                "content_block_stop",  // index 1
                "content_block_start", // text, index 2
                "content_block_delta", // text_delta
                "content_block_stop",  // index 2
                "message_delta",
                "message_stop",
            ]
        );

        let StreamEventData::MessageStart(ms) = &events[0].data else { panic!("message_start") };
        assert_eq!(ms.message.id, "msg_test123");
        assert_eq!(ms.message.role, "assistant");
        assert!(ms.message.content.is_empty());

        let StreamEventData::ContentBlockStart(tool) = &events[1].data else { panic!("start") };
        assert_eq!(tool.index, 0);
        assert_eq!(tool.content_block.block_type, "server_tool_use");
        assert_eq!(tool.content_block.id, "srvtoolu_test123");

        let StreamEventData::ContentBlockStart(search) = &events[3].data else { panic!("start") };
        assert_eq!(search.index, 1);
        assert_eq!(search.content_block.block_type, "web_search_tool_result");

        let StreamEventData::ContentBlockStart(text) = &events[5].data else { panic!("start") };
        assert_eq!(text.index, 2);
        assert_eq!(text.content_block.block_type, "text");
        // Empty on the start, real text on the delta.
        assert_eq!(text.content_block.text.as_deref(), Some(""));

        let StreamEventData::ContentBlockDelta(delta) = &events[6].data else { panic!("delta") };
        assert_eq!(delta.index, 2);
        assert_eq!(delta.delta.delta_type, "text_delta");
        assert_eq!(delta.delta.text, "Here is the answer.");

        let StreamEventData::MessageDelta(md) = &events[8].data else { panic!("message_delta") };
        assert_eq!(md.delta.stop_reason, "end_turn");
        assert_eq!(md.usage, DeltaUsage { input_tokens: 20, output_tokens: 10 });
    }

    #[test]
    fn a_buffered_terminal_is_just_the_message() {
        let mut state = WebSearchStreamState::new();
        let resp = final_response();
        let Some(WriterOutput::Json(got)) = state.terminal(false, "test-model", 0, &resp) else {
            panic!("expected buffered JSON");
        };
        assert_eq!(*got, resp);
        assert!(state.terminal_sent());
    }

    #[test]
    fn the_terminal_is_idempotent_so_a_second_message_stop_can_never_go_out() {
        let mut state = WebSearchStreamState::new();
        assert!(state.terminal(true, "m", 0, &final_response()).is_some());
        assert!(state.terminal(true, "m", 0, &final_response()).is_none());
    }

    #[test]
    fn the_terminal_carries_on_from_where_the_passthrough_left_the_stream() {
        // The passthrough opened block 0 and left it open. The terminal must
        // not re-send message_start, must close block 0, and must start its own
        // blocks at index 1.
        let mut state = WebSearchStreamState::new();
        let mut w = AnthropicWriter::new("msg_test123", "test-model", true, 0);
        let WriterOutput::Events(passthrough) = w.write_response(&ChatResponse {
            model: "test-model".to_string(),
            message: Message {
                role: "assistant".into(),
                content: "Looking...".into(),
                ..Default::default()
            },
            ..Default::default()
        }) else {
            panic!("expected events");
        };
        assert_eq!(
            names(&passthrough),
            vec!["message_start", "content_block_start", "content_block_delta"]
        );
        state.observe_passthrough(&passthrough);

        let mut response = final_response();
        response.content = vec![ContentBlock::text("Done.")];
        let Some(WriterOutput::Events(events)) = state.terminal(true, "test-model", 0, &response)
        else {
            panic!("expected events");
        };

        assert_eq!(
            names(&events),
            vec![
                "content_block_stop",  // closes the passthrough's block 0
                "content_block_start", // text, index 1
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let StreamEventData::ContentBlockStop(stop) = &events[0].data else { panic!("stop") };
        assert_eq!(stop.index, 0);
        let StreamEventData::ContentBlockStart(start) = &events[1].data else { panic!("start") };
        assert_eq!(start.index, 1);
    }

    #[test]
    fn a_passthrough_that_already_reached_message_stop_blocks_the_terminal() {
        let mut state = WebSearchStreamState::new();
        state.observe_passthrough(&[StreamEvent::new(
            "message_stop",
            StreamEventData::MessageStop(MessageStopEvent {
                event_type: "message_stop".to_string(),
            }),
        )]);
        assert!(state.terminal_sent());
        assert!(state.terminal(true, "m", 0, &final_response()).is_none());
    }

    #[test]
    fn the_estimate_only_fills_message_start_when_the_real_input_count_is_zero() {
        let mut state = WebSearchStreamState::new();
        let mut response = final_response();
        response.usage = Usage { input_tokens: 0, output_tokens: 0 };

        let Some(WriterOutput::Events(events)) = state.terminal(true, "test-model", 99, &response)
        else {
            panic!("expected events");
        };
        let StreamEventData::MessageStart(ms) = &events[0].data else { panic!("message_start") };
        assert_eq!(ms.message.usage, Usage { input_tokens: 99, output_tokens: 0 });
    }

    #[test]
    fn observed_usage_is_a_running_maximum_so_a_later_zero_cannot_wipe_it() {
        let mut state = WebSearchStreamState::new();
        state.record_observed_usage(&Metrics {
            prompt_eval_count: 10,
            eval_count: 4,
            ..Default::default()
        });
        state.record_observed_usage(&Metrics::default());
        assert_eq!(state.current_observed_usage(), Usage { input_tokens: 10, output_tokens: 4 });

        state.record_observed_usage(&Metrics {
            prompt_eval_count: 12,
            eval_count: 3,
            ..Default::default()
        });
        assert_eq!(state.current_observed_usage(), Usage { input_tokens: 12, output_tokens: 4 });
    }

    #[test]
    fn only_the_generation_that_kept_running_during_the_loop_is_added_to_the_usage() {
        let mut state = WebSearchStreamState::new();
        state.record_observed_usage(&Metrics {
            prompt_eval_count: 10,
            eval_count: 4,
            ..Default::default()
        });

        // The chunk that triggered the loop is the fresher one on output.
        let base = state.begin_loop(&Metrics {
            prompt_eval_count: 9,
            eval_count: 6,
            ..Default::default()
        });
        assert_eq!(base, Usage { input_tokens: 10, output_tokens: 6 });

        // The original generation kept going while the loop ran.
        state.record_observed_usage(&Metrics {
            prompt_eval_count: 10,
            eval_count: 9,
            ..Default::default()
        });

        let mut usage = Usage { input_tokens: 11, output_tokens: 8 };
        state.apply_observed_usage_delta(&mut usage);
        // input: no growth since the base, so nothing added. output: 9 - 6 = 3.
        assert_eq!(usage, Usage { input_tokens: 11, output_tokens: 11 });
    }

    #[test]
    fn the_usage_delta_never_goes_backwards() {
        let mut state = WebSearchStreamState::new();
        state.record_observed_usage(&Metrics {
            prompt_eval_count: 5,
            eval_count: 5,
            ..Default::default()
        });
        state.begin_loop(&Metrics { prompt_eval_count: 50, eval_count: 50, ..Default::default() });

        let mut usage = Usage { input_tokens: 50, output_tokens: 50 };
        state.apply_observed_usage_delta(&mut usage);
        assert_eq!(usage, Usage { input_tokens: 50, output_tokens: 50 });
    }
}
