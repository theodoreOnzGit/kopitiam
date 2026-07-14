use serde::{Deserialize, Serialize};

/// Who authored a [`Message`] in a [`CompletionRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Instructions/context assembled by `kopitiam-workflow`'s context
    /// builder — never hand-typed by a human in this crate.
    System,
    /// The task or question being posed to the model.
    User,
    /// A prior model response, present when a workflow is multi-turn.
    Assistant,
}

/// One turn in a [`CompletionRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// A request to a [`crate::ModelAdapter`].
///
/// This is deliberately shaped like a chat completion, since every adapter
/// KOPITIAM plans to support (local Qwen, Claude, GPT, Gemini) speaks that
/// shape natively. `messages` is expected to already contain whatever
/// context `kopitiam-workflow`'s `ContextBuilder` assembled — adapters must
/// not go looking for additional context themselves.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    /// Upper bound on generated tokens, if the caller wants one. `None`
    /// leaves the limit to the adapter/model's own default.
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

impl CompletionRequest {
    pub fn new(messages: impl IntoIterator<Item = Message>) -> Self {
        Self { messages: messages.into_iter().collect(), max_tokens: None }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

/// A [`crate::ModelAdapter`]'s reply to a [`CompletionRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    /// Identifies which model actually produced `content` (e.g.
    /// `"qwen2.5-coder-7b"`, `"claude-sonnet-5"`), independent of the
    /// adapter's own [`crate::ModelAdapter::name`] — an adapter may serve
    /// more than one model.
    pub model: String,
}
