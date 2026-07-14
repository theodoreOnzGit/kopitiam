use anyhow::Result;

use crate::{CompletionRequest, CompletionResponse, ModelAdapter, Role};

/// A [`ModelAdapter`] that invokes no model at all: it echoes the last
/// [`Role::User`] message back as the response.
///
/// This exists so `kopitiam-workflow` (and its tests) can exercise the full
/// `load state -> collect facts -> build context -> invoke model -> validate
/// -> persist` pipeline deterministically, without a local Qwen build or
/// network access — useful in CI and while developing workflows before a
/// real adapter is wired in.
#[derive(Debug, Default, Clone, Copy)]
pub struct EchoAdapter;

impl ModelAdapter for EchoAdapter {
    fn name(&self) -> &str {
        "echo"
    }

    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let content = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        Ok(CompletionResponse { content, model: self.name().to_string() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    #[test]
    fn echoes_the_last_user_message() {
        let request = CompletionRequest::new([
            Message::system("you are a helpful assistant"),
            Message::user("first"),
            Message::assistant("ack"),
            Message::user("second"),
        ]);

        let response = EchoAdapter.complete(&request).unwrap();
        assert_eq!(response.content, "second");
        assert_eq!(response.model, "echo");
    }

    #[test]
    fn empty_when_there_is_no_user_message() {
        let request = CompletionRequest::new([Message::system("system only")]);
        let response = EchoAdapter.complete(&request).unwrap();
        assert_eq!(response.content, "");
    }
}
