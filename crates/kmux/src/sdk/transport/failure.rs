use std::io;

use crate::proto::Response;

use crate::sdk::RmuxError;

#[derive(Clone, Debug)]
pub(super) struct TransportFailure {
    kind: io::ErrorKind,
    message: String,
    protocol_error: Option<crate::proto::RmuxError>,
}

impl TransportFailure {
    pub(super) fn io(error: io::Error) -> Self {
        Self {
            kind: error.kind(),
            message: error.to_string(),
            protocol_error: None,
        }
    }

    pub(super) fn frame(error: crate::proto::RmuxError) -> Self {
        let message = error.to_string();
        Self {
            kind: io::ErrorKind::InvalidData,
            message,
            protocol_error: Some(error),
        }
    }

    pub(super) fn eof() -> Self {
        Self {
            kind: io::ErrorKind::UnexpectedEof,
            message: "rmux daemon closed the transport".to_owned(),
            protocol_error: None,
        }
    }

    pub(super) fn invalid_data(message: impl Into<String>) -> Self {
        Self {
            kind: io::ErrorKind::InvalidData,
            message: message.into(),
            protocol_error: None,
        }
    }

    pub(super) fn mismatched_response(expected: &'static str, actual: &'static str) -> Self {
        Self {
            kind: io::ErrorKind::InvalidData,
            message: format!(
                "rmux daemon sent `{actual}` response for pending `{expected}` request"
            ),
            protocol_error: None,
        }
    }

    pub(super) fn unsolicited_response(response: &Response) -> Self {
        Self {
            kind: io::ErrorKind::InvalidData,
            message: format!(
                "rmux daemon sent unsolicited `{}` response",
                response.command_name()
            ),
            protocol_error: None,
        }
    }

    pub(super) fn actor_closed() -> Self {
        Self {
            kind: io::ErrorKind::BrokenPipe,
            message: "rmux transport actor is closed".to_owned(),
            protocol_error: None,
        }
    }

    pub(super) const fn is_eof(&self) -> bool {
        matches!(self.kind, io::ErrorKind::UnexpectedEof)
    }

    pub(super) fn to_error(&self, operation: &str) -> RmuxError {
        RmuxError::transport(operation, io::Error::new(self.kind, self.message.clone()))
    }

    pub(super) fn to_error_for_command(
        &self,
        operation: &str,
        command_name: &'static str,
    ) -> RmuxError {
        if command_name == "handshake" {
            if let Some(error) = self.protocol_error.as_ref() {
                return handshake_protocol_error(error);
            }
        }

        self.to_error(operation)
    }
}

fn handshake_protocol_error(error: &crate::proto::RmuxError) -> RmuxError {
    match error {
        crate::proto::RmuxError::Decode(message) => RmuxError::unsupported(
            crate::sdk::diagnostics::FEATURE_PROTOCOL_CAPABILITIES,
            format!(
                "upgrade the rmux daemon to one that advertises `{}` before using SDK capability negotiation; {message}",
                crate::proto::CAPABILITY_HANDSHAKE
            ),
        ),
        error => RmuxError::protocol(error.clone()),
    }
}
