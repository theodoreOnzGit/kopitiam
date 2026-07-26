use crate::core::command_parser::CommandParser;
use crate::proto::OptionName;

use crate::server::pane_terminals::HandlerState;

pub(in crate::server::handler) fn command_parser_from_state(state: &HandlerState) -> CommandParser {
    CommandParser::new()
        .with_environment_store(&state.environment)
        .with_command_aliases(
            state
                .options
                .resolve_array_values(None, OptionName::CommandAlias),
        )
}
