use std::fmt;

use simvec::VectorSearchError;

/// Error surfaced by the command shell.
#[derive(Debug)]
pub struct CliError {
    message: String,
}

impl CliError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn json(command: &str, err: serde_json::Error) -> Self {
        Self::new(format!("{command}: invalid JSON: {err}"))
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CliError {}

impl From<VectorSearchError> for CliError {
    fn from(err: VectorSearchError) -> Self {
        Self::new(err.to_string())
    }
}

impl From<std::io::Error> for CliError {
    fn from(err: std::io::Error) -> Self {
        Self::new(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, CliError>;
