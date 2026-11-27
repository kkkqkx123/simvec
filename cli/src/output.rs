use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use serde::Serialize;

use crate::error::{CliError, Result};

/// JSON rendering style for command results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Json,
    Pretty,
}

enum Destination {
    Stdout,
    File(File),
}

/// Result writer owned by the shell. Errors always stay on stderr and never
/// follow an `.output` redirection.
pub struct Output {
    mode: OutputMode,
    destination: Destination,
}

impl Output {
    pub fn new() -> Self {
        Self {
            mode: OutputMode::Json,
            destination: Destination::Stdout,
        }
    }

    pub fn set_mode(&mut self, mode: OutputMode) {
        self.mode = mode;
    }

    pub fn redirect(&mut self, path: &Path) -> Result<()> {
        self.destination = Destination::File(File::create(path)?);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.destination = Destination::Stdout;
    }

    pub fn print_json<T: Serialize>(&mut self, value: &T) -> Result<()> {
        let body = match self.mode {
            OutputMode::Json => serde_json::to_string(value),
            OutputMode::Pretty => serde_json::to_string_pretty(value),
        }
        .map_err(|err| CliError::new(format!("failed to serialize result: {err}")))?;
        self.write_line(&body)
    }

    pub fn print_line(&mut self, line: &str) -> Result<()> {
        self.write_line(line)
    }

    pub fn print_u64(&mut self, value: u64) -> Result<()> {
        self.write_line(&value.to_string())
    }

    pub fn print_bool(&mut self, value: bool) -> Result<()> {
        self.write_line(if value { "1" } else { "0" })
    }

    fn write_line(&mut self, line: &str) -> Result<()> {
        match &mut self.destination {
            Destination::Stdout => {
                let mut stdout = io::stdout().lock();
                writeln!(stdout, "{line}")?;
                stdout.flush()?;
            }
            Destination::File(file) => {
                writeln!(file, "{line}")?;
                file.flush()?;
            }
        }
        Ok(())
    }
}
