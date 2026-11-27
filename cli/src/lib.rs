pub mod commands;
pub mod dto;
pub mod error;
pub mod lexer;
pub mod output;
pub mod shell;

use std::io::IsTerminal;

use crate::error::CliError;
use crate::shell::Shell;

const USAGE: &str = "usage: simvec [OPTIONS] <engine-dir> [command]\n\nOptions:\n  --batch          read commands from stdin without prompts\n  --interactive    force the interactive prompt\n  -h, --help       print help\n  -V, --version    print version\n";

/// Parse arguments and run the shell, returning the process exit code.
pub fn run(args: Vec<String>) -> i32 {
    match parse_args(args) {
        Ok(Invocation::Help) => {
            print!("{USAGE}");
            0
        }
        Ok(Invocation::Version) => {
            println!("simvec-cli {}", env!("CARGO_PKG_VERSION"));
            0
        }
        Ok(Invocation::Invalid(message)) => {
            eprintln!("{message}\n{USAGE}");
            2
        }
        Ok(Invocation::Run {
            path,
            command,
            batch,
            interactive,
        }) => match Shell::open(&path) {
            Err(err) => {
                eprintln!("Error: {err}");
                1
            }
            Ok(mut shell) => {
                shell.set_bail(batch || command.is_some() || !interactive && !std::io::stdin().is_terminal());
                match command {
                    Some(line) => shell.execute_line(&line),
                    None if interactive || std::io::stdin().is_terminal() => {
                        shell.run_interactive()
                    }
                    None => shell.run_batch(std::io::stdin().lock()),
                }
                shell.exit_code()
            }
        },
        Err(err) => {
            eprintln!("Error: {err}");
            2
        }
    }
}

enum Invocation {
    Help,
    Version,
    Invalid(String),
    Run {
        path: String,
        command: Option<String>,
        batch: bool,
        interactive: bool,
    },
}

fn parse_args(args: Vec<String>) -> std::result::Result<Invocation, CliError> {
    let mut path = None;
    let mut command = None;
    let mut batch = false;
    let mut interactive = false;
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Invocation::Help),
            "-V" | "--version" => return Ok(Invocation::Version),
            "--batch" => batch = true,
            "--interactive" => interactive = true,
            other if other.starts_with('-') => {
                return Ok(Invocation::Invalid(format!("unknown option: {other}")))
            }
            other if path.is_none() => path = Some(other.to_string()),
            other if command.is_none() => command = Some(other.to_string()),
            other => {
                return Ok(Invocation::Invalid(format!("unexpected argument: {other}")))
            }
        }
    }
    if batch && interactive {
        return Ok(Invocation::Invalid(
            "--batch and --interactive cannot be combined".to_string(),
        ));
    }
    let Some(path) = path else {
        return Ok(Invocation::Invalid("engine directory is required".to_string()));
    };
    Ok(Invocation::Run {
        path,
        command,
        batch,
        interactive,
    })
}
