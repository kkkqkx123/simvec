use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use simvec::LocalVectorEngine;

use crate::commands;
use crate::error::{CliError, Result};
use crate::lexer::{self, Split};
use crate::output::{Output, OutputMode};

const DOT_COMMANDS: &[(&str, &str)] = &[
    (".open <path>", "open an engine root directory"),
    (".close", "close the current engine"),
    (".quit", "exit the shell"),
    (".help [command]", "print command help"),
    (".tables", "list loaded collections"),
    (".schema <collection>", "print collection configuration"),
    (".info <collection>", "print collection information"),
    (".output <path>", "redirect results to a file"),
    (".stdout", "restore results to stdout"),
    (".read <path>", "execute commands from a file"),
    (".echo on|off", "echo commands before execution"),
    (".bail on|off", "stop scripts on the first error"),
    (".timer on|off", "time engine commands on stderr"),
    (".mode json|pretty", "select JSON rendering"),
    (".version", "print the shell version"),
];

/// Interactive and batch command processor.
pub struct Shell {
    engine: Option<LocalVectorEngine>,
    output: Output,
    pending: String,
    bail: bool,
    echo: bool,
    timer: bool,
    quit: bool,
    had_error: bool,
}

impl Shell {
    pub fn open(path: &str) -> Result<Self> {
        let engine = LocalVectorEngine::open(path)?;
        Ok(Self {
            engine: Some(engine),
            output: Output::new(),
            pending: String::new(),
            bail: false,
            echo: false,
            timer: false,
            quit: false,
            had_error: false,
        })
    }

    pub fn set_bail(&mut self, bail: bool) {
        self.bail = bail;
    }

    pub fn exit_code(&self) -> i32 {
        i32::from(self.had_error)
    }

    pub fn run_batch<R: BufRead>(&mut self, reader: R) {
        for line in reader.lines() {
            match line {
                Ok(line) => self.execute_line(&line),
                Err(err) => self.report_error(err.into()),
            }
            if self.quit {
                break;
            }
        }
    }

    pub fn run_interactive(&mut self) {
        let mut line = String::new();
        loop {
            let prompt = if self.pending.is_empty() {
                "simvec> "
            } else {
                "...> "
            };
            print!("{prompt}");
            let _ = io::stdout().flush();
            line.clear();
            match io::stdin().read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => self.execute_line(&line),
                Err(err) => {
                    self.report_error(err.into());
                    break;
                }
            }
            if self.quit {
                break;
            }
        }
    }

    /// Execute one physical line, joining continuation lines when a quote is
    /// still open. Errors are reported once and recorded in the exit status.
    pub fn execute_line(&mut self, line: &str) {
        if self.pending.is_empty() {
            let trimmed = line.trim_start();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                return;
            }
        }

        let candidate = if self.pending.is_empty() {
            line.to_string()
        } else {
            format!("{}\n{}", self.pending, line)
        };
        match lexer::split(&candidate) {
            Ok(Split::Ready(argv)) => {
                self.pending.clear();
                if !argv.is_empty() {
                    let _ = self.execute_argv(&argv);
                }
            }
            Ok(Split::Continuation) => {
                self.pending = candidate;
            }
            Err(err) => {
                self.pending.clear();
                self.report_error(CliError::new(err));
            }
        }
    }

    fn execute_argv(&mut self, argv: &[String]) -> Result<()> {
        if self.echo {
            eprintln!("+ {}", argv.join(" "));
        }
        let result = if argv[0].starts_with('.') {
            self.execute_dot(argv)
        } else {
            let engine = self
                .engine
                .as_ref()
                .ok_or_else(|| CliError::new("no engine is open; use .open <path>"))?;
            let started = self.timer.then(Instant::now);
            let result = commands::execute(engine, &mut self.output, argv);
            if let (Some(started), Ok(())) = (started, &result) {
                eprintln!("run time: {:?}", started.elapsed());
            }
            result
        };
        if let Err(err) = result {
            self.report_error(err);
        }
        Ok(())
    }

    fn execute_dot(&mut self, argv: &[String]) -> Result<()> {
        match argv[0].as_str() {
            ".open" => {
                self.exact(argv, 2, ".open <path>")?;
                let engine = LocalVectorEngine::open(&argv[1])?;
                self.engine = Some(engine);
                Ok(())
            }
            ".close" => {
                self.exact(argv, 1, ".close")?;
                self.engine = None;
                Ok(())
            }
            ".quit" | ".exit" => {
                self.exact(argv, 1, ".quit")?;
                self.quit = true;
                Ok(())
            }
            ".help" => {
                if argv.len() > 2 {
                    return Err(CliError::new("usage: .help [command]"));
                }
                let target = argv.get(1).map(String::as_str);
                match target {
                    Some(name) if name.starts_with('.') => Self::print_dot_help(Some(name)),
                    other => commands::print_help(other),
                }
                Ok(())
            }
            ".tables" => {
                self.exact(argv, 1, ".tables")?;
                let names = self.engine()?.collection_names();
                self.output.print_json(&names)
            }
            ".schema" => {
                self.exact(argv, 2, ".schema <collection>")?;
                let args = vec!["config".to_string(), argv[1].clone()];
                self.execute_engine(&args)
            }
            ".info" => {
                self.exact(argv, 2, ".info <collection>")?;
                let args = vec!["info".to_string(), argv[1].clone()];
                self.execute_engine(&args)
            }
            ".output" => {
                self.exact(argv, 2, ".output <path>")?;
                self.output.redirect(Path::new(&argv[1]))
            }
            ".stdout" => {
                self.exact(argv, 1, ".stdout")?;
                self.output.reset();
                Ok(())
            }
            ".read" => {
                self.exact(argv, 2, ".read <path>")?;
                let file = File::open(&argv[1])?;
                self.run_batch(BufReader::new(file));
                Ok(())
            }
            ".echo" => {
                self.exact(argv, 2, ".echo on|off")?;
                self.echo = parse_flag(&argv[1], ".echo")?;
                Ok(())
            }
            ".bail" => {
                self.exact(argv, 2, ".bail on|off")?;
                self.bail = parse_flag(&argv[1], ".bail")?;
                Ok(())
            }
            ".timer" => {
                self.exact(argv, 2, ".timer on|off")?;
                self.timer = parse_flag(&argv[1], ".timer")?;
                Ok(())
            }
            ".mode" => {
                self.exact(argv, 2, ".mode json|pretty")?;
                match argv[1].as_str() {
                    "json" => self.output.set_mode(OutputMode::Json),
                    "pretty" => self.output.set_mode(OutputMode::Pretty),
                    other => return Err(CliError::new(format!("unknown mode: {other}"))),
                }
                Ok(())
            }
            ".version" => {
                self.exact(argv, 1, ".version")?;
                eprintln!("simvec-cli {}", env!("CARGO_PKG_VERSION"));
                Ok(())
            }
            other => Err(CliError::new(format!(
                "unknown command: {other}; run .help for a command list"
            ))),
        }
    }

    fn execute_engine(&mut self, argv: &[String]) -> Result<()> {
        let engine = self
            .engine
            .as_ref()
            .ok_or_else(|| CliError::new("no engine is open; use .open <path>"))?;
        commands::execute(engine, &mut self.output, argv)
    }

    fn engine(&self) -> Result<&LocalVectorEngine> {
        self.engine
            .as_ref()
            .ok_or_else(|| CliError::new("no engine is open; use .open <path>"))
    }

    fn exact(&self, argv: &[String], expected: usize, usage: &str) -> Result<()> {
        if argv.len() == expected {
            Ok(())
        } else {
            Err(CliError::new(format!("usage: {usage}")))
        }
    }

    fn print_dot_help(command: Option<&str>) {
        match command {
            Some(name) => {
                if let Some((usage, summary)) = DOT_COMMANDS
                    .iter()
                    .find(|(usage, _)| usage.split_whitespace().next() == Some(name))
                {
                    eprintln!("{usage}\n  {summary}");
                } else {
                    eprintln!("unknown command: {name}");
                }
            }
            None => {
                eprintln!("shell commands:");
                for (usage, summary) in DOT_COMMANDS {
                    eprintln!("  {usage}\n      {summary}");
                }
            }
        }
    }

    fn report_error(&mut self, err: CliError) {
        eprintln!("Error: {err}");
        self.had_error = true;
        if self.bail {
            self.quit = true;
        }
    }
}

fn parse_flag(value: &str, command: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(CliError::new(format!(
            "{command}: expected on/off or 1/0, got {value:?}"
        ))),
    }
}
