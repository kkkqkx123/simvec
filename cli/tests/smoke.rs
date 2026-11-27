use std::io::Write;
use std::process::{Command, Stdio};

use tempfile::tempdir;

fn invoke(root: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_simvec"))
        .arg(root)
        .arg(args.join(" "))
        .output()
        .expect("run simvec shell")
}

#[test]
fn one_shot_commands_cover_the_core_path() {
    let dir = tempdir().unwrap();
    let root = dir.path();

    let created = invoke(
        root,
        &[
            "create",
            "docs",
            "'{\"vector_size\":2,\"distance\":\"Cosine\",\"index\":\"Flat\"}'",
        ],
    );
    assert!(created.status.success(), "{}", String::from_utf8_lossy(&created.stderr));

    let upserted = invoke(
        root,
        &[
            "upsert",
            "docs",
            "'[{\"id\":\"a\",\"vector\":[1,0],\"payload\":{\"tag\":\"x\"}}]'",
        ],
    );
    assert!(
        upserted.status.success(),
        "{}",
        String::from_utf8_lossy(&upserted.stderr)
    );

    let count = invoke(root, &["count", "docs"]);
    assert!(count.status.success());
    assert_eq!(String::from_utf8_lossy(&count.stdout).trim(), "1");

    let searched = invoke(
        root,
        &["search", "docs", "'{\"vector\":[1,0],\"limit\":2}'"],
    );
    assert!(
        searched.status.success(),
        "{}",
        String::from_utf8_lossy(&searched.stderr)
    );
    let body = String::from_utf8_lossy(&searched.stdout);
    assert!(body.contains("\"id\":\"a\""), "{body}");

    let fetched = invoke(root, &["get", "docs", "a"]);
    assert!(fetched.status.success());
    assert!(String::from_utf8_lossy(&fetched.stdout).contains("\"tag\":\"x\""));
}

#[test]
fn batch_mode_reads_commands_from_stdin() {
    let dir = tempdir().unwrap();
    let script = concat!(
        "create docs '{\"vector_size\":2,\"distance\":\"Dot\",\"index\":\"Flat\"}'\n",
        ".mode pretty\n",
        "list\n",
        ".quit\n",
    );

    let mut child = Command::new(env!("CARGO_BIN_EXE_simvec"))
        .arg("--batch")
        .arg(dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("\"docs\""));
}

#[test]
fn unknown_command_is_reported_with_nonzero_exit() {
    let dir = tempdir().unwrap();
    let output = invoke(dir.path(), &["unknown"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown command"));
}
