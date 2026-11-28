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
fn snapshot_commands_round_trip() {
    let dir = tempdir().unwrap();
    // Snapshots must not live inside the engine root: a directory carrying a
    // `generation.bin` is opened as a collection on the next engine open.
    let root = dir.path().join("engine");
    let snap = dir.path().join("snapshot");
    let restored = root.join("restored");

    let created = invoke(
        &root,
        &[
            "create",
            "docs",
            "'{\"vector_size\":2,\"distance\":\"Dot\",\"index\":\"Flat\"}'",
        ],
    );
    assert!(
        created.status.success(),
        "{}",
        String::from_utf8_lossy(&created.stderr)
    );
    let upserted = invoke(
        &root,
        &["upsert", "docs", "'[{\"id\":\"a\",\"vector\":[1,0]}]'"],
    );
    assert!(
        upserted.status.success(),
        "{}",
        String::from_utf8_lossy(&upserted.stderr)
    );

    let status = invoke(&root, &["maintenance_status", "docs"]);
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let body = String::from_utf8_lossy(&status.stdout);
    assert!(body.contains("\"healthy\":true"), "{body}");
    assert!(body.contains("\"segments\""), "{body}");

    let snap_arg = snap.to_str().unwrap();
    let captured = invoke(&root, &["snapshot", "docs", snap_arg]);
    assert!(
        captured.status.success(),
        "{}",
        String::from_utf8_lossy(&captured.stderr)
    );
    assert!(String::from_utf8_lossy(&captured.stdout).contains("\"generation\""));

    let verified = invoke(&root, &["verify_snapshot", snap_arg]);
    assert!(
        verified.status.success(),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    assert!(String::from_utf8_lossy(&verified.stdout).contains("\"files\""));

    // A second snapshot reuses the unchanged entries of the first one.
    let incremental = dir.path().join("snapshot_incr");
    let incr = invoke(
        &root,
        &[
            "snapshot_incremental",
            "docs",
            snap_arg,
            incremental.to_str().unwrap(),
        ],
    );
    assert!(
        incr.status.success(),
        "{}",
        String::from_utf8_lossy(&incr.stderr)
    );
    assert!(String::from_utf8_lossy(&incr.stdout).contains("\"reused_files\":"));
    assert!(std::fs::read_dir(&incremental).unwrap().count() > 0);

    let restored_out = invoke(
        &root,
        &["restore_snapshot", snap_arg, restored.to_str().unwrap()],
    );
    assert!(
        restored_out.status.success(),
        "{}",
        String::from_utf8_lossy(&restored_out.stderr)
    );
    // A fresh process discovers the restored collection at open.
    let counted = invoke(&root, &["count", "restored"]);
    assert!(counted.status.success());
    assert_eq!(String::from_utf8_lossy(&counted.stdout).trim(), "1");
}

#[test]
fn unknown_command_is_reported_with_nonzero_exit() {
    let dir = tempdir().unwrap();
    let output = invoke(dir.path(), &["unknown"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown command"));
}
