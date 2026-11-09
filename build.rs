//! Header generation for the C API.
//!
//! When the `c_api` feature is enabled and `cbindgen` is available,
//! `include/simvec.h` is regenerated from `src/c_api` so the committed
//! header cannot drift from the Rust sources. Without the tool the build
//! keeps the committed header and stays green for offline environments.

use std::process::Command;

const HEADER: &str = "include/simvec.h";
const CONFIG: &str = "cbindgen.toml";

fn main() {
    println!("cargo:rerun-if-changed={CONFIG}");
    println!("cargo:rerun-if-changed=src/c_api.rs");
    println!("cargo:rerun-if-changed=src/c_api");

    if std::env::var_os("CARGO_FEATURE_C_API").is_none() {
        return;
    }

    let have_tool = Command::new("cbindgen")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false);
    if !have_tool {
        println!("cargo:warning=cbindgen not found; keeping committed {HEADER}");
        return;
    }

    let output = Command::new("cbindgen")
        .args(["-c", CONFIG, "-o", HEADER, "."])
        .output()
        .expect("failed to spawn cbindgen");
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stdout));
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        panic!("cbindgen failed while generating {HEADER}");
    }
}
