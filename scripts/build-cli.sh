#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
exec cargo build --release --manifest-path cli/Cargo.toml --target-dir target "$@"
