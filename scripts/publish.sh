#!/usr/bin/env bash
set -euo pipefail

# Publish the simvec crate to crates.io. The crate root is resolved
# relative to this script so it can be invoked from any directory.
cd "$(dirname "$0")/.."

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# The package name comes from the manifest, not the directory name.
name=$(awk '
    /^\[package\]/ { in_pkg = 1; next }
    /^\[/ { in_pkg = 0 }
    in_pkg && /^name[[:space:]]*=/ {
        sub(/^[^"]*"/, ""); sub(/".*/, ""); print; exit
    }
' Cargo.toml)
version=$(sed -n 's/^version *= *"\(.*\)".*/\1/p' Cargo.toml | head -1)

echo "=== Publishing $name ==="
# Skip if this version is already on crates.io (User-Agent is required
# by the crates.io API).
if curl -sf -A "simvec-publish-script" -o /dev/null "https://crates.io/api/v1/crates/$name/$version"; then
    echo "Skipping $name v$version: already published."
    exit 0
fi

if $DRY_RUN; then
    cargo publish --dry-run --allow-dirty --registry crates-io
else
    cargo publish --allow-dirty --registry crates-io
fi

echo "Published $name v$version."
