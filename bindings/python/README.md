# simvec Python binding (ctypes over libsimvec)

Pure-Python wrapper around the simvec C API. The package ships no native
code: build the Rust dynamic library first and point the wrapper at it.

```sh
cargo build --features c_api
SIMVEC_LIB=/home/kkkqkx/code/simvec/target/debug/libsimvec.so python -m pytest tests -q
```

Configs, filters, payloads and queries are plain dicts serialized to the
JSON shapes the Rust side already understands. Vectors accept any
sequence of floats and cross the boundary as contiguous float32 arrays.

Bundling the shared library into wheels is follow-up distribution work
and intentionally out of scope for this binding revision.
