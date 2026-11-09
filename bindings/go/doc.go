// Package simvec is a thin Go binding over the simvec C API.
//
// The package ships no native code. Build the Rust shared library first
// and keep the binding and the library from the same checkout. Complex
// values cross the boundary as JSON text reusing the shapes the Rust side
// already understands, while vectors cross as contiguous float32 slices.
//
// An Engine owns one native handle rooted at a directory. Engines are safe
// for concurrent use by multiple goroutines, except that Close must not run
// concurrently with in-flight calls on the same Engine. Close is idempotent,
// and a finalizer closes engines that the caller forgot to close.
package simvec
