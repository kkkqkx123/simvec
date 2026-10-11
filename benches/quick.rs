//! Shared run convention for the simvec benchmark binaries.
//!
//! Default is the full run at promotion scale; setting `QUICK=1` shrinks the
//! corpus and shortens the criterion windows for a seconds-long smoke run.
//! Numbers from different gears are never comparable to each other, and
//! numbers are only comparable on the same machine (see the bench baseline
//! record for the machine fingerprint).
//!
//! Each benchmark binary pulls this in with one line:
//! `#[path = "quick.rs"] mod quick;`
//! It is not a bench target itself (`autobenches = false` in Cargo.toml).

use criterion::Criterion;
use std::time::Duration;

/// Whether the run requests the fast smoke gear (`QUICK` set in the environment).
pub fn quick_mode() -> bool {
    std::env::var("QUICK").is_ok()
}

/// Pick the full-scale value or the smoke-scale value.
pub fn scale(full: usize, quick: usize) -> usize {
    if quick_mode() {
        quick
    } else {
        full
    }
}

/// Open a criterion group, shortening the windows for the smoke gear.
///
/// The full gear keeps the criterion defaults untouched so existing baselines
/// stay valid; the smoke gear uses short windows because each sample is cheap.
pub fn benchmark_group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    if quick_mode() {
        group.measurement_time(Duration::from_millis(800));
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
    }
    group
}
