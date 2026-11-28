//! Crash-window injection points for durability tests.
//!
//! A crash is only interesting at the boundaries where two pieces of durable
//! state disagree for a moment: after the log fsync but before the apply,
//! after the apply but before the ACK, after the tracker rename but before
//! the manifest advance. Unit tests cannot stop the process there, and a
//! timing-based `kill` is a flake generator, so the write paths call
//! [`fault_point`] at those boundaries and the test arms the point from the
//! child process's environment.
//!
//! `SIMVEC_FAULT=name[,name...]` arms points for the whole process. The list
//! is parsed once, on the first call, and an unarmed process pays a single
//! relaxed load per point — the same shape as the `lock-metrics` feature, and
//! nothing more than a predictable branch on a `OnceLock::get`.
//!
//! A hit aborts the process with [`std::process::abort`] rather than a clean
//! exit: a test must not be able to pass because the runtime ran destructors
//! that flushed the very state the window is about.
//!
//! Names are part of the test contract, so they are stable strings declared by
//! the caller (`fault_point("apply_before_ack")`). The table of live points is
//! in `docs/plan/storage-next-phase-stages.md` §2.2.

use std::sync::OnceLock;

/// Environment variable naming the points to arm, comma separated.
const FAULT_ENV: &str = "SIMVEC_FAULT";

/// Parsed `SIMVEC_FAULT` for this process.
///
/// `None` means "no injection in this process"; the outer `OnceLock` caches
/// that decision so the common path never touches the environment again.
static ARMED: OnceLock<Option<Vec<String>>> = OnceLock::new();

fn armed() -> Option<&'static [String]> {
    ARMED.get_or_init(|| {
        let raw = std::env::var(FAULT_ENV).ok()?;
        let points: Vec<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_ascii_lowercase)
            .collect();
        if points.is_empty() {
            None
        } else {
            tracing::warn!(points = ?points, "fault injection armed");
            Some(points)
        }
    })
    .as_deref()
}

/// Abort the process when `name` is armed in `SIMVEC_FAULT`.
///
/// Called from the write paths, so the unarmed path is one atomic load and a
/// branch. The wait is intentional: the point sits where a crash is legal, so
/// tests must not observe anything the process wrote after it.
#[inline]
pub(crate) fn fault_point(name: &'static str) {
    debug_assert!(
        POINTS.contains(&name),
        "fault point {name} is not in the documented table"
    );
    let Some(points) = armed() else {
        return;
    };
    if points.iter().any(|point| point == name) {
        tracing::error!(point = name, "fault injection: aborting at crash window");
        std::process::abort();
    }
}

/// Every point name this build can stop at, for the crash-matrix test to
/// enumerate instead of hard-coding the list twice.
pub(crate) const POINTS: &[&str] = &[
    "wal_after_append",
    "apply_before_ack",
    "wal_before_truncate",
    "checkpoint_after_flush",
    "checkpoint_after_tracker_publish",
    "compaction_after_publish",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unarmed_process_never_stops() {
        // Whatever the parent test binary did, this process has no
        // `SIMVEC_FAULT` unless a runner set one: assert the parser agrees
        // with the environment rather than assuming it.
        let expected = std::env::var(FAULT_ENV)
            .ok()
            .map(|raw| !raw.trim().is_empty())
            .unwrap_or(false);
        assert_eq!(armed().is_some(), expected);
    }

    #[test]
    fn the_point_table_has_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for point in POINTS {
            assert!(seen.insert(*point), "duplicate fault point {point}");
        }
    }
}
