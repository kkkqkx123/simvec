//! Allocation-counting report for build and search phases.
//!
//! Installs a process-local counting global allocator, then runs one ingest
//! batch phase, one HNSW index build and two query sets (unfiltered and
//! highly-selective filtered) against the local engine, printing per-phase
//! allocation counts and byte totals. Report-only: no assertions. This is
//! the "customize alloc hook" measurement route — no external dependency, zero
//! effect on library builds because the allocator lives in this binary.
//!
//! Run: `cargo bench -p simvec --bench alloc_stats_bench`
//!
//! Run convention: default is the full 10k-point run; `QUICK=1` shrinks to
//! 1k points / 20 queries for a seconds-long smoke run.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use rand::{Rng, SeedableRng};

/// Counting wrapper around the system allocator; overhead is two relaxed
/// atomic adds per allocation, fine for a dedicated reporting binary.
struct CountingAllocator;

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Default)]
struct Snapshot {
    alloc_count: u64,
    alloc_bytes: u64,
    dealloc_count: u64,
    dealloc_bytes: u64,
}

fn snapshot() -> Snapshot {
    Snapshot {
        alloc_count: ALLOC_COUNT.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        dealloc_count: DEALLOC_COUNT.load(Ordering::Relaxed),
        dealloc_bytes: DEALLOC_BYTES.load(Ordering::Relaxed),
    }
}

fn delta_since(base: Snapshot) -> Snapshot {
    let now = snapshot();
    Snapshot {
        alloc_count: now.alloc_count - base.alloc_count,
        alloc_bytes: now.alloc_bytes - base.alloc_bytes,
        dealloc_count: now.dealloc_count - base.dealloc_count,
        dealloc_bytes: now.dealloc_bytes - base.dealloc_bytes,
    }
}

const DIM: usize = 64;
const POINTS: usize = 10_000;
const QUERIES: usize = 200;

fn main() {
    let t_started = std::time::Instant::now();
    // Warm-up: engine open + first-touch pages, excluded from every phase.
    let dir = tempfile::tempdir().unwrap();
    let engine = simvec::LocalVectorEngine::open(dir.path().join("vec")).unwrap();
    engine
        .create_collection(
            "col",
            &simvec::CollectionConfig::new(DIM, simvec::DistanceMetric::Euclid)
                .with_index_type(simvec::IndexType::HNSW),
        )
        .unwrap();
    eprintln!("[phase] setup: {:?}", t_started.elapsed());

    let mut rng = rand::rngs::StdRng::seed_from_u64(7);

    // Fast smoke gears: full run keeps the issue's 10k-point scale, QUICK=1
    // drops to 1k points so the report finishes in seconds.
    let (points, n_queries) = if std::env::var("QUICK").is_ok() {
        (1_000, 20)
    } else {
        (POINTS, QUERIES)
    };

    // Phase 1: batched ingest.
    let base = snapshot();
    let t_phase = std::time::Instant::now();
    for chunk in (0..points).step_by(1_000) {
        let batch: Vec<simvec::VectorPoint> = (chunk..(chunk + 1_000).min(points))
            .map(|i| {
                simvec::VectorPoint::new(
                    i as u64,
                    (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect(),
                )
            })
            .collect();
        engine.upsert_batch("col", &batch).unwrap();
    }
    eprintln!("[phase] ingest: {:?}", t_phase.elapsed());
    let ingest = delta_since(base);

    // Phase 2: HNSW graph build.
    let base = snapshot();
    let t_phase = std::time::Instant::now();
    assert!(engine.build_index("col").unwrap());
    eprintln!("[phase] build: {:?}", t_phase.elapsed());
    let build = delta_since(base);

    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    // Phase 3: unfiltered ANN queries.
    let base = snapshot();
    let t_phase = std::time::Instant::now();
    for q in &queries {
        let _ = engine
            .search(
                "col",
                &simvec::SearchQuery::new(q.clone(), 10).with_knn(10, Some(40)),
            )
            .unwrap();
    }
    eprintln!("[phase] search unfiltered: {:?}", t_phase.elapsed());
    let unfiltered = delta_since(base);

    // Phase 4: highly-selective filtered queries (worst case: retry chain).
    let base = snapshot();
    let t_phase = std::time::Instant::now();
    let filter =
        simvec::VectorFilter::new().must(simvec::FilterCondition::match_value("tag", "missing"));
    for (i, q) in queries.iter().enumerate() {
        let _ = engine
            .search(
                "col",
                &simvec::SearchQuery::new(q.clone(), 10)
                    .with_filter(filter.clone())
                    .with_knn(10, Some(40)),
            )
            .unwrap();
        if (i + 1) % 100 == 0 {
            eprintln!(
                "[phase] filtered {}/{}: {:?}",
                i + 1,
                QUERIES,
                t_phase.elapsed()
            );
        }
    }
    eprintln!("[phase] search filtered: {:?}", t_phase.elapsed());
    let filtered = delta_since(base);

    println!(
        "\n=== allocation stats ({} points, dim {}, {} queries) ===",
        points, DIM, n_queries
    );
    println!(
        "{:<28}{:>14}{:>16}{:>14}{:>16}",
        "phase", "allocs", "alloc bytes", "frees", "free bytes"
    );
    for (name, s) in [
        ("ingest (batched upsert)", ingest),
        ("hnsw build", build),
        ("search xN unfiltered", unfiltered),
        ("search xN filtered-miss", filtered),
    ] {
        println!(
            "{:<28}{:>14}{:>16}{:>14}{:>16}",
            name, s.alloc_count, s.alloc_bytes, s.dealloc_count, s.dealloc_bytes
        );
    }
}
