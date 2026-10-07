//! Persist CRC32 overhead baseline.
//!
//! Quantifies `crc32fast::hash` cost inside `index/persist.rs::write_tagged`
//! relative to the full `save`/`save_hnsw` path (postcard serialization +
//! CRC + `File::create`/`write_all`/`sync_all`). Ratio must stay `< 10%`:
//! payloads are multi-MB memory-linear scans while the file write dominates
//! on a tmp+rename path.
//!
//! Payload shapes for the benchmark:
//! - `index.bin` (IVF): lists=256, dim=128, live=100K, ~500KB payload
//! - `hnsw.bin`: dim=128, m=16, live=100K, ~12–15MB payload (`m*2*live` edges)
//!
//! Run: `cargo bench -p simvec --bench persist_crc_bench -- --warm-up-time 1 --measurement-time 3`

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

const IVF_MAGIC: [u8; 4] = *b"VIVF";
const HNSW_MAGIC: [u8; 4] = *b"VHSW";
const VERSION: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DummyIvf {
    lists: u32,
    dim: usize,
    distance: u8,
    built_at_live_count: u64,
    baseline_mean_dist: f32,
    centroids: Vec<Vec<f32>>,
    slot_list: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DummyHnswNode {
    slot: u32,
    level: u8,
    version: u8,
    neighbors: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DummyHnsw {
    dim: usize,
    distance: u8,
    m: usize,
    ef_construct: usize,
    ef_search: usize,
    entry: Option<(u32, i32)>,
    built_at_live_count: u64,
    nodes: Vec<DummyHnswNode>,
}

fn write_tagged<T: Serialize>(
    path: &Path,
    magic: &[u8; 4],
    version: u16,
    data: &T,
) -> std::io::Result<()> {
    let bytes = postcard::to_stdvec(data).expect("postcard");
    let crc = crc32fast::hash(&bytes);
    let mut file = File::create(path)?;
    file.write_all(magic)?;
    file.write_all(&version.to_le_bytes())?;
    file.write_all(&crc.to_le_bytes())?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn make_ivf(lists: u32, dim: usize, live: usize) -> DummyIvf {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE01);
    let centroids: Vec<Vec<f32>> = (0..lists)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let slot_list: Vec<u32> = (0..live)
        .map(|_| {
            if rng.gen_bool(0.05) {
                u32::MAX
            } else {
                rng.gen_range(0..lists)
            }
        })
        .collect();
    DummyIvf {
        lists,
        dim,
        distance: 0,
        built_at_live_count: live as u64,
        baseline_mean_dist: 0.5,
        centroids,
        slot_list,
    }
}

fn hnsw_level(rng: &mut StdRng, m: usize) -> u8 {
    // Geometric with p = 1 / m (pgvector / HNSW paper). Cap at 6 to avoid outliers.
    let ml = 1.0 / (m as f64).ln();
    let r: f64 = rng.gen();
    let lvl = (-r.ln() * ml).floor() as i32;
    lvl.clamp(0, 6) as u8
}

fn make_hnsw(live: usize, dim: usize, m: usize) -> DummyHnsw {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE02);
    let mut nodes = Vec::with_capacity(live);
    for slot in 0..live as u32 {
        let level = hnsw_level(&mut rng, m);
        let mut neighbors = Vec::with_capacity(level as usize + 1);
        for lc in 0..=level {
            let cap = if lc == 0 { m * 2 } else { m };
            // Average fill ~ cap/2 to emulate real graph sparsity while keeping payload large.
            let len = rng.gen_range(cap / 2..=cap);
            let neigh: Vec<u32> = (0..len).map(|_| rng.gen_range(0..live as u32)).collect();
            neighbors.push(neigh);
        }
        nodes.push(DummyHnswNode {
            slot,
            level,
            version: rng.gen_range(1..=15),
            neighbors,
        });
    }
    DummyHnsw {
        dim,
        distance: 0,
        m,
        ef_construct: 100,
        ef_search: 40,
        entry: Some((0, nodes[0].level as i32)),
        built_at_live_count: live as u64,
        nodes,
    }
}

fn bench_persist_crc(c: &mut Criterion) {
    // One-time fixtures; keep outside `b.iter` so criterion measures only the
    // CRC / save kernels. Construction is reported once via eprintln for the
    // baseline report.
    let t0 = Instant::now();
    let ivf = make_ivf(256, 128, 100_000);
    let ivf_gen = t0.elapsed();
    let t0 = Instant::now();
    let ivf_bytes = postcard::to_stdvec(&ivf).expect("ivf postcard");
    let ivf_ser = t0.elapsed();
    let t0 = Instant::now();
    let hnsw = make_hnsw(100_000, 128, 16);
    let hnsw_gen = t0.elapsed();
    let t0 = Instant::now();
    let hnsw_bytes = postcard::to_stdvec(&hnsw).expect("hnsw postcard");
    let hnsw_ser = t0.elapsed();

    eprintln!(
        "[persist_crc] fixtures: ivf gen {:?} ser {:?} (payload {} bytes), hnsw gen {:?} ser {:?} (payload {} bytes)",
        ivf_gen,
        ivf_ser,
        ivf_bytes.len(),
        hnsw_gen,
        hnsw_ser,
        hnsw_bytes.len()
    );

    // CRC vs save_total for each tier. `save_total` re-serializes each iter to
    // include postcard cost (the real write_tagged does so).
    let mut group = c.benchmark_group("persist_crc");

    // IVF CRC only (hash of already-serialized payload) — pure memory scan.
    group.bench_function(BenchmarkId::new("ivf_crc", "100k"), |b| {
        b.iter(|| {
            let crc = crc32fast::hash(black_box(&ivf_bytes));
            black_box(crc)
        })
    });

    // IVF postcard+CRC only (without file I/O) — isolates hash fraction of the in-memory part.
    group.bench_function(BenchmarkId::new("ivf_serialize_plus_crc", "100k"), |b| {
        b.iter(|| {
            let bytes = postcard::to_stdvec(black_box(&ivf)).unwrap();
            let crc = crc32fast::hash(&bytes);
            black_box((bytes.len(), crc))
        })
    });

    // IVF full save (serialize + CRC + File::create/write/sync). Use a fresh tmp
    // path per iter to emulate persist::save's tmp+rename (rename elided to keep
    // the tmp file path stable across iters; the dominating cost is the write+fsync).
    group.bench_function(BenchmarkId::new("ivf_save_total", "100k"), |b| {
        let dir = tempfile::tempdir().unwrap();
        let tmp = dir.path().join("ivf_tmp.bin");
        b.iter(|| {
            write_tagged(black_box(&tmp), &IVF_MAGIC, VERSION, black_box(&ivf)).unwrap();
            black_box(&tmp);
        })
    });

    // HNSW CRC only.
    group.bench_function(BenchmarkId::new("hnsw_crc", "100k"), |b| {
        b.iter(|| {
            let crc = crc32fast::hash(black_box(&hnsw_bytes));
            black_box(crc)
        })
    });

    group.bench_function(BenchmarkId::new("hnsw_serialize_plus_crc", "100k"), |b| {
        b.iter(|| {
            let bytes = postcard::to_stdvec(black_box(&hnsw)).unwrap();
            let crc = crc32fast::hash(&bytes);
            black_box((bytes.len(), crc))
        })
    });

    group.bench_function(BenchmarkId::new("hnsw_save_total", "100k"), |b| {
        let dir = tempfile::tempdir().unwrap();
        let tmp = dir.path().join("hnsw_tmp.bin");
        b.iter(|| {
            write_tagged(black_box(&tmp), &HNSW_MAGIC, VERSION, black_box(&hnsw)).unwrap();
            black_box(&tmp);
        })
    });

    // Tiny manual ratio report at bench setup time (not measured per-iter, just
    // for the baseline doc). Run a few syncs to amortize cold page faults.
    {
        let dir = tempfile::tempdir().unwrap();
        let ivf_tmp = dir.path().join("ratio_ivf.bin");
        let hnsw_tmp = dir.path().join("ratio_hnsw.bin");
        const ITERS: usize = 20;
        let ivf_t_crc = {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(crc32fast::hash(&ivf_bytes));
            }
            t0.elapsed()
        };
        let ivf_t_save = {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                write_tagged(&ivf_tmp, &IVF_MAGIC, VERSION, &ivf).unwrap();
            }
            t0.elapsed()
        };
        let ivf_ratio = ivf_t_crc.as_secs_f64() / ivf_t_save.as_secs_f64().max(1e-9);
        let hnsw_t_crc = {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                black_box(crc32fast::hash(&hnsw_bytes));
            }
            t0.elapsed()
        };
        let hnsw_t_save = {
            let t0 = Instant::now();
            for _ in 0..ITERS {
                write_tagged(&hnsw_tmp, &HNSW_MAGIC, VERSION, &hnsw).unwrap();
            }
            t0.elapsed()
        };
        let hnsw_ratio = hnsw_t_crc.as_secs_f64() / hnsw_t_save.as_secs_f64().max(1e-9);
        eprintln!(
            "[persist_crc] manual ratio ({} iters): ivf_crc_ratio={:.4} (crc {:?} / save {:?}), hnsw_crc_ratio={:.4} (crc {:?} / save {:?})",
            ITERS,
            ivf_ratio,
            ivf_t_crc / ITERS as u32,
            ivf_t_save / ITERS as u32,
            hnsw_ratio,
            hnsw_t_crc / ITERS as u32,
            hnsw_t_save / ITERS as u32
        );
        if ivf_ratio >= 0.10 || hnsw_ratio >= 0.10 {
            eprintln!("[persist_crc] WARNING: crc ratio >= 10% (threshold for observation item)");
        }
    }

    group.finish();
}

criterion_group!(benches, bench_persist_crc);
criterion_main!(benches);
