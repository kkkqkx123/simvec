use super::*;
use crate::types::DistanceMetric;

fn config() -> IvfConfig {
    IvfConfig {
        lists: Some(4),
        min_build_points: 1,
        sample_limit: 1024,
        kmeans_max_iter: 5,
        drift_threshold: 0.10,
        drift_check_interval: 3,
        default_nprobe: 2,
        auto_promotion: true,
        max_probes: None,
    }
}

/// Two separated blobs of 8-dim vectors; returns (vectors, blob id).
fn blobs(n_per_blob: usize) -> Vec<(Vec<f32>, usize)> {
    let mut out: Vec<(Vec<f32>, usize)> = Vec::new();
    for (blob, center) in [[0.0f32; 8], [50.0; 8]].iter().enumerate() {
        for i in 0..n_per_blob {
            let mut v = *center;
            v[i % 8] += i as f32 * 0.1;
            out.push((v.to_vec(), blob));
        }
    }
    out
}

fn mmap_from(vectors: &[Vec<f32>]) -> (tempfile::TempDir, Vec<Arc<Mmap>>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vectors.bin");
    let mut bytes = Vec::with_capacity(vectors.len() * 8 * 4);
    for v in vectors {
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(&path, &bytes).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    (dir, vec![Arc::new(mmap)])
}

#[test]
fn test_assign_and_membership() {
    let data = blobs(4);
    let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
    let (_dir, mmaps) = mmap_from(&vectors);
    let slots: Vec<u32> = (0..data.len() as u32).collect();

    let index = IvfIndex::build(
        &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
        &config(),
    )
    .unwrap();

    assert_eq!(index.list_count(), 4);
    // Every slot is assigned exactly once across all lists.
    let total: usize = index.lists.iter().map(|l| l.read().len()).sum();
    assert_eq!(total, data.len());
    for slot in 0..data.len() as u32 {
        assert_ne!(index.slot_list.read()[slot as usize], UNASSIGNED);
    }
}

#[test]
fn test_reassignment_moves_slot_between_lists() {
    let data = blobs(4);
    let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
    let (_dir, mmaps) = mmap_from(&vectors);
    let slots: Vec<u32> = (0..data.len() as u32).collect();
    let index = IvfIndex::build(
        &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
        &config(),
    )
    .unwrap();

    // Reassigning the same slot with a vector from the other blob must
    // not duplicate it in two lists.
    let other_blob_vector = vec![50.0; 8];
    let before_total: usize = index.lists.iter().map(|l| l.read().len()).sum();
    index.assign_slot(0, &other_blob_vector);
    let after_total: usize = index.lists.iter().map(|l| l.read().len()).sum();
    assert_eq!(before_total, after_total);
    let occurrences: usize = index
        .lists
        .iter()
        .map(|l| l.read().iter().filter(|&&s| s == 0).count())
        .sum();
    assert_eq!(occurrences, 1);
}

#[test]
fn test_probe_candidates_skips_tombstones_and_scores_all() {
    let data = blobs(4);
    let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
    let (_dir, mmaps) = mmap_from(&vectors);
    let slots: Vec<u32> = (0..data.len() as u32).collect();
    let index = IvfIndex::build(
        &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
        &config(),
    )
    .unwrap();

    let mut bits = bitvec::bitvec![0; data.len()];
    bits.set(0, true);
    let tombstones = crate::storage::TombstoneBits::from_bits(bits);

    let query = vec![0.0f32; 8];
    let candidates = index
        .probe_candidates(
            &query,
            4,
            &[],
            &IvfSearchContext {
                tombstones: &tombstones,
                vectors: &mmaps,
                segment_slots: 16,
                filter_mask: None,
            },
        )
        .unwrap();
    assert_eq!(candidates.len(), data.len() - 1, "tombstoned slot skipped");
    assert!(!candidates.iter().any(|&(_, s)| s == 0));

    // nprobe clamped to lists; probing everything covers every live point.
    let exact = index
        .probe_candidates(
            &query,
            99,
            &[],
            &IvfSearchContext {
                tombstones: &tombstones,
                vectors: &mmaps,
                segment_slots: 16,
                filter_mask: None,
            },
        )
        .unwrap();
    assert_eq!(exact.len(), candidates.len());
}

#[test]
fn test_pending_slots_are_probed() {
    let mut vectors: Vec<Vec<f32>> = blobs(2).into_iter().map(|(v, _)| v).collect();
    // One extra mapped slot that the index does not know about yet.
    vectors.push(vec![50.0; 8]);
    let (_dir, mmaps) = mmap_from(&vectors);
    let known: Vec<u32> = (0..4).collect();
    let index = IvfIndex::build(
        &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &known, &mmaps, 16),
        &config(),
    )
    .unwrap();

    // A slot unknown to the index but passed via `extra_slots` must show
    // up in probe results.
    let extra = vec![vectors.len() as u32 - 1];
    let tombstones = crate::storage::TombstoneBits::from_bits(bitvec::bitvec![0; vectors.len()]);
    let candidates = index
        .probe_candidates(
            &[50.0; 8],
            4,
            &extra,
            &IvfSearchContext {
                tombstones: &tombstones,
                vectors: &mmaps,
                segment_slots: 16,
                filter_mask: None,
            },
        )
        .unwrap();
    assert!(candidates
        .iter()
        .any(|&(_, s)| s == vectors.len() as u32 - 1));

    // Rebuild the tombstone table with the extra slot marked deleted.
    let mut bits = bitvec::bitvec![0; vectors.len()];
    bits.set(vectors.len() - 1, true);
    let tombstones = crate::storage::TombstoneBits::from_bits(bits);
    let candidates = index
        .probe_candidates(
            &[50.0; 8],
            4,
            &extra,
            &IvfSearchContext {
                tombstones: &tombstones,
                vectors: &mmaps,
                segment_slots: 16,
                filter_mask: None,
            },
        )
        .unwrap();
    assert!(!candidates
        .iter()
        .any(|&(_, s)| s == vectors.len() as u32 - 1));
}

#[test]
fn test_adopt_range_covers_new_slots() {
    let vectors: Vec<Vec<f32>> = blobs(2).into_iter().map(|(v, _)| v).collect();
    let (_dir, mmaps) = mmap_from(&vectors);
    let first_half: Vec<u32> = (0..4).collect();
    let index = IvfIndex::build(
        &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &first_half, &mmaps, 16),
        &config(),
    )
    .unwrap();

    let tombstones = crate::storage::TombstoneBits::from_bits(bitvec::bitvec![0; vectors.len()]);
    index.adopt_range(4, vectors.len() as u32, &tombstones, &mmaps, 16);
    let total: usize = index.lists.iter().map(|l| l.read().len()).sum();
    assert_eq!(total, vectors.len());
}

#[test]
fn test_drift_ratio_detects_moved_points() {
    let vectors: Vec<Vec<f32>> = blobs(4).into_iter().map(|(v, _)| v).collect();
    let (_dir, mmaps) = mmap_from(&vectors);
    let slots: Vec<u32> = (0..vectors.len() as u32).collect();
    let index = IvfIndex::build(
        &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
        &config(),
    )
    .unwrap();

    let samples = IvfIndex::sample_plan(1024, vectors.len() as u32);
    let tombstones = crate::storage::TombstoneBits::from_bits(bitvec::bitvec![0; vectors.len()]);
    let clean = index.drift_ratio(&samples, &tombstones, &mmaps, 16);
    assert_eq!(clean, 0.0, "assignments straight after build have no drift");

    // Re-point half the slots as if their vectors had moved to a
    // brand-new region far outside the trained clusters: the label then
    // refers to a centroid that is nowhere near the stored vector, so the
    // measured mean distance explodes relative to the baseline.
    let far = [500.0f32; 8];
    for slot in 0..vectors.len() / 2 {
        index.assign_slot(slot as u32, &far);
    }
    let dirty = index.drift_ratio(&samples, &tombstones, &mmaps, 16);
    assert!(dirty > 1.0, "expected substantial drift, got {dirty}");
}

#[test]
fn test_sample_plan_stride() {
    assert!(IvfIndex::sample_plan(100, 0).is_empty());
    assert_eq!(IvfIndex::sample_plan(100, 10).len(), 10);
    let sampled = IvfIndex::sample_plan(4, 100);
    assert!(sampled.len() <= 4);
    assert!(sampled.iter().all(|&s| s < 100));
}

#[test]
fn test_should_check_drift_resets_counter() {
    let data = blobs(2);
    let vectors: Vec<Vec<f32>> = data.into_iter().map(|(v, _)| v).collect();
    let (_dir, mmaps) = mmap_from(&vectors);
    let slots: Vec<u32> = (0..vectors.len() as u32).collect();
    let index = IvfIndex::build(
        &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
        &config(),
    )
    .unwrap();

    assert!(!index.should_check_drift());
    for _ in 0..3 {
        index.note_upsert();
    }
    assert!(index.should_check_drift());
    // Counter was reset by the swap above.
    assert!(!index.should_check_drift());
}

#[test]
fn test_persisted_roundtrip_preserves_membership() {
    let data = blobs(4);
    let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
    let (_dir, mmaps) = mmap_from(&vectors);
    let slots: Vec<u32> = (0..data.len() as u32).collect();
    let original = IvfIndex::build(
        &IndexBuildParams::new("col", 8, DistanceMetric::Euclid, &slots, &mmaps, 16),
        &config(),
    )
    .unwrap();

    let restored = IvfIndex::from_persisted(original.to_persisted(), config());
    assert_eq!(
        restored.baseline_mean_dist, original.baseline_mean_dist,
        "drift baseline must survive the roundtrip"
    );
    let mut a: Vec<(u32, u32)> = (0..data.len() as u32)
        .map(|s| (s, original.slot_list.read()[s as usize]))
        .collect();
    let mut b: Vec<(u32, u32)> = (0..data.len() as u32)
        .map(|s| (s, restored.slot_list.read()[s as usize]))
        .collect();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
}
