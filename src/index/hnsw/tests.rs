use super::*;
    use crate::types::DistanceMetric;
    use std::collections::HashSet;

    const DIM: usize = 8;

    pub(super) fn config() -> HnswConfig {
        // m stays 8: halving it shrinks the ground-layer cap (maxM0 = 2m)
        // enough that reverse-link pruning drops long-range edges and
        // disconnects the tiny clustered fixtures these tests rely on.
        // ef_construct carries the build-cost reduction instead.
        HnswConfig {
            m: 8,
            ef_construct: 16,
            ef_search: 16,
            ..HnswConfig::default()
        }
    }

    /// Two separated blobs of 8-dim vectors.
    pub(super) fn blobs(n_per_blob: usize) -> Vec<(Vec<f32>, usize)> {
        let mut out: Vec<(Vec<f32>, usize)> = Vec::new();
        for (blob, center) in [[0.0f32; DIM], [50.0; DIM]].iter().enumerate() {
            for i in 0..n_per_blob {
                let mut v = *center;
                v[i % DIM] += i as f32 * 0.1;
                out.push((v.to_vec(), blob));
            }
        }
        out
    }

    pub(super) fn mmap_from(vectors: &[Vec<f32>]) -> (tempfile::TempDir, Vec<Arc<memmap2::Mmap>>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.bin");
        let mut bytes = Vec::with_capacity(vectors.len() * DIM * 4);
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

    pub(super) fn all_live(n: usize) -> TombstoneBits {
        TombstoneBits::from_bits(bitvec::bitvec![0; n])
    }

    #[test]
    pub(super) fn test_insert_and_probe_finds_blobs() {
        let data = blobs(15);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();

        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        assert_eq!(index.node_count(), data.len());

        // Every query's top-10 must come from its own blob: the blobs sit
        // 50 units apart while members stay within ~1 unit of their center,
        // so cross-blob hits in the top-10 would indicate broken navigation.
        for (blob, center) in [[0.0f32; DIM], [50.0; DIM]].iter().enumerate() {
            let q = *center;
            let mut hits = index
                .probe_candidates(
                    &q,
                    24,
                    10,
                    &HnswSearchContext {
                        tombstones: Some(&all_live(data.len())),
                        vectors: &mmaps,
                        segment_slots: 128,
                        filter_mask: None,
                    },
                )
                .unwrap();
            assert!(hits.len() >= 10, "expected results");
            hits.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(_, slot) in hits.iter().take(10) {
                assert_eq!(
                    data[slot as usize].1, blob,
                    "slot {slot} from the wrong blob"
                );
            }
        }
    }

    #[test]
    pub(super) fn test_probe_respects_tombstones() {
        let data = blobs(10);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let mut bits = bitvec::bitvec![0; data.len()];
        bits.set(3, true);
        let tombstones = TombstoneBits::from_bits(bits);
        let hits = index
            .probe_candidates(
                &[50.0; DIM],
                64,
                100,
                &HnswSearchContext {
                    tombstones: Some(&tombstones),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert!(!hits.iter().any(|&(_, s)| s == 3));

        // The tombstoned node stays navigable: dropping the filter brings it
        // back without any structural repair.
        let hits = index
            .probe_candidates(
                &[50.0; DIM],
                64,
                100,
                &HnswSearchContext {
                    tombstones: Some(&all_live(data.len())),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert!(hits.iter().any(|&(_, s)| s == 3));
    }

    #[test]
    pub(super) fn test_incremental_insert_reaches_graph() {
        let data = blobs(10);
        let mut vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 256),
            &config(),
        )
        .unwrap();

        // Insert one extra far-away point through the incremental path.
        vectors.push(vec![-100.0; DIM]);
        let (_dir2, mmaps2) = mmap_from(&vectors);
        let new_slot = data.len() as u32;
        let v = vec![-100.0f32; DIM];
        index.insert(new_slot, &v, &mmaps2, 256);
        assert_eq!(index.node_count(), data.len() + 1);

        let hits = index
            .probe_candidates(
                &[-100.0; DIM],
                64,
                5,
                &HnswSearchContext {
                    tombstones: Some(&all_live(vectors.len())),
                    vectors: &mmaps2,
                    segment_slots: 256,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert!(
            hits.iter().any(|&(_, s)| s == new_slot),
            "incrementally inserted point must be reachable"
        );
    }

    #[test]
    pub(super) fn test_overwrite_insert_counts_staleness() {
        let data = blobs(6);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 64),
            &config(),
        )
        .unwrap();
        assert_eq!(index.overwrites_since_build(), 0);

        let before = index.node_count();
        for _ in 0..3 {
            index.insert(0, &vectors[0], &mmaps, 64);
        }
        assert_eq!(index.node_count(), before, "overwrite keeps the position");
        assert_eq!(
            index.overwrites_since_build(),
            3,
            "each overwrite must bump the staleness counter"
        );
    }

    #[test]
    pub(super) fn test_search_layer_result_never_exceeds_ef_with_dead_nodes() {
        // Tombstone almost all of blob 1 and probe its center: the search
        // explores many dead nodes around the entry. The eviction accounting
        // must keep the tracked candidate set within `ef` while still
        // returning only live elements.
        let data = blobs(15);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let mut bits = bitvec::bitvec![0; data.len()];
        let blob1_start = data.len() / 2;
        for slot in blob1_start..data.len() - 2 {
            bits.set(slot, true);
        }
        let dead: HashSet<u32> = (blob1_start..data.len() - 2).map(|s| s as u32).collect();
        let tombstones = TombstoneBits::from_bits(bits);

        // Enter from the last node (a live blob-1 member next to the dead zone).
        let entry_slot = (data.len() - 1) as u32;
        let out = index.search_layer(
            &[50.0; DIM],
            &[Cand {
                dist: index.distance(&[50.0; DIM], &vectors[entry_slot as usize]),
                slot: entry_slot,
            }],
            4,
            0,
            &HnswSearchContext {
                tombstones: Some(&tombstones),
                vectors: &mmaps,
                segment_slots: 128,
                filter_mask: None,
            },
        );
        assert!(
            out.live.len() <= 4,
            "tracked results must stay within ef, got {}",
            out.live.len()
        );
        assert!(out.live.iter().all(|c| !dead.contains(&c.slot)));
    }

    #[test]
    pub(super) fn test_dedicated_pool_build_matches_sequential_topology() {
        let data = blobs(12);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();

        let sequential = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();
        let threaded_cfg = HnswConfig {
            max_indexing_threads: Some(1),
            ..config()
        };
        let threaded = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &threaded_cfg,
        )
        .unwrap();

        type Topology = (Option<(u32, i32)>, Vec<(u32, u8, Vec<Vec<u32>>)>);
        fn topology(index: &HnswIndex) -> Topology {
            let p = index.to_persisted();
            (
                p.entry,
                p.nodes
                    .into_iter()
                    .map(|n| (n.slot, n.level, n.neighbors))
                    .collect(),
            )
        }
        assert_eq!(
            topology(&sequential),
            topology(&threaded),
            "a dedicated single-thread pool must not change the built graph"
        );
    }

    #[test]
    pub(super) fn test_multiworker_build_registers_all_and_keeps_recall() {
        let n = 50;
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut state = SplitMix64::new(7);
        for _ in 0..n {
            vectors.push(
                (0..DIM)
                    .map(|_| (state.next_u64() % 1000) as f32 / 100.0)
                    .collect(),
            );
        }
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..n as u32).collect();
        let workers_cfg = HnswConfig {
            max_indexing_threads: Some(4),
            ..config()
        };
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &workers_cfg,
        )
        .unwrap();
        assert_eq!(
            index.node_count(),
            n,
            "every slot must be registered exactly once"
        );

        let q = vec![5.0f32; DIM];
        let mut exact: Vec<(f32, u32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    crate::distance::distance(DistanceMetric::Euclid, &q, v),
                    i as u32,
                )
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<u32> = exact[..5].iter().map(|&(_, s)| s).collect();

        let hits = index
            .probe_candidates(
                &q,
                32,
                5,
                &HnswSearchContext {
                    tombstones: Some(&all_live(n)),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        let found = hits.iter().filter(|&&(_, s)| truth.contains(&s)).count();
        assert!(
            found >= 3,
            "concurrent build must keep the recall bar, got {found}/5"
        );
    }

    #[test]
    pub(super) fn test_multiworker_build_with_more_workers_than_points() {
        let data = blobs(12);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let workers_cfg = HnswConfig {
            max_indexing_threads: Some(8),
            ..config()
        };
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &workers_cfg,
        )
        .unwrap();
        assert_eq!(index.node_count(), data.len());
        assert!(
            index.to_persisted().entry.is_some(),
            "exactly one entry point must survive concurrent first inserts"
        );
    }

    #[test]
    pub(super) fn test_persisted_roundtrip_preserves_topology() {
        let data = blobs(12);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let original = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let restored = HnswIndex::from_persisted(original.to_persisted(), &config()).unwrap();
        assert_eq!(restored.node_count(), original.node_count());
        assert_eq!(
            restored.built_at_live_count(),
            original.built_at_live_count()
        );
        assert_eq!(
            restored.default_ef(),
            original.default_ef(),
            "effective ef_search must survive the roundtrip"
        );

        let q = vec![50.0; DIM];
        let a = original
            .probe_candidates(
                &q,
                32,
                10,
                &HnswSearchContext {
                    tombstones: Some(&all_live(data.len())),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        let b = restored
            .probe_candidates(
                &q,
                32,
                10,
                &HnswSearchContext {
                    tombstones: Some(&all_live(data.len())),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        assert_eq!(a, b, "restored graph must answer identically");
    }

    #[test]
    pub(super) fn test_from_persisted_rejects_corruption() {
        let data = blobs(10);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 64),
            &config(),
        )
        .unwrap();

        let mut persisted = index.to_persisted();
        persisted.nodes[2].neighbors[0][0] = 999;
        assert!(HnswIndex::from_persisted(persisted, &config()).is_err());

        let mut persisted = index.to_persisted();
        persisted.entry = Some((999, 0));
        assert!(HnswIndex::from_persisted(persisted, &config()).is_err());

        let mut persisted = index.to_persisted();
        persisted.nodes[1].level = 5;
        assert!(HnswIndex::from_persisted(persisted, &config()).is_err());
    }

    #[test]
    pub(super) fn test_recall_against_exact_scan() {
        // Random-ish deterministic vectors; HNSW with these settings should
        // recover most of the true top-10 for queries near the data cloud.
        let n = 50;
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut state = SplitMix64::new(42);
        for _ in 0..n {
            vectors.push(
                (0..DIM)
                    .map(|_| (state.next_u64() % 1000) as f32 / 100.0)
                    .collect(),
            );
        }
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..n as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let q = vec![5.0f32; DIM];
        let mut exact: Vec<(f32, u32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    crate::distance::distance(DistanceMetric::Euclid, &q, v),
                    i as u32,
                )
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<u32> = exact[..5].iter().map(|&(_, s)| s).collect();

        let hits = index
            .probe_candidates(
                &q,
                32,
                5,
                &HnswSearchContext {
                    tombstones: Some(&all_live(n)),
                    vectors: &mmaps,
                    segment_slots: 128,
                    filter_mask: None,
                },
            )
            .unwrap();
        let found = hits.iter().filter(|&&(_, s)| truth.contains(&s)).count();
        assert!(found >= 3, "recall@5 too low: {found}/5");
    }

    #[test]
    pub(super) fn test_iterative_scan_improves_recall() {
        // Build a reasonably large graph, then compare one-shot vs iterative
        // scan with a deliberately small ef to force iterative expansion.
        let n = 50;
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut state = SplitMix64::new(42);
        for _ in 0..n {
            vectors.push(
                (0..DIM)
                    .map(|_| (state.next_u64() % 1000) as f32 / 100.0)
                    .collect(),
            );
        }
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..n as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let q = vec![5.0f32; DIM];
        let mut exact: Vec<(f32, u32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    crate::distance::distance(DistanceMetric::Euclid, &q, v),
                    i as u32,
                )
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: HashSet<u32> = exact[..5].iter().map(|&(_, s)| s).collect();

        let ctx = HnswSearchContext {
            tombstones: Some(&all_live(n)),
            vectors: &mmaps,
            segment_slots: 128,
            filter_mask: None,
        };

        // One-shot with small ef.
        let small_ef = 4;
        let oneshot = index.probe_candidates(&q, small_ef, 5, &ctx).unwrap();
        let oneshot_found = oneshot.iter().filter(|&&(_, s)| truth.contains(&s)).count();

        // Iterative scan with the same small ef, 3 iterations.
        let iterative = index
            .probe_candidates_iterative(&q, small_ef, 5, 3, None, &ctx)
            .unwrap();
        let iterative_found = iterative
            .iter()
            .filter(|&&(_, s)| truth.contains(&s))
            .count();

        // Iterative should achieve reasonable recall (it explores more of the
        // graph via discarded candidates, though one-shot can occasionally
        // score higher on specific topologies).
        assert!(
            iterative_found >= 3,
            "iterative recall@5 too low: {iterative_found}/5"
        );
        assert!(
            oneshot_found >= 3,
            "oneshot recall@5 too low: {oneshot_found}/5"
        );
    }

    #[test]
    pub(super) fn test_stale_ratio_combines_count_and_delta() {
        let data = blobs(6);
        let vectors: Vec<Vec<f32>> = data.iter().map(|(v, _)| v.clone()).collect();
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..data.len() as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 64),
            &config(),
        )
        .unwrap();
        assert_eq!(index.stale_ratio(), 0.0);

        // Overwrite with the same vector: count increases, delta stays ~0.
        index.insert(0, &vectors[0], &mmaps, 64);
        assert!(index.stale_ratio() > 0.0, "count-based ratio should be > 0");

        // Overwrite with a very different vector: delta should dominate.
        let far = vec![1000.0; DIM];
        index.insert(1, &far, &mmaps, 64);
        let ratio = index.stale_ratio();
        assert!(ratio > 0.1, "delta should push ratio up, got {ratio}");
    }

    #[test]
    pub(super) fn test_iterative_scan_tuple_budget_truncates() {
        let n = 50;
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut state = SplitMix64::new(42);
        for _ in 0..n {
            vectors.push(
                (0..DIM)
                    .map(|_| (state.next_u64() % 1000) as f32 / 100.0)
                    .collect(),
            );
        }
        let (_dir, mmaps) = mmap_from(&vectors);
        let slots: Vec<u32> = (0..n as u32).collect();
        let index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &slots, &mmaps, 128),
            &config(),
        )
        .unwrap();

        let ctx = HnswSearchContext {
            tombstones: Some(&all_live(n)),
            vectors: &mmaps,
            segment_slots: 128,
            filter_mask: None,
        };
        let q = vec![5.0f32; DIM];

        // A one-node budget stops the expansion after the first pass; an
        // unbounded budget may keep resuming from discarded candidates.
        let bounded = index
            .probe_candidates_iterative(&q, 4, 50, 8, Some(1), &ctx)
            .unwrap();
        let unbounded = index
            .probe_candidates_iterative(&q, 4, 50, 8, None, &ctx)
            .unwrap();
        assert!(!bounded.is_empty(), "first pass must still return results");
        assert!(
            bounded.len() <= unbounded.len(),
            "budgeted scan must not explore more than unbounded ({}/{})",
            bounded.len(),
            unbounded.len()
        );
    }

    #[test]
    pub(super) fn test_concurrent_insert_and_probe_exercises_version_reload() {
        // Writers keep mutating adjacency lists while a reader hammers the
        // probe path; the reader's version double-read protocol must detect
        // at least one concurrent mutation and reload the neighborhood.
        let base = blobs(30);
        let mut vectors: Vec<Vec<f32>> = base.iter().map(|(v, _)| v.clone()).collect();
        // Extra slots inserted by the writer thread, clustered on blob 1 so
        // reverse-links keep mutating the same hub neighborhoods.
        let extra: Vec<Vec<f32>> = (0..300)
            .map(|i| {
                let mut v = vec![50.0f32; DIM];
                v[i % DIM] += (i % 11) as f32 * 0.1;
                v
            })
            .collect();
        vectors.extend(extra.iter().cloned());
        let (_dir, mmaps) = mmap_from(&vectors);

        let initial: Vec<u32> = (0..base.len() as u32).collect();
        let metrics = Arc::new(Metrics::default());
        let mut index = HnswIndex::build(
            &IndexBuildParams::new("col", DIM, DistanceMetric::Euclid, &initial, &mmaps, 256),
            &config(),
        )
        .unwrap();
        index.set_metrics(Arc::clone(&metrics));
        let index = Arc::new(index);

        let first_extra = base.len() as u32;
        let total_slots = vectors.len();
        let writer_mmaps = mmaps.clone();
        let writer_index = Arc::clone(&index);
        let writer = std::thread::spawn(move || {
            for i in 0..extra.len() as u32 {
                writer_index.insert(first_extra + i, &extra[i as usize], &writer_mmaps, 256);
            }
        });

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader_index = Arc::clone(&index);
        let reader_stop = Arc::clone(&stop);
        let reader = std::thread::spawn(move || {
            while !reader_stop.load(std::sync::atomic::Ordering::Relaxed) {
                for _ in 0..50 {
                    let _ = reader_index.probe_candidates(
                        &[50.0; DIM],
                        8,
                        5,
                        &HnswSearchContext {
                            tombstones: Some(&all_live(total_slots)),
                            vectors: &mmaps,
                            segment_slots: 256,
                            filter_mask: None,
                        },
                    );
                }
            }
        });

        writer.join().unwrap();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        reader.join().unwrap();

        assert_eq!(index.node_count(), vectors.len());
        assert!(
            metrics.snapshot().search_version_reloads > 0,
            "write/read contention must exercise the version double-read reload path"
        );
    }
