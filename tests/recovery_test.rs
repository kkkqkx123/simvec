//! WAL replay idempotency integration tests.

use sim_vec::storage::CollectionStore;
use sim_vec::storage::{Wal, WalPoint, WalRecord, WalTxn};
use sim_vec::types::{CollectionConfig, DistanceMetric, PointId, SearchQuery, VectorPoint};

fn config() -> CollectionConfig {
    CollectionConfig::new(4, DistanceMetric::Cosine)
}

fn euclid_config() -> CollectionConfig {
    CollectionConfig::new(4, DistanceMetric::Euclid)
}

fn point(id: u64) -> VectorPoint {
    VectorPoint::new(id, vec![id as f32, 0.0, 0.0, 0.0])
}

#[test]
fn test_reopen_replays_wal_exactly_once() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");

    {
        let store = CollectionStore::create(&store_dir, "col", &config()).unwrap();
        store
            .apply_txn(&WalTxn {
                txn_id: 1,
                ops: vec![WalRecord::Upsert {
                    point: WalPoint::from_point(&point(1)).unwrap(),
                }],
            })
            .unwrap();
        store
            .apply_txn(&WalTxn {
                txn_id: 2,
                ops: vec![WalRecord::Upsert {
                    point: WalPoint::from_point(&point(2)).unwrap(),
                }],
            })
            .unwrap();
        store
            .apply_txn(&WalTxn {
                txn_id: 3,
                ops: vec![WalRecord::Upsert {
                    point: WalPoint::from_point(&point(3)).unwrap(),
                }],
            })
            .unwrap();
    }

    // Reopen N times: the WAL must replay idempotently (upserts overwrite by
    // id, never duplicate).
    for _ in 0..3 {
        let store = CollectionStore::open(&store_dir).unwrap();
        assert_eq!(store.count(), 3);
        assert_eq!(store.meta().last_applied_txn, 3);
    }
}

#[test]
fn test_replay_delete_batch_and_compact_marker() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");

    {
        let store = CollectionStore::create(&store_dir, "col", &config()).unwrap();
        store
            .apply_txn(&WalTxn {
                txn_id: 1,
                ops: vec![
                    WalRecord::Upsert {
                        point: WalPoint::from_point(&point(1)).unwrap(),
                    },
                    WalRecord::Upsert {
                        point: WalPoint::from_point(&point(2)).unwrap(),
                    },
                    WalRecord::Upsert {
                        point: WalPoint::from_point(&point(3)).unwrap(),
                    },
                ],
            })
            .unwrap();
    }

    // Append a delete-batch + compact marker directly to the WAL (simulating a
    // crash after WAL fsync before memory apply).
    {
        let wal = Wal::open_or_create(&store_dir.join("wal.bin")).unwrap();
        wal.append(&WalTxn {
            txn_id: 2,
            ops: vec![WalRecord::DeleteBatch {
                point_ids: vec!["1".to_string(), "2".to_string()],
            }],
        })
        .unwrap();
        wal.append(&WalTxn {
            txn_id: 3,
            ops: vec![WalRecord::Compact],
        })
        .unwrap();
    }

    let store = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(store.count(), 1, "batch delete must apply during replay");
    assert!(store.get(&PointId::Num(1)).unwrap().is_none());
    assert!(store.get(&PointId::Num(2)).unwrap().is_none());
    assert_eq!(
        store.get(&PointId::Num(3)).unwrap().unwrap().vector,
        point(3).vector
    );
    assert_eq!(store.meta().last_applied_txn, 3);
}

#[test]
fn test_partial_wal_record_is_tolerated() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");

    {
        let store = CollectionStore::create(&store_dir, "col", &config()).unwrap();
        store
            .apply_txn(&WalTxn {
                txn_id: 1,
                ops: vec![WalRecord::Upsert {
                    point: WalPoint::from_point(&point(1)).unwrap(),
                }],
            })
            .unwrap();
        store
            .apply_txn(&WalTxn {
                txn_id: 2,
                ops: vec![WalRecord::Upsert {
                    point: WalPoint::from_point(&point(2)).unwrap(),
                }],
            })
            .unwrap();
    }

    // Crash mid-append: a length prefix claiming bytes that were never
    // written.
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(store_dir.join("wal.bin"))
            .unwrap();
        f.write_all(&4096u32.to_le_bytes()).unwrap();
        f.sync_all().unwrap();
    }

    let store = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(store.count(), 2, "truncated tail record must be ignored");
    assert!(store.get(&PointId::Num(2)).unwrap().is_some());
}

#[test]
fn test_upsert_overwrite_after_replay_does_not_duplicate() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");

    {
        let store = CollectionStore::create(&store_dir, "col", &config()).unwrap();
        store
            .apply_txn(&WalTxn {
                txn_id: 1,
                ops: vec![WalRecord::Upsert {
                    point: WalPoint::from_point(&point(1)).unwrap(),
                }],
            })
            .unwrap();
    }

    // Same point id with a different vector, logged after a crash that lost
    // the memory apply of txn 2.
    {
        let wal = Wal::open_or_create(&store_dir.join("wal.bin")).unwrap();
        let mut updated = point(1);
        updated.vector = vec![9.0, 9.0, 9.0, 9.0];
        wal.append(&WalTxn {
            txn_id: 2,
            ops: vec![WalRecord::Upsert {
                point: WalPoint::from_point(&updated).unwrap(),
            }],
        })
        .unwrap();
    }

    let store = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(store.count(), 1);
    assert_eq!(
        store.get(&PointId::Num(1)).unwrap().unwrap().vector,
        vec![9.0, 9.0, 9.0, 9.0],
        "replay overwrite must win"
    );
}

#[test]
fn test_replay_reconciles_stale_meta_counts() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");

    {
        let store = CollectionStore::create(&store_dir, "col", &config()).unwrap();
        store
            .apply_txn(&WalTxn {
                txn_id: 1,
                ops: vec![WalRecord::Upsert {
                    point: WalPoint::from_point(&point(1)).unwrap(),
                }],
            })
            .unwrap();
        // Crash window: WAL written but meta.bin not yet persisted with the
        // live_count bump. Append directly to the WAL.
        let wal = Wal::open_or_create(&store_dir.join("wal.bin")).unwrap();
        wal.append(&WalTxn {
            txn_id: 2,
            ops: vec![WalRecord::Upsert {
                point: WalPoint::from_point(&point(2)).unwrap(),
            }],
        })
        .unwrap();
    }

    let store = CollectionStore::open(&store_dir).unwrap();
    assert_eq!(store.count(), 2, "live_count must be reconciled from slots");
    assert_eq!(store.meta().last_applied_txn, 2);
}

#[test]
fn test_hnsw_pending_slots_visible_after_restart() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col");

    {
        let store = CollectionStore::create(&store_dir, "col", &euclid_config()).unwrap();
        let points: Vec<VectorPoint> = (0..60u64)
            .map(|i| VectorPoint::new(i, vec![i as f32, (i % 7) as f32 * 0.1, 1.0, 2.0]))
            .collect();
        let ops: Vec<WalRecord> = points
            .iter()
            .map(|p| WalRecord::Upsert {
                point: WalPoint::from_point(p).unwrap(),
            })
            .collect();
        store.apply_ops(&ops).unwrap();
        assert!(store.build_index().unwrap(), "index published");

        // Late writes after publication land in the pending queue and are
        // not part of hnsw.bin's coverage.
        for l in 0..5u64 {
            let v = vec![-100.0 - l as f32, 0.0, 0.0, 0.0];
            let p = VectorPoint::new(format!("late{l}"), v);
            store
                .apply_ops(&[WalRecord::Upsert {
                    point: WalPoint::from_point(&p).unwrap(),
                }])
                .unwrap();
        }
    }

    // Reopen: hnsw.bin covers only the first 60 slots; WAL replay restores
    // the late ones into pending, and approximate search must still see
    // them via the exact-scored pending path.
    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert!(reopened.has_index(), "hnsw.bin rehydrated");
    assert_eq!(reopened.count(), 65);
    for l in 0..5u64 {
        let q = vec![-100.0 - l as f32, 0.0, 0.0, 0.0];
        let hits = reopened.search(&SearchQuery::new(q, 1)).unwrap();
        assert_eq!(hits.len(), 1, "no hit for late{l}");
        assert_eq!(
            hits[0].id.to_string(),
            format!("late{l}"),
            "post-restart ANN must surface the late write"
        );
    }
}

#[test]
fn test_index_bin_payload_bitflip_triggers_crc_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_ivf");
    let cfg = CollectionConfig::new(4, DistanceMetric::Cosine)
        .with_index_type(sim_vec::types::IndexType::IVF)
        .with_ivf(sim_vec::types::IvfConfig {
            lists: Some(2),
            min_build_points: 1,
            sample_limit: 64,
            kmeans_max_iter: 5,
            drift_threshold: 0.10,
            drift_check_interval: u64::MAX,
            default_nprobe: 1,
            auto_promotion: false,
            max_probes: None,
        });

    {
        let store = CollectionStore::create(&store_dir, "col_ivf", &cfg).unwrap();
        let points: Vec<VectorPoint> = (0..20u64)
            .map(|i| VectorPoint::new(i, vec![i as f32 * 0.1, 1.0, 2.0, 3.0]))
            .collect();
        let ops: Vec<WalRecord> = points
            .iter()
            .map(|p| WalRecord::Upsert {
                point: WalPoint::from_point(p).unwrap(),
            })
            .collect();
        store.apply_ops(&ops).unwrap();
        assert!(store.build_index().unwrap(), "IVF index must publish");
        assert!(store.has_index(), "index published before corruption");
        // Exact search ground truth before corruption.
        let hits = store
            .search(&SearchQuery::new(vec![0.1, 1.0, 2.0, 3.0], 1))
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    // Flip an arbitrary payload byte after the 10-byte header (magic+version+crc).
    let index_path = store_dir.join("index.bin");
    let mut bytes = std::fs::read(&index_path).unwrap();
    assert!(bytes.len() > 12, "index.bin must contain payload");
    // Payload starts at byte 10; flipping byte 10 must break CRC.
    bytes[10] ^= 0xFF;
    std::fs::write(&index_path, &bytes).unwrap();

    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert!(
        !reopened.has_index(),
        "corrupt index.bin payload must fall back to exact scan"
    );
    assert_eq!(reopened.count(), 20, "live count must survive fallback");
    assert!(
        !index_path.exists(),
        "corrupt index.bin must be deleted on load, so next save starts clean"
    );
    assert!(
        reopened.metrics().snapshot().index_load_fallbacks >= 1,
        "CRC mismatch must increment index_load_fallbacks"
    );
    // Search must still succeed via exact scan with perfect recall.
    let hits = reopened
        .search(&SearchQuery::new(vec![0.1, 1.0, 2.0, 3.0], 1))
        .unwrap();
    assert_eq!(hits.len(), 1);
}

#[test]
fn test_hnsw_bin_payload_bitflip_triggers_crc_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let store_dir = dir.path().join("col_hnsw");
    let cfg = CollectionConfig::new(4, DistanceMetric::Euclid);

    {
        let store = CollectionStore::create(&store_dir, "col_hnsw", &cfg).unwrap();
        let points: Vec<VectorPoint> = (0..20u64)
            .map(|i| VectorPoint::new(i, vec![i as f32, 0.0, 0.0, 0.0]))
            .collect();
        let ops: Vec<WalRecord> = points
            .iter()
            .map(|p| WalRecord::Upsert {
                point: WalPoint::from_point(p).unwrap(),
            })
            .collect();
        store.apply_ops(&ops).unwrap();
        assert!(store.build_index().unwrap(), "HNSW index must publish");
        assert!(store.has_index());
    }

    let hnsw_path = store_dir.join("hnsw.bin");
    let mut bytes = std::fs::read(&hnsw_path).unwrap();
    assert!(bytes.len() > 12, "hnsw.bin must contain payload");
    // Flip a payload byte (header is 10 bytes).
    let flip_off = 11.min(bytes.len() - 1);
    bytes[flip_off] ^= 0xA5;
    std::fs::write(&hnsw_path, &bytes).unwrap();

    let reopened = CollectionStore::open(&store_dir).unwrap();
    assert!(
        !reopened.has_index(),
        "corrupt hnsw.bin payload must fall back to exact scan"
    );
    assert_eq!(reopened.count(), 20);
    assert!(
        !hnsw_path.exists(),
        "corrupt hnsw.bin must be deleted on load"
    );
    assert!(
        reopened.metrics().snapshot().index_load_fallbacks >= 1,
        "CRC mismatch must increment index_load_fallbacks for HNSW"
    );
    let hits = reopened
        .search(&SearchQuery::new(vec![0.0, 0.0, 0.0, 0.0], 1))
        .unwrap();
    assert_eq!(hits.len(), 1);
}
