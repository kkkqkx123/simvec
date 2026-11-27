//! Regression: payload allocation must stay inside one page.
//!
//! `ValuePointer` is `(page_id, block_offset, len)` and pages are fixed-size
//! files, so a run that straddles a page boundary both grows the page file past
//! `page_size` and makes the pointer unreadable by the same process that wrote
//! it. The allocator used to search the global block space for a contiguous
//! run, which produced exactly that. These tests use a small page (8 blocks) to
//! make the boundary reachable.

use simvec::storage::{CollectionStore, WalPoint, WalRecord};
use simvec::types::{
    CollectionConfig, DistanceMetric, Payload, PointId, StorageConfig, VectorPoint,
};

fn point(id: u64, filler: usize) -> VectorPoint {
    let mut p = VectorPoint::new(PointId::Num(id), vec![1.0, 0.0, 0.0]);
    p.payload = Some(Payload::from([(
        "a".to_string(),
        serde_json::json!("x".repeat(filler)),
    )]));
    p
}

fn upsert(store: &CollectionStore, id: u64, filler: usize) {
    store
        .apply_ops(&[WalRecord::Upsert {
            point: WalPoint::from_point(&point(id, filler)).unwrap(),
        }])
        .expect("upsert");
}

fn page_sizes(collection_dir: &std::path::Path) -> Vec<u64> {
    let mut sizes = Vec::new();
    for entry in std::fs::read_dir(collection_dir.join("segments/0000000000/payloads_store/pages")).expect("pages dir")
    {
        let entry = entry.expect("page entry");
        sizes.push(entry.metadata().expect("page meta").len());
    }
    sizes.sort_unstable();
    sizes
}

fn payload_len(store: &CollectionStore, id: u64) -> Option<usize> {
    store
        .get(&PointId::Num(id))
        .expect("get")
        .and_then(|p| p.payload)
        .and_then(|m| m.get("a").and_then(|v| v.as_str()).map(|s| s.len()))
}

#[test]
fn allocation_never_straddles_a_page() {
    let dir = tempfile::tempdir().unwrap();
    let storage = StorageConfig {
        payload_block_size: 128,
        payload_page_size: 1024, // 8 blocks per page
        ..StorageConfig::default()
    };
    let cfg = CollectionConfig::new(3, DistanceMetric::Dot);
    let collection_dir = dir.path().join("probe");
    let store = CollectionStore::create_with_storage(&collection_dir, "probe", &cfg, &storage)
        .expect("create");

    upsert(&store, 0, 300); // 3 blocks: 0..2
    upsert(&store, 1, 300); // 3 blocks: 3..5
    upsert(&store, 2, 200); // 2 blocks: 6..7 -> page 0 is now full
    upsert(&store, 3, 300); // 3 blocks: page 1
                            // Free page 1's blocks without allocating anything else.
    store
        .apply_ops(&[WalRecord::SetPayload {
            point_id: "3".to_string(),
            payload: None,
        }])
        .expect("drop payload");
    // Overwrite slot 2 with a 5-block payload: the only free run (page 0 tail
    // plus page 1 head) used to straddle the boundary.
    upsert(&store, 2, 600);

    let sizes = page_sizes(&store.data_dir());
    assert!(!sizes.is_empty());
    for size in &sizes {
        assert_eq!(
            *size, storage.payload_page_size as u64,
            "page files must stay exactly one page long: {sizes:?}"
        );
    }

    // The writing process must be able to read every point back immediately
    // (this used to fail with "payload read out of bounds").
    for id in 0..4u64 {
        let expected = match id {
            0 | 1 => Some(300),
            2 => Some(600),
            3 => None, // payload explicitly dropped above
            _ => None,
        };
        assert_eq!(payload_len(&store, id), expected, "point {id}");
    }

    store.checkpoint().expect("checkpoint");
    drop(store);

    let store = CollectionStore::open(&collection_dir).expect("reopen");
    assert_eq!(store.count(), 4);
    assert_eq!(payload_len(&store, 2), Some(600));
    assert_eq!(payload_len(&store, 1), Some(300));
    for size in page_sizes(&store.data_dir()) {
        assert_eq!(size, storage.payload_page_size as u64);
    }
}

#[test]
fn payload_larger_than_a_page_is_rejected_before_logging() {
    let dir = tempfile::tempdir().unwrap();
    let storage = StorageConfig {
        payload_block_size: 128,
        payload_page_size: 1024,
        ..StorageConfig::default()
    };
    let cfg = CollectionConfig::new(3, DistanceMetric::Dot);
    let collection_dir = dir.path().join("probe");
    let store = CollectionStore::create_with_storage(&collection_dir, "probe", &cfg, &storage)
        .expect("create");

    // ~4 KiB of JSON cannot fit in a 1 KiB page.
    let err = store
        .apply_ops(&[WalRecord::Upsert {
            point: WalPoint::from_point(&point(1, 4096)).unwrap(),
        }])
        .expect_err("oversized payload must be rejected");
    assert!(
        err.to_string().contains("exceeds the"),
        "unexpected error: {err}"
    );

    // Nothing was logged, so the collection still opens and has no points.
    store.checkpoint().expect("checkpoint");
    drop(store);
    let store = CollectionStore::open(&collection_dir).expect("reopen");
    assert_eq!(store.count(), 0);
}
