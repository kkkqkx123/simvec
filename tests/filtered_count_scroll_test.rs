//! Filtered count / filtered scroll acceptance: reconciliation, re-ingest
//! iteration and version cleanup on one collection with multi-version data.

use simvec::storage::{CollectionStore, WalPoint, WalRecord};
use simvec::types::{
    CollectionConfig, ConditionType, DistanceMetric, FilterCondition, GeoPoint, GeoRadius, PointId,
    RangeCondition, ValuesCountCondition, VectorFilter, VectorPoint,
};

fn config(dim: usize) -> CollectionConfig {
    CollectionConfig::new(dim, DistanceMetric::Cosine)
}

fn unit(i: u64, dim: usize) -> VectorPoint {
    let mut v = vec![0.0f32; dim];
    v[(i as usize) % dim] = 1.0;
    for (j, x) in v.iter_mut().enumerate() {
        if j != (i as usize) % dim {
            *x = ((i as usize * 7 + j * 13) % 100) as f32 * 0.001;
        }
    }
    VectorPoint::new(i, v)
}

fn upsert_logged(store: &CollectionStore, point: &VectorPoint) {
    store
        .apply_ops(&[WalRecord::Upsert {
            point: WalPoint::from_point(point).unwrap(),
        }])
        .unwrap();
}

fn payload(json: serde_json::Value) -> Option<simvec::types::Payload> {
    Some(
        json.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    )
}

/// Single collection with two versions plus edge payloads for every filter kind.
fn seed_versions(store: &CollectionStore) {
    let dim = 4;
    // v1: ids 0..6, v2: ids 10..14, plus id 99 without payload.
    for i in 0..6u64 {
        let mut p = unit(i, dim);
        p.payload = payload(serde_json::json!({
            "version": "v1",
            "color": if i % 2 == 0 { "red" } else { "blue" },
            "price": (i as f64) * 10.0,
            "tags": if i % 3 == 0 { vec!["hot", "new"] } else { vec!["cold"] },
            "note": format!("item-{i}"),
        }));
        upsert_logged(store, &p);
    }
    for i in 10..15u64 {
        let mut p = unit(i, dim);
        p.payload = payload(serde_json::json!({
            "version": "v2",
            "color": "green",
            "price": 100.0 + (i as f64),
            "tags": vec!["hot"],
            "note": format!("item-{i}"),
        }));
        upsert_logged(store, &p);
    }
    upsert_logged(store, &unit(99, dim));
    // Nested + geo witness for v1 id 0.
    let mut p0 = store.get(&PointId::Num(0)).unwrap().unwrap();
    p0.payload = payload(serde_json::json!({
        "version": "v1",
        "color": "red",
        "price": 0.0,
        "tags": ["hot", "new"],
        "note": "item-0",
        "addresses": [{"city": "lyon"}, {"city": "paris"}],
        "location": {"lat": 48.8566, "lon": 2.3522},
    }));
    upsert_logged(store, &p0);
}

fn version_filter(version: &str) -> VectorFilter {
    VectorFilter::new().must(FilterCondition::match_value("version", version))
}

fn collect_all_scroll(
    store: &CollectionStore,
    filter: &VectorFilter,
    limit: usize,
) -> Vec<VectorPoint> {
    let mut offset: Option<PointId> = None;
    let mut all = Vec::new();
    loop {
        let (page, next) = store
            .scroll_by_filter(filter, limit, offset.as_ref(), Some(true), Some(false))
            .unwrap();
        let done_empty = page.is_empty();
        all.extend(page);
        offset = next;
        if done_empty || offset.is_none() {
            break;
        }
    }
    all
}

#[test]
fn filtered_count_covers_condition_kinds() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(dir.path().join("col"), "col", &config(4)).unwrap();
    seed_versions(&store);

    // Must / must_not / should / min_should combine with AND across groups.
    assert_eq!(
        store.count_by_filter(&version_filter("v1")).unwrap(),
        6,
        "v1 reconciliation count"
    );
    assert_eq!(store.count_by_filter(&version_filter("v2")).unwrap(), 5);
    assert_eq!(
        store.count_by_filter(&VectorFilter::new()).unwrap(),
        store.count(),
        "empty filter equals total"
    );
    assert_eq!(
        store
            .count_by_filter(
                &VectorFilter::new().must(FilterCondition::match_value("color", "red"))
            )
            .unwrap(),
        3,
        "v1 even ids are red"
    );
    assert_eq!(
        store
            .count_by_filter(
                &VectorFilter::new().must(FilterCondition::match_any(
                    "color",
                    vec![serde_json::json!("red"), serde_json::json!("green")]
                ))
            )
            .unwrap(),
        8
    );
    assert_eq!(
        store
            .count_by_filter(
                &VectorFilter::new().must(FilterCondition::range(
                    "price",
                    RangeCondition::new().gte(100.0)
                ))
            )
            .unwrap(),
        5,
        "v2 prices dominate"
    );
    assert_eq!(
        store
            .count_by_filter(&VectorFilter::new().must(FilterCondition::is_null("missing")))
            .unwrap(),
        store.count(),
        "absent field is null for every point"
    );
    assert_eq!(
        store
            .count_by_filter(
                &VectorFilter::new()
                    .must(FilterCondition::has_id(vec!["0".into(), "10".into()]))
            )
            .unwrap(),
        2
    );
    assert_eq!(
        store
            .count_by_filter(
                &VectorFilter::new().must(FilterCondition::values_count(
                    "tags",
                    ValuesCountCondition::new().gte(2)
                ))
            )
            .unwrap(),
        2,
        "only id 0 and v1 id 0/3 carry two tags"
    );
    assert_eq!(
        store
            .count_by_filter(
                &VectorFilter::new().must(FilterCondition::contains("note", "item-1"))
            )
            .unwrap(),
        6,
        "item-1 prefix matches 1, 10..14"
    );
    assert_eq!(
        store
            .count_by_filter(
                &VectorFilter::new().must(FilterCondition::new(
                    "note",
                    ConditionType::StartsWith {
                        prefix: "item-1".to_string()
                    }
                ))
            )
            .unwrap(),
        6
    );
    let nested = VectorFilter::new().must(FilterCondition::new(
        "addresses",
        ConditionType::Nested {
            filter: Box::new(
                VectorFilter::new().must(FilterCondition::match_value("city", "lyon")),
            ),
        },
    ));
    assert_eq!(store.count_by_filter(&nested).unwrap(), 1);
    let geo = VectorFilter::new().must(FilterCondition::new(
        "location",
        ConditionType::GeoRadius(GeoRadius::new(GeoPoint::new(48.8566, 2.3522), 1000.0)),
    ));
    assert_eq!(store.count_by_filter(&geo).unwrap(), 1);
    let negated = VectorFilter::new().must(FilterCondition::new(
        "color",
        ConditionType::Not {
            condition: Box::new(FilterCondition::match_value("color", "red")),
        },
    ));
    assert_eq!(
        store.count_by_filter(&negated).unwrap(),
        store.count() - 3,
        "negation counts non-red plus payload-less"
    );
}

#[test]
fn filtered_scroll_pagination_matches_full_scan() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(dir.path().join("col"), "col", &config(4)).unwrap();
    seed_versions(&store);
    let filter = version_filter("v2");

    let (single, _) = store
        .scroll_by_filter(&filter, 100, None, Some(true), Some(false))
        .unwrap();
    assert_eq!(single.len(), 5);
    let paged = collect_all_scroll(&store, &filter, 2);
    assert_eq!(paged.len(), 5);
    let single_ids: Vec<String> = single.iter().map(|p| p.id.to_string()).collect();
    let paged_ids: Vec<String> = paged.iter().map(|p| p.id.to_string()).collect();
    assert_eq!(single_ids, paged_ids, "pagination must equal full scan");
    assert_eq!(store.count_by_filter(&filter).unwrap(), 5);

    // Sparse filter: short page must not imply exhaustion.
    let sparse = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
    let (first, cursor) = store
        .scroll_by_filter(&sparse, 2, None, Some(true), Some(false))
        .unwrap();
    assert_eq!(first.len(), 2);
    assert!(cursor.is_some());
    let rest = collect_all_scroll(&store, &sparse, 2);
    assert_eq!(rest.len(), 3);

    // Trimming switches only affect serialization, never membership.
    let (trimmed, _) = store
        .scroll_by_filter(&filter, 100, None, Some(false), Some(false))
        .unwrap();
    assert_eq!(trimmed.len(), 5);
    assert!(trimmed.iter().all(|p| p.payload.is_none()));
    assert!(trimmed.iter().all(|p| p.vector.is_empty()));
}

#[test]
fn tombstones_excluded_and_delete_count_matches() {
    let dir = tempfile::tempdir().unwrap();
    let store = CollectionStore::create(dir.path().join("col"), "col", &config(4)).unwrap();
    seed_versions(&store);
    let filter = version_filter("v1");
    assert_eq!(store.count_by_filter(&filter).unwrap(), 6);

    store
        .apply_ops(&[WalRecord::Delete {
            point_id: "0".to_string(),
        }])
        .unwrap();
    assert_eq!(
        store.count_by_filter(&filter).unwrap(),
        5,
        "tombstoned point leaves the count"
    );
    let pages = collect_all_scroll(&store, &filter, 2);
    assert_eq!(pages.len(), 5);
    assert!(!pages.iter().any(|p| p.id.to_string() == "0"));

    // Differential check: count on a twin equals delete return on the original.
    let dir2 = tempfile::tempdir().unwrap();
    let twin = CollectionStore::create(dir2.path().join("col"), "col", &config(4)).unwrap();
    seed_versions(&twin);
    twin
        .apply_ops(&[WalRecord::Delete {
            point_id: "0".to_string(),
        }])
        .unwrap();
    let counted = twin.count_by_filter(&version_filter("v2")).unwrap();
    let deleted = twin.delete_by_filter(&version_filter("v2")).unwrap();
    assert_eq!(counted, deleted);
    assert_eq!(counted, 5);
    assert_eq!(twin.count_by_filter(&version_filter("v2")).unwrap(), 0);
}

#[test]
fn engine_wrappers_agree_with_store() {
    let dir = tempfile::tempdir().unwrap();
    let engine = simvec::LocalVectorEngine::open(dir.path().join("eng")).unwrap();
    engine.create_collection("docs", &config(4)).unwrap();
    for i in 0..6u64 {
        let mut p = unit(i, 4);
        p.payload = payload(serde_json::json!({"version": if i < 3 { "v1" } else { "v2" }}));
        engine.upsert("docs", p).unwrap();
    }
    let filter = version_filter("v1");
    assert_eq!(engine.count_by_filter("docs", &filter).unwrap(), 3);
    let (page, next) = engine
        .scroll_by_filter("docs", &filter, 2, None, Some(true), Some(false))
        .unwrap();
    assert_eq!(page.len(), 2);
    let (tail, _) = engine
        .scroll_by_filter("docs", &filter, 2, next.as_ref(), Some(true), Some(false))
        .unwrap();
    assert_eq!(tail.len(), 1);
    assert!(engine.count_by_filter("ghost", &filter).is_err());
}

#[cfg(feature = "wasm")]
#[test]
fn memory_form_parity_on_counts_and_id_sets() {
    use std::collections::HashSet;
    let memory = simvec::memory_engine::MemoryEngine::new();
    memory.create_collection("docs", &config(4)).unwrap();
    for i in 0..6u64 {
        let mut p = unit(i, 4);
        p.payload = payload(serde_json::json!({"version": if i < 3 { "v1" } else { "v2" }}));
        memory.upsert("docs", p).unwrap();
    }
    let filter = version_filter("v1");
    assert_eq!(memory.count_by_filter("docs", &filter).unwrap(), 3);
    let (all, _) = memory
        .scroll_by_filter("docs", &filter, 100, None, Some(true), Some(false))
        .unwrap();
    assert_eq!(all.len(), 3);
    let got: HashSet<String> = all.iter().map(|p| p.id.to_string()).collect();
    assert_eq!(
        got,
        HashSet::from(["0".to_string(), "1".to_string(), "2".to_string()])
    );
}

#[test]
fn async_trait_filtered_reads_match_sync() {
    use futures::StreamExt;
    use simvec::engine::{LocalVectorEngine, VectorEngine};
    let dir = tempfile::tempdir().unwrap();
    let engine = LocalVectorEngine::open(dir.path().join("eng")).unwrap();
    engine.create_collection("docs", &config(4)).unwrap();
    for i in 0..6u64 {
        let mut p = unit(i, 4);
        p.payload = payload(serde_json::json!({"version": if i < 3 { "v1" } else { "v2" }}));
        engine.upsert("docs", p).unwrap();
    }
    let filter = version_filter("v1");
    futures::executor::block_on(async {
        let counted = <LocalVectorEngine as VectorEngine>::count_by_filter(&engine, "docs", filter.clone())
            .await
            .unwrap();
        assert_eq!(counted, engine.count_by_filter("docs", &filter).unwrap());
        let (page, next) =
            <LocalVectorEngine as VectorEngine>::scroll_by_filter(&engine, "docs", filter.clone(), 2, None, Some(true), Some(false))
                .await
                .unwrap();
        assert_eq!(page.len(), 2);
        let (tail, _) = <LocalVectorEngine as VectorEngine>::scroll_by_filter(
            &engine,
            "docs",
            filter.clone(),
            2,
            next.as_ref(),
            Some(true),
            Some(false),
        )
        .await
        .unwrap();
        assert_eq!(tail.len(), 1);
        let mut stream = <LocalVectorEngine as VectorEngine>::scroll_by_filter_stream(
            &engine,
            "docs",
            filter.clone(),
            2,
            Some(true),
            Some(false),
        )
        .await
        .unwrap();
        let mut streamed = Vec::new();
        while let Some(item) = stream.next().await {
            streamed.push(item.unwrap());
        }
        assert_eq!(streamed.len(), 3);
        let mut plain = <LocalVectorEngine as VectorEngine>::scroll_stream(&engine, "docs", 2, Some(true), Some(false))
            .await
            .unwrap();
        let mut plain_points = Vec::new();
        while let Some(item) = plain.next().await {
            plain_points.push(item.unwrap());
        }
        assert_eq!(plain_points.len(), 6);
        assert!(<LocalVectorEngine as VectorEngine>::count_by_filter(&engine, "ghost", filter.clone())
            .await
            .is_err());
    });
}
