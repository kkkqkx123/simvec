use std::collections::HashMap;

use crate::types::*;

#[test]
fn test_search_query_new() {
    let q = SearchQuery::new(vec![1.0, 2.0], 10);
    assert_eq!(q.primary_vector(), vec![1.0, 2.0]);
    assert_eq!(q.limit, 10);
    assert_eq!(q.with_payload, Some(true));
    assert!(q.with_vector.is_none());
}

#[test]
fn test_search_query_with_offset() {
    let q = SearchQuery::new(vec![1.0], 10).with_offset(5);
    assert_eq!(q.offset, Some(5));
}

#[test]
fn test_search_query_with_score_threshold() {
    let q = SearchQuery::new(vec![1.0], 10).with_score_threshold(0.5);
    assert_eq!(q.score_threshold, Some(0.5));
}

#[test]
fn test_search_query_with_payload() {
    let q = SearchQuery::new(vec![1.0], 10).with_payload(false);
    assert_eq!(q.with_payload, Some(false));
}

#[test]
fn test_search_query_with_vector() {
    let q = SearchQuery::new(vec![1.0], 10).with_vector(true);
    assert_eq!(q.with_vector, Some(true));
}

#[test]
fn test_search_query_with_nprobe() {
    let q = SearchQuery::new(vec![1.0], 10).with_nprobe(64);
    assert_eq!(q.nprobe, Some(64));
}

#[test]
fn test_search_query_effective_limit_default() {
    let q = SearchQuery::new(vec![1.0], 10);
    assert_eq!(q.effective_limit(), 10);
}

#[test]
fn test_search_query_effective_limit_topk() {
    let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::TopK(5));
    assert_eq!(q.effective_limit(), 5);
}

#[test]
fn test_search_query_effective_limit_knn() {
    let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::KNN {
        k: 20,
        ef_search: Some(100),
    });
    assert_eq!(q.effective_limit(), 20);
}

#[test]
fn test_search_query_effective_limit_range() {
    let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::Range {
        radius: 0.5,
        max_results: Some(30),
    });
    assert_eq!(q.effective_limit(), 30);
}

#[test]
fn test_search_query_hnsw_ef_default() {
    let q = SearchQuery::new(vec![1.0], 10);
    assert!(q.hnsw_ef().is_none());
}

#[test]
fn test_search_query_hnsw_ef_knn() {
    let q = SearchQuery::new(vec![1.0], 10).with_knn(5, Some(128));
    assert_eq!(q.hnsw_ef(), Some(128));
}

#[test]
fn test_search_query_hnsw_ef_topk() {
    let q = SearchQuery::new(vec![1.0], 10).with_search_mode(SearchMode::TopK(5));
    assert!(q.hnsw_ef().is_none());
}

#[test]
fn test_search_query_with_knn_sets_limit() {
    let q = SearchQuery::new(vec![1.0], 10).with_knn(42, None);
    assert_eq!(q.limit, 42);
    assert!(matches!(q.search_mode, Some(SearchMode::KNN { k: 42, .. })));
}

#[test]
fn test_search_query_with_range_sets_limit() {
    let q = SearchQuery::new(vec![1.0], 10).with_range(0.3, Some(25));
    assert_eq!(q.limit, 25);
    assert_eq!(q.score_threshold, Some(0.3));
}

#[test]
fn test_search_query_with_range_no_max_keeps_limit() {
    let q = SearchQuery::new(vec![1.0], 10).with_range(0.3, None);
    assert_eq!(q.limit, 10);
}

#[test]
fn test_search_result_new() {
    let r = SearchResult::new(42u64, 0.95);
    assert_eq!(r.id, PointId::Num(42));
    assert!((r.score - 0.95).abs() < f32::EPSILON);
    assert!(r.payload.is_none());
    assert!(r.vector.is_none());
}

#[test]
fn test_search_result_with_payload() {
    let mut payload = HashMap::new();
    payload.insert("key".into(), serde_json::json!("val"));
    let r = SearchResult::new("1", 0.5).with_payload(payload);
    assert!(r.payload.is_some());
}

#[test]
fn test_search_result_with_vector() {
    let r = SearchResult::new("id", 0.1).with_vector(vec![1.0, 2.0]);
    assert_eq!(r.vector, Some(vec![1.0, 2.0]));
}

#[test]
fn test_search_results_empty() {
    let r = SearchResults::new(vec![]);
    assert!(r.is_empty());
    assert_eq!(r.len(), 0);
}

#[test]
fn test_search_results_non_empty() {
    let results = vec![SearchResult::new(1u64, 0.9), SearchResult::new(2u64, 0.8)];
    let r = SearchResults::new(results);
    assert!(!r.is_empty());
    assert_eq!(r.len(), 2);
    assert_eq!(r.total, Some(2));
}

#[test]
fn test_search_results_from_vec() {
    let results = vec![SearchResult::new(1u64, 0.9)];
    let r: SearchResults = results.into();
    assert_eq!(r.len(), 1);
}

#[test]
fn test_batch_search_query_new() {
    let q = SearchQuery::new(vec![1.0], 5);
    let batch = BatchSearchQuery::new(vec![q]);
    assert_eq!(batch.queries.len(), 1);
}

#[test]
fn test_batch_search_query_from() {
    let q = SearchQuery::new(vec![1.0], 5);
    let batch: BatchSearchQuery = vec![q].into();
    assert_eq!(batch.queries.len(), 1);
}

#[test]
fn test_search_mode_topk() {
    let mode = SearchMode::TopK(10);
    match mode {
        SearchMode::TopK(k) => assert_eq!(k, 10),
        _ => panic!("expected TopK"),
    }
}

#[test]
fn test_search_mode_knn() {
    let mode = SearchMode::KNN {
        k: 20,
        ef_search: Some(200),
    };
    match mode {
        SearchMode::KNN { k, ef_search } => {
            assert_eq!(k, 20);
            assert_eq!(ef_search, Some(200));
        }
        _ => panic!("expected KNN"),
    }
}

#[test]
fn test_search_mode_range() {
    let mode = SearchMode::Range {
        radius: 0.7,
        max_results: Some(50),
    };
    match mode {
        SearchMode::Range {
            radius,
            max_results,
        } => {
            assert!((radius - 0.7).abs() < f32::EPSILON);
            assert_eq!(max_results, Some(50));
        }
        _ => panic!("expected Range"),
    }
}

#[test]
fn test_search_query_score_threshold_range_mode() {
    let q = SearchQuery::new(vec![1.0], 10).with_range(0.5, None);
    assert_eq!(q.score_threshold(), Some(0.5));
}

#[test]
fn test_search_query_score_threshold_default() {
    let q = SearchQuery::new(vec![1.0], 10);
    assert!(q.score_threshold().is_none());
}
