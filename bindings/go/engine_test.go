package simvec

import (
	"fmt"
	"sync"
	"testing"
)

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func flatConfig() map[string]any {
	return map[string]any{"vector_size": 4, "distance": "Cosine", "index": "Flat"}
}

func payloadOf(t *testing.T, point map[string]any) map[string]any {
	t.Helper()
	payload, ok := point["payload"].(map[string]any)
	if !ok {
		t.Fatalf("point payload has unexpected shape: %v", point["payload"])
	}
	return payload
}

func numberOf(t *testing.T, doc map[string]any, key string) float64 {
	t.Helper()
	value, ok := doc[key].(float64)
	if !ok {
		t.Fatalf("document field %q has unexpected shape: %v", key, doc[key])
	}
	return value
}

func TestLifecycle(t *testing.T) {
	engine, err := NewEngine(t.TempDir())
	must(t, err)
	defer func() { must(t, engine.Close()) }()

	must(t, engine.CreateCollection("docs", flatConfig()))
	exists, err := engine.CollectionExists("docs")
	must(t, err)
	if !exists {
		t.Fatal("collection should exist after creation")
	}
	names, err := engine.ListCollections()
	must(t, err)
	if len(names) != 1 || names[0] != "docs" {
		t.Fatalf("unexpected collection list: %v", names)
	}

	must(t, engine.Upsert("docs", "1", []float32{1, 0, 0, 0}, map[string]any{"title": "hello"}))
	must(t, engine.UpsertBatch("docs", []any{
		map[string]any{"id": "2", "vector": []float32{0, 1, 0, 0}, "payload": map[string]any{"title": "world"}},
	}))
	count, err := engine.Count("docs")
	must(t, err)
	if count != 2 {
		t.Fatalf("unexpected count: %d", count)
	}
	if engine.LastCode() != CodeOK {
		t.Fatalf("unexpected last code: %d", engine.LastCode())
	}

	hits, err := engine.SearchVector("docs", []float32{1, 0, 0, 0}, 5, nil, -1, true, false)
	must(t, err)
	if len(hits) != 2 {
		t.Fatalf("unexpected hit count: %d", len(hits))
	}
	if fmt.Sprintf("%v", hits[0]["id"]) != "1" {
		t.Fatalf("unexpected top hit: %v", hits[0]["id"])
	}
	if payloadOf(t, hits[0])["title"] != "hello" {
		t.Fatalf("unexpected top payload: %v", hits[0]["payload"])
	}

	queryHits, err := engine.Search("docs", map[string]any{"vector": []float32{1, 0, 0, 0}, "limit": 5})
	must(t, err)
	if len(queryHits) != 2 {
		t.Fatalf("unexpected query hit count: %d", len(queryHits))
	}

	point, err := engine.Get("docs", "1")
	must(t, err)
	vector, ok := point["vector"].([]any)
	if !ok || len(vector) != 4 {
		t.Fatalf("unexpected point vector: %v", point["vector"])
	}

	must(t, engine.MergePayload("docs", "1", map[string]any{"lang": "en"}))
	point, err = engine.Get("docs", "1")
	must(t, err)
	if payloadOf(t, point)["lang"] != "en" {
		t.Fatalf("merge did not apply: %v", point["payload"])
	}
	must(t, engine.SetPayloadField("docs", "1", "topic", "intro"))
	point, err = engine.Get("docs", "1")
	must(t, err)
	if payloadOf(t, point)["topic"] != "intro" {
		t.Fatalf("field write did not apply: %v", point["payload"])
	}
	must(t, engine.DeletePayloadKeys("docs", "1", []string{"topic"}))
	point, err = engine.Get("docs", "1")
	must(t, err)
	if _, found := payloadOf(t, point)["topic"]; found {
		t.Fatalf("field delete did not apply: %v", point["payload"])
	}

	must(t, engine.CreatePayloadIndex("docs", "title", "keyword"))
	indexes, err := engine.ListPayloadIndexes("docs")
	must(t, err)
	if len(indexes) != 1 || indexes[0] != (PayloadIndex{Field: "title", Schema: "keyword"}) {
		t.Fatalf("unexpected payload indexes: %v", indexes)
	}
	removed, err := engine.DeletePayloadIndex("docs", "title")
	must(t, err)
	if !removed {
		t.Fatal("expected payload index deletion to report a hit")
	}
	removed, err = engine.DeletePayloadIndex("docs", "title")
	must(t, err)
	if removed {
		t.Fatal("repeated payload index deletion should report a miss")
	}

	page, err := engine.Scroll("docs", 10, nil, true, false)
	must(t, err)
	if len(page.Points) != 2 {
		t.Fatalf("unexpected page size: %d", len(page.Points))
	}
	if page.NextOffset == nil || *page.NextOffset != "2" {
		t.Fatalf("unexpected page cursor: %v", page.NextOffset)
	}
	tail, err := engine.Scroll("docs", 10, page.NextOffset, true, false)
	must(t, err)
	if len(tail.Points) != 0 {
		t.Fatalf("expected empty tail page, got %d points", len(tail.Points))
	}

	info, err := engine.CollectionInfo("docs")
	must(t, err)
	if numberOf(t, info, "vector_count") != 2 {
		t.Fatalf("unexpected collection info: %v", info)
	}
	config, err := engine.CollectionConfig("docs")
	must(t, err)
	if numberOf(t, config, "vector_size") != 4 {
		t.Fatalf("unexpected collection config: %v", config)
	}
	metrics, err := engine.CollectionMetrics("docs")
	must(t, err)
	if metrics == nil {
		t.Fatal("expected collection metrics document")
	}

	indexed, err := engine.HasIndex("docs")
	must(t, err)
	if indexed {
		t.Fatal("index should not exist before building")
	}
	quantized, err := engine.HasQuantization("docs")
	must(t, err)
	if quantized {
		t.Fatal("quantization should not exist before building")
	}
	if _, err := engine.BuildIndex("docs"); err != nil {
		t.Fatalf("build index failed: %v", err)
	}
	if _, err := engine.BuildQuantization("docs"); err != nil {
		t.Fatalf("build quantization failed: %v", err)
	}
	must(t, engine.DrainPending("docs"))
	must(t, engine.DropPublishedIndex("docs"))
	must(t, engine.DropIndex("docs"))
	must(t, engine.RunMaintenanceSweep())

	must(t, engine.Delete("docs", "2"))
	count, err = engine.Count("docs")
	must(t, err)
	if count != 1 {
		t.Fatalf("unexpected count after delete: %d", count)
	}
	if _, err := engine.Get("docs", "2"); !IsNotFound(err) {
		t.Fatalf("expected not-found error, got %v", err)
	}
	if engine.LastCode() == CodeOK {
		t.Fatal("expected nonzero last code after failure")
	}
	if engine.LastMessage() == "" {
		t.Fatal("expected last message after failure")
	}

	must(t, engine.DeleteCollection("docs"))
	exists, err = engine.CollectionExists("docs")
	must(t, err)
	if exists {
		t.Fatal("collection should be gone after deletion")
	}
}

func TestBatchAndFilteredDeletes(t *testing.T) {
	engine, err := NewEngine(t.TempDir())
	must(t, err)
	defer func() { must(t, engine.Close()) }()

	must(t, engine.CreateCollection("batch", map[string]any{"vector_size": 3, "distance": "Cosine", "index": "Flat"}))
	must(t, engine.UpsertBatch("batch", []any{
		map[string]any{"id": "a", "vector": []float32{1, 0, 0}, "payload": map[string]any{"color": "red"}},
		map[string]any{"id": "b", "vector": []float32{0, 1, 0}, "payload": map[string]any{"color": "blue"}},
		map[string]any{"id": "c", "vector": []float32{0, 0, 1}, "payload": map[string]any{"color": "red"}},
	}))
	count, err := engine.Count("batch")
	must(t, err)
	if count != 3 {
		t.Fatalf("unexpected count: %d", count)
	}

	must(t, engine.ApplyTxn([]any{
		map[string]any{"op": "upsert", "collection": "batch", "point": map[string]any{"id": "d", "vector": []float32{1, 1, 0}}},
		map[string]any{"op": "delete", "collection": "batch", "id": "d"},
	}))
	count, err = engine.Count("batch")
	must(t, err)
	if count != 3 {
		t.Fatalf("unexpected count after txn: %d", count)
	}

	deleted, err := engine.DeleteByFilter("batch", map[string]any{
		"must": []any{
			map[string]any{"field": "color", "condition": map[string]any{"Match": map[string]any{"value": "red"}}},
		},
	})
	must(t, err)
	if deleted != 2 {
		t.Fatalf("unexpected filtered delete count: %d", deleted)
	}
	count, err = engine.Count("batch")
	must(t, err)
	if count != 1 {
		t.Fatalf("unexpected count after filtered delete: %d", count)
	}

	must(t, engine.DeleteBatch("batch", []string{"b"}))
	count, err = engine.Count("batch")
	must(t, err)
	if count != 0 {
		t.Fatalf("unexpected count after batch delete: %d", count)
	}
	if _, err := engine.Compact("batch"); err != nil {
		t.Fatalf("compact failed: %v", err)
	}

	must(t, engine.DeleteCollection("batch"))
	if _, err := engine.Get("batch", "b"); !IsNotFound(err) {
		t.Fatalf("expected not-found error, got %v", err)
	}
}

func TestPromoteAndDefaults(t *testing.T) {
	engine, err := NewEngine(t.TempDir())
	must(t, err)
	defer func() { must(t, engine.Close()) }()

	must(t, engine.CreateCollection("live", map[string]any{"vector_size": 2, "distance": "Cosine", "index": "Flat"}))
	must(t, engine.Upsert("live", "1", []float32{1, 0}, nil))
	must(t, engine.CreateCollection("live.rebuild-7", map[string]any{"vector_size": 2, "distance": "Cosine", "index": "Flat"}))
	must(t, engine.Upsert("live.rebuild-7", "2", []float32{0, 1}, nil))
	temps, err := engine.ListTempCollections()
	must(t, err)
	if len(temps) != 1 || temps[0] != "live.rebuild-7" {
		t.Fatalf("unexpected temp collections: %v", temps)
	}

	must(t, engine.PromoteTempCollection("live", "live.rebuild-7", ".old-7"))
	point, err := engine.Get("live", "2")
	must(t, err)
	if len(point["vector"].([]any)) != 2 {
		t.Fatalf("unexpected promoted point: %v", point)
	}
	backups, err := engine.ListOldBackups("live")
	must(t, err)
	if len(backups) != 1 || backups[0] != "live.old-7" {
		t.Fatalf("unexpected backups: %v", backups)
	}
	backup, err := engine.RestorePromoteBackup("live")
	must(t, err)
	if backup != "live.old-7" {
		t.Fatalf("unexpected restored backup: %q", backup)
	}
	point, err = engine.Get("live", "1")
	must(t, err)
	if len(point["vector"].([]any)) != 2 {
		t.Fatalf("unexpected restored point: %v", point)
	}

	hnsw, err := engine.GetDefaultHnswConfig()
	must(t, err)
	if hnsw != nil {
		t.Fatalf("expected unset hnsw default, got %v", hnsw)
	}
	ivf, err := engine.GetDefaultIvfConfig()
	must(t, err)
	if ivf != nil {
		t.Fatalf("expected unset ivf default, got %v", ivf)
	}
	quant, err := engine.GetDefaultQuantizationConfig()
	must(t, err)
	if quant != nil {
		t.Fatalf("expected unset quantization default, got %v", quant)
	}

	must(t, engine.SetDefaultHnswConfig(map[string]any{"m": 8, "ef_construct": 32}))
	must(t, engine.SetDefaultIvfConfig(map[string]any{
		"lists":                nil,
		"min_build_points":     1000,
		"sample_limit":         1024,
		"kmeans_max_iter":      10,
		"drift_threshold":      0.1,
		"drift_check_interval": 1000,
		"default_nprobe":       4,
		"auto_promotion":       false,
		"max_probes":           nil,
	}))
	must(t, engine.SetDefaultQuantizationConfig(map[string]any{"quant_type": map[string]any{"Binary": map[string]any{"always_ram": true}}}))
	hnsw, err = engine.GetDefaultHnswConfig()
	must(t, err)
	if numberOf(t, hnsw, "m") != 8 {
		t.Fatalf("unexpected hnsw default: %v", hnsw)
	}
	ivf, err = engine.GetDefaultIvfConfig()
	must(t, err)
	if ivf == nil {
		t.Fatal("expected ivf default after setting")
	}
	quant, err = engine.GetDefaultQuantizationConfig()
	must(t, err)
	if quant == nil {
		t.Fatal("expected quantization default after setting")
	}
}

func TestErrorsAndClose(t *testing.T) {
	engine, err := NewEngine(t.TempDir())
	must(t, err)

	must(t, engine.CreateCollection("docs", flatConfig()))
	if err := engine.CreateCollection("docs", flatConfig()); !IsAlreadyExists(err) {
		t.Fatalf("expected already-exists error, got %v", err)
	} else if CodeOf(err) != CodeAlreadyExists {
		t.Fatalf("unexpected status code: %d", CodeOf(err))
	}
	if _, err := engine.Get("docs", "missing"); !IsNotFound(err) {
		t.Fatalf("expected not-found error, got %v", err)
	}

	must(t, engine.Close())
	if !engine.Closed() {
		t.Fatal("engine should report closed")
	}
	must(t, engine.Close())
	if err := engine.CreateCollection("docs", flatConfig()); !IsMisuse(err) {
		t.Fatalf("expected misuse on closed engine, got %v", err)
	}
	if engine.LastCode() != CodeMisuse {
		t.Fatalf("unexpected last code on closed engine: %d", engine.LastCode())
	}
	if engine.LastMessage() != "engine is closed" {
		t.Fatalf("unexpected last message on closed engine: %q", engine.LastMessage())
	}
}

func TestConcurrentUse(t *testing.T) {
	engine, err := NewEngine(t.TempDir())
	must(t, err)
	defer func() { must(t, engine.Close()) }()

	must(t, engine.CreateCollection("shared", flatConfig()))

	const writers = 8
	const perWriter = 10
	var group sync.WaitGroup
	failures := make(chan error, writers*2)
	for writer := 0; writer < writers; writer++ {
		group.Add(1)
		go func(slot int) {
			defer group.Done()
			for i := 0; i < perWriter; i++ {
				id := fmt.Sprintf("g%d-p%d", slot, i)
				vector := []float32{float32(i + 1), float32(slot + 1), 0.5, 0.25}
				if err := engine.Upsert("shared", id, vector, nil); err != nil {
					failures <- err
					return
				}
			}
			if _, err := engine.SearchVector("shared", []float32{1, 0, 0, 0}, 5, nil, -1, false, false); err != nil {
				failures <- err
			}
		}(writer)
	}
	group.Wait()
	close(failures)
	for failure := range failures {
		t.Fatalf("concurrent use failed: %v", failure)
	}
	count, err := engine.Count("shared")
	must(t, err)
	if count != writers*perWriter {
		t.Fatalf("unexpected count after concurrent writes: %d", count)
	}
}
