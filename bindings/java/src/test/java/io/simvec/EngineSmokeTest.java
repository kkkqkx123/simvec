package io.simvec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Smoke coverage for the JNI binding, mirroring the other bindings. */
class EngineSmokeTest {
    private static Map<String, Object> flatConfig(int dims) {
        return Map.of("vector_size", dims, "distance", "Cosine", "index", "Flat");
    }

    private static Map<String, Object> point(String id, float[] vector, Map<String, Object> payload) {
        return Map.of("id", id, "vector", toList(vector), "payload", payload);
    }

    private static List<Float> toList(float[] vector) {
        List<Float> out = new ArrayList<>(vector.length);
        for (float v : vector) {
            out.add(v);
        }
        return out;
    }

    @Test
    void lifecycle(@TempDir Path root) {
        try (Engine engine = new Engine(root.toString())) {
            engine.createCollection("docs", flatConfig(4));
            assertTrue(engine.collectionExists("docs"));
            assertEquals(List.of("docs"), engine.listCollections());

            engine.upsert("docs", "1", new float[]{1, 0, 0, 0}, Map.of("title", "hello"));
            engine.upsertBatch("docs", List.of(
                    point("2", new float[]{0, 1, 0, 0}, Map.of("title", "world"))));
            assertEquals(0, engine.lastCode());
            assertEquals(2, engine.count("docs"));

            List<Map<String, Object>> hits =
                    engine.searchVector("docs", new float[]{1, 0, 0, 0}, 5);
            assertEquals(2, hits.size());
            assertEquals("1", String.valueOf(hits.get(0).get("id")));

            List<Map<String, Object>> queryHits = engine.search("docs",
                    Map.of("vector", toList(new float[]{1, 0, 0, 0}), "limit", 5));
            assertEquals(2, queryHits.size());

            Map<String, Object> fetched = engine.get("docs", "1");
            assertNotNull(fetched.get("vector"));

            engine.mergePayload("docs", "1", Map.of("lang", "en"));
            assertEquals("en", payloadOf(engine, "1").get("lang"));
            engine.setPayloadField("docs", "1", "topic", "intro");
            assertEquals("intro", payloadOf(engine, "1").get("topic"));
            engine.deletePayloadKeys("docs", "1", List.of("topic"));
            assertFalse(payloadOf(engine, "1").containsKey("topic"));

            engine.createPayloadIndex("docs", "title", "keyword");
            assertEquals(List.of(new Engine.PayloadIndex("title", "keyword")),
                    engine.listPayloadIndexes("docs"));
            assertTrue(engine.deletePayloadIndex("docs", "title"));
            assertFalse(engine.deletePayloadIndex("docs", "title"));

            Engine.ScrollPage page = engine.scroll("docs", 10);
            assertEquals(2, page.points().size());
            Engine.ScrollPage tail = engine.scroll("docs", 10, page.nextOffset(), true, false);
            assertEquals(0, tail.points().size());

            assertEquals(2L, numberOf(engine.collectionInfo("docs"), "vector_count"));
            assertEquals(4L, numberOf(engine.collectionConfig("docs"), "vector_size"));
            assertNotNull(engine.collectionMetrics("docs"));

            assertFalse(engine.hasIndex("docs"));
            assertFalse(engine.hasQuantization("docs"));
            engine.buildIndex("docs");
            engine.buildQuantization("docs");
            engine.drainPending("docs");
            engine.dropPublishedIndex("docs");
            engine.dropIndex("docs");
            engine.runMaintenanceSweep();

            engine.delete("docs", "2");
            assertEquals(1, engine.count("docs"));
            NotFoundError thrown = assertThrows(NotFoundError.class, () -> engine.get("docs", "2"));
            assertEquals(12, thrown.getCode());
            assertTrue(engine.lastCode() != 0);
            assertFalse(engine.lastMessage().isEmpty());

            engine.deleteCollection("docs");
            assertFalse(engine.collectionExists("docs"));
        }
    }

    @Test
    void batchAndFilteredDeletes(@TempDir Path root) {
        try (Engine engine = new Engine(root.toString())) {
            engine.createCollection("batch", flatConfig(3));
            engine.upsertBatch("batch", List.of(
                    point("a", new float[]{1, 0, 0}, Map.of("color", "red")),
                    point("b", new float[]{0, 1, 0}, Map.of("color", "blue")),
                    point("c", new float[]{0, 0, 1}, Map.of("color", "red"))));
            assertEquals(3, engine.count("batch"));

            engine.applyTxn(List.of(
                    Map.of("op", "upsert", "collection", "batch",
                            "point", Map.of("id", "d", "vector", toList(new float[]{1, 1, 0}))),
                    Map.of("op", "delete", "collection", "batch", "id", "d")));
            assertEquals(3, engine.count("batch"));

            long deleted = engine.deleteByFilter("batch",
                    Map.of("must", List.of(Map.of("field", "color",
                            "condition", Map.of("Match", Map.of("value", "red"))))));
            assertEquals(2, deleted);
            assertEquals(1, engine.count("batch"));

            engine.deleteBatch("batch", List.of("b"));
            assertEquals(0, engine.count("batch"));
            engine.compact("batch");

            engine.deleteCollection("batch");
            assertThrows(NotFoundError.class, () -> engine.get("batch", "b"));
        }
    }

    @Test
    void promoteAndDefaults(@TempDir Path root) {
        try (Engine engine = new Engine(root.toString())) {
            engine.createCollection("live", flatConfig(2));
            engine.upsert("live", "1", new float[]{1, 0}, null);
            engine.createCollection("live.rebuild-7", flatConfig(2));
            engine.upsert("live.rebuild-7", "2", new float[]{0, 1}, null);
            assertEquals(List.of("live.rebuild-7"), engine.listTempCollections());

            engine.promoteTempCollection("live", "live.rebuild-7", ".old-7");
            assertEquals(List.of(0.0, 1.0), engine.get("live", "2").get("vector"));
            assertEquals(List.of("live.old-7"), engine.listOldBackups("live"));
            assertEquals("live.old-7", engine.restorePromoteBackup("live"));
            assertEquals(List.of(1.0, 0.0), engine.get("live", "1").get("vector"));

            assertNull(engine.getDefaultHnswConfig());
            assertNull(engine.getDefaultIvfConfig());
            assertNull(engine.getDefaultQuantizationConfig());
            engine.setDefaultHnswConfig(Map.of("m", 8, "ef_construct", 32));
            engine.setDefaultIvfConfig(Map.of(
                    "min_build_points", 1000,
                    "sample_limit", 1024,
                    "kmeans_max_iter", 10,
                    "drift_threshold", 0.1,
                    "drift_check_interval", 1000,
                    "default_nprobe", 4,
                    "auto_promotion", false));
            engine.setDefaultQuantizationConfig(
                    Map.of("quant_type", Map.of("Binary", Map.of("always_ram", true))));
            assertEquals(8L, numberOf(asMap(engine.getDefaultHnswConfig()), "m"));
            assertNotNull(engine.getDefaultIvfConfig());
            assertNotNull(engine.getDefaultQuantizationConfig());

            // Repeatable close and use-after-close misuse.
            engine.close();
            engine.close();
            assertTrue(engine.isClosed());
            assertThrows(MisuseError.class, () -> engine.count("live"));
        }
    }

    @Test
    void concurrentUse(@TempDir Path root) throws Exception {
        try (Engine engine = new Engine(root.toString())) {
            engine.createCollection("race", flatConfig(4));
            ExecutorService pool = Executors.newFixedThreadPool(8);
            try {
                List<Future<?>> pending = new ArrayList<>();
                for (int i = 0; i < 32; i++) {
                    final int n = i;
                    pending.add(pool.submit(() -> {
                        engine.upsert("race", "p" + n, new float[]{1, 0, 0, 0}, null);
                        engine.searchVector("race", new float[]{1, 0, 0, 0}, 5);
                        return null;
                    }));
                }
                for (Future<?> task : pending) {
                    task.get(30, TimeUnit.SECONDS);
                }
            } finally {
                pool.shutdownNow();
            }
            assertEquals(32, engine.count("race"));
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> payloadOf(Engine engine, String id) {
        return (Map<String, Object>) engine.get("docs", id).get("payload");
    }

    private static long numberOf(Map<String, Object> doc, String key) {
        return ((Number) doc.get(key)).longValue();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object value) {
        return (Map<String, Object>) value;
    }

}
