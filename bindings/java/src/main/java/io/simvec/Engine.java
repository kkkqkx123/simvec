package io.simvec;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.lang.ref.Cleaner;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Owner of one native engine handle rooted at a directory.
 *
 * <p>Configs, filters, payloads and queries are plain maps or lists serialized
 * to the JSON shapes the Rust side already understands. Vectors cross as
 * contiguous float arrays; {@code double[]} overloads copy with narrowing
 * conversion. Point ids are strings; convert integer ids with
 * {@code String.valueOf} before calling.
 *
 * <p>Methods are synchronized: instances are safe for concurrent use, with the
 * cost that concurrent searches serialize. {@link #close()} is idempotent, and
 * a cleaner closes engines the caller forgot to close, but explicit closing
 * (preferably try-with-resources) is still expected.
 */
public final class Engine implements AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final TypeReference<List<String>> STRING_LIST = new TypeReference<>() {
    };
    private static final TypeReference<Map<String, Object>> STRING_MAP = new TypeReference<>() {
    };
    private static final TypeReference<List<Map<String, Object>>> MAP_LIST = new TypeReference<>() {
    };

    /** One paginated scan result. A null next offset starts from the beginning. */
    public record ScrollPage(List<Map<String, Object>> points, String nextOffset) {
    }

    /** One payload index declaration on a collection. */
    public record PayloadIndex(String field, String schema) {
    }

    private final Cleaner.Cleanable cleanable;
    private final CloseState closeState;
    private long handle;
    private boolean closed;

    private static final class CloseState implements Runnable {
        private long handle;

        CloseState(long handle) {
            this.handle = handle;
        }

        @Override
        public void run() {
            if (handle != 0) {
                NativeLib.nClose(handle);
                handle = 0;
            }
        }
    }

    public Engine(String dirPath) {
        NativeLib.ensureLoaded();
        long[] out = new long[1];
        int code = NativeLib.nOpen(utf8(dirPath), out);
        if (code != SimvecError.OK) {
            throw SimvecError.raiseForCode(code, text(NativeLib.nLastErrorText()));
        }
        this.handle = out[0];
        this.closeState = new CloseState(out[0]);
        this.cleanable = CLEANER.register(this, closeState);
    }

    /** Idempotent close; safe to call more than once. */
    @Override
    public synchronized void close() {
        if (closed) {
            return;
        }
        closed = true;
        handle = 0;
        cleanable.clean();
    }

    /** Whether the engine has been closed. */
    public synchronized boolean isClosed() {
        return closed;
    }

    /** Most recent status code recorded on the native handle. */
    public synchronized int lastCode() {
        if (closed) {
            return SimvecError.MISUSE;
        }
        return NativeLib.nErrcode(handle);
    }

    /** Most recent error text recorded on the native handle. */
    public synchronized String lastMessage() {
        if (closed) {
            return "engine is closed";
        }
        return text(NativeLib.nLastError(handle));
    }

    // -- collections --

    public synchronized void createCollection(String name, Object config) {
        long h = requireOpen();
        check(NativeLib.nCreateCollection(h, utf8(name), json(config)), h);
    }

    public synchronized void deleteCollection(String name) {
        long h = requireOpen();
        check(NativeLib.nDeleteCollection(h, utf8(name)), h);
    }

    public synchronized boolean collectionExists(String name) {
        long h = requireOpen();
        int[] out = new int[1];
        check(NativeLib.nCollectionExists(h, utf8(name), out), h);
        return out[0] != 0;
    }

    public synchronized List<String> listCollections() {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nListCollections(h, out), h);
        return readList(out[0]);
    }

    public synchronized Map<String, Object> collectionInfo(String name) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nCollectionInfo(h, utf8(name), out), h);
        return readMap(out[0]);
    }

    public synchronized Map<String, Object> collectionConfig(String name) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nCollectionConfig(h, utf8(name), out), h);
        return readMap(out[0]);
    }

    public synchronized void promoteTempCollection(String live, String temp, String backupSuffix) {
        long h = requireOpen();
        check(NativeLib.nPromoteTempCollection(h, utf8(live), utf8(temp), utf8(backupSuffix)), h);
    }

    public synchronized String restorePromoteBackup(String live) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nRestorePromoteBackup(h, utf8(live), out), h);
        return readString(out[0]);
    }

    public synchronized List<String> listTempCollections() {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nListTempCollections(h, out), h);
        return readList(out[0]);
    }

    public synchronized List<String> listOldBackups(String live) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nListOldBackups(h, utf8(live), out), h);
        return readList(out[0]);
    }

    // -- writes --

    public synchronized void upsert(String collection, String id, float[] vector, Object payload) {
        long h = requireOpen();
        check(NativeLib.nUpsert(h, utf8(collection), utf8(id), vector, json(payload)), h);
    }

    /** Narrowing overload: doubles are cast to float element-wise. */
    public synchronized void upsert(String collection, String id, double[] vector, Object payload) {
        upsert(collection, id, toFloat32(vector), payload);
    }

    public synchronized void upsertBatch(String collection, Object points) {
        long h = requireOpen();
        check(NativeLib.nUpsertBatch(h, utf8(collection), json(points)), h);
    }

    public synchronized void delete(String collection, String id) {
        long h = requireOpen();
        check(NativeLib.nDelete(h, utf8(collection), utf8(id)), h);
    }

    public synchronized void deleteBatch(String collection, List<String> ids) {
        long h = requireOpen();
        check(NativeLib.nDeleteBatch(h, utf8(collection), json(ids)), h);
    }

    public synchronized long deleteByFilter(String collection, Object filter) {
        long h = requireOpen();
        long[] out = new long[1];
        check(NativeLib.nDeleteByFilter(h, utf8(collection), json(filter), out), h);
        return out[0];
    }

    public synchronized void setPayload(String collection, String id, Object payload) {
        long h = requireOpen();
        check(NativeLib.nSetPayload(h, utf8(collection), utf8(id), json(payload)), h);
    }

    public synchronized void mergePayload(String collection, String id, Object fields) {
        long h = requireOpen();
        check(NativeLib.nMergePayload(h, utf8(collection), utf8(id), json(fields)), h);
    }

    public synchronized void deletePayloadKeys(String collection, String id, List<String> keys) {
        long h = requireOpen();
        check(NativeLib.nDeletePayloadKeys(h, utf8(collection), utf8(id), json(keys)), h);
    }

    public synchronized void setPayloadField(String collection, String id, String key, Object value) {
        long h = requireOpen();
        check(NativeLib.nSetPayloadField(h, utf8(collection), utf8(id), utf8(key), json(value)), h);
    }

    public synchronized void applyTxn(Object ops) {
        long h = requireOpen();
        check(NativeLib.nApplyTxn(h, json(ops)), h);
    }

    public synchronized void createPayloadIndex(String collection, String field, String schema) {
        long h = requireOpen();
        check(NativeLib.nCreatePayloadIndex(h, utf8(collection), utf8(field), utf8(schema)), h);
    }

    public synchronized boolean deletePayloadIndex(String collection, String field) {
        long h = requireOpen();
        int[] out = new int[1];
        check(NativeLib.nDeletePayloadIndex(h, utf8(collection), utf8(field), out), h);
        return out[0] != 0;
    }

    public synchronized List<PayloadIndex> listPayloadIndexes(String collection) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nListPayloadIndexes(h, utf8(collection), out), h);
        List<Map<String, Object>> raw = readMapList(out[0]);
        List<PayloadIndex> indexes = new ArrayList<>(raw.size());
        for (Map<String, Object> entry : raw) {
            indexes.add(new PayloadIndex(String.valueOf(entry.get("field")),
                    String.valueOf(entry.get("schema"))));
        }
        return indexes;
    }

    // -- reads --

    public synchronized List<Map<String, Object>> search(String collection, Object query) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nSearch(h, utf8(collection), json(query), out), h);
        return readMapList(out[0]);
    }

    public synchronized List<Map<String, Object>> searchVector(String collection, float[] vector,
            long limit) {
        return searchVector(collection, vector, limit, null, -1.0f, true, false);
    }

    public synchronized List<Map<String, Object>> searchVector(String collection, double[] vector,
            long limit) {
        return searchVector(collection, toFloat32(vector), limit, null, -1.0f, true, false);
    }

    public synchronized List<Map<String, Object>> searchVector(String collection, float[] vector,
            long limit, Object filter, float scoreThreshold, boolean withPayload,
            boolean withVector) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nSearchVector(h, utf8(collection), vector, limit, json(filter),
                scoreThreshold, withPayload, withVector, out), h);
        return readMapList(out[0]);
    }

    public synchronized List<Map<String, Object>> searchVector(String collection, double[] vector,
            long limit, Object filter, float scoreThreshold, boolean withPayload,
            boolean withVector) {
        return searchVector(collection, toFloat32(vector), limit, filter, scoreThreshold,
                withPayload, withVector);
    }

    public synchronized Map<String, Object> get(String collection, String id) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nGet(h, utf8(collection), utf8(id), out), h);
        return readMap(out[0]);
    }

    public synchronized long count(String collection) {
        long h = requireOpen();
        long[] out = new long[1];
        check(NativeLib.nCount(h, utf8(collection), out), h);
        return out[0];
    }

    public synchronized long countByFilter(String collection, Object filter) {
        long h = requireOpen();
        long[] out = new long[1];
        check(NativeLib.nCountByFilter(h, utf8(collection), json(filter), out), h);
        return out[0];
    }

    public synchronized ScrollPage scroll(String collection, long limit) {
        return scroll(collection, limit, null, true, false);
    }

    public synchronized ScrollPage scrollByFilter(String collection, Object filter, long limit) {
        return scrollByFilter(collection, filter, limit, null, true, false);
    }

    public synchronized ScrollPage scrollByFilter(String collection, Object filter, long limit,
            String offset, boolean withPayload, boolean withVector) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nScrollByFilter(h, utf8(collection), limit, utf8(offset), json(filter),
                withPayload, withVector, out), h);
        Map<String, Object> page = readMap(out[0]);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> points = (List<Map<String, Object>>) page.get("points");
        Object cursor = page.get("next_offset");
        return new ScrollPage(points == null ? List.of() : points,
                cursor == null ? null : String.valueOf(cursor));
    }

    public synchronized ScrollPage scroll(String collection, long limit, String offset,
            boolean withPayload, boolean withVector) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nScroll(h, utf8(collection), limit, utf8(offset), withPayload,
                withVector, out), h);
        Map<String, Object> page = readMap(out[0]);
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> points = (List<Map<String, Object>>) page.get("points");
        Object cursor = page.get("next_offset");
        return new ScrollPage(points == null ? List.of() : points,
                cursor == null ? null : String.valueOf(cursor));
    }

    // -- maintenance --

    public synchronized boolean buildIndex(String collection) {
        long h = requireOpen();
        int[] out = new int[1];
        check(NativeLib.nBuildIndex(h, utf8(collection), out), h);
        return out[0] != 0;
    }

    public synchronized boolean buildQuantization(String collection) {
        long h = requireOpen();
        int[] out = new int[1];
        check(NativeLib.nBuildQuantization(h, utf8(collection), out), h);
        return out[0] != 0;
    }

    public synchronized boolean hasIndex(String collection) {
        long h = requireOpen();
        int[] out = new int[1];
        check(NativeLib.nHasIndex(h, utf8(collection), out), h);
        return out[0] != 0;
    }

    public synchronized boolean hasQuantization(String collection) {
        long h = requireOpen();
        int[] out = new int[1];
        check(NativeLib.nHasQuantization(h, utf8(collection), out), h);
        return out[0] != 0;
    }

    public synchronized void dropIndex(String collection) {
        long h = requireOpen();
        check(NativeLib.nDropIndex(h, utf8(collection)), h);
    }

    public synchronized void dropPublishedIndex(String collection) {
        long h = requireOpen();
        check(NativeLib.nDropPublishedIndex(h, utf8(collection)), h);
    }

    public synchronized long compact(String collection) {
        long h = requireOpen();
        long[] out = new long[1];
        check(NativeLib.nCompact(h, utf8(collection), out), h);
        return out[0];
    }

    public synchronized void drainPending(String collection) {
        long h = requireOpen();
        check(NativeLib.nDrainPending(h, utf8(collection)), h);
    }

    public synchronized Map<String, Object> collectionMetrics(String collection) {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nCollectionMetrics(h, utf8(collection), out), h);
        return readMap(out[0]);
    }

    public synchronized void runMaintenanceSweep() {
        long h = requireOpen();
        check(NativeLib.nRunMaintenanceSweep(h), h);
    }

    // -- engine defaults --

    public synchronized void setDefaultHnswConfig(Object config) {
        long h = requireOpen();
        check(NativeLib.nSetDefaultHnswConfig(h, json(config)), h);
    }

    public synchronized Object getDefaultHnswConfig() {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nGetDefaultHnswConfig(h, out), h);
        return readValue(out[0]);
    }

    public synchronized void setDefaultIvfConfig(Object config) {
        long h = requireOpen();
        check(NativeLib.nSetDefaultIvfConfig(h, json(config)), h);
    }

    public synchronized Object getDefaultIvfConfig() {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nGetDefaultIvfConfig(h, out), h);
        return readValue(out[0]);
    }

    public synchronized void setDefaultQuantizationConfig(Object config) {
        long h = requireOpen();
        check(NativeLib.nSetDefaultQuantizationConfig(h, json(config)), h);
    }

    public synchronized Object getDefaultQuantizationConfig() {
        long h = requireOpen();
        byte[][] out = new byte[1][];
        check(NativeLib.nGetDefaultQuantizationConfig(h, out), h);
        return readValue(out[0]);
    }

    // -- helpers --

    private long requireOpen() {
        if (closed || handle == 0) {
            throw new MisuseError(SimvecError.MISUSE, "engine is closed");
        }
        return handle;
    }

    private void check(int code, long handle) {
        if (code == SimvecError.OK) {
            return;
        }
        throw SimvecError.raiseForCode(code, text(NativeLib.nLastError(handle)));
    }

    private static byte[] utf8(String value) {
        if (value == null) {
            return null;
        }
        return value.getBytes(StandardCharsets.UTF_8);
    }

    private static String text(byte[] value) {
        if (value == null) {
            return "";
        }
        return new String(value, StandardCharsets.UTF_8);
    }

    private static byte[] json(Object value) {
        if (value == null) {
            return null;
        }
        try {
            return MAPPER.writeValueAsBytes(value);
        } catch (JsonProcessingException e) {
            throw new MisuseError(SimvecError.MISUSE, "value is not JSON serializable");
        }
    }

    private static List<String> readList(byte[] raw) {
        try {
            List<String> parsed = MAPPER.readValue(raw, STRING_LIST);
            return parsed == null ? List.of() : parsed;
        } catch (Exception e) {
            throw new SimvecError(SimvecError.INTERNAL, "failed to decode engine response");
        }
    }

    private static Map<String, Object> readMap(byte[] raw) {
        try {
            return MAPPER.readValue(raw, STRING_MAP);
        } catch (Exception e) {
            throw new SimvecError(SimvecError.INTERNAL, "failed to decode engine response");
        }
    }

    private static List<Map<String, Object>> readMapList(byte[] raw) {
        try {
            List<Map<String, Object>> parsed = MAPPER.readValue(raw, MAP_LIST);
            return parsed == null ? List.of() : parsed;
        } catch (Exception e) {
            throw new SimvecError(SimvecError.INTERNAL, "failed to decode engine response");
        }
    }

    private static String readString(byte[] raw) {
        try {
            return MAPPER.readValue(raw, String.class);
        } catch (Exception e) {
            throw new SimvecError(SimvecError.INTERNAL, "failed to decode engine response");
        }
    }

    private static Object readValue(byte[] raw) {
        try {
            return MAPPER.readValue(raw, Object.class);
        } catch (Exception e) {
            throw new SimvecError(SimvecError.INTERNAL, "failed to decode engine response");
        }
    }

    private static float[] toFloat32(double[] vector) {
        float[] out = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            out[i] = (float) vector[i];
        }
        return out;
    }
}
