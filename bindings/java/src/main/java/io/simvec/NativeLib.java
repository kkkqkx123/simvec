package io.simvec;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Thin JNI declarations over the simvec C API plus library location.
 *
 * <p>All text crosses as UTF-8 bytes (never modified UTF-8): every string
 * parameter is a {@code byte[]} and every JSON output is a {@code byte[][]}
 * out slot holding standard UTF-8. Null means absent, matching the C rule
 * that null inputs are borrowed and outputs are allocated by the library.
 */
final class NativeLib {
    private NativeLib() {
    }

    static {
        load();
    }

    static void ensureLoaded() {
        // Loading happens in the static initializer; the call keeps the reference.
    }

    private static void load() {
        List<String> tried = new ArrayList<>();
        String explicitBridge = firstNonBlank(
                System.getProperty("simvec.jni.lib"), System.getenv("SIMVEC_JNI_LIB"));
        if (explicitBridge != null) {
            tried.add(explicitBridge);
            preloadRustLib();
            System.load(explicitBridge);
            return;
        }
        Path userDir = Paths.get(System.getProperty("user.dir", ".")).toAbsolutePath();
        Path workspace = userDir.resolve("../../").normalize();
        for (String profile : new String[]{"debug", "release"}) {
            for (String name : bridgeNames()) {
                Path candidate = workspace.resolve("target").resolve(profile).resolve(name);
                tried.add(candidate.toString());
                if (Files.isRegularFile(candidate)) {
                    preloadRustLib();
                    System.load(candidate.toString());
                    return;
                }
            }
        }
        preloadRustLib();
        try {
            System.loadLibrary("simvec_jni");
            return;
        } catch (UnsatisfiedLinkError first) {
            throw new UnsatisfiedLinkError(
                    "libsimvec_jni not found; build it with scripts/build-native.sh after "
                            + "`cargo build --features c_api`, then set SIMVEC_JNI_LIB, "
                            + "use -Dsimvec.jni.lib=<path>, or rely on java.library.path. "
                            + "Tried: " + String.join(", ", tried)
                            + ". Original: " + first.getMessage());
        }
    }

    private static void preloadRustLib() {
        String explicit = System.getenv("SIMVEC_LIB");
        if (explicit != null && !explicit.isBlank()) {
            System.load(explicit);
        }
    }

    private static String firstNonBlank(String first, String second) {
        if (first != null && !first.isBlank()) {
            return first;
        }
        if (second != null && !second.isBlank()) {
            return second;
        }
        return null;
    }

    private static String[] bridgeNames() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("mac")) {
            return new String[]{"libsimvec_jni.dylib"};
        }
        if (os.contains("win")) {
            return new String[]{"simvec_jni.dll"};
        }
        return new String[]{"libsimvec_jni.so"};
    }

    static native int nOpen(byte[] pathUtf8, long[] handleOut);

    static native byte[] nLastErrorText();

    static native int nClose(long handle);

    static native int nErrcode(long handle);

    static native byte[] nLastError(long handle);

    static native int nCreateCollection(long handle, byte[] name, byte[] configJson);

    static native int nDeleteCollection(long handle, byte[] name);

    static native int nCollectionExists(long handle, byte[] name, int[] out);

    static native int nListCollections(long handle, byte[][] out);

    static native int nCollectionInfo(long handle, byte[] name, byte[][] out);

    static native int nCollectionConfig(long handle, byte[] name, byte[][] out);

    static native int nUpsert(long handle, byte[] collection, byte[] id, float[] vector, byte[] payloadJson);

    static native int nUpsertBatch(long handle, byte[] collection, byte[] pointsJson);

    static native int nDelete(long handle, byte[] collection, byte[] id);

    static native int nDeleteBatch(long handle, byte[] collection, byte[] idsJson);

    static native int nDeleteByFilter(long handle, byte[] collection, byte[] filterJson, long[] out);

    static native int nSetPayload(long handle, byte[] collection, byte[] id, byte[] payloadJson);

    static native int nMergePayload(long handle, byte[] collection, byte[] id, byte[] fieldsJson);

    static native int nDeletePayloadKeys(long handle, byte[] collection, byte[] id, byte[] keysJson);

    static native int nCreatePayloadIndex(long handle, byte[] collection, byte[] field, byte[] schema);

    static native int nDeletePayloadIndex(long handle, byte[] collection, byte[] field, int[] out);

    static native int nListPayloadIndexes(long handle, byte[] collection, byte[][] out);

    static native int nSearch(long handle, byte[] collection, byte[] queryJson, byte[][] out);

    static native int nSearchVector(long handle, byte[] collection, float[] vector, long limit,
            byte[] filterJson, float scoreThreshold, boolean withPayload, boolean withVector,
            byte[][] out);

    static native int nGet(long handle, byte[] collection, byte[] id, byte[][] out);

    static native int nCount(long handle, byte[] collection, long[] out);

    static native int nCountByFilter(long handle, byte[] collection, byte[] filterJson, long[] out);

    static native int nScroll(long handle, byte[] collection, long limit, byte[] offset,
            boolean withPayload, boolean withVector, byte[][] out);

    static native int nScrollByFilter(long handle, byte[] collection, long limit, byte[] offset,
            byte[] filterJson, boolean withPayload, boolean withVector, byte[][] out);

    static native int nBuildIndex(long handle, byte[] collection, int[] out);

    static native int nBuildQuantization(long handle, byte[] collection, int[] out);

    static native int nHasIndex(long handle, byte[] collection, int[] out);

    static native int nHasQuantization(long handle, byte[] collection, int[] out);

    static native int nDropIndex(long handle, byte[] collection);

    static native int nDropPublishedIndex(long handle, byte[] collection);

    static native int nCompact(long handle, byte[] collection, long[] out);

    static native int nDrainPending(long handle, byte[] collection);

    static native int nCollectionMetrics(long handle, byte[] collection, byte[][] out);

    static native int nRunMaintenanceSweep(long handle);

    static native int nSetPayloadField(long handle, byte[] collection, byte[] id, byte[] key,
            byte[] valueJson);

    static native int nApplyTxn(long handle, byte[] opsJson);

    static native int nPromoteTempCollection(long handle, byte[] live, byte[] temp, byte[] backupSuffix);

    static native int nRestorePromoteBackup(long handle, byte[] live, byte[][] out);

    static native int nListTempCollections(long handle, byte[][] out);

    static native int nListOldBackups(long handle, byte[] live, byte[][] out);

    static native int nSetDefaultHnswConfig(long handle, byte[] configJson);

    static native int nGetDefaultHnswConfig(long handle, byte[][] out);

    static native int nSetDefaultIvfConfig(long handle, byte[] configJson);

    static native int nGetDefaultIvfConfig(long handle, byte[][] out);

    static native int nSetDefaultQuantizationConfig(long handle, byte[] configJson);

    static native int nGetDefaultQuantizationConfig(long handle, byte[][] out);
}
