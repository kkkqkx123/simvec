//! Regression: one unopenable collection must not take the engine down.
//!
//! Collection loading used to propagate the first error with `?`, so a single
//! corrupt `meta.bin` made every collection unreachable. The loader now
//! quarantines the damaged directory as `<name>.suspect-<ts>` and keeps going.

use simvec::{CollectionConfig, DistanceMetric, LocalVectorEngine};

fn config() -> CollectionConfig {
    CollectionConfig::new(3, DistanceMetric::Dot)
}

fn data_dir(collection: &std::path::Path) -> std::path::PathBuf {
    std::fs::read_dir(collection.join("generations"))
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.is_dir())
        .expect("published generation")
}

#[test]
fn corrupt_collection_is_quarantined_and_the_rest_load() {
    let dir = tempfile::tempdir().unwrap();
    {
        let engine = LocalVectorEngine::open(dir.path()).unwrap();
        engine.create_collection("good", &config()).unwrap();
        engine.create_collection("bad", &config()).unwrap();
    }

    // Flip one byte of the payload body: the CRC envelope rejects it.
    let meta_path = data_dir(&dir.path().join("bad")).join("meta.bin");
    let mut bytes = std::fs::read(&meta_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&meta_path, &bytes).unwrap();

    let engine = LocalVectorEngine::open(dir.path()).expect("engine must still open");
    assert!(
        engine.collection_exists("good"),
        "healthy collection must load"
    );
    assert!(
        !engine.collection_exists("bad"),
        "corrupt collection must not be loaded"
    );

    assert!(!dir.path().join("bad").exists(), "original dir moved away");
    let quarantined: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("bad.suspect-"))
        .collect();
    assert_eq!(
        quarantined.len(),
        1,
        "expected exactly one quarantine dir, got {quarantined:?}"
    );

    // The quarantine marker keeps it invisible on the next start too. Note
    // `list_collections` intentionally reports physical directories (startup
    // recovery needs them); the logical set excludes `*.suspect-*`.
    let engine = LocalVectorEngine::open(dir.path()).expect("second open");
    assert!(engine.collection_exists("good"));
    let logical: Vec<String> = engine
        .list_collections()
        .into_iter()
        .filter(|name| !name.contains(".suspect-"))
        .collect();
    assert_eq!(logical, vec!["good".to_string()]);
}
