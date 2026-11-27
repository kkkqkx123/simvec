//! Published data generations.
//!
//! A collection directory keeps its WAL and its type marker at the root, but
//! every serving data file lives in `generations/g<20-digit id>/`. A checkpoint
//! mutates the published generation in place; a compaction builds a complete
//! new generation in a staging directory and publishes it by renaming the
//! directory and then atomically replacing `generation.bin`.
//!
//! The manifest is the commit point. A crash before publishing it leaves the
//! previous generation serving and an unreferenced staging directory behind;
//! a crash after publishing it leaves a complete new generation and a WAL that
//! has not yet been truncated. Recovery replays from the manifest's lsn, so an
//! untruncated prefix cannot be applied twice.
//!
//! Manifest writes use the checksummed snapshot envelope, so a torn manifest
//! reads as missing and `Generation::open` reports corruption instead of
//! silently falling back to an older generation.

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::snapshot;
use crate::error::{Result, VectorSearchError};

pub(crate) const GENERATIONS_DIR: &str = "generations";
const MANIFEST_FILE: &str = "generation.bin";
const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    generation: u64,
    lsn: u64,
}

/// The currently published generation of a collection.
#[derive(Debug, Clone)]
pub(crate) struct Generation {
    root: PathBuf,
    manifest: Manifest,
}

/// A fully built generation waiting to be published.
pub(crate) struct StagedGeneration {
    path: PathBuf,
    id: u64,
}

impl Generation {
    /// Create and publish generation 1 for a new collection.
    pub(crate) fn create(root: &Path) -> Result<Self> {
        let root = root.to_path_buf();
        fs::create_dir_all(root.join(GENERATIONS_DIR))?;
        let generation = Self {
            root,
            manifest: Manifest {
                version: FORMAT_VERSION,
                generation: 1,
                lsn: 0,
            },
        };
        let staged = generation.stage()?;
        sync_tree(staged.path())?;
        fs::rename(staged.path(), generation.path())?;
        sync_dir(&generation.root.join(GENERATIONS_DIR))?;
        generation.write_manifest()?;
        Ok(generation)
    }

    /// Load the published generation. A missing or corrupt manifest is not
    /// recoverable by guessing a directory: doing so could pair a new
    /// generation's files with an old WAL watermark.
    pub(crate) fn open(root: &Path) -> Result<Self> {
        let root = root.to_path_buf();
        let path = root.join(MANIFEST_FILE);
        let Some(manifest) = snapshot::read::<Manifest>(&path)? else {
            return Err(VectorSearchError::CorruptData(format!(
                "missing or corrupt generation manifest {}",
                path.display()
            )));
        };
        if manifest.version != FORMAT_VERSION || manifest.generation == 0 {
            return Err(VectorSearchError::CorruptData(format!(
                "unsupported generation manifest in {}",
                path.display()
            )));
        }
        let generation = Self { root, manifest };
        if !generation.path().is_dir() {
            return Err(VectorSearchError::CorruptData(format!(
                "published generation directory {} is missing",
                generation.path().display()
            )));
        }
        Ok(generation)
    }

    /// Directory holding the published data files.
    pub(crate) fn path(&self) -> PathBuf {
        self.root
            .join(GENERATIONS_DIR)
            .join(format!("g{:020}", self.manifest.generation))
    }

    /// Lsn whose effects are durable in this generation. WAL recovery never
    /// goes past it, regardless of the log's own watermark.
    pub(crate) fn lsn(&self) -> u64 {
        self.manifest.lsn
    }

    /// Publish a new lsn for the same generation, after its files are durable
    /// and before the WAL is truncated.
    pub(crate) fn set_lsn(&mut self, lsn: u64) -> Result<()> {
        self.manifest.lsn = lsn;
        self.write_manifest()
    }

    /// Create the next generation's staging directory.
    pub(crate) fn stage(&self) -> Result<StagedGeneration> {
        let id = self.manifest.generation.checked_add(1).ok_or_else(|| {
            VectorSearchError::Internal("generation id overflow".to_string())
        })?;
        let generations = self.root.join(GENERATIONS_DIR);
        let path = generations.join(format!(".g{id:020}-staging"));
        // A previous publish may have renamed the directory but crashed
        // before replacing the manifest. That generation was never visible,
        // so both it and any stale staging directory are disposable.
        let final_path = generations.join(format!("g{id:020}"));
        if final_path.exists() {
            fs::remove_dir_all(&final_path)?;
        }
        if path.exists() {
            fs::remove_dir_all(&path)?;
        }
        fs::create_dir_all(&path)?;
        Ok(StagedGeneration { path, id })
    }

    /// Publish a complete staged generation and return the new manifest.
    ///
    /// Callers must sync the staged files before this call; `fsync_tree`
    /// covers directories created by helpers that only sync file contents.
    pub(crate) fn publish(self, staged: StagedGeneration, lsn: u64) -> Result<Self> {
        sync_tree(staged.path())?;
        let target = self.root
            .join(GENERATIONS_DIR)
            .join(format!("g{:020}", staged.id));
        if target.exists() {
            return Err(VectorSearchError::Internal(format!(
                "generation directory {} already exists",
                target.display()
            )));
        }
        fs::rename(staged.path(), &target)?;
        sync_dir(&self.root.join(GENERATIONS_DIR))?;
        let next = Self {
            root: self.root,
            manifest: Manifest {
                version: FORMAT_VERSION,
                generation: staged.id,
                lsn,
            },
        };
        next.write_manifest()?;
        Ok(next)
    }

    /// Remove a superseded generation once no reader can hold its files.
    pub(crate) fn retire(&self) -> Result<()> {
        if self.path().exists() {
            fs::remove_dir_all(self.path())?;
            sync_dir(&self.root.join(GENERATIONS_DIR))?;
        }
        Ok(())
    }

    fn write_manifest(&self) -> Result<()> {
        snapshot::write(&self.root.join(MANIFEST_FILE), &self.manifest)
    }
}

impl StagedGeneration {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Remove an unpublished generation after a failed build.
    pub(crate) fn discard(&self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Sync every file and directory below `root`, including `root` itself.
pub(crate) fn sync_tree(root: &Path) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            sync_tree(&path)?;
        } else if kind.is_file() {
            File::open(&path)?.sync_all()?;
        }
    }
    sync_dir(root)
}

fn sync_dir(dir: &Path) -> Result<()> {
    File::open(dir)?.sync_all()?;
    Ok(())
}
