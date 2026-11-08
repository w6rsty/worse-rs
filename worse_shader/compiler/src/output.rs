use crate::{Error, Spirv};
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static TEMP_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

/// A batch sink consumes each result immediately; offline compilation need not
/// retain the batch's blobs. Returning Err records a per-job output failure.
pub trait OutputSink {
    fn write(&mut self, job: usize, artifacts: Spirv) -> Result<(), Error>;
}

#[derive(Default)]
pub struct MemoryOutput {
    pub artifacts: Vec<(usize, Spirv)>,
}

impl OutputSink for MemoryOutput {
    fn write(&mut self, job: usize, artifacts: Spirv) -> Result<(), Error> {
        self.artifacts.push((job, artifacts));
        Ok(())
    }
}

pub struct FileOutput {
    directory: PathBuf,
    names: Vec<String>,
}

impl FileOutput {
    /// Names are relative file stems, one per job; duplicate destinations are rejected.
    pub fn new(directory: impl Into<PathBuf>, names: Vec<String>) -> Result<Self, Error> {
        let mut seen = std::collections::HashSet::new();
        for name in &names {
            if name.is_empty()
                || Path::new(name).components().count() != 1
                || name.contains(['/', '\\', ':'])
                || name == "."
                || name == ".."
                || !seen.insert(name.to_lowercase())
            {
                return Err(Error::InvalidBatch("output names must be unique file stems"));
            }
        }
        Ok(Self {
            directory: directory.into(),
            names,
        })
    }
}

impl OutputSink for FileOutput {
    fn write(&mut self, job: usize, artifacts: Spirv) -> Result<(), Error> {
        let name = self
            .names
            .get(job)
            .ok_or(Error::InvalidBatch("missing output filename"))?;
        if let Some(bytes) = artifacts.code_bytes() {
            atomic_write(&self.directory.join(format!("{name}.spv")), bytes)?;
        }
        if let Some(bytes) = artifacts.asm_bytes() {
            atomic_write(&self.directory.join(format!("{name}.spvasm")), bytes)?;
        }
        if let Some(bytes) = artifacts.debug_data() {
            atomic_write(&self.directory.join(format!("{name}.debug.spv")), bytes)?;
        }
        Ok(())
    }
}

pub struct BothOutput<A, B>(pub A, pub B);
impl<A: OutputSink, B: OutputSink> OutputSink for BothOutput<A, B> {
    fn write(&mut self, job: usize, artifacts: Spirv) -> Result<(), Error> {
        self.0.write(job, artifacts.clone())?;
        self.1.write(job, artifacts)
    }
}
