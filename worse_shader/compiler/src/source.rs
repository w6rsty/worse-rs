use crate::Error;
use slang_auto::Blob;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Component, Path, PathBuf},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_SERVICE: AtomicU64 = AtomicU64::new(1);
/// Owned generated sources transfer their allocation; shared sources retain
/// their Arc. Borrowed inputs are copied once when entering the queue.
pub enum SourceBytes {
    Owned(Vec<u8>),
    Shared(Arc<[u8]>),
}
impl From<Vec<u8>> for SourceBytes {
    fn from(value: Vec<u8>) -> Self {
        Self::Owned(value)
    }
}
impl From<String> for SourceBytes {
    fn from(value: String) -> Self {
        Self::Owned(value.into_bytes())
    }
}
impl From<Arc<[u8]>> for SourceBytes {
    fn from(value: Arc<[u8]>) -> Self {
        Self::Shared(value)
    }
}
impl From<&[u8]> for SourceBytes {
    fn from(value: &[u8]) -> Self {
        Self::Owned(value.to_vec())
    }
}
impl<const N: usize> From<&[u8; N]> for SourceBytes {
    fn from(value: &[u8; N]) -> Self {
        Self::from(value.as_slice())
    }
}
impl From<&str> for SourceBytes {
    fn from(value: &str) -> Self {
        Self::from(value.as_bytes())
    }
}
impl SourceBytes {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Shared(v) => v,
        }
    }
    fn into_blob(self) -> Blob {
        match self {
            Self::Owned(v) => Blob::from_vec(v),
            Self::Shared(v) => Blob::from_bytes(v),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleUnitToken {
    service: u64,
    slot: u32,
    generation: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct ModuleUnit {
    pub name: String,
    pub path: PathBuf,
    pub bytes: Option<Blob>,
}

struct Slot {
    order: u64,
    generation: u32,
    unit: Option<ModuleUnit>,
    shared_import: bool,
}

/// FIFO input registry. File contents are frozen on first preparation; refreshing
/// a file creates a new token so existing jobs keep their original snapshot.
pub struct ModuleQueue {
    next_order: u64,
    service: u64,
    slots: Vec<Slot>,
    free: Vec<u32>,
}

impl Default for ModuleQueue {
    fn default() -> Self {
        Self {
            next_order: 0,
            service: NEXT_SERVICE.fetch_add(1, Ordering::Relaxed),
            slots: Vec::new(),
            free: Vec::new(),
        }
    }
}

pub(crate) fn normalize(path: &Path) -> std::io::Result<PathBuf> {
    let path = std::path::absolute(path)?;
    let mut result = PathBuf::new();
    for part in path.components() {
        match part {
            Component::ParentDir => {
                result.pop();
            }
            Component::CurDir => {}
            other => result.push(other.as_os_str()),
        }
    }
    // Slang uses UTF-8 paths on all platforms; reject lossy identities.
    if result.to_str().is_none_or(|path| path.contains('\0')) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path must be UTF-8 without NULs",
        ));
    }
    Ok(result)
}

impl ModuleQueue {
    fn push(&mut self, unit: ModuleUnit) -> Result<ModuleUnitToken, Error> {
        let order = self.next_order;
        self.next_order = order
            .checked_add(1)
            .ok_or(Error::InvalidBatch("module queue sequence exhausted"))?;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.order = order;
            slot.unit = Some(unit);
            slot.shared_import = false;
            return Ok(ModuleUnitToken {
                service: self.service,
                slot: index,
                generation: slot.generation,
            });
        }
        let slot = u32::try_from(self.slots.len()).map_err(|_| Error::InvalidBatch("too many module units"))?;
        self.slots.push(Slot {
            order,
            generation: 1,
            unit: Some(unit),
            shared_import: false,
        });
        Ok(ModuleUnitToken {
            service: self.service,
            slot,
            generation: 1,
        })
    }

    pub fn enqueue_file(&mut self, path: impl AsRef<Path>) -> Result<ModuleUnitToken, Error> {
        let path = normalize(path.as_ref())?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or(Error::InvalidBatch("source has no module name"))?
            .to_owned();
        self.push(ModuleUnit {
            name,
            path,
            bytes: None,
        })
    }

    pub fn enqueue_memory(
        &mut self,
        name: impl Into<String>,
        path: impl AsRef<Path>,
        bytes: impl Into<SourceBytes>,
    ) -> Result<ModuleUnitToken, Error> {
        let name = name.into();
        let path = normalize(path.as_ref())?;
        let bytes = bytes.into();
        // Preserve text API NUL validation at the ownership boundary, once.
        if bytes.bytes().contains(&0) {
            return Err(slang_auto::Error::from(std::ffi::CString::new(bytes.bytes()).unwrap_err()).into());
        }
        if name.contains('\0') {
            return Err(slang_auto::Error::from(std::ffi::CString::new(name).unwrap_err()).into());
        }
        self.push(ModuleUnit {
            name,
            path,
            bytes: Some(bytes.into_blob()),
        })
    }

    pub fn refresh_file(&mut self, token: ModuleUnitToken) -> Result<ModuleUnitToken, Error> {
        let path = self.get(token)?.path.clone();
        self.enqueue_file(path)
    }
    /// Mount an in-memory file for imports/includes. Release the old token
    /// before mounting a changed revision at the same path. Unlike job roots,
    /// registered import modules are eligible for persistent IR caching.
    pub fn register_import_memory(
        &mut self,
        path: impl AsRef<Path>,
        bytes: impl Into<SourceBytes>,
    ) -> Result<ModuleUnitToken, Error> {
        let path = normalize(path.as_ref())?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or(Error::InvalidBatch("import has no name"))?
            .to_owned();
        let token = self.enqueue_memory(name, path, bytes)?;
        self.slots[token.slot as usize].shared_import = true;
        Ok(token)
    }
    pub(crate) fn imports(&self) -> impl Iterator<Item = &ModuleUnit> {
        self.slots
            .iter()
            .filter(|slot| slot.shared_import)
            .filter_map(|slot| slot.unit.as_ref())
    }

    pub fn release(&mut self, token: ModuleUnitToken) -> Result<(), Error> {
        self.get(token)?;
        let slot = &mut self.slots[token.slot as usize];
        slot.unit = None;
        if let Some(generation) = slot.generation.checked_add(1) {
            slot.generation = generation;
            self.free.push(token.slot);
        }
        Ok(())
    }

    pub(crate) fn get(&self, token: ModuleUnitToken) -> Result<&ModuleUnit, Error> {
        if token.service != self.service {
            return Err(Error::InvalidToken);
        }
        self.slots
            .get(token.slot as usize)
            .filter(|slot| slot.generation == token.generation)
            .and_then(|slot| slot.unit.as_ref())
            .ok_or(Error::InvalidToken)
    }

    pub(crate) fn prepare(
        &mut self,
        tokens: &HashSet<ModuleUnitToken>,
    ) -> Result<(usize, HashMap<ModuleUnitToken, Rc<Error>>), Error> {
        for &token in tokens {
            self.get(token)?;
        }
        let mut reads = 0;
        let mut fresh: BTreeMap<PathBuf, Result<Blob, Rc<Error>>> = BTreeMap::new();
        let mut errors = HashMap::new();
        let mut order: Vec<_> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.unit.is_some())
            .map(|(index, s)| (s.order, index))
            .collect();
        order.sort_unstable();
        for (_, index) in order {
            let slot = &mut self.slots[index];
            let token = ModuleUnitToken {
                service: self.service,
                slot: index as u32,
                generation: slot.generation,
            };
            if tokens.contains(&token) {
                let unit = slot.unit.as_mut().unwrap();
                if unit.bytes.is_none() {
                    if !fresh.contains_key(&unit.path) {
                        let result = std::fs::read(&unit.path)
                            .map(Blob::from_vec)
                            .map_err(|error| Rc::new(Error::Io(error)));
                        if result.is_ok() {
                            reads += 1;
                        }
                        fresh.insert(unit.path.clone(), result);
                    }
                    match fresh.get(&unit.path).unwrap() {
                        Ok(blob) => unit.bytes = Some(blob.clone()),
                        Err(error) => {
                            errors.insert(token, error.clone());
                        }
                    }
                }
            }
        }
        Ok((reads, errors))
    }
}
