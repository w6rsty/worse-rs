use crate::{BatchStats, CacheConfig, Error, output::atomic_write, source::normalize};
use serde::{Deserialize, Serialize};
use slang_auto as slang;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    rc::Rc,
};

pub(crate) fn digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Manifest {
    pub name: String,
    pub path: PathBuf,
    pub files: BTreeMap<PathBuf, String>,
    pub missing: Vec<PathBuf>,
    pub dependencies: Vec<PathBuf>,
    pub ir_hash: String,
}
struct Node {
    module: slang::Module,
    name: String,
    files: Vec<PathBuf>,
    edges: Vec<PathBuf>,
}

#[derive(Clone)]
pub(crate) struct CachedModule {
    pub manifest: Manifest,
    pub ir: slang::Blob,
    age: u64,
}

pub(crate) struct ModuleCache {
    pub config: CacheConfig,
    entries: BTreeMap<String, CachedModule>,
    clock: u64,
}

impl ModuleCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: BTreeMap::new(),
            clock: 0,
        }
    }

    pub fn key(environment: &str, path: &Path) -> String {
        digest(format!("worse-module-cache-v1\0{environment}\0{}", path.display()).as_bytes())
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn invalidate(&mut self, key: &str) {
        self.entries.remove(key);
        if let Some(dir) = &self.config.directory {
            let _ = std::fs::remove_file(dir.join(format!("{key}.json")));
        }
    }

    fn insert(&mut self, key: String, mut entry: CachedModule) {
        self.clock += 1;
        entry.age = self.clock;
        self.entries.insert(key, entry);
        while self.entries.values().map(|e| e.ir.as_bytes().len()).sum::<usize>() > self.config.max_memory_bytes {
            let Some(oldest) = self.entries.iter().min_by_key(|(_, e)| e.age).map(|(k, _)| k.clone()) else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    fn lookup(&mut self, key: &str) -> Option<CachedModule> {
        if !self.config.enabled {
            return None;
        }
        self.clock += 1;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.age = self.clock;
            return Some(entry.clone());
        }
        let dir = self.config.directory.as_ref()?;
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(dir.join(format!("{key}.json"))).ok()?).ok()?;
        let bytes = std::fs::read(dir.join(format!("{}.ir", manifest.ir_hash))).ok()?;
        if digest(&bytes) != manifest.ir_hash {
            return None;
        }
        let entry = CachedModule {
            manifest,
            ir: slang::Blob::from_vec(bytes),
            age: self.clock,
        };
        self.insert(key.to_owned(), entry.clone());
        Some(entry)
    }

    pub fn save(&mut self, environment: &str, manifest: Manifest, ir: slang::Blob) -> Result<(), Error> {
        if !self.config.enabled {
            return Ok(());
        }
        let key = Self::key(environment, &manifest.path);
        if let Some(dir) = &self.config.directory {
            atomic_write(&dir.join(format!("{}.ir", manifest.ir_hash)), ir.as_bytes())?;
            atomic_write(&dir.join(format!("{key}.json")), &serde_json::to_vec(&manifest)?)?;
        }
        self.insert(key, CachedModule { manifest, ir, age: 0 });
        Ok(())
    }
}

#[derive(Clone)]
struct SourceFile {
    blob: slang::Blob,
    hash: String,
}

/// One immutable view of every successful or failed file lookup in a batch.
pub(crate) struct BatchFs {
    pub cache: Rc<RefCell<ModuleCache>>,
    pub environment: String,
    pub roots: BTreeSet<PathBuf>,
    files: BTreeMap<PathBuf, Option<SourceFile>>,
    checked: BTreeMap<PathBuf, Option<slang::Blob>>,
    pub hits: BTreeSet<PathBuf>,
    pub stats: BatchStats,
    pub warnings: Vec<String>,
    pub allow_cache: bool,
}

impl BatchFs {
    pub fn new(
        cache: Rc<RefCell<ModuleCache>>,
        environment: String,
        roots: BTreeMap<PathBuf, slang::Blob>,
        direct_roots: BTreeSet<PathBuf>,
        reads: usize,
    ) -> Self {
        let files = roots
            .into_iter()
            .map(|(path, blob)| {
                let hash = digest(blob.as_bytes());
                (path, Some(SourceFile { blob, hash }))
            })
            .collect();
        Self {
            cache,
            environment,
            roots: direct_roots,
            files,
            checked: BTreeMap::new(),
            hits: BTreeSet::new(),
            stats: BatchStats {
                file_reads: reads,
                ..Default::default()
            },
            warnings: Vec::new(),
            allow_cache: true,
        }
    }

    fn read(&mut self, path: &Path) -> Result<Option<SourceFile>, Error> {
        if let Some(value) = self.files.get(path) {
            return Ok(value.clone());
        }
        let value = match std::fs::read(path) {
            Ok(bytes) => {
                self.stats.file_reads += 1;
                Some(SourceFile {
                    hash: digest(&bytes),
                    blob: slang::Blob::from_vec(bytes),
                })
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) =>
            {
                None
            }
            Err(e) => return Err(e.into()),
        };
        self.files.insert(path.to_owned(), value.clone());
        Ok(value)
    }

    pub fn load(&mut self, path: &Path) -> Result<Option<slang::Blob>, Error> {
        let path = normalize(path)?;
        if path.extension().is_some_and(|ext| ext == "slang-module") {
            let source = path.with_extension("slang");
            if !self.allow_cache || self.roots.contains(&source) {
                return Ok(None);
            }
            if let Some(value) = self.checked.get(&source) {
                return Ok(value.clone());
            }
            let key = ModuleCache::key(&self.environment, &source);
            let candidate = self.cache.borrow_mut().lookup(&key);
            let mut result = None;
            if let Some(entry) = candidate {
                let mut valid = entry.manifest.path == source;
                for (file, expected) in &entry.manifest.files {
                    if self.read(file)?.is_none_or(|value| value.hash != *expected) {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    for file in &entry.manifest.missing {
                        if self.read(file)?.is_some() {
                            valid = false;
                            break;
                        }
                    }
                }
                if valid {
                    self.stats.cache_hits += 1;
                    self.hits.insert(source.clone());
                    result = Some(entry.ir);
                } else {
                    self.cache.borrow_mut().invalidate(&key);
                }
            }
            self.checked.insert(source, result.clone());
            Ok(result)
        } else {
            Ok(self.read(&path)?.map(|value| value.blob))
        }
    }

    pub fn reject_hits(&mut self) {
        for path in &self.hits {
            self.cache
                .borrow_mut()
                .invalidate(&ModuleCache::key(&self.environment, path));
        }
        self.hits.clear();
        self.checked.clear();
        self.allow_cache = false;
        self.warnings
            .push("Rejected serialized modules; retried the batch from source in a fresh session".into());
    }

    pub fn collect(state: &Rc<RefCell<Self>>, session: &slang::Session) {
        if !state.borrow().cache.borrow().config.enabled {
            return;
        }
        // Serialization may call the filesystem. Do not borrow state during
        // any native module call, including dependency discovery.
        let result = (|| {
            let nodes = Self::discover(session)?;
            let candidates = state.borrow_mut().candidates(nodes)?;
            for (mut manifest, module) in candidates {
                let ir = module.serialize()?;
                manifest.ir_hash = digest(ir.as_bytes());
                let mut state = state.borrow_mut();
                state.cache.borrow_mut().save(&state.environment, manifest, ir)?;
                state.stats.cache_writes += 1;
            }
            Ok::<_, Error>(())
        })();
        if let Err(error) = result {
            state.borrow_mut().warnings.push(format!("Module cache: {error}"));
        }
    }

    fn discover(session: &slang::Session) -> Result<BTreeMap<PathBuf, Node>, Error> {
        let mut nodes = BTreeMap::new();
        for module in session.loaded_modules() {
            let path = module
                .file_path()
                .ok_or(Error::InvalidBatch("module has no file path"))?
                .to_str()
                .map_err(|_| Error::InvalidBatch("module path is not UTF-8"))?;
            let mut path = normalize(Path::new(path))?;
            if path.extension().is_some_and(|s| s == "slang-module") {
                path.set_extension("slang");
            }
            let name = module
                .name()
                .and_then(|n| n.to_str().ok())
                .ok_or(Error::InvalidBatch("module name is not UTF-8"))?
                .to_owned();
            let files = module
                .dependency_files()
                .into_iter()
                .map(|p| {
                    let mut path = normalize(Path::new(
                        p.to_str().map_err(|_| Error::InvalidBatch("dependency is not UTF-8"))?,
                    ))?;
                    if path.extension().is_some_and(|s| s == "slang-module") {
                        path.set_extension("slang");
                    }
                    Ok::<_, Error>(path)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let edges = module
                .dependencies()?
                .into_iter()
                .map(|m| {
                    let mut path = normalize(Path::new(
                        m.file_path()
                            .and_then(|p| p.to_str().ok())
                            .ok_or(Error::InvalidBatch("dependency module has no UTF-8 path"))?,
                    ))?;
                    if path.extension().is_some_and(|s| s == "slang-module") {
                        path.set_extension("slang");
                    }
                    Ok::<_, Error>(path)
                })
                .collect::<Result<Vec<_>, _>>()?;
            nodes.insert(
                path,
                Node {
                    module,
                    name,
                    files,
                    edges,
                },
            );
        }
        // File dependencies provide a conservative closure of imported modules.
        // The experimental interface alone may report no edges in this SDK.
        let identities: BTreeSet<_> = nodes.keys().cloned().collect();
        for (path, node) in &mut nodes {
            node.edges.extend(
                node.files
                    .iter()
                    .filter(|p| *p != path && identities.contains(*p))
                    .cloned(),
            );
            node.edges.sort();
            node.edges.dedup();
        }
        Ok(nodes)
    }
    fn candidates(&mut self, nodes: BTreeMap<PathBuf, Node>) -> Result<Vec<(Manifest, slang::Module)>, Error> {
        let mut candidates = Vec::new();
        let missing: Vec<_> = self
            .files
            .iter()
            .filter(|(p, value)| value.is_none() && p.extension().is_none_or(|s| s != "slang-module"))
            .map(|(p, _)| p.clone())
            .collect();
        for (path, node) in &nodes {
            if self.roots.contains(path) || self.hits.contains(path) {
                continue;
            }
            let mut pending = vec![path.clone()];
            let mut visited = BTreeSet::new();
            let mut files = BTreeMap::new();
            let mut eligible = true;
            while let Some(dependency) = pending.pop() {
                if !visited.insert(dependency.clone()) {
                    continue;
                }
                if self.roots.contains(&dependency) {
                    eligible = false;
                    break;
                }
                let Some(child) = nodes.get(&dependency) else {
                    eligible = false;
                    break;
                };
                pending.extend(child.edges.iter().filter(|p| **p != dependency).cloned());
                for file in child.files.iter().chain(std::iter::once(&dependency)) {
                    if self.roots.contains(file) {
                        eligible = false;
                        break;
                    }
                    if let Some(value) = self.read(file)? {
                        files.insert(file.clone(), value.hash);
                    } else {
                        eligible = false;
                        break;
                    }
                }
            }
            if !eligible {
                continue;
            }
            let manifest = Manifest {
                name: node.name.clone(),
                path: path.clone(),
                files,
                missing: missing.clone(),
                dependencies: node.edges.clone(),
                ir_hash: String::new(),
            };
            candidates.push((manifest, node.module.clone()));
        }
        Ok(candidates)
    }
}
