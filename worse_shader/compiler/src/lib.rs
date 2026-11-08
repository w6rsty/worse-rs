#![doc = include_str!("../README.md")]
#![doc = include_str!("../README.md")]
mod cache;
mod output;
mod source;
mod spirv;
mod types;

pub use output::{BothOutput, FileOutput, MemoryOutput, OutputSink};
use slang_auto as slang;
pub use source::{ModuleQueue, SourceBytes};
pub use spirv::{Spirv, SpirvError};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
    rc::Rc,
};
use thiserror::Error;
type LoadedModule = Result<Rc<slang::Output<slang::Module>>, Rc<Error>>;
pub use types::*;

#[derive(Debug, Error)]
pub enum Error {
    #[error("slang error: {0}")]
    Slang(#[from] slang::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("cache manifest: {0}")]
    Manifest(#[from] serde_json::Error),
    #[error("invalid, expired, or foreign module token")]
    InvalidToken,
    #[error("invalid batch: {0}")]
    InvalidBatch(&'static str),
    #[error("module load failed: {0}")]
    Module(Rc<Error>),
}

#[derive(Debug)]
pub struct JobOutcome {
    pub job: usize,
    pub result: Result<(), Error>,
    pub diagnostics: Vec<String>,
}
#[derive(Debug)]
pub struct BatchReport {
    pub jobs: Vec<JobOutcome>,
    pub stats: BatchStats,
    pub warnings: Vec<String>,
}

/// Thread-bound compiler, input registry and serialized import cache.
/// Each batch uses a fresh native session; artifacts do not borrow the service.
pub struct CompileService {
    global_session: slang::GlobalSession,
    sources: ModuleQueue,
    cache: Rc<RefCell<cache::ModuleCache>>,
}
impl CompileService {
    pub fn new() -> Result<Self, Error> {
        Self::with_cache(CacheConfig::default())
    }
    pub fn with_cache(config: CacheConfig) -> Result<Self, Error> {
        Ok(Self {
            global_session: slang::GlobalSession::new()?,
            sources: ModuleQueue::default(),
            cache: Rc::new(RefCell::new(cache::ModuleCache::new(config))),
        })
    }
    pub fn sources(&mut self) -> &mut ModuleQueue {
        &mut self.sources
    }
    pub fn clear_memory_cache(&mut self) {
        self.cache.borrow_mut().clear();
    }
    pub fn compile(&mut self, config: &BatchConfig, desc: CompileDesc<'_>) -> Result<Spirv, Error> {
        let mut output = MemoryOutput::default();
        let report = self.compile_batch(config, &[desc], &mut output)?;
        report.jobs.into_iter().next().unwrap().result?;
        Ok(output.artifacts.pop().unwrap().1)
    }
    pub fn compile_batch(
        &mut self,
        config: &BatchConfig,
        jobs: &[CompileDesc<'_>],
        output: &mut impl OutputSink,
    ) -> Result<BatchReport, Error> {
        if jobs.is_empty() {
            return Ok(BatchReport {
                jobs: Vec::new(),
                stats: BatchStats::default(),
                warnings: Vec::new(),
            });
        }
        let tokens: HashSet<_> = jobs
            .iter()
            .flat_map(|job| std::iter::once(job.source).chain(job.specialization))
            .collect();
        let (reads, source_errors) = self.sources.prepare(&tokens)?;
        let mut names = HashMap::new();
        let mut mounts: BTreeMap<_, slang::Blob> = BTreeMap::new();
        let mut units = Vec::new();
        // Preserve submission order, rather than HashSet iteration order.
        let mut seen = HashSet::new();
        for token in jobs
            .iter()
            .flat_map(|job| std::iter::once(job.source).chain(job.specialization))
        {
            if !seen.insert(token) {
                continue;
            }
            let unit = self.sources.get(token)?.clone();
            let Some(bytes) = unit.bytes.as_ref() else {
                continue;
            };
            if names
                .insert(unit.name.clone(), unit.path.clone())
                .filter(|previous| *previous != unit.path)
                .is_some()
            {
                return Err(Error::InvalidBatch(
                    "conflicting module names; split into separate batches",
                ));
            }
            if mounts
                .insert(unit.path.clone(), bytes.clone())
                .filter(|previous| previous.as_bytes() != bytes.as_bytes())
                .is_some()
            {
                return Err(Error::InvalidBatch("different revisions of a module in one batch"));
            }
            units.push((token, unit));
        }
        let direct_roots = mounts.keys().cloned().collect();
        for import in self.sources.imports() {
            let bytes = import.bytes.as_ref().unwrap();
            if mounts
                .insert(import.path.clone(), bytes.clone())
                .filter(|previous| previous.as_bytes() != bytes.as_bytes())
                .is_some()
            {
                return Err(Error::InvalidBatch("conflicting mounted import revisions"));
            }
        }
        let emit_asm = jobs.iter().any(|j| j.artifacts.contains(ArtifactMask::ASM));
        let paths: Vec<_> = config
            .search_paths
            .iter()
            .map(|p| source::normalize(p))
            .collect::<Result<_, _>>()?;
        let environment = cache::digest(
            format!(
                "cache-v1|{:?}|{:?}|{emit_asm}|{}",
                config,
                paths,
                slang::build_tag_string().to_string_lossy()
            )
            .as_bytes(),
        );
        let fs = Rc::new(RefCell::new(cache::BatchFs::new(
            self.cache.clone(),
            environment,
            mounts,
            direct_roots,
            reads,
        )));
        let (session, loaded) = loop {
            let state = fs.clone();
            let filesystem = slang::FileSystem::new(move |path| {
                let path = path
                    .to_str()
                    .map_err(|_| slang::Error::InvalidArgument("path is not UTF-8"))?;
                let mut state = state.borrow_mut();
                match state.load(Path::new(path)) {
                    Ok(value) => Ok(value),
                    Err(error) => {
                        state.warnings.push(error.to_string());
                        Err(slang::Error::InvalidArgument("filesystem read failed"))
                    }
                }
            });
            let session = self.create_session(config, &paths, emit_asm, &filesystem)?;
            fs.borrow_mut().stats.sessions += 1;
            let mut loaded: HashMap<_, LoadedModule> = source_errors
                .iter()
                .map(|(token, error)| (*token, Err(error.clone())))
                .collect();
            let mut identities: HashMap<_, LoadedModule> = HashMap::new();
            let mut failed = false;
            for (token, unit) in &units {
                let identity = (unit.name.clone(), unit.path.clone());
                if let Some(value) = identities.get(&identity) {
                    loaded.insert(*token, value.clone());
                    continue;
                }
                fs.borrow_mut().stats.source_loads += 1;
                let value = session
                    .load_module_from_bytes(
                        &unit.name,
                        unit.path.to_str().unwrap(),
                        unit.bytes.as_ref().unwrap().as_bytes(),
                    )
                    .map(Rc::new)
                    .map_err(|e| Rc::new(Error::Slang(e)));
                failed |= value.is_err();
                identities.insert(identity, value.clone());
                loaded.insert(*token, value);
            }
            if failed && !fs.borrow().hits.is_empty() && fs.borrow().allow_cache {
                fs.borrow_mut().reject_hits();
                continue;
            }
            break (session, loaded);
        };
        let mut outcomes = Vec::with_capacity(jobs.len());
        for (index, job) in jobs.iter().enumerate() {
            let mut diagnostics = Vec::new();
            let result = (|| {
                let source = loaded
                    .get(&job.source)
                    .unwrap()
                    .as_ref()
                    .map_err(|e| Error::Module(e.clone()))?;
                if !source.diagnostics.is_empty() {
                    diagnostics.push(source.diagnostics.clone());
                }
                let entry = take(
                    source
                        .value
                        .find_and_check_entry_point(job.entry.name, job.entry.stage.into())?,
                    &mut diagnostics,
                );
                let mut views = vec![source.value.as_component_ref(), entry.as_component_ref()];
                if let Some(token) = job.specialization {
                    let implementation = loaded
                        .get(&token)
                        .unwrap()
                        .as_ref()
                        .map_err(|e| Error::Module(e.clone()))?;
                    if !implementation.diagnostics.is_empty() {
                        diagnostics.push(implementation.diagnostics.clone());
                    }
                    views.push(implementation.value.as_component_ref());
                }
                let program = take(session.compose_refs(&views)?, &mut diagnostics);
                let linked = take(program.link()?, &mut diagnostics);
                let mut artifacts = Spirv {
                    code: None,
                    asm: None,
                    dbg: None,
                };
                if job.artifacts.contains(ArtifactMask::DEBUG) {
                    if !config
                        .debug_info
                        .is_some_and(|d| d.flags.contains(DebugFlags::SEPARATE))
                    {
                        return Err(Error::InvalidBatch(
                            "DEBUG artifact requires separate debug information",
                        ));
                    }
                    let result = take(linked.entry_point_compile_result(0, 0)?, &mut diagnostics);
                    if result.item_count() < 2 {
                        return Err(Error::InvalidBatch("compiler returned no separate debug data"));
                    }
                    artifacts.dbg = Some(result.item(1)?);
                    if job.artifacts.contains(ArtifactMask::CODE) {
                        artifacts.code = Some(result.item(0)?);
                    }
                } else if job.artifacts.contains(ArtifactMask::CODE) {
                    artifacts.code = Some(take(linked.entry_point_code(0, 0)?, &mut diagnostics));
                }
                if job.artifacts.contains(ArtifactMask::ASM) {
                    artifacts.asm = Some(take(linked.entry_point_code(0, 1)?, &mut diagnostics));
                }
                output.write(index, artifacts)
            })();
            outcomes.push(JobOutcome {
                job: index,
                result,
                diagnostics,
            });
        }
        cache::BatchFs::collect(&fs, &session);
        let mut state = fs.borrow_mut();
        Ok(BatchReport {
            jobs: outcomes,
            stats: state.stats,
            warnings: std::mem::take(&mut state.warnings),
        })
    }
    fn create_session(
        &self,
        config: &BatchConfig,
        paths: &[std::path::PathBuf],
        emit_asm: bool,
        filesystem: &slang::FileSystem,
    ) -> Result<slang::Session, Error> {
        let profile = self.global_session.find_profile(&config.profile)?;
        let capability = self.global_session.find_capability("spvDescriptorHeapEXT")?;
        let debug = config.debug_info.unwrap_or(DebugInfo {
            level: DebugInfoLevel::None,
            flags: DebugFlags::empty(),
        });
        let options = [
            slang::CompilerOption::capability(capability),
            slang::CompilerOption::optimization(config.optimization),
            slang::CompilerOption::debug_information(debug.level),
            slang::CompilerOption::emit_spirv_directly(true),
            slang::CompilerOption::emit_separate_debug(debug.flags.contains(DebugFlags::SEPARATE)),
            slang::CompilerOption::vulkan_use_entry_point_name(true),
        ];
        let targets = [
            slang::TargetDesc::spirv().profile(profile),
            slang::TargetDesc::spirv_asm().profile(profile),
        ];
        let paths: Vec<_> = paths.iter().map(|p| p.to_str().unwrap()).collect();
        let defines: Vec<_> = config.defines.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        Ok(self.global_session.create_session_desc(
            &slang::SessionDesc::new()
                .targets(&targets[..if emit_asm { 2 } else { 1 }])
                .compiler_options(&options)
                .matrix_layout(config.matrix_layout)
                .search_paths(&paths)
                .defines(&defines)
                .filesystem(filesystem),
        )?)
    }
}
fn take<T>(output: slang::Output<T>, diagnostics: &mut Vec<String>) -> T {
    if !output.diagnostics.is_empty() {
        diagnostics.push(output.diagnostics);
    }
    output.value
}
