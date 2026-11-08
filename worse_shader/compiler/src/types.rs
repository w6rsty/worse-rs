use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub use crate::source::ModuleUnitToken;
pub use slang_auto::{DebugInfoLevel, MatrixLayout, OptimizationLevel, Stage};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DebugFlags: u8 {
        const SEPARATE = 1 << 1;
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ArtifactMask: u8 {
        const CODE = 1 << 0;
        const ASM = 1 << 1;
        const DEBUG = 1 << 2;
        const ALL = Self::CODE.bits() | Self::ASM.bits() | Self::DEBUG.bits();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DebugInfo {
    pub level: DebugInfoLevel,
    pub flags: DebugFlags,
}

/// Configuration shared by every job in one native Slang session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchConfig {
    pub matrix_layout: MatrixLayout,
    pub optimization: OptimizationLevel,
    pub debug_info: Option<DebugInfo>,
    pub search_paths: Vec<PathBuf>,
    pub defines: Vec<(String, String)>,
    pub profile: String,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            matrix_layout: MatrixLayout::ColumnMajor,
            optimization: OptimizationLevel::Default,
            debug_info: None,
            search_paths: Vec::new(),
            defines: Vec::new(),
            profile: "spirv_1_5".into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EntryPointRef<'a> {
    pub name: &'a str,
    pub stage: Stage,
}

impl<'a> EntryPointRef<'a> {
    pub const fn new(name: &'a str, stage: Stage) -> Self {
        Self { name, stage }
    }
}

/// A cheap job descriptor. Specialization is a module explicitly composed into
/// this job (for link-time definitions); it is not a generic type argument.
#[derive(Debug, Clone, Copy)]
pub struct CompileDesc<'a> {
    pub source: ModuleUnitToken,
    pub specialization: Option<ModuleUnitToken>,
    pub entry: EntryPointRef<'a>,
    pub artifacts: ArtifactMask,
}

impl<'a> CompileDesc<'a> {
    pub fn new(source: ModuleUnitToken, entry: EntryPointRef<'a>) -> Self {
        Self {
            source,
            specialization: None,
            entry,
            artifacts: ArtifactMask::CODE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub enabled: bool,
    pub directory: Option<PathBuf>,
    pub max_memory_bytes: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            directory: None,
            max_memory_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct BatchStats {
    pub sessions: usize,
    pub source_loads: usize,
    pub file_reads: usize,
    pub cache_hits: usize,
    pub cache_writes: usize,
}
