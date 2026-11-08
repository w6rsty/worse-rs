use std::{path::PathBuf, sync::Arc};

use slang_auto::sys;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationLevel {
    #[doc = "< Don't optimize at all."]
    None = 0,
    #[doc = "< Default optimization level: balance code quality and compilation time."]
    Default = 1,
    #[doc = "< Optimize aggressively."]
    High = 2,
    #[doc = "< Include optimizations that may take a very long time, or may involve severe space-vs-speed tradeoffs"]
    Maximal = 3,
}

impl From<OptimizationLevel> for sys::SlangOptimizationLevel {
    fn from(value: OptimizationLevel) -> Self {
        match value {
            OptimizationLevel::None => sys::SlangOptimizationLevel_SLANG_OPTIMIZATION_LEVEL_NONE,
            OptimizationLevel::Default => sys::SlangOptimizationLevel_SLANG_OPTIMIZATION_LEVEL_DEFAULT,
            OptimizationLevel::High => sys::SlangOptimizationLevel_SLANG_OPTIMIZATION_LEVEL_HIGH,
            OptimizationLevel::Maximal => sys::SlangOptimizationLevel_SLANG_OPTIMIZATION_LEVEL_MAXIMAL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DebugInfoLevel {
    #[doc = "< Don't emit debug information at all."]
    None = 0,
    #[doc = "< Emit as little debug information as possible, while still supporting stack trackers."]
    Minimal = 1,
    #[doc = "< Emit whatever is the standard level of debug information for each target."]
    Standard = 2,
    #[doc = "< Emit as much debug information as possible for each target."]
    Maximal = 3,
}

impl From<DebugInfoLevel> for sys::SlangDebugInfoLevel {
    fn from(value: DebugInfoLevel) -> Self {
        match value {
            DebugInfoLevel::None => sys::SlangDebugInfoLevel_SLANG_DEBUG_INFO_LEVEL_NONE,
            DebugInfoLevel::Minimal => sys::SlangDebugInfoLevel_SLANG_DEBUG_INFO_LEVEL_MINIMAL,
            DebugInfoLevel::Standard => sys::SlangDebugInfoLevel_SLANG_DEBUG_INFO_LEVEL_STANDARD,
            DebugInfoLevel::Maximal => sys::SlangDebugInfoLevel_SLANG_DEBUG_INFO_LEVEL_MAXIMAL,
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DebugFlags: u8 {
        const EMIT_ASM = 1 << 0; // Add spirv-asm target
        const SEPARATE = 1 << 1; // -separate-debug-info
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,

    Mesh,
    Task,

    RayGeneration,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss,
    Callable,
}

impl From<Stage> for sys::SlangStage {
    fn from(value: Stage) -> Self {
        match value {
            Stage::Vertex => sys::SlangStage_SLANG_STAGE_VERTEX,
            Stage::Fragment => sys::SlangStage_SLANG_STAGE_FRAGMENT,
            Stage::Compute => sys::SlangStage_SLANG_STAGE_COMPUTE,
            Stage::Mesh => sys::SlangStage_SLANG_STAGE_MESH,
            Stage::Task => sys::SlangStage_SLANG_STAGE_AMPLIFICATION,
            Stage::RayGeneration => sys::SlangStage_SLANG_STAGE_RAY_GENERATION,
            Stage::Intersection => sys::SlangStage_SLANG_STAGE_INTERSECTION,
            Stage::AnyHit => sys::SlangStage_SLANG_STAGE_ANY_HIT,
            Stage::ClosestHit => sys::SlangStage_SLANG_STAGE_CLOSEST_HIT,
            Stage::Miss => sys::SlangStage_SLANG_STAGE_MISS,
            Stage::Callable => sys::SlangStage_SLANG_STAGE_CALLABLE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DebugInfo {
    pub level: DebugInfoLevel,
    pub flags: DebugFlags,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EntryPoint {
    pub name: String,
    pub stage: Stage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FileSource {
    File(PathBuf),
    Memory {
        virtual_name: String,
        virutal_path: String,
        payload: Arc<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileDesc {
    pub source: FileSource,
    pub entry: EntryPoint,
    pub matrix_layout: MatrixLayout,
    pub optimization: OptimizationLevel,
    /// `None` disables debug information; use `DebugFlags::SEPARATE` for a sidecar.
    pub debug_info: Option<DebugInfo>,
    pub specialization: Option<FileSource>,
}
