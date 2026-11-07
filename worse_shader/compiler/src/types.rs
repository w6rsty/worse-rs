use std::{path::PathBuf, sync::Arc};

use slang_rs as slang;

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

impl From<OptimizationLevel> for slang::OptimizationLevel {
    fn from(value: OptimizationLevel) -> Self {
        match value {
            OptimizationLevel::None => slang::OptimizationLevel::None,
            OptimizationLevel::Default => slang::OptimizationLevel::Default,
            OptimizationLevel::High => slang::OptimizationLevel::High,
            OptimizationLevel::Maximal => slang::OptimizationLevel::Maximal,
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

impl From<DebugInfoLevel> for slang::DebugInfoLevel {
    fn from(value: DebugInfoLevel) -> Self {
        match value {
            DebugInfoLevel::None => slang::DebugInfoLevel::None,
            DebugInfoLevel::Minimal => slang::DebugInfoLevel::Minimal,
            DebugInfoLevel::Standard => slang::DebugInfoLevel::Standard,
            DebugInfoLevel::Maximal => slang::DebugInfoLevel::Maximal,
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

impl From<Stage> for slang::Stage {
    fn from(value: Stage) -> Self {
        match value {
            Stage::Vertex => slang::Stage::Vertex,
            Stage::Fragment => slang::Stage::Fragment,
            Stage::Compute => slang::Stage::Compute,
            Stage::Mesh => slang::Stage::Mesh,
            Stage::Task => slang::Stage::Amplification,
            Stage::RayGeneration => slang::Stage::RayGeneration,
            Stage::Intersection => slang::Stage::Intersection,
            Stage::AnyHit => slang::Stage::AnyHit,
            Stage::ClosestHit => slang::Stage::ClosestHit,
            Stage::Miss => slang::Stage::Miss,
            Stage::Callable => slang::Stage::Callable,
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
        payload: Arc<str>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileDesc {
    pub source: FileSource,
    pub entry: EntryPoint,
    pub matrix_layout: MatrixLayout,
    pub optimization: OptimizationLevel,
    /// Provides None for strip (-g0 --separate-debug-info)
    pub debug_info: Option<DebugInfo>,
    pub specialization: Option<FileSource>,
}
