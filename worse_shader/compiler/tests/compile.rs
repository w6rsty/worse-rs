use std::{path::PathBuf, sync::Arc};

use worse_shader_compiler::{
    CompileDesc, CompileService, DebugFlags, DebugInfo, DebugInfoLevel, EntryPoint, Error, FileSource, MatrixLayout,
    OptimizationLevel, SpirvError, Stage,
};

// Intentionally omit [shader]: CompileDesc.entry.stage must reach Slang.
const SOURCE: &str = r#"
RWStructuredBuffer<float> output;
[numthreads(4, 1, 1)]
void dispatch_main(uint3 id : SV_DispatchThreadID) { output[id.x] = float(id.x) * 2.0; }
"#;

fn descriptor(source: &str) -> CompileDesc {
    CompileDesc {
        source: FileSource::Memory {
            virtual_name: "migration_test".into(),
            virutal_path: "migration_test.slang".into(),
            payload: Arc::new(source.into()),
        },
        entry: EntryPoint {
            name: "dispatch_main".into(),
            stage: Stage::Compute,
        },
        matrix_layout: MatrixLayout::ColumnMajor,
        optimization: OptimizationLevel::Default,
        debug_info: None,
        specialization: None,
    }
}

#[test]
fn memory_source_keeps_spirv_profile_entry_name_and_owned_outputs() {
    let service = CompileService::new().unwrap();
    let mut desc = descriptor(SOURCE);
    desc.debug_info = Some(DebugInfo {
        level: DebugInfoLevel::None,
        flags: DebugFlags::EMIT_ASM,
    });
    let compiled = service.compile(desc).unwrap();
    drop(service);
    let words = compiled.words().unwrap();
    assert_eq!(words[0], 0x0723_0203);
    assert_eq!(words[1], 0x0001_0500, "must retain the spirv_1_5 profile");
    let asm = compiled.asm_str().unwrap();
    assert!(asm.contains("OpEntryPoint GLCompute"), "{asm}");
    assert!(asm.contains("\"dispatch_main\""), "{asm}");
    assert!(asm.contains("LocalSize 4 1 1"), "{asm}");
    assert!(compiled.debug_bytes().is_err_and(|err| err == SpirvError::NoDebugInfo));
}

#[test]
fn file_source_resolves_relative_includes() {
    let mut desc = descriptor(SOURCE);
    desc.source = FileSource::File(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compute.slang"));
    let compiled = CompileService::new().unwrap().compile(desc).unwrap();
    assert_eq!(compiled.words().unwrap()[0], 0x0723_0203);
    assert!(compiled.asm_str().is_none());
    assert!(compiled.debug_bytes().is_err());
}

#[test]
fn compilation_errors_and_ffi_strings_are_reported() {
    let service = CompileService::new().unwrap();
    let error = service.compile(descriptor("this is not Slang !!!")).err().unwrap();
    assert!(
        matches!(error, Error::Slang(slang_auto::Error::Slang { ref diagnostics, .. })
        if diagnostics.contains("error"))
    );
    let mut missing = descriptor(SOURCE);
    missing.entry.name = "missing_entry".into();
    assert!(service.compile(missing).is_err());
    let mut nul = descriptor(SOURCE);
    nul.entry.name = "dispatch_main\0invalid".into();
    assert!(matches!(
        service.compile(nul),
        Err(Error::Slang(slang_auto::Error::InteriorNul(_)))
    ));
    let nul_source = descriptor("invalid\0source");
    assert!(matches!(
        service.compile(nul_source),
        Err(Error::Slang(slang_auto::Error::InteriorNul(_)))
    ));
}

#[test]
fn separate_debug_artifact_survives_service_drop() {
    let service = CompileService::new().unwrap();
    let mut desc = descriptor(SOURCE);
    desc.debug_info = Some(DebugInfo {
        level: DebugInfoLevel::Standard,
        flags: DebugFlags::SEPARATE | DebugFlags::EMIT_ASM,
    });
    let compiled = service.compile(desc).unwrap();
    drop(service);
    assert_eq!(compiled.words().unwrap()[0], 0x0723_0203);
    let debug = compiled.debug_bytes().expect("Slang should emit a debug sidecar");
    assert!(debug.len() >= 20);
    assert_eq!(&debug[..4], &0x0723_0203u32.to_le_bytes());
    assert!(!compiled.asm_str().unwrap().is_empty());
}

#[test]
fn matrix_layout_changes_buffer_decorations() {
    let source = r#"
struct Params { float3x4 transform; };
ConstantBuffer<Params> params;
RWStructuredBuffer<float> output;
[numthreads(4, 1, 1)]
void dispatch_main(uint3 id : SV_DispatchThreadID) {
    output[id.x] = params.transform[id.x % 3][id.x % 4];
}
"#;
    let service = CompileService::new().unwrap();
    let mut decorations = Vec::new();
    for layout in [MatrixLayout::RowMajor, MatrixLayout::ColumnMajor] {
        let mut desc = descriptor(source);
        desc.matrix_layout = layout;
        desc.debug_info = Some(DebugInfo {
            level: DebugInfoLevel::None,
            flags: DebugFlags::EMIT_ASM,
        });
        let spirv = service.compile(desc).unwrap();
        let asm = spirv.asm_str().unwrap();
        let major = asm
            .lines()
            .find(|line| line.contains("OpMemberDecorate") && (line.contains("RowMajor") || line.contains("ColMajor")))
            .expect("matrix buffer should have a layout decoration");
        decorations.push(major.contains("RowMajor"));
    }
    assert_ne!(decorations[0], decorations[1]);
}
