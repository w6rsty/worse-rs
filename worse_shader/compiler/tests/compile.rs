use worse_shader_compiler::*;
const SOURCE: &str = r#"
RWStructuredBuffer<float> output;
[numthreads(4, 1, 1)]
void dispatch_main(uint3 id : SV_DispatchThreadID) { output[id.x] = float(id.x) * 2.0; }
"#;
fn token(service: &mut CompileService, source: &str) -> ModuleUnitToken {
    service
        .sources()
        .enqueue_memory("migration_test", "migration_test.slang", source.as_bytes())
        .unwrap()
}
fn job(source: ModuleUnitToken, artifacts: ArtifactMask) -> CompileDesc<'static> {
    CompileDesc {
        artifacts,
        ..CompileDesc::new(source, EntryPointRef::new("dispatch_main", Stage::Compute))
    }
}
#[test]
fn memory_source_keeps_profile_entry_name_and_owned_outputs() {
    let mut service = CompileService::new().unwrap();
    let source = token(&mut service, SOURCE);
    let compiled = service
        .compile(
            &BatchConfig::default(),
            job(source, ArtifactMask::CODE | ArtifactMask::ASM),
        )
        .unwrap();
    drop(service);
    assert_eq!(compiled.words().unwrap()[0], 0x0723_0203);
    assert_eq!(compiled.words().unwrap()[1], 0x0001_0500);
    let asm = compiled.asm_str().unwrap();
    for text in ["OpEntryPoint GLCompute", "\"dispatch_main\"", "LocalSize 4 1 1"] {
        assert!(asm.contains(text), "{asm}");
    }
    assert_eq!(compiled.debug_bytes().unwrap_err(), SpirvError::NoDebugInfo);
}
#[test]
fn file_source_resolves_relative_includes() {
    let mut service = CompileService::new().unwrap();
    let source = service
        .sources()
        .enqueue_file(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/compute.slang"))
        .unwrap();
    let compiled = service
        .compile(&BatchConfig::default(), job(source, ArtifactMask::CODE))
        .unwrap();
    assert_eq!(compiled.words().unwrap()[0], 0x0723_0203);
    assert!(compiled.asm_str().is_none());
}
#[test]
fn errors_and_invalid_tokens_are_reported() {
    let mut service = CompileService::new().unwrap();
    let bad = token(&mut service, "this is not Slang !!!");
    assert!(
        service
            .compile(&BatchConfig::default(), job(bad, ArtifactMask::CODE))
            .unwrap_err()
            .to_string()
            .contains("error")
    );
    let source = token(&mut service, SOURCE);
    let mut desc = job(source, ArtifactMask::CODE);
    desc.entry.name = "dispatch_main\0invalid";
    assert!(matches!(
        service.compile(&BatchConfig::default(), desc),
        Err(Error::Slang(slang_auto::Error::InteriorNul(_)))
    ));
    assert!(
        service
            .sources()
            .enqueue_memory("nul", "nul.slang", b"bad\0source".as_slice())
            .is_err()
    );
    service.sources().release(source).unwrap();
    assert!(matches!(
        service.compile(&BatchConfig::default(), job(source, ArtifactMask::CODE)),
        Err(Error::InvalidToken)
    ));
    let mut other = CompileService::new().unwrap();
    assert!(matches!(
        other.compile(&BatchConfig::default(), job(bad, ArtifactMask::CODE)),
        Err(Error::InvalidToken)
    ));
}
#[test]
fn separate_debug_artifact_survives_service_drop() {
    let mut service = CompileService::new().unwrap();
    let source = token(&mut service, SOURCE);
    let config = BatchConfig {
        debug_info: Some(DebugInfo {
            level: DebugInfoLevel::Standard,
            flags: DebugFlags::SEPARATE,
        }),
        ..Default::default()
    };
    let compiled = service.compile(&config, job(source, ArtifactMask::ALL)).unwrap();
    drop(service);
    assert_eq!(compiled.words().unwrap()[0], 0x0723_0203);
    assert_eq!(&compiled.debug_bytes().unwrap()[..4], &0x0723_0203u32.to_le_bytes());
    assert!(!compiled.asm_str().unwrap().is_empty());
}
#[test]
fn matrix_layout_changes_buffer_decorations() {
    let source = r#"
struct Params { float3x4 transform; };
ConstantBuffer<Params> params;
RWStructuredBuffer<float> output;
[numthreads(4,1,1)] void dispatch_main(uint3 id:SV_DispatchThreadID) { output[id.x] = params.transform[id.x%3][id.x%4]; }
"#;
    let mut service = CompileService::new().unwrap();
    let token = token(&mut service, source);
    let mut decorations = Vec::new();
    for layout in [MatrixLayout::RowMajor, MatrixLayout::ColumnMajor] {
        let config = BatchConfig {
            matrix_layout: layout,
            ..Default::default()
        };
        let compiled = service
            .compile(&config, job(token, ArtifactMask::CODE | ArtifactMask::ASM))
            .unwrap();
        let asm = compiled.asm_str().unwrap();
        let major = asm
            .lines()
            .find(|line| line.contains("OpMemberDecorate") && (line.contains("RowMajor") || line.contains("ColMajor")))
            .unwrap();
        decorations.push(major.contains("RowMajor"));
    }
    assert_ne!(decorations[0], decorations[1]);
}
