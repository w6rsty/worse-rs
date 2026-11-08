use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
use worse_shader_compiler::*;
static ID: AtomicUsize = AtomicUsize::new(0);
struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "worse-compiler-{}-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn write(&self, name: &str, bytes: &str) {
        std::fs::write(self.0.join(name), bytes).unwrap();
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
const ROOT: &str = r#"
import common;
RWStructuredBuffer<float> output;
[numthreads(4,1,1)] void main(uint3 id:SV_DispatchThreadID) { output[id.x] = transform(float(id.x)); }
"#;
fn config(dir: &Path) -> BatchConfig {
    BatchConfig {
        search_paths: vec![dir.to_owned()],
        ..Default::default()
    }
}
fn jobs(source: ModuleUnitToken) -> [CompileDesc<'static>; 2] {
    [CompileDesc::new(source, EntryPointRef::new("main", Stage::Compute)); 2]
}
fn success(report: &BatchReport) {
    assert!(report.jobs.iter().all(|j| j.result.is_ok()), "{report:?}");
    assert!(report.warnings.is_empty(), "{report:?}");
}
#[test]
fn shared_imports_cache_across_sessions_and_invalidate_includes() {
    let files = Files::new();
    files.write(
        "common.slang",
        "#include \"value.slangh\"\npublic float transform(float x) { return x * VALUE; }",
    );
    files.write("value.slangh", "#define VALUE 2.0\n");
    let mut service = CompileService::new().unwrap();
    let root = service
        .sources()
        .enqueue_memory("root", files.0.join("root.slang"), ROOT.as_bytes())
        .unwrap();
    let mut output = MemoryOutput::default();
    let cold = service
        .compile_batch(&config(&files.0), &jobs(root), &mut output)
        .unwrap();
    success(&cold);
    assert_eq!(cold.stats.sessions, 1);
    assert_eq!(cold.stats.source_loads, 1);
    assert_eq!(cold.stats.file_reads, 2, "one read of each shared source/include");
    assert_eq!(cold.stats.cache_writes, 1, "direct root must not be serialized");
    let before = output.artifacts[0].1.bytes().to_vec();
    let warm = service
        .compile_batch(&config(&files.0), &jobs(root), &mut MemoryOutput::default())
        .unwrap();
    success(&warm);
    assert_eq!(warm.stats.cache_hits, 1);
    files.write("value.slangh", "#define VALUE 7.0\n");
    let mut changed = MemoryOutput::default();
    let invalidated = service
        .compile_batch(&config(&files.0), &jobs(root), &mut changed)
        .unwrap();
    success(&invalidated);
    assert_eq!(invalidated.stats.cache_hits, 0);
    assert_ne!(before, changed.artifacts[0].1.bytes());
}
#[test]
fn disk_cache_reopens_and_corruption_falls_back_to_source() {
    let files = Files::new();
    files.write("common.slang", "public float transform(float x) { return x + 3.0; }");
    let cache_dir = files.0.join("cache");
    let cache = CacheConfig {
        directory: Some(cache_dir.clone()),
        ..Default::default()
    };
    for iteration in 0..3 {
        let mut service = CompileService::with_cache(cache.clone()).unwrap();
        let root = service
            .sources()
            .enqueue_memory("root", files.0.join("root.slang"), ROOT.as_bytes())
            .unwrap();
        let report = service
            .compile_batch(&config(&files.0), &jobs(root), &mut MemoryOutput::default())
            .unwrap();
        success(&report);
        assert_eq!(report.stats.cache_hits, usize::from(iteration == 1));
        if iteration == 1 {
            for file in std::fs::read_dir(&cache_dir).unwrap() {
                let path = file.unwrap().path();
                if path.extension().is_some_and(|s| s == "ir") {
                    std::fs::write(path, b"damaged").unwrap();
                }
            }
        }
    }
}
#[test]
fn individual_errors_continue_and_both_sinks_share_artifacts() {
    let files = Files::new();
    let mut service = CompileService::new().unwrap();
    let source = service
        .sources()
        .enqueue_memory(
            "simple",
            files.0.join("simple.slang"),
            b"[numthreads(1,1,1)] void main() {}".as_slice(),
        )
        .unwrap();
    let mut descriptors = jobs(source);
    descriptors[0].entry.name = "missing";
    let mut output = BothOutput(
        MemoryOutput::default(),
        FileOutput::new(files.0.join("out"), vec!["bad".into(), "good".into()]).unwrap(),
    );
    let report = service
        .compile_batch(&BatchConfig::default(), &descriptors, &mut output)
        .unwrap();
    assert!(report.jobs[0].result.is_err());
    assert!(report.jobs[1].result.is_ok());
    assert_eq!(output.0.artifacts.len(), 1);
    assert_eq!(
        std::fs::read(files.0.join("out/good.spv")).unwrap(),
        output.0.artifacts[0].1.bytes()
    );
    // Existing destinations can be atomically replaced in an offline rebuild.
    let repeat = service
        .compile_batch(
            &BatchConfig::default(),
            &descriptors[1..],
            &mut FileOutput::new(files.0.join("out"), vec!["good".into()]).unwrap(),
        )
        .unwrap();
    assert!(repeat.jobs[0].result.is_ok());
    assert!(FileOutput::new(&files.0, vec!["duplicate".into(), "duplicate".into()]).is_err());
}
#[test]
fn empty_batches_and_conflicting_revisions_are_validated() {
    let mut service = CompileService::new().unwrap();
    let empty = service
        .compile_batch(&BatchConfig::default(), &[], &mut MemoryOutput::default())
        .unwrap();
    assert_eq!(empty.stats.sessions, 0);
    let a = service
        .sources()
        .enqueue_memory("same", "same.slang", b"void a(){}".as_slice())
        .unwrap();
    let b = service
        .sources()
        .enqueue_memory("same", "same.slang", b"void b(){}".as_slice())
        .unwrap();
    let mut tasks = jobs(a);
    tasks[1].source = b;
    assert!(matches!(
        service.compile_batch(&BatchConfig::default(), &tasks, &mut MemoryOutput::default()),
        Err(Error::InvalidBatch(_))
    ));
}

#[test]
fn diamond_imports_reuse_dependencies_and_configuration_partitions_cache() {
    let files = Files::new();
    files.write("leaf.slang", "public float leaf(float x) { return x + 1.0; }");
    files.write(
        "left.slang",
        "import leaf; public float left(float x) { return leaf(x); }",
    );
    files.write(
        "right.slang",
        "import leaf; public float right(float x) { return leaf(x) * 2.0; }",
    );
    let root_text = b"import left; import right; RWStructuredBuffer<float> o; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){o[id.x]=left(float(id.x))+right(float(id.x));}";
    let mut service = CompileService::new().unwrap();
    let root = service
        .sources()
        .enqueue_memory("diamond", files.0.join("diamond.slang"), root_text.as_slice())
        .unwrap();
    let config = config(&files.0);
    let cold = service
        .compile_batch(&config, &jobs(root), &mut MemoryOutput::default())
        .unwrap();
    success(&cold);
    assert_eq!(cold.stats.file_reads, 3);
    assert_eq!(cold.stats.cache_writes, 3);
    let warm = service
        .compile_batch(&config, &jobs(root), &mut MemoryOutput::default())
        .unwrap();
    success(&warm);
    assert_eq!(warm.stats.cache_hits, 3);
    // A different direct input still shares imports, but its IR is not persisted.
    let next = service
        .sources()
        .enqueue_memory("diamond2", files.0.join("diamond2.slang"), root_text.as_slice())
        .unwrap();
    let next_report = service
        .compile_batch(&config, &jobs(next), &mut MemoryOutput::default())
        .unwrap();
    success(&next_report);
    assert_eq!(next_report.stats.cache_hits, 3);
    let changed = BatchConfig {
        defines: vec![("OTHER_CONFIGURATION".into(), "1".into())],
        ..config.clone()
    };
    let other = service
        .compile_batch(&changed, &jobs(root), &mut MemoryOutput::default())
        .unwrap();
    success(&other);
    assert_eq!(other.stats.cache_hits, 0);
    files.write("leaf.slang", "public float leaf(float x) { return x + 9.0; }");
    let invalidated = service
        .compile_batch(&config, &jobs(root), &mut MemoryOutput::default())
        .unwrap();
    success(&invalidated);
    assert_eq!(invalidated.stats.cache_hits, 0);
}

#[test]
fn file_tokens_freeze_source_and_refresh_explicitly() {
    let files = Files::new();
    let root = "RWStructuredBuffer<float> o; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){o[id.x]=2.0;}";
    files.write("snapshot.slang", root);
    let mut service = CompileService::new().unwrap();
    let token = service.sources().enqueue_file(files.0.join("snapshot.slang")).unwrap();
    let desc = jobs(token)[0];
    let before = service.compile(&BatchConfig::default(), desc).unwrap();
    files.write("snapshot.slang", &root.replace("2.0", "7.0"));
    let same = service.compile(&BatchConfig::default(), desc).unwrap();
    assert_eq!(before.bytes(), same.bytes());
    let refreshed = service.sources().refresh_file(token).unwrap();
    let after = service.compile(&BatchConfig::default(), jobs(refreshed)[0]).unwrap();
    assert_ne!(before.bytes(), after.bytes());
}

#[test]
fn specialization_module_is_explicitly_linked_and_not_persisted() {
    let files = Files::new();
    let mut service = CompileService::new().unwrap();
    let root = service
        .sources()
        .enqueue_memory(
            "material",
            files.0.join("material.slang"),
            br#"
        extern static const float selected;
        RWStructuredBuffer<float> output;
        [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID) { output[id.x] = selected * float(id.x); }
    "#
            .as_slice(),
        )
        .unwrap();
    let implementation = service
        .sources()
        .enqueue_memory(
            "implementation",
            files.0.join("implementation.slang"),
            b"export static const float selected = 3.0;".as_slice(),
        )
        .unwrap();
    let mut desc = jobs(root)[0];
    desc.specialization = Some(implementation);
    let other = service
        .sources()
        .enqueue_memory(
            "implementation2",
            files.0.join("implementation2.slang"),
            "export static const float selected = 7.0;",
        )
        .unwrap();
    let other_desc = CompileDesc {
        specialization: Some(other),
        ..desc
    };
    let mut sink = MemoryOutput::default();
    let report = service
        .compile_batch(&BatchConfig::default(), &[desc, other_desc], &mut sink)
        .unwrap();
    success(&report);
    assert_eq!(report.stats.source_loads, 3);
    assert_eq!(report.stats.cache_writes, 0);
    assert!(!sink.artifacts[0].1.bytes().is_empty());
    assert_ne!(sink.artifacts[0].1.bytes(), sink.artifacts[1].1.bytes());
}

#[test]
fn memory_import_updates_invalidate_ir_without_disk_reads() {
    let files = Files::new();
    let mut service = CompileService::new().unwrap();
    let root = service
        .sources()
        .enqueue_memory("root", files.0.join("root.slang"), ROOT.as_bytes())
        .unwrap();
    let shared = service
        .sources()
        .register_import_memory(
            files.0.join("common.slang"),
            b"public float transform(float x) { return x + 1.0; }".as_slice(),
        )
        .unwrap();
    for iteration in 0..2 {
        let report = service
            .compile_batch(&config(&files.0), &jobs(root), &mut MemoryOutput::default())
            .unwrap();
        success(&report);
        assert_eq!(report.stats.file_reads, 0);
        assert_eq!(report.stats.cache_hits, iteration);
    }
    service.sources().release(shared).unwrap();
    let replacement = service
        .sources()
        .register_import_memory(
            files.0.join("common.slang"),
            b"public float transform(float x) { return x + 5.0; }".as_slice(),
        )
        .unwrap();
    assert_ne!(replacement, shared, "recycled slots must advance generation");
    assert!(matches!(service.sources().release(shared), Err(Error::InvalidToken)));
    let report = service
        .compile_batch(&config(&files.0), &jobs(root), &mut MemoryOutput::default())
        .unwrap();
    success(&report);
    assert_eq!(report.stats.file_reads, 0);
    assert_eq!(report.stats.cache_hits, 0);
}

#[test]
fn missing_input_does_not_discard_other_jobs() {
    let files = Files::new();
    let mut service = CompileService::new().unwrap();
    let bad = service.sources().enqueue_file(files.0.join("missing.slang")).unwrap();
    let good = service
        .sources()
        .enqueue_memory(
            "valid",
            files.0.join("valid.slang"),
            "[numthreads(1,1,1)] void main() {}",
        )
        .unwrap();
    let mut sink = MemoryOutput::default();
    let report = service
        .compile_batch(&BatchConfig::default(), &[jobs(bad)[0], jobs(good)[0]], &mut sink)
        .unwrap();
    assert!(report.jobs[0].result.is_err());
    assert!(report.jobs[1].result.is_ok());
    assert_eq!(sink.artifacts.len(), 1);
}

#[test]
fn a_new_higher_priority_dependency_invalidates_cached_importers() {
    let files = Files::new();
    let high = files.0.join("high");
    let low = files.0.join("low");
    let middle = files.0.join("middle");
    std::fs::create_dir_all(&high).unwrap();
    std::fs::create_dir_all(&low).unwrap();
    std::fs::create_dir_all(&middle).unwrap();
    std::fs::write(low.join("leaf.slang"), "public float leaf(float x) { return x + 1.0; }").unwrap();
    std::fs::write(
        middle.join("common.slang"),
        "import leaf; public float transform(float x) { return leaf(x); }",
    )
    .unwrap();
    let mut service = CompileService::new().unwrap();
    let root = service
        .sources()
        .enqueue_memory("root", files.0.join("root.slang"), ROOT)
        .unwrap();
    let config = BatchConfig {
        search_paths: vec![high.clone(), low, middle],
        ..Default::default()
    };
    let mut before = MemoryOutput::default();
    success(&service.compile_batch(&config, &jobs(root), &mut before).unwrap());
    let warm = service
        .compile_batch(&config, &jobs(root), &mut MemoryOutput::default())
        .unwrap();
    success(&warm);
    assert_eq!(warm.stats.cache_hits, 2);
    std::fs::write(
        high.join("leaf.slang"),
        "public float leaf(float x) { return x + 8.0; }",
    )
    .unwrap();
    let mut after = MemoryOutput::default();
    let report = service.compile_batch(&config, &jobs(root), &mut after).unwrap();
    success(&report);
    assert_eq!(report.stats.cache_hits, 0);
    assert_ne!(before.artifacts[0].1.bytes(), after.artifacts[0].1.bytes());
}

#[test]
fn caching_can_be_disabled_or_evicted_without_changing_results() {
    let files = Files::new();
    files.write("common.slang", "public float transform(float x) { return x + 2.0; }");
    for enabled in [false, true] {
        let mut service = CompileService::with_cache(CacheConfig {
            enabled,
            max_memory_bytes: 0,
            directory: None,
        })
        .unwrap();
        let root = service
            .sources()
            .enqueue_memory("root", files.0.join("root.slang"), ROOT)
            .unwrap();
        for _ in 0..2 {
            let report = service
                .compile_batch(&config(&files.0), &jobs(root), &mut MemoryOutput::default())
                .unwrap();
            success(&report);
            assert_eq!(report.stats.cache_hits, 0);
            assert_eq!(report.stats.cache_writes, usize::from(enabled));
        }
    }
}
