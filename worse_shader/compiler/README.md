# worse_shader_compiler

Thread-bound Slang compiler with queued immutable sources, batches, serialized
import caches and streaming output. `CompileService` owns the registry and cache;
each batch creates one native session. Use a separate service on each compiler
thread. `slang-auto` owns the FFI and native interfaces.

```rust,no_run
use worse_shader_compiler::*;

let mut service = CompileService::with_cache(CacheConfig {
    directory: Some("target/shader-cache".into()),
    ..Default::default()
})?;
let source = service.sources().enqueue_file("shaders/material.slang")?;
let jobs = [
    CompileDesc::new(source, EntryPointRef::new("vertex_main", Stage::Vertex)),
    CompileDesc::new(source, EntryPointRef::new("fragment_main", Stage::Fragment)),
];
let config = BatchConfig {
    search_paths: vec!["shaders".into()],
    ..Default::default()
};
let mut output = MemoryOutput::default();
let report = service.compile_batch(&config, &jobs, &mut output)?;
for job in report.jobs {
    for diagnostic in job.diagnostics { eprint!("{diagnostic}"); }
    job.result?;
}
for (job, spirv) in output.artifacts {
    println!("job {job}: {} bytes", spirv.bytes().len());
}
service.sources().release(source)?;
# Ok::<(), Error>(())
```

## Inputs and snapshots

`enqueue_memory(name, virtual_path, source)` accepts `String`, `Vec<u8>`,
`Arc<[u8]>`, `&str`, or `&[u8]`. Owned String/Vec storage is transferred to the
native blob without repacking; Arc inputs share their storage. Borrowed inputs
are copied once at submission. Ordinary Rust text does not need CString.

File units are read in queue order on first preparation, and their contents
remain frozen while their token is live. `refresh_file(token)` queues a new
snapshot. `release(token)` frees the backing source and invalidates the handle;
recycled slots advance generation. Foreign-service and expired tokens are errors.
Direct inputs do not enter the persistent compiled-module cache, but repeated
jobs share their source snapshot and loaded module inside a batch.

`register_import_memory(path, source)` mounts an in-memory import or include
through the same filesystem used by Slang. Release the old registration before
registering a new revision at the same path. These imported modules are eligible
for IR caching. A module explicitly submitted as a job root or specialization
in the same batch takes priority as a nonpersistent direct input.

## Batches and specialization

Session options live in `BatchConfig`: profile, matrix layout, optimization,
debug settings, search paths and defines. Every job in one call shares them.
Different macro environments require separate batches. Different revisions of
one module or conflicting module names in one batch are rejected.

`specialization: Some(token)` explicitly composes an implementation module with
the root and entry point. For example the root can declare
`extern static const float material_scale;`, with the implementation defining
`export static const float material_scale = 2.0;`. This is link-time module
specialization; generic type/value arguments are not exposed by this API yet.

`ArtifactMask` requests code, assembly and/or separate debug data per job. ASM
adds a second target to the batch. DEBUG requires `BatchConfig.debug_info` with
`DebugFlags::SEPARATE`. `compile(&config, job)` is the one-job memory-output
convenience method. Configuration/token failures return an outer error; source,
entry, link, codegen and output failures are recorded per job and later jobs
continue. Diagnostics and cache warnings are returned, not automatically logged.

## Module cache

The memory cache is enabled by default. Disable it with
`CacheConfig { enabled: false, ..Default::default() }`. `directory: Some(path)`
also persists checked module IR and JSON manifests to disk. The memory byte
limit bounds retained IR bytes; source tokens and manifest overhead are separate.
Eviction is LRU. `clear_memory_cache()` leaves on-disk records available.

Cache identity includes schema, compiler build identifier, full batch
configuration, target set and normalized search paths. Manifests contain content
hashes, module dependency closures and failed path lookups. Import/include files
are hashed once per batch, including cache-hit validation. This is strict content
validation rather than a watcher-driven no-IO mode. In-memory registrations need
no disk read. Direct roots and modules depending on them are not persisted.

The filesystem has stable absolute path identities. Module discovery combines
loaded modules, file dependency closures and the SDK's experimental dependency
interface. The graph conservatively includes transitive edges. If discovery is
unsupported, compilation still works and cache collection reports a warning.
Serialized modules do not include their imports; imports resolve on demand
through this same cache-aware filesystem in the next session.

IR has a BLAKE3 checksum. Missing, damaged or incompatible records fall back to
source; native rejection of cached IR retries root loading once in a fresh
session. Disk records use temporary-file replacement, with the manifest published
after its checksummed IR. `BatchReport.stats` reports sessions, direct source
loads, successful source file reads, accepted cache candidates and cache writes.
Cache hits measure accepted IR candidates, not a promise that Slang performs no
additional validation. Disk garbage collection of unreferenced IR files is not
automatic; the configured cache directory may be removed between runs.

## Output and memory

`MemoryOutput` retains artifacts. `FileOutput::new(directory, names)` writes
`.spv`, `.spvasm`, and `.debug.spv` outputs as requested. Names correspond to job
indices and must be unique relative stems. `BothOutput(memory, files)` shares the
same blobs between sinks. Custom `OutputSink` implementations can process each
result immediately, bounding offline output memory. Each file replacement is
atomic; a batch or a set of several artifact files is not a filesystem transaction.

Code, debug and serialized blobs own their buffers and do not keep an old batch
session alive. Blob cloning adjusts reference counts without copying bytes.
`words()` borrows aligned native-endian SPIR-V and copies only when conversion
is required. Native compiler operations remain thread-bound; cross-thread
artifact delivery requires explicitly owned transferable storage.
