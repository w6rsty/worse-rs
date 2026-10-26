//! Engine logging, built on [`tracing`].
//!
//! # Overview
//!
//! A single call to [`init`] installs a global subscriber that fans every log
//! record out to up to four independent sinks. Each sink is optional, so the
//! same module serves both the engine (long-lived, multithreaded, needs crash
//! forensics) and the tool-chain binaries (short-lived, console only).
//!
//! | Sink            | Backing store            | Blocks the caller? | Survives a hard crash? |
//! |-----------------|--------------------------|--------------------|------------------------|
//! | Console         | stdout / stderr          | configurable       | n/a                    |
//! | Rolling file    | `<prefix>.<date>.log`    | no (worker thread) | no                     |
//! | Crash channel   | `<prefix>.critical.log`  | yes (unbuffered)   | yes                    |
//! | Ring buffer     | memory                   | yes (a mutex push) | only via a dump        |
//!
//! ## Why a separate crash channel?
//!
//! The rolling file sink hands records to a background worker thread through a
//! bounded channel. That keeps the render thread from ever touching the disk,
//! but it means records live in memory for a short while before they land.
//! `tracing_appender`'s writer has a no-op `flush`; the only guaranteed flush
//! happens when its `WorkerGuard` is dropped, and destructors do not run on
//! `abort`, `SIGSEGV`, a GPU driver reset, or [`std::process::exit`].
//!
//! The crash channel therefore bypasses the worker entirely: every record is
//! written straight to the OS with one `write` syscall. It is filtered down to
//! `ERROR` and above (see [`WriteFileConfig::crash_from`]) so the syscall cost
//! is paid only when something has already gone wrong.
//!
//! # Quick start
//!
//! Engine, with everything enabled:
//!
//! ```ignore
//! mod log;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Pin the local UTC offset before any thread is spawned. See the
//!     // "Time zones" gotcha below.
//!     log::capture_local_time_zone_offset();
//!
//!     // The guard must stay alive for the whole program: dropping it flushes
//!     // and shuts down the async writers.
//!     let _guard = log::init(log::LogConfig::engine())?;
//!
//!     let mut frame_index = 0u64;
//!     loop {
//!         log::set_frame(frame_index);
//!         tracing::info!(dt_ms = 16.7, "frame tick");
//!         frame_index += 1;
//!     }
//! }
//! ```
//!
//! A tool-chain binary; console only, no threads, no files:
//!
//! ```ignore
//! fn main() {
//!     let _guard = log::init(log::LogConfig::cli()).expect("init log");
//!     tracing::info!("scanning assets");
//!     tracing::error!("aborting");
//! }
//! ```
//!
//! Anything in between is expressed by building the config directly. Here the
//! console is on but files are not, which is the usual state early in a
//! project:
//!
//! ```ignore
//! let config = log::LogConfig {
//!     filter: String::from("info,worse::render=debug,wgpu_core=warn"),
//!     write_console_config: Some(log::WriteConsoleConfig::default()),
//!     write_file_config: None,
//!     ring_capacity: 256,
//!     panic_hook: true,
//! };
//! let _guard = log::init(config)?;
//! ```
//!
//! # Emitting records
//!
//! Use the [`tracing`] macros directly; this module deliberately does not wrap
//! them. Prefer structured fields over pre-formatted strings, because fields
//! are only evaluated once the filter has decided the record is enabled:
//!
//! ```ignore
//! tracing::debug!(entity = id, pos = ?transform.translation, "entity moved");
//! ```
//!
//! Two extras are provided:
//!
//! - `critical!(...)` logs at `ERROR`, forces the record onto the crash
//!   channel, and dumps the ring buffer. Use it on the way out of an
//!   unrecoverable failure.
//! - Adding a `sync = true` field to any record routes it to the crash channel
//!   regardless of level. The field name is part of the callsite metadata, so
//!   the routing decision is cached per callsite and costs nothing at runtime.
//!
//! # Runtime control
//!
//! [`set_filter`] swaps the filter while the process runs, which is what an
//! in-game console command would call. [`recent`] returns the tail of the ring
//! buffer for display in that same console. [`dropped_lines`] reports how many
//! records the async writers discarded; wire it into a debug HUD, because the
//! writers are lossy by design and will silently drop rather than stall the
//! caller.
//!
//! # Filter syntax
//!
//! The filter string uses [`tracing_subscriber::filter::Targets`], for example
//! `"info,worse::render=debug,wgpu_core=warn"`. A bare level sets the default;
//! `target=level` pairs override it by module-path prefix, so
//! `worse::render=debug` also covers `worse::render::mesh`.
//!
//! This is a subset of `EnvFilter`: span-field predicates such as
//! `[span{field=value}]` are not supported. That is deliberate. Those
//! predicates defeat `tracing`'s per-callsite interest cache and force a
//! dynamic evaluation on every event, which is exactly what a per-frame logger
//! cannot afford. Dropping them also keeps the `regex` dependency out of the
//! build.
//!
//! Third-party crates that log through the `log` facade (wgpu, winit, naga)
//! are captured automatically, because installing the subscriber also installs
//! a `LogTracer` bridge. They are verbose at `debug`, so keep entries like
//! `wgpu_core=warn` in the default filter.
//!
//! # Cargo.toml
//!
//! ```toml
//! [dependencies]
//! tracing = "0.1"
//! tracing-appender = "0.2"
//! tracing-subscriber = { version = "0.3", features = ["time", "local-time"] }
//! time = { version = "0.3", features = ["formatting", "parsing", "local-offset", "macros"] }
//! ```
//!
//! `tracing-subscriber`'s `env-filter` feature is not needed and is left off on
//! purpose; see "Filter syntax" above.
//!
//! Consider having the final binary — not this module — declare
//! `tracing = { version = "0.1", features = ["release_max_level_info"] }`.
//! That compiles `trace!` and `debug!` out of release builds entirely, which
//! matters inside hot loops. It belongs in the binary because Cargo unifies
//! features across the graph, so a library enabling it would impose the choice
//! on every consumer.
//!
//! # Gotchas
//!
//! **The guard must outlive everything you want logged.** [`init`] returns a
//! [`LoggerGuard`]; binding it to `_` instead of a named variable drops it
//! immediately and silently disables the async sinks.
//!
//! **`std::process::exit` skips destructors.** Anything still sitting in an
//! async writer's queue is lost. Short-lived binaries should set
//! [`WriteConsoleConfig::blocking`] to `true` — [`LogConfig::cli`] already
//! does — or return an `ExitCode` from `main` instead of calling `exit`.
//!
//! **Time zones.** [`capture_local_time_zone_offset`] pins the offset once at
//! startup. Recent versions of `time` can read the offset safely from a
//! multithreaded process, so this is no longer strictly required, but pinning
//! it keeps timestamps stable across a daylight-saving transition in a
//! long-running session and guards against the dependency being downgraded.
//! Call it before spawning threads; if it is never called, the first timestamp
//! formats the offset lazily and falls back to UTC on failure.
//!
//! **[`TimerKind::None`] also disables span timing.** Suppressing the
//! timestamp column suppresses the `time.busy` / `time.idle` fields that
//! `tracing` reports when a span closes. The two cannot be combined.
//!
//! **The `sync = true` marker is visible in the output.** The ring buffer hides
//! it, but the console and file sinks use `tracing`'s stock formatter, which
//! does not know it is an internal routing flag. It renders as a trailing
//! `sync=true`.
//!
//! **[`WriteFileConfig::directory`] defaults to a relative path.** That
//! resolves against the working directory, which is the project root under
//! `cargo run` but is unpredictable when a shipped build is launched from a
//! shortcut or a game store client. Before shipping, resolve it against the
//! executable's directory or a platform data directory.
//!
//! # Performance notes
//!
//! Records are dropped, never queued indefinitely: both async writers run in
//! lossy mode, so a full queue discards the record instead of blocking the
//! caller. Watch [`dropped_lines`] to know when the queue length needs raising.
//!
//! Spans cost a registry insertion plus an enter/exit pair. Per-frame or
//! per-subsystem spans are fine; per-entity spans are not.
//!
//! Running the console through the async writer reorders it against direct
//! `println!` output, since one path goes through a worker thread and the other
//! does not. Set [`WriteConsoleConfig::blocking`] when that ordering matters.

use std::collections::VecDeque;
use std::fmt::Write as FmtWrite;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use tracing::field::{Field, Visit};
use tracing::level_filters::LevelFilter;
use tracing::{Event, Subscriber};
use tracing_appender::non_blocking::{ErrorCounter, NonBlockingBuilder, WorkerGuard};
use tracing_appender::rolling::RollingFileAppender;
use tracing_subscriber::filter::{Targets, filter_fn};
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::fmt::{self, MakeWriter};
use tracing_subscriber::layer::{Context, Layer, Layered, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{Registry, reload};

/// Re-exported so call sites can reach the macros through this module.
/// Emitting through `tracing` directly is equally fine.
#[allow(unused_imports)]
pub use tracing::{debug, error, info, instrument, trace, warn};
pub use tracing_appender::rolling::Rotation;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which standard stream the console sink writes to.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleTarget {
    Stdout,
    Stderr,
}

/// Shape of the timestamp column on the console sink.
///
/// Variants a given binary does not select are not dead code; they are the
/// vocabulary of the configuration API.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimerKind {
    /// No timestamp column. Also disables span timing; see the module docs.
    None,
    /// Seconds since process start, e.g. `12.345678900s`. Convenient when
    /// correlating log output against a frame graph.
    Uptime,
    /// RFC 3339 in UTC, e.g. `2026-07-28T14:59:34.756369Z`. This is the stock
    /// `tracing` format.
    Utc,
    /// Local wall-clock time, e.g. `23:34:40.707`. The date is omitted because
    /// it is already in the rolling file's name. Pair with
    /// [`capture_local_time_zone_offset`].
    Local,
}

/// Console sink settings. Absent from [`LogConfig`] means no console output.
#[derive(Clone, Debug)]
pub struct WriteConsoleConfig {
    pub target: ConsoleTarget,
    /// Colour and style escape codes. Turn off when the output is piped.
    pub ansi: bool,
    pub timer_kind: TimerKind,
    /// Write on the calling thread instead of handing off to a worker.
    ///
    /// Required for anything that may call [`std::process::exit`], since that
    /// skips the destructor that would otherwise flush the worker's queue.
    /// Costs one syscall per record, which is irrelevant for a binary that
    /// prints a few dozen lines and unacceptable inside a frame loop.
    pub blocking: bool,
    /// Bounded queue depth for the worker, in records. Ignored when
    /// [`Self::blocking`] is set. A full queue drops records; see
    /// [`dropped_lines`].
    pub queue_length: usize,
}

impl Default for WriteConsoleConfig {
    fn default() -> Self {
        Self {
            target: ConsoleTarget::Stdout,
            ansi: true,
            timer_kind: TimerKind::Local,
            blocking: false,
            queue_length: 8192,
        }
    }
}

/// File sink settings. Absent from [`LogConfig`] means the filesystem is never
/// touched — no directory is created and no handle is opened.
#[derive(Clone, Debug)]
pub struct WriteFileConfig {
    /// Created if missing. Relative paths resolve against the working
    /// directory; see the module docs before shipping.
    pub directory: PathBuf,
    /// Leading component of the file names, e.g. `engine` produces
    /// `engine.2026-07-28.log` and `engine.critical.log`.
    pub file_prefix: String,
    pub rotation_frequency: Rotation,
    /// Rotated files retained; older ones are deleted.
    pub max_files: usize,
    /// Bounded queue depth for the worker, in records.
    pub queue_length: usize,
    /// Write the rolling file on the calling thread instead of a worker.
    ///
    /// Turn this on temporarily when chasing a crash or an ordering problem:
    /// it removes the window in which records can be lost, at the cost of
    /// putting disk latency in the caller's path.
    pub force_sync: bool,
    /// Records at this level or more severe are additionally written, without
    /// buffering, to `<file_prefix>.critical.log`. `None` disables the crash
    /// channel.
    ///
    /// Records carrying a `sync = true` field take this path at any level.
    pub crash_from: Option<LevelFilter>,
}

impl Default for WriteFileConfig {
    fn default() -> Self {
        Self {
            directory: PathBuf::from("logs"),
            file_prefix: String::from("engine"),
            rotation_frequency: Rotation::DAILY,
            max_files: 8,
            queue_length: 8192,
            force_sync: false,
            crash_from: Some(LevelFilter::ERROR),
        }
    }
}

/// Top-level configuration passed to [`init`].
#[derive(Clone, Debug)]
pub struct LogConfig {
    /// `Targets` DSL, for example `"info,worse::render=debug,wgpu_core=warn"`.
    /// See the module docs for the supported subset.
    pub filter: String,
    pub write_console_config: Option<WriteConsoleConfig>,
    pub write_file_config: Option<WriteFileConfig>,
    /// Records retained in the in-memory ring buffer. `0` skips the layer
    /// entirely, which also skips the per-record string formatting it does.
    pub ring_capacity: usize,
    /// Install a panic hook that logs the payload and backtrace through the
    /// crash channel and dumps the ring buffer. Leave this off if something
    /// else already owns the hook, such as `color-eyre`.
    pub panic_hook: bool,
}

/// Reads the initial filter from the `WORSE_LOG` environment variable, falling
/// back to `info`.
fn default_filter() -> String {
    std::env::var("WORSE_LOG").unwrap_or_else(|_| String::from("info"))
}

impl LogConfig {
    /// Tool-chain preset: blocking stderr, no timestamps, no files, no ring
    /// buffer, no panic hook. Spawns no threads and creates no directories, so
    /// it is safe in a binary that exits via [`std::process::exit`].
    #[allow(dead_code)]
    pub fn cli_preset() -> Self {
        Self {
            filter: default_filter(),
            write_console_config: Some(WriteConsoleConfig {
                target: ConsoleTarget::Stderr,
                ansi: true,
                timer_kind: TimerKind::None,
                blocking: true,
                queue_length: 0,
            }),
            write_file_config: None,
            ring_capacity: 0,
            panic_hook: false,
        }
    }

    /// Engine preset: async console, rolling file, crash channel, ring buffer
    /// and panic hook, all on.
    #[allow(dead_code)]
    pub fn engine_preset() -> Self {
        Self {
            filter: default_filter(),
            write_console_config: Some(WriteConsoleConfig::default()),
            write_file_config: None,
            ring_capacity: 512,
            panic_hook: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum LogError {
    /// Creating the log directory or opening the crash file failed.
    Io(io::Error),
    /// The filter string did not parse, or the rolling appender rejected the
    /// file settings.
    Config(String),
    /// Another subscriber already holds the global slot. [`init`] is not
    /// idempotent; this most often means it was called twice.
    AlreadyInitialized,
    /// A runtime accessor was called before [`init`] succeeded.
    NotInitialized,
}

impl std::fmt::Display for LogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogError::Io(e) => write!(f, "Log IO error: {e}"),
            LogError::Config(s) => write!(f, "Log config error: {s}"),
            LogError::AlreadyInitialized => write!(f, "A global subscriber is already installed"),
            LogError::NotInitialized => write!(f, "Logger not initialized"),
        }
    }
}

impl std::error::Error for LogError {}

impl From<io::Error> for LogError {
    fn from(e: io::Error) -> Self {
        LogError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Frame counter
// ---------------------------------------------------------------------------

/// Sentinel meaning [`set_frame`] has never been called in this process, in
/// which case the frame prefix is omitted from ring buffer entries. This is
/// what lets the same module serve a frame loop and a one-shot binary without
/// a configuration switch.
const NEVER_SET_FRAME_COUNTER: u64 = u64::MAX;

static FRAME_COUNTER: AtomicU64 = AtomicU64::new(NEVER_SET_FRAME_COUNTER);

/// Records the current frame index, to be stamped onto subsequent ring buffer
/// entries. Call once per frame from the main loop.
///
/// This is deliberately a plain atomic rather than a parameter threaded
/// through the logging API, so that this module never has to depend on any
/// engine type.
#[inline]
#[allow(dead_code)]
pub fn set_frame(n: u64) {
    FRAME_COUNTER.store(n, Ordering::Relaxed);
}

/// The current frame index, or `None` if [`set_frame`] has never been called.
#[inline]
pub fn frame() -> Option<u64> {
    match FRAME_COUNTER.load(Ordering::Relaxed) {
        NEVER_SET_FRAME_COUNTER => None,
        n => Some(n),
    }
}

// ---------------------------------------------------------------------------
// Local time zone
// ---------------------------------------------------------------------------

static LOCAL_TIME_ZONE_OFFSET: OnceLock<time::UtcOffset> = OnceLock::new();

/// Resolves and pins the local UTC offset for the lifetime of the process.
///
/// Call it as the first statement in `main`, before any thread is spawned.
/// Pinning keeps timestamps monotonic across a daylight-saving transition and
/// avoids paying for the lookup repeatedly. If resolution fails the offset
/// falls back to UTC rather than erroring.
#[allow(dead_code)]
pub fn capture_local_time_zone_offset() {
    let _ = LOCAL_TIME_ZONE_OFFSET.set(time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC));
}

/// The pinned offset, resolving it lazily if [`capture_local_time_zone_offset`]
/// was never called.
fn local_time_zone_offset() -> time::UtcOffset {
    *LOCAL_TIME_ZONE_OFFSET.get_or_init(|| time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC))
}

// ---------------------------------------------------------------------------
// Unbuffered file writer (crash channel)
// ---------------------------------------------------------------------------

/// A shared append-only file handle with no userspace buffering.
///
/// Every write becomes one `write` syscall, so the record reaches the kernel
/// before the call returns and survives the process being killed. It does not
/// survive a power loss; that would need an `fsync` per record, which is not
/// worth the cost here.
#[derive(Clone)]
pub struct SyncFileWriter(Arc<Mutex<File>>);

impl SyncFileWriter {
    fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self(Arc::new(Mutex::new(file))))
    }

    /// Writes a pre-formatted blob, bypassing the subscriber. Used by
    /// [`dump_ring`], which has already done its own formatting.
    fn write_raw(&self, s: &str) {
        let mut file = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let _ = file.write_all(s.as_bytes());
    }
}

/// Held for the duration of one formatted record, serialising writers.
pub struct SyncGuard<'a>(MutexGuard<'a, File>);

impl Write for SyncGuard<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl<'a> MakeWriter<'a> for SyncFileWriter {
    type Writer = SyncGuard<'a>;

    fn make_writer(&'a self) -> Self::Writer {
        // A poisoned mutex only means some other thread panicked mid-write;
        // the file handle itself is still usable, and refusing to log during
        // a panic is the opposite of what is wanted here.
        SyncGuard(self.0.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

// ---------------------------------------------------------------------------
// In-memory ring buffer
// ---------------------------------------------------------------------------

/// Fixed-capacity FIFO of formatted records, retaining the most recent ones.
#[derive(Clone)]
struct Ring {
    inner: Arc<Mutex<VecDeque<String>>>,
    capacity: usize,
}

impl Ring {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
        }
    }

    /// Appends a record, evicting the oldest once at capacity.
    fn push(&self, line: String) {
        let mut buffer = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if buffer.len() >= self.capacity {
            buffer.pop_front();
        }
        buffer.push_back(line);
    }

    /// Clones out the newest `n` records, oldest first.
    fn snapshot(&self, n: usize) -> Vec<String> {
        let buffer = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let skip = buffer.len().saturating_sub(n);
        buffer.iter().skip(skip).cloned().collect()
    }
}

/// Layer that formats each record into [`Ring`].
///
/// This exists because the async file sink can lose the last few hundred
/// milliseconds of output when the process dies abruptly. The ring buffer is
/// written synchronously into memory, so dumping it from a panic hook recovers
/// exactly that lost tail. It doubles as the backing store for an in-game
/// console overlay.
struct RingLayer {
    ring: Ring,
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for RingLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let mut line = String::with_capacity(128);
        if let Some(frame) = frame() {
            let _ = write!(line, "[frame{frame:>7}]");
        }
        let _ = write!(line, "[{:>5}] {}: ", metadata.level(), metadata.target());
        event.record(&mut RingVisitor { out: &mut line });
        self.ring.push(line);
    }
}

/// Flattens a record's fields into a single line.
///
/// The `message` field is the record body and is written bare; everything else
/// is appended as `name=value`. The `sync` marker is skipped because it is an
/// internal routing flag rather than something the reader cares about.
struct RingVisitor<'a> {
    out: &'a mut String,
}

impl Visit for RingVisitor<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "message" => self.out.push_str(value),
            "sync" => {}
            n => {
                let _ = write!(self.out, " {n}={value}");
            }
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn core::fmt::Debug) {
        match field.name() {
            "message" => {
                let _ = write!(self.out, "{value:?}");
            }
            "sync" => {}
            n => {
                let _ = write!(self.out, " {n}={value:?}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

/// The layers are boxed and collected into a `Vec` so that sinks can be
/// included or omitted at runtime while still presenting one concrete type to
/// the registry.
type LayerStack = Vec<Box<dyn Layer<Registry> + Send + Sync>>;
type Stacked = Layered<LayerStack, Registry>;
type FilterHandle = reload::Handle<Targets, Stacked>;

/// Handles that the free functions below need after [`init`] returns. Kept
/// separate from [`LoggerGuard`] so that the guard can own the shutdown
/// semantics while these stay reachable from anywhere.
struct Global {
    reload: FilterHandle,
    crash: Option<SyncFileWriter>,
    ring: Option<Ring>,
    dropped: Vec<ErrorCounter>,
}

static GLOBAL: OnceLock<Global> = OnceLock::new();

/// Keeps the async writers alive.
///
/// Dropping this shuts down the worker threads and flushes whatever they still
/// hold, so it must outlive every record worth keeping. Bind it to a named
/// variable in `main`; binding to `_` drops it immediately.
#[must_use = "dropping the guard shuts down the async writers and loses queued records"]
pub struct LoggerGuard {
    _appenders: Vec<WorkerGuard>,
    emit_shutdown: bool,
}

impl Drop for LoggerGuard {
    fn drop(&mut self) {
        // A sentinel for post-mortem analysis: this is the last record to go
        // through the full pipeline, so a log file that does not end with it
        // was produced by a process that did not exit normally.
        //
        // Only worth emitting when there is a file to inspect later; on a
        // console-only binary it would just be noise on every run.
        if self.emit_shutdown {
            tracing::info!("Shutdown logger");
        }
    }
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

/// Builds the configured sinks and installs the global subscriber.
///
/// Returns [`LogError::AlreadyInitialized`] rather than panicking if a
/// subscriber is already installed, which keeps multi-binary test runs
/// well-behaved.
pub fn init(config: LogConfig) -> Result<LoggerGuard, LogError> {
    let targets: Targets = config
        .filter
        .parse()
        .map_err(|e| LogError::Config(format!("bad filter {:?}: {e}", config.filter)))?;

    // Wrapping the filter in a reload layer is what makes `set_filter` work.
    // Reloading rebuilds `tracing`'s callsite interest cache, so a level change
    // takes effect on records that were previously compiled out of the fast
    // path.
    let (filter_layer, reload_handle) = reload::Layer::new(targets);

    let mut guards: Vec<WorkerGuard> = Vec::new();
    let mut dropped: Vec<ErrorCounter> = Vec::new();
    let mut stack: LayerStack = Vec::new();
    let mut crash: Option<SyncFileWriter> = None;

    // --- Console -----------------------------------------------------------
    if let Some(write_console_config) = &config.write_console_config {
        // `BoxMakeWriter` erases the difference between a bare stream and a
        // worker-backed one so the four cases converge on a single type.
        let make: BoxMakeWriter = match (write_console_config.target, write_console_config.blocking) {
            (ConsoleTarget::Stdout, true) => BoxMakeWriter::new(io::stdout),
            (ConsoleTarget::Stderr, true) => BoxMakeWriter::new(io::stderr),
            (ConsoleTarget::Stdout, false) => {
                let (non_blocking_writer, guard) = NonBlockingBuilder::default()
                    .buffered_lines_limit(write_console_config.queue_length)
                    // Lossy: a full queue drops the record instead of stalling
                    // the caller. Never let logging pace the frame loop.
                    .lossy(true)
                    .finish(io::stdout());
                dropped.push(non_blocking_writer.error_counter());
                guards.push(guard);
                BoxMakeWriter::new(non_blocking_writer)
            }
            (ConsoleTarget::Stderr, false) => {
                let (non_blocking, guard) = NonBlockingBuilder::default()
                    .buffered_lines_limit(write_console_config.queue_length)
                    .lossy(true)
                    .finish(io::stderr());
                dropped.push(non_blocking.error_counter());
                guards.push(guard);
                BoxMakeWriter::new(non_blocking)
            }
        };

        let layer_base = fmt::layer().with_ansi(write_console_config.ansi).with_writer(make);

        // Each timer produces a differently-typed layer, so the branches box
        // separately rather than reassigning one binding.
        let layer: Box<dyn Layer<Registry> + Send + Sync> = match write_console_config.timer_kind {
            TimerKind::None => Box::new(layer_base.without_time()),
            TimerKind::Uptime => Box::new(layer_base.with_timer(fmt::time::uptime())),
            TimerKind::Utc => Box::new(layer_base),
            TimerKind::Local => Box::new(layer_base.with_timer(fmt::time::OffsetTime::new(
                local_time_zone_offset(),
                // Compile-time parsed: a malformed description is a build
                // error, and the result is a `&'static` slice with no
                // per-record allocation. The date is omitted because the
                // rolling file name already carries it.
                time::macros::format_description!("[hour]:[minute]:[second].[subsecond digits:3]"),
            ))),
        };
        stack.push(layer);
    }

    // --- Rolling file and crash channel ------------------------------------
    if let Some(write_file_config) = &config.write_file_config {
        std::fs::create_dir_all(&write_file_config.directory)?;

        let appender = RollingFileAppender::builder()
            .rotation(write_file_config.rotation_frequency.clone())
            .filename_prefix(&write_file_config.file_prefix)
            .filename_suffix("log")
            .max_log_files(write_file_config.max_files)
            .build(&write_file_config.directory)
            .map_err(|e| LogError::Config(e.to_string()))?;

        // ANSI is off on every file sink: escape codes are noise in a file and
        // break naive log parsers.
        if write_file_config.force_sync {
            stack.push(Box::new(fmt::layer().with_ansi(false).with_writer(appender)));
        } else {
            let (non_blocking_writer, guard) = NonBlockingBuilder::default()
                .buffered_lines_limit(write_file_config.queue_length)
                .lossy(true)
                .finish(appender);
            dropped.push(non_blocking_writer.error_counter());
            guards.push(guard);
            stack.push(Box::new(fmt::layer().with_ansi(false).with_writer(non_blocking_writer)));
        }

        if let Some(level) = write_file_config.crash_from {
            let sync_writer = SyncFileWriter::new(
                write_file_config
                    .directory
                    .join(format!("{}.critical.log", write_file_config.file_prefix)),
            )?;
            crash = Some(sync_writer.clone());
            stack.push(Box::new(
                fmt::layer()
                    .with_ansi(false)
                    .with_writer(sync_writer)
                    // This predicate reads only callsite metadata — the level
                    // and the set of field names — both of which are static
                    // per callsite, so `tracing` caches the decision and the
                    // check does not recur per record.
                    .with_filter(filter_fn(move |meta| {
                        *meta.level() <= level || meta.fields().field("sync").is_some()
                    })),
            ))
        }
    }

    // --- Ring buffer -------------------------------------------------------
    let ring = if config.ring_capacity > 0 {
        let inner = Ring::new(config.ring_capacity);
        stack.push(Box::new(RingLayer { ring: inner.clone() }));
        Some(inner)
    } else {
        None
    };

    // The filter is layered on top of the stack, which makes it global: a
    // record rejected here never reaches any sink.
    tracing_subscriber::registry()
        .with(stack)
        .with(filter_layer)
        .try_init()
        .map_err(|_| LogError::AlreadyInitialized)?;

    let _ = GLOBAL.set(Global {
        reload: reload_handle,
        crash,
        ring,
        dropped,
    });

    if config.panic_hook {
        install_panic_hook();
    }

    Ok(LoggerGuard {
        _appenders: guards,
        // Only meaningful when there is a file to inspect afterwards.
        emit_shutdown: config.write_file_config.is_some(),
    })
}

// ---------------------------------------------------------------------------
// Runtime accessors
// ---------------------------------------------------------------------------

/// Replaces the active filter, e.g. from an in-game console command.
///
/// Takes the same DSL as [`LogConfig::filter`]. A malformed string leaves the
/// previous filter in place and returns [`LogError::Config`].
#[allow(dead_code)]
pub fn set_filter(dsl: &str) -> Result<(), LogError> {
    let global = GLOBAL.get().ok_or(LogError::NotInitialized)?;
    let targets: Targets = dsl
        .parse()
        .map_err(|e| LogError::Config(format!("bad filter {dsl:?}: {e}")))?;
    global
        .reload
        .reload(targets)
        .map_err(|e| LogError::Config(e.to_string()))
}

/// The newest `n` records held in the ring buffer, oldest first.
///
/// Empty when the ring buffer is disabled or the logger is not initialised.
#[allow(dead_code)]
pub fn recent(n: usize) -> Vec<String> {
    GLOBAL
        .get()
        .and_then(|global| global.ring.as_ref())
        .map(|ring| ring.snapshot(n))
        .unwrap_or_default()
}

/// Records discarded because an async writer's queue was full, summed across
/// sinks.
///
/// A non-zero and climbing value means the queue length is too small for the
/// current log volume, or a subsystem is logging far more than intended. Worth
/// surfacing on a debug HUD, since the drops are otherwise silent.
#[allow(dead_code)]
pub fn dropped_lines() -> usize {
    GLOBAL
        .get()
        .map(|global| global.dropped.iter().map(|counter| counter.dropped_lines()).sum())
        .unwrap_or(0)
}

/// Writes the whole ring buffer out synchronously, framed by marker lines.
///
/// Goes to the crash channel when one is configured, otherwise to stderr, so
/// the call is never a silent no-op on a console-only build. Called
/// automatically by the panic hook and by `critical!`.
#[allow(dead_code)]
pub fn dump_ring() {
    let Some(global) = GLOBAL.get() else { return };
    let Some(ring) = global.ring.as_ref() else {
        return;
    };

    let lines = ring.snapshot(usize::MAX);
    let mut buffer = String::with_capacity(lines.len() * 96 + 64);
    match frame() {
        Some(f) => {
            let _ = writeln!(buffer, "---- ring dump (frame {f}) ----");
        }
        None => buffer.push_str("---- ring dump ----\n"),
    }
    for line in &lines {
        buffer.push_str(line);
        buffer.push('\n');
    }
    buffer.push_str("---- end ring dump ----\n");

    match global.crash.as_ref() {
        Some(writer) => writer.write_raw(&buffer),
        None => {
            let mut stderr = io::stderr().lock();
            let _ = stderr.write_all(buffer.as_bytes());
        }
    }
}

/// Logs panics through the crash channel and dumps the ring buffer, then
/// delegates to whatever hook was previously installed.
///
/// The `sync = true` marker is what forces the panic record onto the
/// unbuffered path; the async sinks are unlikely to flush before the process
/// goes away.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let backtrace = std::backtrace::Backtrace::force_capture();
        match frame() {
            Some(frame) => {
                tracing::error!(sync = true, "PANIC @frame {frame}: {info}\n{backtrace}")
            }
            None => tracing::error!(sync = true, "PANIC {info}\n{backtrace}"),
        }
        dump_ring();
        previous(info);
    }));
}

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

/// Logs at `ERROR`, forces the record onto the unbuffered crash channel, and
/// dumps the ring buffer.
///
/// For failures you are about to die from, where the tail of the async log is
/// likely to be lost.
///
/// The expansion refers to `$crate::log::dump_ring`, which assumes this file is
/// mounted as `log` at the crate root. Adjust the path if it moves.
///
/// ```ignore
/// critical!(device = ?adapter.info(), "GPU device lost");
/// ```
#[macro_export]
macro_rules! critical {
    ($($arg:tt)*) => {{
        ::tracing::error!(sync = true, $($arg)*);
        $crate::log::dump_ring();
    }};
}
