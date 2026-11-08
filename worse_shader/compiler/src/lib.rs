mod slang_api;
mod spirv;
mod types;

use slang_auto as slang;
use thiserror::Error;
use tracing::{info, warn};

pub use spirv::{Spirv, SpirvError};
pub use types::*;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to create GlobalSession")]
    FailedToCreateGlobalSsesion,
    #[error("failed to create Session")]
    FailedToCreateSsesion,
    #[error("entry point {0} not exits")]
    EntryPointNotExits(String),
    #[error("slang error: {0}")]
    Slang(#[from] slang::Error),
}

pub struct CompileService {
    global_session: slang::GlobalSession,
}

impl CompileService {
    pub fn new() -> Result<Self, Error> {
        let global_session = slang::GlobalSession::new()?;
        info!(
            "created slang compile service({})",
            slang::build_tag_string().to_string_lossy()
        );
        Ok(Self { global_session })
    }

    pub fn compile(&self, desc: CompileDesc) -> Result<Spirv, Error> {
        info!("compiling {:?}", &desc);

        let emit_asm = desc
            .debug_info
            .is_some_and(|debug| debug.flags.contains(DebugFlags::EMIT_ASM));
        let separate_debug_info = desc
            .debug_info
            .is_some_and(|debug| debug.flags.contains(DebugFlags::SEPARATE));
        let session = slang_api::create_session(&self.global_session, &desc)?;

        let module = log_diagnostics(match &desc.source {
            FileSource::File(path) => {
                let file_name = path
                    .to_str()
                    .ok_or(slang::Error::InvalidArgument("Shader source path is not valid UTF-8"))?;
                session.load_module(file_name)?
            }
            FileSource::Memory {
                virtual_name,
                virutal_path,
                payload,
            } => session.load_module_from_source(virtual_name, virutal_path, payload)?,
        });
        let entry_point =
            log_diagnostics(module.find_and_check_entry_point(&desc.entry.name, desc.entry.stage.into())?);
        let program = log_diagnostics(session.compose(&[&module.as_component(), &entry_point.as_component()])?);
        let linked_program = log_diagnostics(program.link()?);
        let asm = if emit_asm {
            Some(log_diagnostics(linked_program.entry_point_code(0, 1)?))
        } else {
            None
        };
        let code = log_diagnostics(linked_program.entry_point_code(0, 0)?);
        let dbg = if separate_debug_info {
            slang_api::separate_debug_data(&linked_program)?
        } else {
            None
        };
        Ok(Spirv { code, asm, dbg })
    }
}

fn log_diagnostics<T>(output: slang::Output<T>) -> T {
    if !output.diagnostics.is_empty() {
        warn!("{}", output.diagnostics);
    }
    output.value
}
