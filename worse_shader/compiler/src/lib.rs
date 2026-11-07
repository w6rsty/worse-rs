mod types;

use arrayvec::ArrayVec;
use slang_rs as slang;
use thiserror::Error;
use tracing::info;
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
        let global_session = slang::GlobalSession::new().ok_or(Error::FailedToCreateGlobalSsesion)?;
        info!("created slang compile service({})", global_session.build_tag_string());
        Ok(Self { global_session })
    }

    pub fn compile(&self, desc: CompileDesc) -> Result<Vec<Target>, Error> {
        let descrptor_heap_ext = self.global_session.find_capability("spvDescriptorHeapEXT");
        let spirv_profile = self.global_session.find_profile("spirv_1_5");

        let (debug_info_level, emit_asm, separate_debug_info) = match desc.debug_info {
            Some(debug_info) => (
                debug_info.level.into(),
                debug_info.flags.contains(DebugFlags::EMIT_ASM),
                debug_info.flags.contains(DebugFlags::SEPARATE),
            ),
            None => (slang::DebugInfoLevel::None, false, false),
        };

        let matrix_layout_options = match desc.matrix_layout {
            MatrixLayout::RowMajor => slang::CompilerOptions::default().matrix_layout_row(true),
            MatrixLayout::ColumnMajor => slang::CompilerOptions::default().matrix_layout_column(true),
        };
        let options = matrix_layout_options
            .capability(descrptor_heap_ext)
            .profile(spirv_profile)
            .optimization(desc.optimization.into())
            .debug_information(debug_info_level)
            .emit_spirv_directly(true)
            .separate_debug_info(separate_debug_info)
            .vulkan_use_entry_point_name(true);

        let mut targets = ArrayVec::<_, 2>::new();
        targets.push(slang::TargetDesc::default().format(slang::CompileTarget::Spirv));
        if emit_asm {
            targets.push(slang::TargetDesc::default().format(slang::CompileTarget::SpirvAsm));
        }

        let session = self
            .global_session
            .create_session(&slang::SessionDesc::default().options(&options).targets(&targets))
            .ok_or(Error::FailedToCreateSsesion)?;

        let module = match &desc.source {
            FileSource::File(path) => {
                let file_name = &path.to_str().unwrap();
                session.load_module(file_name)?
            }
            FileSource::Memory {
                virtual_name,
                virutal_path,
                payload,
            } => session.load_module_from_source_string(&virtual_name, &virutal_path, payload)?,
        };

        let entry_point = module
            .find_entry_point_by_name(&desc.entry.name)
            .ok_or(Error::EntryPointNotExits(desc.entry.name.clone()))?;

        let program = session.create_composite_component_type(&[module.into(), entry_point.into()])?;
        let linked_program = program.link()?;

        let mut codes = Vec::new();
        let spirv_code = linked_program.entry_point_code(0, 0)?;
        codes.push(Target {
            kind: TargetKind::Spirv,
            blob: spirv_code,
        });

        if emit_asm {
            let spirv_asm_code = linked_program.entry_point_code(0, 1)?;
            codes.push(Target {
                kind: TargetKind::SpirvAsm,
                blob: spirv_asm_code,
            });
        }

        info!("finished compiling, {:?}", &desc);

        Ok(codes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetKind {
    Spirv,
    SpirvAsm,
}

pub struct Target {
    kind: TargetKind,
    blob: slang::Blob,
}

impl Target {
    pub fn test_print(&self) {
        if self.kind == TargetKind::SpirvAsm {
            println!("{}", self.blob.as_str().unwrap())
        }
    }

    pub fn to_spirv_byte_code(&self) -> Result<Vec<u32>, SpirvError> {
        if self.kind != TargetKind::Spirv {
            return Err(SpirvError::WrongSource);
        }

        let bytes = self.blob.as_slice();
        if bytes.len() % 4 != 0 || bytes.len() < 4 {
            return Err(SpirvError::InvalidLength);
        }

        let mut words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let magic_number = words[0];

        match magic_number {
            0x0723_0203 => {}
            0x0302_2307 => words.iter_mut().for_each(|w| *w = w.swap_bytes()),
            _ => return Err(SpirvError::InvalidMagicNumber(magic_number)),
        }
        Ok(words)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpirvError {
    WrongSource,
    InvalidLength,
    InvalidMagicNumber(u32),
}
