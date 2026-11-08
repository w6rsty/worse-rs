use std::borrow::Cow;

use slang_auto as slang;

const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_MAGIC_INVERSED: u32 = 0x0302_2307;

#[derive(Debug, Clone)]
pub struct Spirv {
    pub(crate) code: Option<slang::Blob>,
    pub(crate) asm: Option<slang::Blob>,
    pub(crate) dbg: Option<slang::Blob>,
}

impl Spirv {
    pub fn code_bytes(&self) -> Option<&[u8]> {
        self.code.as_ref().map(slang::Blob::as_bytes)
    }

    pub fn asm_bytes(&self) -> Option<&[u8]> {
        self.asm.as_ref().map(slang::Blob::as_bytes)
    }

    pub fn debug_data(&self) -> Option<&[u8]> {
        self.dbg.as_ref().map(slang::Blob::as_bytes)
    }

    pub fn bytes(&self) -> &[u8] {
        self.code_bytes().unwrap_or_default()
    }

    pub fn words(&self) -> Result<Cow<'_, [u32]>, SpirvError> {
        spirv_words(self.code_bytes().ok_or(SpirvError::NoCode)?)
    }

    pub fn debug_bytes(&self) -> Result<&[u8], SpirvError> {
        self.dbg
            .as_ref()
            .map(slang::Blob::as_bytes)
            .ok_or(SpirvError::NoDebugInfo)
    }

    pub fn debug_words(&self) -> Result<Cow<'_, [u32]>, SpirvError> {
        spirv_words(self.debug_bytes()?)
    }

    pub fn asm_str(&self) -> Option<Cow<'_, str>> {
        self.asm.as_ref().map(slang::Blob::to_string_lossy)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpirvError {
    NoCode,
    InvalidLength,
    InvalidMagicNumber(u32),
    NoDebugInfo,
}

fn spirv_words(bytes: &[u8]) -> Result<Cow<'_, [u32]>, SpirvError> {
    if bytes.len() < 4 || bytes.len() & 3 != 0 {
        return Err(SpirvError::InvalidLength);
    }

    match bytemuck::try_cast_slice::<u8, u32>(bytes) {
        Ok(words) if words[0] == SPIRV_MAGIC => Ok(Cow::Borrowed(words)),
        _ => copy_spirv_words(bytes).map(Cow::Owned),
    }
}

fn copy_spirv_words(bytes: &[u8]) -> Result<Vec<u32>, SpirvError> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() || bytes.len() < 4 {
        return Err(SpirvError::InvalidLength);
    }

    let mut words: Vec<u32> = chunks.map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    match words[0] {
        SPIRV_MAGIC => {}
        SPIRV_MAGIC_INVERSED => words.iter_mut().for_each(|w| *w = w.swap_bytes()),
        magic => return Err(SpirvError::InvalidMagicNumber(magic)),
    }
    Ok(words)
}
