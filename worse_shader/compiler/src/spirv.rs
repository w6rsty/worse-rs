use std::borrow::Cow;

use slang_auto as slang;

const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_MAGIC_INVERSED: u32 = 0x0302_2307;

#[derive(Debug)]
pub struct Spirv {
    pub(crate) code: slang::Blob,
    pub(crate) asm: Option<slang::Blob>,
    pub(crate) dbg: Option<slang::Blob>,
}

impl Spirv {
    pub fn bytes(&self) -> &[u8] {
        self.code.as_bytes()
    }

    pub fn words(&self) -> Result<Cow<'_, [u32]>, SpirvError> {
        spirv_words(self.code.as_bytes())
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
        self.asm.as_ref().map(|b| String::from_utf8_lossy(b.as_bytes()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpirvError {
    InvalidLength,
    InvalidMagicNumber(u32),
    NoDebugInfo,
}

fn spirv_words(bytes: &[u8]) -> Result<Cow<'_, [u32]>, SpirvError> {
    if bytes.len() < 4 || bytes.len() % 4 != 0 {
        return Err(SpirvError::InvalidLength);
    }

    if let Ok(words) = bytemuck::try_cast_slice::<u8, u32>(bytes) {
        if words[0] == SPIRV_MAGIC {
            return Ok(Cow::Borrowed(words));
        }
    }

    // fallback
    copy_spirv_words(bytes).map(Cow::Owned)
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
