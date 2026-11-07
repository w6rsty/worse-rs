use ash::vk;

pub const MAX_ENTRY_NAME_LENGTH: usize = 64;
pub const MAX_COLOR_TARGETS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    R8Srgb,
    RG8Srgb,
    RGBA8Srgb,
    BGRA8Srgb,

    RGBA4Unorm,
    R5G5B5A1Unorm,
    R5G6B5Unorm,

    R8Unorm,
    RG8Unorm,
    RGBA8Unorm,
    BGRA8Unorm,
    R16Unorm,
    RG16Unorm,
    RGBA16Unorm,

    R8UInt,
    RG8UInt,
    RGBA8UInt,
    BGRA8UInt,
    R16UInt,
    RG16UInt,
    RGBA16UInt,
    R32UInt,
    RG32UInt,
    RGB32UInt,
    RGBA32UInt,

    R16Float,
    RG16Float,
    RGBA16Float,
    R32Float,
    RG32Float,
    RGB32Float,
    RGBA32Float,

    RGB10A2Unorm,
    RG11B10Float,

    D16Unorm,
    D24UnormS8UInt,
    D32Float,
    S8UInt,
    D32FloatS8UInt,

    EACRG,
    ASTC4X4Srgb,
    ASTC4X4Unorm,
    BC3Srgb,
    BC3Unorm,
    BC5RG,
    BC6HUFloat,
    BC6HSFloat,
    BC7Srgb,
    BC7Unorm,

    Undefined,
}

impl From<PixelFormat> for vk::Format {
    fn from(value: PixelFormat) -> Self {
        match value {
            PixelFormat::R8Srgb => vk::Format::R8_SRGB,
            PixelFormat::RG8Srgb => vk::Format::R8G8_SRGB,
            PixelFormat::RGBA8Srgb => vk::Format::R8G8B8A8_SRGB,
            PixelFormat::BGRA8Srgb => vk::Format::B8G8R8A8_SRGB,

            PixelFormat::RGBA4Unorm => vk::Format::R4G4B4A4_UNORM_PACK16,
            PixelFormat::R5G5B5A1Unorm => vk::Format::R5G5B5A1_UNORM_PACK16,
            PixelFormat::R5G6B5Unorm => vk::Format::R5G6B5_UNORM_PACK16,

            PixelFormat::R8Unorm => vk::Format::R8_UNORM,
            PixelFormat::RG8Unorm => vk::Format::R8G8_UNORM,
            PixelFormat::RGBA8Unorm => vk::Format::R8G8B8A8_UNORM,
            PixelFormat::BGRA8Unorm => vk::Format::B8G8R8A8_UNORM,
            PixelFormat::R16Unorm => vk::Format::R16_UNORM,
            PixelFormat::RG16Unorm => vk::Format::R16G16_UNORM,
            PixelFormat::RGBA16Unorm => vk::Format::R16G16B16A16_UNORM,

            PixelFormat::R8UInt => vk::Format::R8_UINT,
            PixelFormat::RG8UInt => vk::Format::R8G8_UINT,
            PixelFormat::RGBA8UInt => vk::Format::R8G8B8A8_UINT,
            PixelFormat::BGRA8UInt => vk::Format::B8G8R8A8_UINT,
            PixelFormat::R16UInt => vk::Format::R16_UINT,
            PixelFormat::RG16UInt => vk::Format::R16G16_UINT,
            PixelFormat::RGBA16UInt => vk::Format::R16G16B16A16_UINT,
            PixelFormat::R32UInt => vk::Format::R32_UINT,
            PixelFormat::RG32UInt => vk::Format::R32G32_UINT,
            PixelFormat::RGB32UInt => vk::Format::R32G32B32_UINT,
            PixelFormat::RGBA32UInt => vk::Format::R32G32B32A32_UINT,

            PixelFormat::R16Float => vk::Format::R16_SFLOAT,
            PixelFormat::RG16Float => vk::Format::R16G16_SFLOAT,
            PixelFormat::RGBA16Float => vk::Format::R16G16B16A16_SFLOAT,
            PixelFormat::R32Float => vk::Format::R32_SFLOAT,
            PixelFormat::RG32Float => vk::Format::R32G32_SFLOAT,
            PixelFormat::RGB32Float => vk::Format::R32G32B32_SFLOAT,
            PixelFormat::RGBA32Float => vk::Format::R32G32B32A32_SFLOAT,

            PixelFormat::RGB10A2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
            PixelFormat::RG11B10Float => vk::Format::B10G11R11_UFLOAT_PACK32,

            PixelFormat::D16Unorm => vk::Format::D16_UNORM,
            PixelFormat::D24UnormS8UInt => vk::Format::D24_UNORM_S8_UINT,
            PixelFormat::D32Float => vk::Format::D32_SFLOAT,
            PixelFormat::S8UInt => vk::Format::S8_UINT,
            PixelFormat::D32FloatS8UInt => vk::Format::D32_SFLOAT_S8_UINT,

            PixelFormat::EACRG => vk::Format::EAC_R11G11_UNORM_BLOCK,
            PixelFormat::ASTC4X4Srgb => vk::Format::ASTC_4X4_SRGB_BLOCK,
            PixelFormat::ASTC4X4Unorm => vk::Format::ASTC_4X4_UNORM_BLOCK,
            PixelFormat::BC3Srgb => vk::Format::BC3_SRGB_BLOCK,
            PixelFormat::BC3Unorm => vk::Format::BC3_UNORM_BLOCK,
            PixelFormat::BC5RG => vk::Format::BC5_UNORM_BLOCK,
            PixelFormat::BC6HUFloat => vk::Format::BC6H_UFLOAT_BLOCK,
            PixelFormat::BC6HSFloat => vk::Format::BC6H_SFLOAT_BLOCK,
            PixelFormat::BC7Srgb => vk::Format::BC7_SRGB_BLOCK,
            PixelFormat::BC7Unorm => vk::Format::BC7_UNORM_BLOCK,

            PixelFormat::Undefined => vk::Format::UNDEFINED,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    SrgbNonLinear,
}

impl From<ColorSpace> for vk::ColorSpaceKHR {
    fn from(value: ColorSpace) -> Self {
        match value {
            ColorSpace::SrgbNonLinear => vk::ColorSpaceKHR::SRGB_NONLINEAR,
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BufferUsageFlags: u32 {
        const TRANSFER_SRC          = 1 << 0;
        const TRANSFER_DST          = 1 << 1;
        const UNIFORM               = 1 << 2;
        const STORAGE               = 1 << 3;
        const INDEX                 = 1 << 4;
        const VERTEX                = 1 << 5;
        const INDIRECT              = 1 << 6;
        const DESCRIPTOR_HEAP       = 1 << 7;
        const SHADER_DEVICE_ADDRESS = 1 << 8;
        // vk::SharingMode::CONCURRENT
        const SHARED                = 1 << 9;
    }
}

impl From<BufferUsageFlags> for vk::BufferUsageFlags {
    fn from(value: BufferUsageFlags) -> Self {
        let mut result = vk::BufferUsageFlags::empty();

        if value.contains(BufferUsageFlags::TRANSFER_SRC) {
            result |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if value.contains(BufferUsageFlags::TRANSFER_DST) {
            result |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if value.contains(BufferUsageFlags::UNIFORM) {
            result |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if value.contains(BufferUsageFlags::STORAGE) {
            result |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if value.contains(BufferUsageFlags::INDEX) {
            result |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if value.contains(BufferUsageFlags::VERTEX) {
            result |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if value.contains(BufferUsageFlags::INDIRECT) {
            result |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }
        if value.contains(BufferUsageFlags::DESCRIPTOR_HEAP) {
            result |= vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT;
        }
        if value.contains(BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            result |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }

        result
    }
}

impl BufferUsageFlags {
    pub fn universal_bits() -> Self {
        Self::SHADER_DEVICE_ADDRESS | Self::INDEX | Self::INDIRECT | Self::TRANSFER_SRC | Self::TRANSFER_DST
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureDimension {
    Dimension1D,
    Dimension2D,
    Dimension3D,
    Cube,
    Dimension2DArray,
    CubeArray,
}

impl From<TextureDimension> for vk::ImageType {
    fn from(value: TextureDimension) -> Self {
        match value {
            TextureDimension::Dimension1D => vk::ImageType::TYPE_1D,
            TextureDimension::Dimension3D => vk::ImageType::TYPE_3D,
            TextureDimension::Dimension2D
            | TextureDimension::Cube
            | TextureDimension::Dimension2DArray
            | TextureDimension::CubeArray => vk::ImageType::TYPE_2D,
        }
    }
}

impl From<TextureDimension> for vk::ImageViewType {
    fn from(value: TextureDimension) -> Self {
        match value {
            TextureDimension::Dimension1D => vk::ImageViewType::TYPE_1D,
            TextureDimension::Dimension2D => vk::ImageViewType::TYPE_2D,
            TextureDimension::Dimension3D => vk::ImageViewType::TYPE_3D,
            TextureDimension::Cube => vk::ImageViewType::CUBE,
            TextureDimension::Dimension2DArray => vk::ImageViewType::TYPE_2D_ARRAY,
            TextureDimension::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TextureUsageFlags : u32 {
        const SAMPLED                  = 1 << 0;
        const STORAGE                  = 1 << 1;
        const COLOR_ATTACHMENT         = 1 << 2;
        const DEPTH_STENCIL_ATTACHMENT = 1 << 3;
        const TRANSFER_SRC             = 1 << 4;
        const TRANSFER_DST             = 1 << 5;
    }
}

impl From<TextureUsageFlags> for vk::ImageUsageFlags {
    fn from(value: TextureUsageFlags) -> Self {
        let mut result = vk::ImageUsageFlags::empty();

        if value.contains(TextureUsageFlags::SAMPLED) {
            result |= vk::ImageUsageFlags::SAMPLED;
        }
        if value.contains(TextureUsageFlags::STORAGE) {
            result |= vk::ImageUsageFlags::STORAGE;
        }
        if value.contains(TextureUsageFlags::COLOR_ATTACHMENT) {
            result |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if value.contains(TextureUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
            result |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if value.contains(TextureUsageFlags::TRANSFER_SRC) {
            result |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if value.contains(TextureUsageFlags::TRANSFER_DST) {
            result |= vk::ImageUsageFlags::TRANSFER_DST;
        }

        result
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TextureCreateFlags : u32 {
        const SHARED = 1 << 0;
        const ALIAS  = 1 << 1;
        const SPARSE = 1 << 2;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleCount {
    Count1 = 0b1,
    Count2 = 0b10,
    Count4 = 0b100,
    Count8 = 0b1000,
    Count16 = 0b10000,
    Count32 = 0b100000,
    Count64 = 0b1000000,
}

impl From<SampleCount> for vk::SampleCountFlags {
    fn from(value: SampleCount) -> Self {
        match value {
            SampleCount::Count1 => vk::SampleCountFlags::TYPE_1,
            SampleCount::Count2 => vk::SampleCountFlags::TYPE_2,
            SampleCount::Count4 => vk::SampleCountFlags::TYPE_4,
            SampleCount::Count8 => vk::SampleCountFlags::TYPE_8,
            SampleCount::Count16 => vk::SampleCountFlags::TYPE_16,
            SampleCount::Count32 => vk::SampleCountFlags::TYPE_32,
            SampleCount::Count64 => vk::SampleCountFlags::TYPE_64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureDescriptorKind {
    Sampled,
    Storage,
}

impl From<TextureDescriptorKind> for vk::DescriptorType {
    fn from(value: TextureDescriptorKind) -> Self {
        match value {
            TextureDescriptorKind::Sampled => vk::DescriptorType::SAMPLED_IMAGE,
            TextureDescriptorKind::Storage => vk::DescriptorType::STORAGE_IMAGE,
        }
    }
}

impl From<TextureDescriptorKind> for vk::ImageUsageFlags {
    fn from(value: TextureDescriptorKind) -> Self {
        match value {
            TextureDescriptorKind::Sampled => vk::ImageUsageFlags::SAMPLED,
            TextureDescriptorKind::Storage => vk::ImageUsageFlags::STORAGE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterKind {
    Nearest,
    Linear,
}

impl From<FilterKind> for vk::Filter {
    fn from(value: FilterKind) -> Self {
        match value {
            FilterKind::Nearest => vk::Filter::NEAREST,
            FilterKind::Linear => vk::Filter::LINEAR,
        }
    }
}

impl From<FilterKind> for vk::SamplerMipmapMode {
    fn from(value: FilterKind) -> Self {
        match value {
            FilterKind::Nearest => vk::SamplerMipmapMode::NEAREST,
            FilterKind::Linear => vk::SamplerMipmapMode::LINEAR,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

impl From<AddressMode> for vk::SamplerAddressMode {
    fn from(value: AddressMode) -> Self {
        match value {
            AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessQual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl From<CompareOp> for vk::CompareOp {
    fn from(value: CompareOp) -> Self {
        match value {
            CompareOp::Never => vk::CompareOp::NEVER,
            CompareOp::Less => vk::CompareOp::LESS,
            CompareOp::Equal => vk::CompareOp::EQUAL,
            CompareOp::LessQual => vk::CompareOp::LESS_OR_EQUAL,
            CompareOp::Greater => vk::CompareOp::GREATER,
            CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
            CompareOp::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
            CompareOp::Always => vk::CompareOp::ALWAYS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolygonMode {
    Fill,
    Line,
}

impl From<PolygonMode> for vk::PolygonMode {
    fn from(value: PolygonMode) -> Self {
        match value {
            PolygonMode::Fill => vk::PolygonMode::FILL,
            PolygonMode::Line => vk::PolygonMode::LINE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullMode {
    None,
    Front,
    Back,
}

impl From<CullMode> for vk::CullModeFlags {
    fn from(value: CullMode) -> Self {
        match value {
            CullMode::None => vk::CullModeFlags::NONE,
            CullMode::Front => vk::CullModeFlags::FRONT,
            CullMode::Back => vk::CullModeFlags::BACK,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrontFace {
    Clockwise,
    CounterClockwise,
}

impl From<FrontFace> for vk::FrontFace {
    fn from(value: FrontFace) -> Self {
        match value {
            FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
            FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    DestinationColor,
    OneMinusDestinationColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationAlpha,
    OneMinusDestinationAlpha,
    SourceAlphaStaurate,
}

impl From<BlendFactor> for vk::BlendFactor {
    fn from(value: BlendFactor) -> Self {
        match value {
            BlendFactor::Zero => vk::BlendFactor::ZERO,
            BlendFactor::One => vk::BlendFactor::ONE,
            BlendFactor::SourceColor => vk::BlendFactor::SRC_COLOR,
            BlendFactor::OneMinusSourceColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            BlendFactor::DestinationColor => vk::BlendFactor::DST_COLOR,
            BlendFactor::OneMinusDestinationColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            BlendFactor::SourceAlpha => vk::BlendFactor::SRC_ALPHA,
            BlendFactor::OneMinusSourceAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DestinationAlpha => vk::BlendFactor::DST_ALPHA,
            BlendFactor::OneMinusDestinationAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
            BlendFactor::SourceAlphaStaurate => vk::BlendFactor::SRC_ALPHA_SATURATE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendOp {
    Add,
    Substract,
    ResverseSubract,
    Minimum,
    Maximum,
}

impl From<BlendOp> for vk::BlendOp {
    fn from(value: BlendOp) -> Self {
        match value {
            BlendOp::Add => vk::BlendOp::ADD,
            BlendOp::Substract => vk::BlendOp::SUBTRACT,
            BlendOp::ResverseSubract => vk::BlendOp::REVERSE_SUBTRACT,
            BlendOp::Minimum => vk::BlendOp::MIN,
            BlendOp::Maximum => vk::BlendOp::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    UInt16,
    UInt32,
}

pub trait IndexType: bytemuck::Pod {
    const VK_TYPE: vk::IndexType;
}

impl IndexType for u16 {
    const VK_TYPE: vk::IndexType = vk::IndexType::UINT16;
}

impl IndexType for u32 {
    const VK_TYPE: vk::IndexType = vk::IndexType::UINT32;
}

impl From<IndexKind> for vk::IndexType {
    fn from(value: IndexKind) -> Self {
        match value {
            IndexKind::UInt16 => vk::IndexType::UINT16,
            IndexKind::UInt32 => vk::IndexType::UINT32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp {
    Load,
    Clear,
    Discard,
}

impl From<LoadOp> for vk::AttachmentLoadOp {
    fn from(value: LoadOp) -> Self {
        match value {
            LoadOp::Load => vk::AttachmentLoadOp::LOAD,
            LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
            LoadOp::Discard => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOp {
    Store,
    Discard,
}

impl From<StoreOp> for vk::AttachmentStoreOp {
    fn from(value: StoreOp) -> Self {
        match value {
            StoreOp::Store => vk::AttachmentStoreOp::STORE,
            StoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

impl From<StencilOp> for vk::StencilOp {
    fn from(value: StencilOp) -> Self {
        match value {
            StencilOp::Keep => vk::StencilOp::KEEP,
            StencilOp::Zero => vk::StencilOp::ZERO,
            StencilOp::Replace => vk::StencilOp::REPLACE,
            StencilOp::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
            StencilOp::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
            StencilOp::Invert => vk::StencilOp::INVERT,
            StencilOp::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
            StencilOp::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct StageFlags : u64 {
        const NONE               = 0;
        const INDIRECT           = 1 << 6;
        const INDEX_INPUT        = 1 << 7;
        const VERTEX             = 1 << 1;
        const MESH               = 1 << 9;
        const DEPTH_STENCIL_TEST = 1 << 8;
        const FRAGMENT           = 1 << 2;
        const COLOR_OUTPUT       = 1 << 4;
        const COMPUTE            = 1 << 3;
        const TRANSFER           = 1 << 0;
        const HOST               = 1 << 5;
        const ALL_COMMANDS       = 1 << 10;
    }
}

impl From<StageFlags> for vk::PipelineStageFlags2 {
    fn from(value: StageFlags) -> Self {
        let mut flags = vk::PipelineStageFlags2::empty();

        if value.contains(StageFlags::INDIRECT) {
            flags |= vk::PipelineStageFlags2::DRAW_INDIRECT;
        }

        if value.contains(StageFlags::INDEX_INPUT) {
            flags |= vk::PipelineStageFlags2::INDEX_INPUT;
        }
        if value.contains(StageFlags::VERTEX) {
            flags |= vk::PipelineStageFlags2::VERTEX_SHADER;
        }
        if value.contains(StageFlags::MESH) {
            flags |= vk::PipelineStageFlags2::MESH_SHADER_EXT;
        }
        if value.contains(StageFlags::DEPTH_STENCIL_TEST) {
            flags |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        }
        if value.contains(StageFlags::FRAGMENT) {
            flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }
        if value.contains(StageFlags::COLOR_OUTPUT) {
            flags |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
        }
        if value.contains(StageFlags::COMPUTE) {
            flags |= vk::PipelineStageFlags2::COMPUTE_SHADER;
        }
        if value.contains(StageFlags::TRANSFER) {
            flags |= vk::PipelineStageFlags2::TRANSFER;
        }
        if value.contains(StageFlags::HOST) {
            flags |= vk::PipelineStageFlags2::HOST;
        }
        if value.contains(StageFlags::ALL_COMMANDS) {
            flags |= vk::PipelineStageFlags2::ALL_COMMANDS;
        }

        flags
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AccessFlags : u64 {
        const NONE                = 0;
        const TRANSFER_READ       = 1 << 0;
        const TRANSFER_WRITE      = 1 << 1;
        const SHADER_READ         = 1 << 2;
        const SHADER_WRITE        = 1 << 3;
        const COLOR_READ          = 1 << 4;
        const COLOR_WRITE         = 1 << 5;
        const DEPTH_STENCIL_READ  = 1 << 6;
        const DPETH_STENICL_WRITE = 1 << 7;
        const INDIRECT_READ       = 1 << 8;
        const INDEX_READ          = 1 << 9;
        const HOST_READ           = 1 << 10;
        const DESCIRPTOR_READ     = 1 << 11;
    }
}

impl From<AccessFlags> for vk::AccessFlags2 {
    fn from(value: AccessFlags) -> Self {
        let mut flags = vk::AccessFlags2::empty();

        if value.contains(AccessFlags::TRANSFER_READ) {
            flags |= vk::AccessFlags2::TRANSFER_READ;
        }
        if value.contains(AccessFlags::TRANSFER_WRITE) {
            flags |= vk::AccessFlags2::TRANSFER_WRITE;
        }
        if value.contains(AccessFlags::SHADER_READ) {
            flags |= vk::AccessFlags2::SHADER_READ;
        }
        if value.contains(AccessFlags::SHADER_WRITE) {
            flags |= vk::AccessFlags2::SHADER_WRITE;
        }
        if value.contains(AccessFlags::COLOR_READ) {
            flags |= vk::AccessFlags2::COLOR_ATTACHMENT_READ;
        }
        if value.contains(AccessFlags::COLOR_WRITE) {
            flags |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
        }
        if value.contains(AccessFlags::DEPTH_STENCIL_READ) {
            flags |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
        }
        if value.contains(AccessFlags::DPETH_STENICL_WRITE) {
            flags |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }
        if value.contains(AccessFlags::INDIRECT_READ) {
            flags |= vk::AccessFlags2::INDEX_READ;
        }
        if value.contains(AccessFlags::HOST_READ) {
            flags |= vk::AccessFlags2::HOST_READ;
        }
        if value.contains(AccessFlags::DESCIRPTOR_READ) {
            flags |= vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT;
        }

        flags
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    CPUVisible,
    GPUOnly,
    ReadBack,
    TextureDescriptorHeap,
    SamplerDescriptorHeap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

impl From<Extent2D> for vk::Extent2D {
    fn from(value: Extent2D) -> Self {
        vk::Extent2D {
            width: value.width,
            height: value.height,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl From<Extent3D> for vk::Extent3D {
    fn from(value: Extent3D) -> Self {
        vk::Extent3D {
            width: value.width,
            height: value.height,
            depth: value.depth,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

impl From<Offset2D> for vk::Offset2D {
    fn from(value: Offset2D) -> Self {
        vk::Offset2D { x: value.x, y: value.y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Offset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl From<Offset3D> for vk::Offset3D {
    fn from(value: Offset3D) -> Self {
        vk::Offset3D {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rect2D {
    pub offset: Offset2D,
    pub extent: Extent2D,
}

impl From<Rect2D> for vk::Rect2D {
    fn from(value: Rect2D) -> Self {
        vk::Rect2D {
            offset: value.offset.into(),
            extent: value.extent.into(),
        }
    }
}

impl Rect2D {
    pub fn offset(mut self, offset: Offset2D) -> Self {
        self.offset = offset;
        self
    }

    pub fn extent(mut self, extent: Extent2D) -> Self {
        self.extent = extent;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureAspect {
    Color,
    Depth,
    Stencil,
}

impl From<TextureAspect> for vk::ImageAspectFlags {
    fn from(value: TextureAspect) -> Self {
        match value {
            TextureAspect::Color => vk::ImageAspectFlags::COLOR,
            TextureAspect::Depth => vk::ImageAspectFlags::DEPTH,
            TextureAspect::Stencil => vk::ImageAspectFlags::STENCIL,
        }
    }
}

pub mod state {
    pub struct Initial;
    pub struct Available;
    pub struct Active;
    pub struct Pending;
    pub struct Finished;
}

#[derive(Debug, Clone, Copy)]
pub struct LabelColor {
    pub value: [f32; 4],
}

impl LabelColor {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { value: [r, g, b, a] }
    }

    pub const BLACK: Self = Self {
        value: [0.0, 0.0, 0.0, 1.0],
    };
    pub const WHITE: Self = Self {
        value: [1.0, 1.0, 1.0, 1.0],
    };
    pub const RED: Self = Self {
        value: [1.0, 0.0, 0.0, 1.0],
    };
    pub const GREEN: Self = Self {
        value: [0.0, 1.0, 0.0, 1.0],
    };
    pub const BLUE: Self = Self {
        value: [0.0, 0.0, 1.0, 1.0],
    };
    pub const YELLOW: Self = Self {
        value: [1.0, 1.0, 0.0, 1.0],
    };
    pub const CYAN: Self = Self {
        value: [0.0, 1.0, 1.0, 1.0],
    };
    pub const MAGENTA: Self = Self {
        value: [1.0, 0.0, 1.0, 1.0],
    };
    pub const TRANSPARENT: Self = Self {
        value: [0.0, 0.0, 0.0, 0.0],
    };
}
