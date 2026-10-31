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

impl PixelFormat {
    pub const fn to_vk(self) -> vk::Format {
        match self {
            Self::R8Srgb => vk::Format::R8_SRGB,
            Self::RG8Srgb => vk::Format::R8G8_SRGB,
            Self::RGBA8Srgb => vk::Format::R8G8B8A8_SRGB,
            Self::BGRA8Srgb => vk::Format::B8G8R8A8_SRGB,

            Self::RGBA4Unorm => vk::Format::R4G4B4A4_UNORM_PACK16,
            Self::R5G5B5A1Unorm => vk::Format::R5G5B5A1_UNORM_PACK16,
            Self::R5G6B5Unorm => vk::Format::R5G6B5_UNORM_PACK16,

            Self::R8Unorm => vk::Format::R8_UNORM,
            Self::RG8Unorm => vk::Format::R8G8_UNORM,
            Self::RGBA8Unorm => vk::Format::R8G8B8A8_UNORM,
            Self::BGRA8Unorm => vk::Format::B8G8R8A8_UNORM,
            Self::R16Unorm => vk::Format::R16_UNORM,
            Self::RG16Unorm => vk::Format::R16G16_UNORM,
            Self::RGBA16Unorm => vk::Format::R16G16B16A16_UNORM,

            Self::R8UInt => vk::Format::R8_UINT,
            Self::RG8UInt => vk::Format::R8G8_UINT,
            Self::RGBA8UInt => vk::Format::R8G8B8A8_UINT,
            Self::BGRA8UInt => vk::Format::B8G8R8A8_UINT,
            Self::R16UInt => vk::Format::R16_UINT,
            Self::RG16UInt => vk::Format::R16G16_UINT,
            Self::RGBA16UInt => vk::Format::R16G16B16A16_UINT,
            Self::R32UInt => vk::Format::R32_UINT,
            Self::RG32UInt => vk::Format::R32G32_UINT,
            Self::RGB32UInt => vk::Format::R32G32B32_UINT,
            Self::RGBA32UInt => vk::Format::R32G32B32A32_UINT,

            Self::R16Float => vk::Format::R16_SFLOAT,
            Self::RG16Float => vk::Format::R16G16_SFLOAT,
            Self::RGBA16Float => vk::Format::R16G16B16A16_SFLOAT,
            Self::R32Float => vk::Format::R32_SFLOAT,
            Self::RG32Float => vk::Format::R32G32_SFLOAT,
            Self::RGB32Float => vk::Format::R32G32B32_SFLOAT,
            Self::RGBA32Float => vk::Format::R32G32B32A32_SFLOAT,

            Self::RGB10A2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
            Self::RG11B10Float => vk::Format::B10G11R11_UFLOAT_PACK32,

            Self::D16Unorm => vk::Format::D16_UNORM,
            Self::D24UnormS8UInt => vk::Format::D24_UNORM_S8_UINT,
            Self::D32Float => vk::Format::D32_SFLOAT,
            Self::S8UInt => vk::Format::S8_UINT,
            Self::D32FloatS8UInt => vk::Format::D32_SFLOAT_S8_UINT,

            Self::EACRG => vk::Format::EAC_R11G11_UNORM_BLOCK,
            Self::ASTC4X4Srgb => vk::Format::ASTC_4X4_SRGB_BLOCK,
            Self::ASTC4X4Unorm => vk::Format::ASTC_4X4_UNORM_BLOCK,
            Self::BC3Srgb => vk::Format::BC3_SRGB_BLOCK,
            Self::BC3Unorm => vk::Format::BC3_UNORM_BLOCK,
            Self::BC5RG => vk::Format::BC5_UNORM_BLOCK,
            Self::BC6HUFloat => vk::Format::BC6H_UFLOAT_BLOCK,
            Self::BC6HSFloat => vk::Format::BC6H_SFLOAT_BLOCK,
            Self::BC7Srgb => vk::Format::BC7_SRGB_BLOCK,
            Self::BC7Unorm => vk::Format::BC7_UNORM_BLOCK,

            Self::Undefined => vk::Format::UNDEFINED,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    SrgbNonLinear,
}

impl ColorSpace {
    pub(crate) fn to_vk(self) -> vk::ColorSpaceKHR {
        match self {
            Self::SrgbNonLinear => vk::ColorSpaceKHR::SRGB_NONLINEAR,
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

impl BufferUsageFlags {
    pub(crate) fn to_vk(self) -> vk::BufferUsageFlags {
        let mut result = vk::BufferUsageFlags::empty();

        if self.contains(Self::TRANSFER_SRC) {
            result |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if self.contains(Self::TRANSFER_DST) {
            result |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if self.contains(Self::UNIFORM) {
            result |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        if self.contains(Self::STORAGE) {
            result |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if self.contains(Self::INDEX) {
            result |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if self.contains(Self::VERTEX) {
            result |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if self.contains(Self::INDIRECT) {
            result |= vk::BufferUsageFlags::INDIRECT_BUFFER;
        }
        if self.contains(Self::DESCRIPTOR_HEAP) {
            result |= vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT;
        }
        if self.contains(Self::SHADER_DEVICE_ADDRESS) {
            result |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }

        result
    }

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

impl TextureDimension {
    pub(crate) fn to_vk_image_type(self) -> vk::ImageType {
        match self {
            Self::Dimension1D => vk::ImageType::TYPE_1D,
            Self::Dimension3D => vk::ImageType::TYPE_3D,
            Self::Dimension2D | Self::Cube | Self::Dimension2DArray | Self::CubeArray => vk::ImageType::TYPE_2D,
        }
    }

    pub(crate) fn to_vk_image_view_type(self) -> vk::ImageViewType {
        match self {
            Self::Dimension1D => vk::ImageViewType::TYPE_1D,
            Self::Dimension2D => vk::ImageViewType::TYPE_2D,
            Self::Dimension3D => vk::ImageViewType::TYPE_3D,
            Self::Cube => vk::ImageViewType::CUBE,
            Self::Dimension2DArray => vk::ImageViewType::TYPE_2D_ARRAY,
            Self::CubeArray => vk::ImageViewType::CUBE_ARRAY,
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

impl TextureUsageFlags {
    pub(crate) fn to_vk(self) -> vk::ImageUsageFlags {
        let mut result = vk::ImageUsageFlags::empty();

        if self.contains(Self::SAMPLED) {
            result |= vk::ImageUsageFlags::SAMPLED;
        }
        if self.contains(Self::STORAGE) {
            result |= vk::ImageUsageFlags::STORAGE;
        }
        if self.contains(Self::COLOR_ATTACHMENT) {
            result |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if self.contains(Self::DEPTH_STENCIL_ATTACHMENT) {
            result |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if self.contains(Self::TRANSFER_SRC) {
            result |= vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if self.contains(Self::TRANSFER_DST) {
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

impl SampleCount {
    pub(crate) fn to_vk(self) -> vk::SampleCountFlags {
        match self {
            Self::Count1 => vk::SampleCountFlags::TYPE_1,
            Self::Count2 => vk::SampleCountFlags::TYPE_2,
            Self::Count4 => vk::SampleCountFlags::TYPE_4,
            Self::Count8 => vk::SampleCountFlags::TYPE_8,
            Self::Count16 => vk::SampleCountFlags::TYPE_16,
            Self::Count32 => vk::SampleCountFlags::TYPE_32,
            Self::Count64 => vk::SampleCountFlags::TYPE_64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureDescriptorKind {
    Sampled,
    Storage,
}

impl TextureDescriptorKind {
    pub(crate) fn to_vk_descriptor_type(self) -> vk::DescriptorType {
        match self {
            Self::Sampled => vk::DescriptorType::SAMPLED_IMAGE,
            Self::Storage => vk::DescriptorType::STORAGE_IMAGE,
        }
    }

    pub(crate) fn to_vk_usage(self) -> vk::ImageUsageFlags {
        match self {
            Self::Sampled => vk::ImageUsageFlags::SAMPLED,
            Self::Storage => vk::ImageUsageFlags::STORAGE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterKind {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
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

impl CompareOp {
    pub fn to_vk(self) -> vk::CompareOp {
        match self {
            Self::Never => vk::CompareOp::NEVER,
            Self::Less => vk::CompareOp::LESS,
            Self::Equal => vk::CompareOp::EQUAL,
            Self::LessQual => vk::CompareOp::LESS_OR_EQUAL,
            Self::Greater => vk::CompareOp::GREATER,
            Self::NotEqual => vk::CompareOp::NOT_EQUAL,
            Self::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
            Self::Always => vk::CompareOp::ALWAYS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolygonMode {
    Fill,
    Line,
}

impl PolygonMode {
    pub(crate) fn to_vk(self) -> vk::PolygonMode {
        match self {
            Self::Fill => vk::PolygonMode::FILL,
            Self::Line => vk::PolygonMode::LINE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullMode {
    None,
    Front,
    Back,
}

impl CullMode {
    pub(crate) fn to_vk(self) -> vk::CullModeFlags {
        match self {
            Self::None => vk::CullModeFlags::NONE,
            Self::Front => vk::CullModeFlags::FRONT,
            Self::Back => vk::CullModeFlags::BACK,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrontFace {
    Clockwise,
    CounterClockwise,
}

impl FrontFace {
    pub(crate) fn to_vk(self) -> vk::FrontFace {
        match self {
            Self::Clockwise => vk::FrontFace::CLOCKWISE,
            Self::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
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

impl BlendFactor {
    pub(crate) fn to_vk(self) -> vk::BlendFactor {
        match self {
            Self::Zero => vk::BlendFactor::ZERO,
            Self::One => vk::BlendFactor::ONE,
            Self::SourceColor => vk::BlendFactor::SRC_COLOR,
            Self::OneMinusSourceColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            Self::DestinationColor => vk::BlendFactor::DST_COLOR,
            Self::OneMinusDestinationColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            Self::SourceAlpha => vk::BlendFactor::SRC_ALPHA,
            Self::OneMinusSourceAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            Self::DestinationAlpha => vk::BlendFactor::DST_ALPHA,
            Self::OneMinusDestinationAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
            Self::SourceAlphaStaurate => vk::BlendFactor::SRC_ALPHA_SATURATE,
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

impl BlendOp {
    pub(crate) fn to_vk(self) -> vk::BlendOp {
        match self {
            Self::Add => vk::BlendOp::ADD,
            Self::Substract => vk::BlendOp::SUBTRACT,
            Self::ResverseSubract => vk::BlendOp::REVERSE_SUBTRACT,
            Self::Minimum => vk::BlendOp::MIN,
            Self::Maximum => vk::BlendOp::MAX,
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

impl IndexKind {
    pub(crate) fn to_vk(self) -> vk::IndexType {
        match self {
            Self::UInt16 => vk::IndexType::UINT16,
            Self::UInt32 => vk::IndexType::UINT32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp {
    Load,
    Clear,
    Discard,
}

impl LoadOp {
    pub(crate) fn to_vk(self) -> vk::AttachmentLoadOp {
        match self {
            Self::Load => vk::AttachmentLoadOp::LOAD,
            Self::Clear => vk::AttachmentLoadOp::CLEAR,
            Self::Discard => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOp {
    Store,
    Discard,
}

impl StoreOp {
    pub(crate) fn to_vk(self) -> vk::AttachmentStoreOp {
        match self {
            Self::Store => vk::AttachmentStoreOp::STORE,
            Self::Discard => vk::AttachmentStoreOp::DONT_CARE,
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

impl StencilOp {
    pub(crate) fn to_vk(self) -> vk::StencilOp {
        match self {
            Self::Keep => vk::StencilOp::KEEP,
            Self::Zero => vk::StencilOp::ZERO,
            Self::Replace => vk::StencilOp::REPLACE,
            Self::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
            Self::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
            Self::Invert => vk::StencilOp::INVERT,
            Self::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
            Self::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
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

impl StageFlags {
    pub(crate) fn to_vk(self) -> vk::PipelineStageFlags2 {
        let mut flags = vk::PipelineStageFlags2::empty();

        if self.contains(Self::INDIRECT) {
            flags |= vk::PipelineStageFlags2::DRAW_INDIRECT;
        }

        if self.contains(Self::INDEX_INPUT) {
            flags |= vk::PipelineStageFlags2::INDEX_INPUT;
        }
        if self.contains(Self::VERTEX) {
            flags |= vk::PipelineStageFlags2::VERTEX_SHADER;
        }
        if self.contains(Self::MESH) {
            flags |= vk::PipelineStageFlags2::MESH_SHADER_EXT;
        }
        if self.contains(Self::DEPTH_STENCIL_TEST) {
            flags |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        }
        if self.contains(Self::FRAGMENT) {
            flags |= vk::PipelineStageFlags2::FRAGMENT_SHADER;
        }
        if self.contains(Self::COLOR_OUTPUT) {
            flags |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
        }
        if self.contains(Self::COMPUTE) {
            flags |= vk::PipelineStageFlags2::COMPUTE_SHADER;
        }
        if self.contains(Self::TRANSFER) {
            flags |= vk::PipelineStageFlags2::TRANSFER;
        }
        if self.contains(Self::HOST) {
            flags |= vk::PipelineStageFlags2::HOST;
        }
        if self.contains(Self::ALL_COMMANDS) {
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

impl AccessFlags {
    pub(crate) fn to_vk(self) -> vk::AccessFlags2 {
        let mut flags = vk::AccessFlags2::empty();

        if self.contains(Self::TRANSFER_READ) {
            flags |= vk::AccessFlags2::TRANSFER_READ;
        }
        if self.contains(Self::TRANSFER_WRITE) {
            flags |= vk::AccessFlags2::TRANSFER_WRITE;
        }
        if self.contains(Self::SHADER_READ) {
            flags |= vk::AccessFlags2::SHADER_READ;
        }
        if self.contains(Self::SHADER_WRITE) {
            flags |= vk::AccessFlags2::SHADER_WRITE;
        }
        if self.contains(Self::COLOR_READ) {
            flags |= vk::AccessFlags2::COLOR_ATTACHMENT_READ;
        }
        if self.contains(Self::COLOR_WRITE) {
            flags |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
        }
        if self.contains(Self::DEPTH_STENCIL_READ) {
            flags |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
        }
        if self.contains(Self::DPETH_STENICL_WRITE) {
            flags |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }
        if self.contains(Self::INDIRECT_READ) {
            flags |= vk::AccessFlags2::INDEX_READ;
        }
        if self.contains(Self::HOST_READ) {
            flags |= vk::AccessFlags2::HOST_READ;
        }
        if self.contains(Self::DESCIRPTOR_READ) {
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

impl Extent2D {
    pub(crate) fn to_vk(self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Extent3D {
    pub(crate) fn to_vk(self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

impl Offset2D {
    pub(crate) fn to_vk(self) -> vk::Offset2D {
        vk::Offset2D { x: self.x, y: self.y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Offset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Offset3D {
    pub(crate) fn to_vk(self) -> vk::Offset3D {
        vk::Offset3D {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rect2D {
    pub offset: Offset2D,
    pub extent: Extent2D,
}

impl Rect2D {
    pub(crate) fn to_vk(self) -> vk::Rect2D {
        vk::Rect2D {
            offset: self.offset.to_vk(),
            extent: self.extent.to_vk(),
        }
    }

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

impl TextureAspect {
    pub(crate) fn to_vk(self) -> vk::ImageAspectFlags {
        match self {
            Self::Color => vk::ImageAspectFlags::COLOR,
            Self::Depth => vk::ImageAspectFlags::DEPTH,
            Self::Stencil => vk::ImageAspectFlags::STENCIL,
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
