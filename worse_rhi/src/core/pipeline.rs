use arrayvec::ArrayVec;
use ash::vk::{self, TaggedStructure};
use std::{
    ffi::CStr,
    hash::{Hash, Hasher},
};
use thiserror::Error;
use tracing::debug;

use crate::core::{
    device::Device,
    resource::TargetView,
    types::*,
    utils::{create_object_debug_name, string_to_cstring_lossy},
};

pub const MAX_ENTRY_NAME_LENGTH: usize = 64;
pub const MAX_COLOR_TARGETS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Error {
    #[error("vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ShaderStageFlags : u32 {
        const TASK        = 1 << 0;
        const MESH        = 1 << 1;
        const VERTEX      = 1 << 2;
        const FRAGMENT    = 1 << 3;

        const COMPUTE      = 1 << 4;

        const ALL_RASTER =
            Self::TASK.bits()
            | Self::MESH.bits()
            | Self::VERTEX.bits()
            | Self::FRAGMENT.bits();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineKind {
    Raster,
    Compute,
}

impl PipelineKind {
    pub fn to_vk_bind_point(self) -> vk::PipelineBindPoint {
        match self {
            Self::Raster => vk::PipelineBindPoint::GRAPHICS,
            Self::Compute => vk::PipelineBindPoint::COMPUTE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderStage<'desc> {
    pub(crate) code: &'desc [u32],
    pub(crate) entry: &'desc str,
}

impl<'desc> ShaderStage<'desc> {
    pub fn new(code: &'desc [u32], entry: &'desc str) -> Self {
        Self { code, entry }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveStages<'desc> {
    Vertex {
        vertex: ShaderStage<'desc>,
    },
    Mesh {
        task: Option<ShaderStage<'desc>>,
        mesh: ShaderStage<'desc>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterizerState {
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub depth_bias_constant: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slop: f32,
}

impl Hash for RasterizerState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.polygon_mode.hash(state);
        self.cull_mode.hash(state);
        self.front_face.hash(state);
        hash_f32(self.depth_bias_constant, state);
        hash_f32(self.depth_bias_clamp, state);
        hash_f32(self.depth_bias_slop, state);
    }
}

fn hash_f32<H: Hasher>(value: f32, state: &mut H) {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    bits.hash(state);
}

impl Default for RasterizerState {
    fn default() -> Self {
        Self {
            polygon_mode: PolygonMode::Fill,
            cull_mode: CullMode::Back,
            front_face: FrontFace::Clockwise,
            depth_bias_constant: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slop: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendComponentState {
    pub source: BlendFactor,
    pub destination: BlendFactor,
    pub operation: BlendOp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorTargetBlendState {
    pub color: BlendComponentState,
    pub alpha: BlendComponentState,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ColorComponentFlags: u32 {
        const R = 0b0001;
        const G = 0b0010;
        const B = 0b0100;
        const A = 0b1000;
        const ALL = Self::R.bits()
            | Self::G.bits()
            | Self::B.bits()
            | Self::A.bits();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorTargetDesc {
    pub format: PixelFormat,
    pub blend_state: Option<ColorTargetBlendState>,
    pub write_mask: ColorComponentFlags,
}

impl ColorTargetDesc {
    pub fn default_with_format(format: PixelFormat) -> Self {
        Self {
            format,
            blend_state: None,
            write_mask: ColorComponentFlags::ALL,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct RasterPipelineDesc<'desc> {
    pub primitive: PrimitiveStages<'desc>,
    pub fragment: ShaderStage<'desc>,
    pub rasterizer: RasterizerState,
    pub color_targets: &'desc [ColorTargetDesc],
    pub depth_format: PixelFormat,
    pub stencil_format: PixelFormat,
}

pub struct Pipeline {
    pub(crate) handle: vk::Pipeline,
    pub(crate) kind: PipelineKind,
}

impl Device {
    pub fn create_compute_pipeline(&self, compute: ShaderStage) -> Result<Pipeline, Error> {
        let root_mapping = [vk::DescriptorSetAndBindingMappingEXT::default()
            .descriptor_set(0)
            .first_binding(0)
            .binding_count(1)
            .resource_mask(vk::SpirvResourceTypeFlagsEXT::UNIFORM_BUFFER)
            .source(vk::DescriptorMappingSourceEXT::PUSH_ADDRESS)
            .source_data({
                let mut push_address_data = vk::DescriptorMappingSourceDataEXT::default();
                push_address_data.push_address_offset = 0;
                push_address_data
            })];

        let mut module_info = vk::ShaderModuleCreateInfo::default().code(compute.code);
        let mut mapping_info = vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(&root_mapping);

        let entry_name = string_to_cstring_lossy(compute.entry);
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .push(&mut module_info)
            .push(&mut mapping_info)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(&entry_name);

        let mut flags_info =
            vk::PipelineCreateFlags2CreateInfo::default().flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);

        let pipeline_infos = [vk::ComputePipelineCreateInfo::default()
            .push(&mut flags_info)
            .stage(stage_info)
            .base_pipeline_index(-1)];
        let pipeline = match unsafe {
            self.shared
                .handle
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
        } {
            Ok(pipelines) => {
                assert_eq!(pipelines.len(), 1);
                pipelines[0]
            }
            Err((pipelines, err)) => {
                assert!(pipelines.is_empty());
                return Err(Error::Vulkan(err));
            }
        };
        debug!("create pipeline {pipeline:?}");
        create_object_debug_name(&self.shared, pipeline, "compute-pipeline");
        Ok(Pipeline {
            handle: pipeline,
            kind: PipelineKind::Compute,
        })
    }

    pub fn create_raster_pipeline(&self, desc: RasterPipelineDesc) -> Result<Pipeline, Error> {
        let root_mappings = [vk::DescriptorSetAndBindingMappingEXT::default()
            .descriptor_set(0)
            .first_binding(0)
            .binding_count(1)
            .resource_mask(vk::SpirvResourceTypeFlagsEXT::UNIFORM_BUFFER)
            .source(vk::DescriptorMappingSourceEXT::PUSH_ADDRESS)
            .source_data({
                let mut push_address_data = vk::DescriptorMappingSourceDataEXT::default();
                push_address_data.push_address_offset = 0;
                push_address_data
            })];

        const MAX_STAGE: usize = 3;
        struct StageInfoSource<'a> {
            stage: vk::ShaderStageFlags,
            code: &'a [u32],
            entry_name: &'a str,
        }
        let mut info_sources = ArrayVec::<StageInfoSource, MAX_STAGE>::new();

        match &desc.primitive {
            PrimitiveStages::Vertex { vertex } => {
                info_sources.push(StageInfoSource {
                    stage: vk::ShaderStageFlags::VERTEX,
                    code: vertex.code,
                    entry_name: vertex.entry,
                });
            }
            PrimitiveStages::Mesh { task, mesh } => {
                if let Some(task) = task {
                    info_sources.push(StageInfoSource {
                        stage: vk::ShaderStageFlags::TASK_EXT,
                        code: task.code,
                        entry_name: task.entry,
                    });
                }

                info_sources.push(StageInfoSource {
                    stage: vk::ShaderStageFlags::MESH_EXT,
                    code: mesh.code,
                    entry_name: mesh.entry,
                });
            }
        };
        info_sources.push(StageInfoSource {
            stage: vk::ShaderStageFlags::FRAGMENT,
            code: desc.fragment.code,
            entry_name: desc.fragment.entry,
        });

        let mut module_infos: ArrayVec<_, MAX_STAGE> = info_sources
            .iter()
            .map(|source| vk::ShaderModuleCreateInfo::default().code(source.code))
            .collect();
        let mut mapping_infos: ArrayVec<_, MAX_STAGE> = info_sources
            .iter()
            .map(|_| vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(&root_mappings))
            .collect();

        let entry_names: ArrayVec<ArrayVec<u8, MAX_ENTRY_NAME_LENGTH>, MAX_STAGE> = info_sources
            .iter()
            .map(|source| {
                let mut buffer = ArrayVec::new();
                buffer
                    .try_extend_from_slice(source.entry_name.as_bytes())
                    .expect("entry name exceeds MAX_ENTRY_NAME_LENGTH");
                buffer.push(0);
                buffer
            })
            .collect();
        let mut raw_entry_names: ArrayVec<_, MAX_STAGE> = entry_names
            .iter()
            .map(|buffer| CStr::from_bytes_with_nul(buffer).unwrap())
            .collect();

        let stage_infos: ArrayVec<_, MAX_STAGE> = info_sources
            .iter()
            .zip(module_infos.iter_mut())
            .zip(mapping_infos.iter_mut())
            .zip(raw_entry_names.iter_mut())
            .map(|(((source, module_info), mapping_info), entry_name)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .push(module_info)
                    .push(mapping_info)
                    .stage(source.stage)
                    .name(*entry_name)
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default();
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(desc.rasterizer.polygon_mode.to_vk())
            .cull_mode(desc.rasterizer.cull_mode.to_vk())
            .front_face(desc.rasterizer.front_face.to_vk())
            .depth_bias_enable(
                desc.rasterizer.depth_bias_constant != 0.0
                    || desc.rasterizer.depth_bias_constant != 0.0
                    || desc.rasterizer.depth_bias_slop != 0.0,
            )
            .depth_bias_constant_factor(desc.rasterizer.depth_bias_constant)
            .depth_bias_clamp(desc.rasterizer.depth_bias_clamp)
            .depth_bias_slope_factor(desc.rasterizer.depth_bias_slop)
            .line_width(1.0);
        let multisample =
            vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let (color_blend_states, color_formats): (ArrayVec<_, MAX_COLOR_TARGETS>, ArrayVec<_, MAX_COLOR_TARGETS>) =
            desc.color_targets
                .iter()
                .map(|color_target| {
                    let mut color_blend_state = vk::PipelineColorBlendAttachmentState::default()
                        .blend_enable(color_target.blend_state.is_some())
                        .color_write_mask(vk::ColorComponentFlags::from_raw(color_target.write_mask.bits()));
                    if let Some(blend_state) = color_target.blend_state {
                        color_blend_state = color_blend_state
                            .src_color_blend_factor(blend_state.color.source.to_vk())
                            .dst_color_blend_factor(blend_state.color.destination.to_vk())
                            .color_blend_op(blend_state.color.operation.to_vk())
                            .src_alpha_blend_factor(blend_state.alpha.source.to_vk())
                            .dst_alpha_blend_factor(blend_state.alpha.destination.to_vk())
                            .alpha_blend_op(blend_state.alpha.operation.to_vk());
                    }
                    (color_blend_state, color_target.format.to_vk())
                })
                .unzip();
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_states);
        let dynamic_states = [
            vk::DynamicState::VIEWPORT_WITH_COUNT,
            vk::DynamicState::SCISSOR_WITH_COUNT,
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
        ];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(desc.depth_format.to_vk())
            .stencil_attachment_format(desc.stencil_format.to_vk());
        let mut pipeline_flags =
            vk::PipelineCreateFlags2CreateInfo::default().flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .push(&mut rendering_info)
            .push(&mut pipeline_flags)
            .stages(&stage_infos)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .base_pipeline_index(-1);
        if matches!(&desc.primitive, PrimitiveStages::Vertex { .. }) {
            pipeline_info = pipeline_info
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly);
        }

        let pipeline_infos = [pipeline_info];
        let pipeline = match unsafe {
            self.shared
                .handle
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
        } {
            Ok(pipelines) => {
                assert_eq!(pipelines.len(), 1);
                pipelines[0]
            }
            Err((pipelines, err)) => {
                assert!(pipelines.is_empty());
                return Err(Error::Vulkan(err));
            }
        };
        debug!("create pipeline {pipeline:?}");
        create_object_debug_name(&self.shared, pipeline, "raster-pipeline");
        Ok(Pipeline {
            handle: pipeline,
            kind: PipelineKind::Raster,
        })
    }

    pub fn destroy_pipeline(&self, pipeline: Pipeline) {
        unsafe { self.shared.handle.destroy_pipeline(pipeline.handle, None) };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ClearColor {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl ClearColor {
    pub(crate) fn to_vk(self) -> vk::ClearValue {
        let mut clear_value = vk::ClearValue::default();
        clear_value.color = vk::ClearColorValue {
            float32: [self.x, self.y, self.z, self.w],
        };
        clear_value
    }
}

impl ClearColor {
    pub const EMPTY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };
    pub const FILL: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        w: 1.0,
    };
}

#[derive(Debug)]
pub struct ColorTarget {
    pub view: TargetView,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: ClearColor,
}

impl ColorTarget {
    pub fn new(view: TargetView, load: LoadOp, store: StoreOp, clear: ClearColor) -> Self {
        Self {
            view,
            load,
            store,
            clear,
        }
    }
}

#[derive(Debug)]
pub struct DepthTarget {
    pub view: TargetView,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: f32,
}

#[derive(Debug)]
pub struct StencilTarget {
    pub view: TargetView,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: u32,
}

#[derive(Debug)]
pub struct RenderingDesc<'desc> {
    pub area: Rect2D,
    pub colors: &'desc [ColorTarget],
    pub depth: Option<DepthTarget>,
    pub stencil: Option<StencilTarget>,
}

impl Default for RenderingDesc<'_> {
    fn default() -> Self {
        Self {
            area: Rect2D::default(),
            colors: &[],
            depth: None,
            stencil: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Viewport {
    pub(crate) fn to_vk(self) -> vk::Viewport {
        vk::Viewport {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
        }
    }
}

pub type Scissor = Rect2D;

#[derive(Debug, Clone, Copy)]
pub struct DepthState {
    pub test: bool,
    pub write: bool,
    pub compare: CompareOp,
}

impl Default for DepthState {
    fn default() -> Self {
        Self {
            test: false,
            write: false,
            compare: CompareOp::Never,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StencilFaceState {
    pub compare: CompareOp,
    pub fail: StencilOp,
    pub pass: StencilOp,
    pub depth_fail: StencilOp,
    pub reference: u32,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            compare: CompareOp::Never,
            fail: StencilOp::Keep,
            pass: StencilOp::Keep,
            depth_fail: StencilOp::Keep,
            reference: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StencilState {
    pub test: bool,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub front: StencilFaceState,
    pub back: StencilFaceState,
}

impl Default for StencilState {
    fn default() -> Self {
        Self {
            test: false,
            compare_mask: 0xFF,
            write_mask: 0xFF,
            front: StencilFaceState::default(),
            back: StencilFaceState::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DepthStencilState {
    pub depth: DepthState,
    pub stencil: StencilState,
}
