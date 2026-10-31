use arrayvec::ArrayVec;
use ash::vk;
use std::{cell::RefCell, marker::PhantomData, ops::Range, sync::Arc};
use thiserror::Error;
use tracing::debug;

use crate::core::{
    AccessFlags, Buffer, IndexKind, StageFlags, Texture, TextureAspect,
    device::DeviceShared,
    pipeline::{DepthStencilState, Pipeline, RenderingDesc, Scissor, Viewport},
    resource::BufferSlice,
    sync::{self, SyncPoint},
    types::*,
    utils::{VulkanCheck, create_object_debug_name},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueKind {
    Main,
    DedicatedCompute,
    DedicatedTransfer,
}

#[derive(Debug, Error)]
pub enum Error {}

pub struct CmdQueue {
    pub(crate) device: Arc<DeviceShared>,
    pub handle: vk::Queue,
    pub kind: QueueKind,
    /// Queue family
    pub family: u32,
    /// Queue index in family
    pub index: u32,
}

impl CmdQueue {
    pub(crate) fn new(device: &Arc<DeviceShared>, kind: QueueKind, family: u32, index: u32) -> Self {
        let queue = unsafe { device.handle.get_device_queue(family, index) };
        debug!("get queue {queue:?}, kind {kind:?}, family {family}");
        Self {
            device: device.clone(),
            handle: queue,
            kind,
            family,
            index,
        }
    }
}

pub trait CmdFrameState {}
impl CmdFrameState for state::Initial {}
impl CmdFrameState for state::Available {}
impl CmdFrameState for state::Active {}
impl CmdFrameState for state::Pending {}

pub struct CmdFrame<'q, S: CmdFrameState> {
    inner: CmdFrameInner<'q>,
    _state: PhantomData<S>,
}

struct CmdFrameInner<'q> {
    queue: &'q CmdQueue,
    pool: vk::CommandPool,
    storage: RefCell<CmdStorage>,
    finish_point: RefCell<Option<SyncPoint>>,
}

impl<'q> CmdFrame<'q, state::Initial> {
    pub fn new(queue: &'q CmdQueue, name: impl AsRef<str>) -> CmdFrame<'q, state::Pending> {
        let device = &queue.device;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue.family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let pool = unsafe { device.handle.create_command_pool(&pool_info, None) }.check(&device);
        debug!(
            "create command pool {pool:?}[{}] queue family {}",
            name.as_ref(),
            queue.family
        );
        create_object_debug_name(&device, pool, name.as_ref());

        CmdFrame {
            inner: CmdFrameInner {
                queue,
                pool,
                storage: RefCell::new(CmdStorage {
                    buffers: Vec::new(),
                    next: 0,
                    batches: Vec::new(),
                }),
                finish_point: RefCell::new(None),
            },
            _state: PhantomData,
        }
    }
}

impl Drop for CmdFrameInner<'_> {
    fn drop(&mut self) {
        let device = &self.queue.device;
        let pool = self.pool;
        unsafe { device.handle.destroy_command_pool(pool, None) };
    }
}

struct CmdStorage {
    buffers: Vec<vk::CommandBuffer>,
    next: usize,
    batches: Vec<CmdEncoderBatch>,
}

impl<'q> CmdFrame<'q, state::Available> {
    pub fn begin(self) -> CmdFrame<'q, state::Active> {
        CmdFrame {
            inner: self.inner,
            _state: PhantomData,
        }
    }
}

impl<'q> CmdFrame<'q, state::Active> {
    fn ensure_capacity(&self) {
        let mut storage = self.inner.storage.borrow_mut();
        let capacity = storage.buffers.len();

        if storage.next < capacity && capacity != 0 {
            return;
        }

        let num_to_grow = if capacity == 0 {
            1
        } else {
            capacity + ((capacity + 1) >> 1)
        };

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.inner.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(num_to_grow as u32);
        let device = &self.inner.queue.device;
        let new_buffers = unsafe { device.handle.allocate_command_buffers(&allocate_info) }.check(device);
        storage.buffers.extend(new_buffers);
    }

    pub fn bump_encoder<'e>(&'e self) -> CmdEncoder<'e, state::Active> {
        self.ensure_capacity();

        let mut storage = self.inner.storage.borrow_mut();
        let index = storage.next;
        let buffer = storage.buffers[index];
        storage.next += 1;

        let device = &self.inner.queue.device;
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.handle.begin_command_buffer(buffer, &begin_info) }.check(device);

        CmdEncoder::<state::Active> {
            inner: CmdEncoderInner {
                index,
                queue: self.inner.queue,
                buffer,
            },
            _state: PhantomData,
        }
    }

    pub fn batch<'e>(
        &'e self,
        encoders: impl IntoIterator<Item = CmdEncoder<'e, state::Finished>>,
        waits: impl IntoIterator<Item = SyncPoint>,
        signals: impl IntoIterator<Item = SyncPoint>,
    ) {
        self.inner.storage.borrow_mut().batches.push(CmdEncoderBatch {
            indices: encoders.into_iter().map(|encoder| encoder.index()).collect(),
            waits: waits.into_iter().collect(),
            signals: signals.into_iter().collect(),
        });
    }

    pub fn end_and_submit<'e>(self, finish_point: SyncPoint) -> CmdFrame<'q, state::Pending> {
        {
            let storage = self.inner.storage.borrow();

            let batches: Vec<(
                Vec<vk::CommandBufferSubmitInfo>,
                Vec<vk::SemaphoreSubmitInfo>,
                Vec<vk::SemaphoreSubmitInfo>,
            )> = storage
                .batches
                .iter()
                .map(|batch| {
                    (
                        batch
                            .indices
                            .iter()
                            .map(|index| {
                                vk::CommandBufferSubmitInfo::default()
                                    .command_buffer(storage.buffers[*index])
                                    .device_mask(1)
                            })
                            .collect(),
                        batch.waits.iter().map(|wait| wait.to_vk()).collect(),
                        batch.signals.iter().map(|signal| signal.to_vk()).collect(),
                    )
                })
                .collect();

            let submits: Vec<vk::SubmitInfo2> = batches
                .iter()
                .map(|batch| {
                    vk::SubmitInfo2::default()
                        .command_buffer_infos(&batch.0)
                        .wait_semaphore_infos(&batch.1)
                        .signal_semaphore_infos(&batch.2)
                })
                .collect();

            if !submits.is_empty() {
                let device = &self.inner.queue.device;
                let queue = self.inner.queue.handle;
                unsafe { device.handle.queue_submit2(queue, &submits, vk::Fence::null()) }.check(device);
            }
        }
        {
            self.inner.storage.borrow_mut().batches.clear();
        }
        {
            *self.inner.finish_point.borrow_mut() = Some(finish_point);
        }

        CmdFrame {
            inner: self.inner,
            _state: PhantomData,
        }
    }
}

impl<'q> CmdFrame<'q, state::Pending> {
    pub fn wait_and_reset(self) -> CmdFrame<'q, state::Available> {
        let device = &self.inner.queue.device;

        if let Some(finish_point) = *self.inner.finish_point.borrow() {
            sync::wait_blocking(device, finish_point);
        }

        {
            let mut storage = self.inner.storage.borrow_mut();
            if !storage.buffers.is_empty() {
                unsafe {
                    device
                        .handle
                        .reset_command_pool(self.inner.pool, vk::CommandPoolResetFlags::empty())
                        .check(device)
                };
            }
            storage.next = 0;
        }

        CmdFrame {
            inner: self.inner,
            _state: PhantomData,
        }
    }
}

struct CmdEncoderBatch {
    indices: Vec<usize>,
    waits: Vec<SyncPoint>,
    signals: Vec<SyncPoint>,
}

pub trait CmdEncoderState {}
impl CmdEncoderState for state::Active {}
impl CmdEncoderState for state::Finished {}

pub struct CmdEncoder<'e, S: CmdEncoderState> {
    pub(crate) inner: CmdEncoderInner<'e>,
    _state: PhantomData<S>,
}

pub(crate) struct CmdEncoderInner<'e> {
    index: usize,
    pub(crate) queue: &'e CmdQueue,
    pub(crate) buffer: vk::CommandBuffer,
}

impl CmdEncoder<'_, state::Active> {
    pub fn bind_pipeline(&self, pipeline: &Pipeline) {
        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_bind_pipeline(buffer, pipeline.kind.to_vk_bind_point(), pipeline.handle) };
    }

    pub fn set_viewport(&self, viewport: Viewport) {
        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_set_viewport_with_count(buffer, &[viewport.to_vk()]) };
    }

    pub fn set_scissor(&self, scissor: Scissor) {
        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_set_scissor_with_count(buffer, &[scissor.to_vk()]) };
    }

    pub fn set_stencil(&self, state: DepthStencilState) {
        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;

        unsafe {
            let depth = &state.depth;
            device.cmd_set_depth_test_enable(buffer, depth.test);
            if state.depth.test {
                device.cmd_set_depth_write_enable(buffer, depth.write);
                device.cmd_set_depth_compare_op(buffer, depth.compare.to_vk());
            }

            let stencil = &state.stencil;
            device.cmd_set_stencil_test_enable(buffer, stencil.test);
            if state.stencil.test {
                device.cmd_set_stencil_op(
                    buffer,
                    vk::StencilFaceFlags::FRONT,
                    stencil.front.fail.to_vk(),
                    stencil.front.pass.to_vk(),
                    stencil.front.depth_fail.to_vk(),
                    stencil.front.compare.to_vk(),
                );
                device.cmd_set_stencil_op(
                    buffer,
                    vk::StencilFaceFlags::BACK,
                    stencil.back.fail.to_vk(),
                    stencil.back.pass.to_vk(),
                    stencil.back.depth_fail.to_vk(),
                    stencil.back.compare.to_vk(),
                );
                device.cmd_set_stencil_compare_mask(buffer, vk::StencilFaceFlags::FRONT_AND_BACK, stencil.compare_mask);
                device.cmd_set_stencil_write_mask(buffer, vk::StencilFaceFlags::FRONT_AND_BACK, stencil.write_mask);
                device.cmd_set_stencil_reference(buffer, vk::StencilFaceFlags::FRONT, stencil.front.reference);
                device.cmd_set_stencil_reference(buffer, vk::StencilFaceFlags::BACK, stencil.back.reference);
            }
        }
    }

    pub fn texture_init_barrier(
        &self,
        texture: &Texture,
        aspect: TextureAspect,
        mip_range: Range<u32>,
        layer_range: Range<u32>,
    ) {
        let barrier_infos = [vk::ImageMemoryBarrier2::default()
            .image(texture.handle)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect.to_vk())
                    .base_mip_level(mip_range.start)
                    .level_count(mip_range.end - mip_range.start)
                    .base_array_layer(layer_range.start)
                    .layer_count(layer_range.end - layer_range.start),
            )];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barrier_infos);

        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_pipeline_barrier2(buffer, &dependency_info) };
    }

    pub fn barrier(
        &self,
        src_stage: StageFlags,
        src_access: AccessFlags,
        dst_stage: StageFlags,
        dst_access: AccessFlags,
    ) {
        let barrier_infos = [vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage.to_vk())
            .src_access_mask(src_access.to_vk())
            .dst_stage_mask(dst_stage.to_vk())
            .dst_access_mask(dst_access.to_vk())];
        let dependency_info = vk::DependencyInfo::default().memory_barriers(&barrier_infos);

        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_pipeline_barrier2(buffer, &dependency_info) };
    }

    pub fn emit_root(&self, root: &Buffer) {
        let address_bytes = root.device_address.to_ne_bytes();
        let root_info =
            vk::PushDataInfoEXT::default().data(vk::HostAddressRangeConstEXT::default().address(&address_bytes));

        let device = &self.inner.queue.device;
        let buffer = self.inner.buffer;
        unsafe { device.extensions.descriptor_heap.cmd_push_data(buffer, &root_info) };
    }

    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_dispatch(buffer, group_count_x, group_count_y, group_count_z) };
    }

    pub fn begin_rendering(&self, desc: RenderingDesc) {
        let flags = vk::RenderingFlags::empty();

        let color_targets: ArrayVec<_, 8> = desc
            .colors
            .iter()
            .map(|color| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(color.view.handle)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(color.load.to_vk())
                    .store_op(color.store.to_vk())
                    .clear_value(color.clear.to_vk())
            })
            .collect();

        let mut rendering_info = vk::RenderingInfo::default()
            .flags(flags)
            .render_area(desc.area.to_vk())
            .layer_count(1)
            .view_mask(0)
            .color_attachments(&color_targets);

        let mut depth_target = vk::RenderingAttachmentInfo::default();
        if let Some(depth) = desc.depth {
            depth_target = depth_target
                .image_view(depth.view.handle)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(depth.load.to_vk())
                .store_op(depth.store.to_vk())
                .clear_value({
                    let mut clear_value = vk::ClearValue::default();
                    clear_value.depth_stencil = unsafe { clear_value.depth_stencil.depth(depth.clear) };
                    clear_value
                });
            rendering_info = rendering_info.depth_attachment(&depth_target);
        }

        let mut stencil_target = vk::RenderingAttachmentInfo::default();
        if let Some(stencil) = desc.stencil {
            stencil_target = stencil_target
                .image_view(stencil.view.handle)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(stencil.load.to_vk())
                .store_op(stencil.store.to_vk())
                .clear_value({
                    let mut clear_value = vk::ClearValue::default();
                    clear_value.depth_stencil = unsafe { clear_value.depth_stencil.stencil(stencil.clear) };
                    clear_value
                });
            rendering_info = rendering_info.stencil_attachment(&stencil_target);
        }

        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_begin_rendering(buffer, &rendering_info) };
    }

    pub fn end_rendering(&self) {
        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_end_rendering(buffer) }
    }

    pub fn draw(&self, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_draw(buffer, vertex_count, instance_count, first_vertex, first_instance) };
    }

    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
        indices: BufferSlice,
        kind: IndexKind,
    ) {
        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        let extension = &self.inner.queue.device.extensions.device_address_commands;

        let bind_info = vk::BindIndexBuffer3InfoKHR::default()
            .address_range(
                vk::DeviceAddressRangeKHR::default()
                    .address(indices.device_base)
                    .size(indices.size),
            )
            .address_flags(vk::AddressCommandFlagsKHR::FULLY_BOUND)
            .index_type(kind.to_vk());
        unsafe { extension.cmd_bind_index_buffer3(buffer, &bind_info) };
        unsafe {
            device.cmd_draw_indexed(
                buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        };
    }

    pub fn draw_mesh(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        let buffer = self.inner.buffer;
        let extension = &self.inner.queue.device.extensions.mesh_shader;
        unsafe { extension.cmd_draw_mesh_tasks(buffer, group_count_x, group_count_y, group_count_z) };
    }
}

impl<'e> CmdEncoder<'e, state::Active> {
    pub fn finish(self) -> CmdEncoder<'e, state::Finished> {
        let device = &self.inner.queue.device;
        let buffer = self.inner.buffer;
        unsafe { device.handle.end_command_buffer(buffer) }.check(&device);
        CmdEncoder::<state::Finished> {
            inner: self.inner,
            _state: PhantomData,
        }
    }
}

impl CmdEncoder<'_, state::Finished> {
    pub(crate) fn index(self) -> usize {
        self.inner.index
    }
}
