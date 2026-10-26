use arrayvec::ArrayVec;
use ash::vk::{self, TaggedStructure};
use std::sync::Arc;
use thiserror::Error;
use tracing::debug;

use crate::core::{
    CmdQueue, TargetView, Timeline,
    command::CmdEncoder,
    device::{Device, DeviceShared},
    instance::InstanceShared,
    sync::SyncPoint,
    types::*,
    utils::{VulkanCheck, create_object_debug_name, state},
};

pub const MAX_FRAMES_IN_FLIGHT: usize = 4;
pub const MAX_SWAPCHAIN_IMAGES: usize = 8;
/// Retired swapchains waiting for destruction. When full, the oldest is destroyed with a blocking wait.
const MAX_RETIRED: usize = 8;
/// Rebuild + acquire attempts per `begin_frame` before giving up on this frame.
const MAX_REBUILD_ATTEMPTS: usize = 3;

#[derive(Debug, Error)]
pub enum SwapchainCreateError {
    #[error("instance support for surface is not available")]
    InstanceSurfaceSupportNotAvailable,

    #[error("device support for swapchain is not available")]
    DeviceSwapchainSupportNotAvailable,

    #[error("skip creating swapchain because window is minimized")]
    WindowMinimized,

    #[error("unsupported surface format: {0:?}")]
    UnsupportedSurfaceFormat(vk::SurfaceFormatKHR),

    #[error("unsupported alpha composite: {0:?}")]
    UnsupportedAlphaComposite(SwapchainAlphaComposite),

    #[error("swapchain returned more than {} images", MAX_SWAPCHAIN_IMAGES)]
    TooManyImages,

    #[error("vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

#[derive(Debug, Error)]
pub enum SwapchainError {
    #[error(transparent)]
    Create(#[from] SwapchainCreateError),

    #[error("vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentMode {
    Immediate,
    Mailbox,
    Fifo,
    FifoRelaxed,
}

impl PresentMode {
    pub fn to_vk(self) -> vk::PresentModeKHR {
        match self {
            PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
            PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
            PresentMode::Fifo => vk::PresentModeKHR::FIFO,
            PresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapchainAlphaComposite {
    Opaque,
    Transparent,
}

impl SwapchainAlphaComposite {
    pub fn to_vk(self) -> vk::CompositeAlphaFlagsKHR {
        match self {
            Self::Opaque => vk::CompositeAlphaFlagsKHR::OPAQUE,
            Self::Transparent => vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SwapchainCreateOptions {
    pub format: PixelFormat,
    pub color_space: ColorSpace,
    /// Requested swapchain image count (clamped to surface limits and `MAX_SWAPCHAIN_IMAGES`).
    pub image_count: u32,
    /// Number of frame slots (clamped to `1..=MAX_FRAMES_IN_FLIGHT`). Independent from the image count.
    pub frames_in_flight: u32,
    pub present_mode: PresentMode,
    pub alpha_composite: SwapchainAlphaComposite,
    /// Application's window extent
    pub window_extent: Extent2D,
}

impl SwapchainCreateOptions {
    pub fn default_with_count(image_count: u32, frames_in_flight: u32) -> Self {
        Self {
            format: PixelFormat::RGBA8Srgb,
            color_space: ColorSpace::SrgbNonLinear,
            image_count,
            frames_in_flight,
            present_mode: PresentMode::Mailbox,
            alpha_composite: SwapchainAlphaComposite::Opaque,
            window_extent: Extent2D::default(),
        }
    }
}

/// Token for one acquired image. Must be submitted and then passed to [`Swapchain::present`].
///
/// Contract:
/// - the image is in `UNDEFINED` layout and must end in `PRESENT_SRC_KHR`;
/// - the submit must wait `wait_semaphore` at a stage covering the first use of the image;
/// - the submit must signal `signal_semaphore` (binary) and the timeline with the `finish_value`
///   later passed to `present`.
#[must_use = "an acquired frame must be submitted and presented"]
pub struct SwapchainFrame {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub extent: Extent2D,
    pub format: PixelFormat,
    pub(crate) image_index: u32,
    /// Index for the user's own per-frame resources.
    pub(crate) frame_index: usize,
    pub(crate) wait_semaphore: vk::Semaphore,
    pub(crate) signal_semaphore: vk::Semaphore,
    pub(crate) generation: u64,
}

impl SwapchainFrame {
    pub fn acquisition_wait(&self) -> SyncPoint {
        SyncPoint::binary(self.wait_semaphore, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
    }

    pub fn presentation_signal(&self) -> SyncPoint {
        SyncPoint::binary(self.signal_semaphore, vk::PipelineStageFlags2::ALL_COMMANDS)
    }
}

struct FrameSlot {
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    present_fence: vk::Fence,
    /// Timeline value of the last submit that used this slot.
    finish_value: u64,
    /// `Some(generation)` while `present_fence` is going to be signaled by a present on that generation.
    pending_present: Option<u64>,
}

impl FrameSlot {
    fn new(device: &DeviceShared) -> Self {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default();

        let image_available = unsafe { device.handle.create_semaphore(&semaphore_info, None) }.check(device);
        create_object_debug_name(device, image_available, "image_available");
        let render_finished = unsafe { device.handle.create_semaphore(&semaphore_info, None) }.check(device);
        create_object_debug_name(device, render_finished, "render_finished");
        let present_fence = unsafe { device.handle.create_fence(&fence_info, None) }.check(device);
        create_object_debug_name(device, present_fence, "present_fence");

        Self {
            image_available,
            render_finished,
            present_fence,
            finish_value: 0,
            pending_present: None,
        }
    }

    /// Blocks until the pending present (if any) released its semaphore, then resets the fence.
    fn wait_present(&mut self, device: &DeviceShared) {
        if self.pending_present.take().is_some() {
            let fences = [self.present_fence];
            unsafe {
                device.handle.wait_for_fences(&fences, true, u64::MAX).check(device);
                device.handle.reset_fences(&fences).check(device);
            }
        }
    }

    fn destroy(&self, device: &DeviceShared) {
        unsafe {
            device.handle.destroy_semaphore(self.image_available, None);
            device.handle.destroy_semaphore(self.render_finished, None);
            device.handle.destroy_fence(self.present_fence, None);
        }
    }
}

struct SwapchainState {
    generation: u64,
    handle: vk::SwapchainKHR,
    extent: Extent2D,
    format: PixelFormat,
    /// Indexed by `vkGetSwapchainImagesKHR`
    images: ArrayVec<vk::Image, MAX_SWAPCHAIN_IMAGES>,
    views: ArrayVec<vk::ImageView, MAX_SWAPCHAIN_IMAGES>,
    /// Timeline value of the last present submitted on this swapchain.
    last_use_value: u64,
}

impl SwapchainState {
    fn into_retired(self) -> Retired {
        Retired {
            generation: self.generation,
            handle: self.handle,
            views: self.views,
            last_use_value: self.last_use_value,
        }
    }
}

struct Retired {
    generation: u64,
    handle: vk::SwapchainKHR,
    views: ArrayVec<vk::ImageView, MAX_SWAPCHAIN_IMAGES>,
    last_use_value: u64,
}

impl Retired {
    /// Caller must guarantee that the GPU and all presents are done with this swapchain.
    fn destroy(&self, device: &DeviceShared) {
        unsafe {
            for view in &self.views {
                device.handle.destroy_image_view(*view, None);
            }
            if let Some(ext) = device.extensions.swapchain.as_ref() {
                ext.destroy_swapchain(self.handle, None);
            }
        }
    }
}

struct BuildParams {
    extent: vk::Extent2D,
    image_count: u32,
    composite_alpha: vk::CompositeAlphaFlagsKHR,
    transform: vk::SurfaceTransformFlagsKHR,
}

pub struct Swapchain {
    device: Arc<DeviceShared>,
    instance: Arc<InstanceShared>,
    options: SwapchainCreateOptions,
    /// `None`: not created yet, minimized, or the last creation failed.
    current: Option<SwapchainState>,
    retired: ArrayVec<Retired, MAX_RETIRED>,
    /// Survive swapchain rebuilds.
    slots: ArrayVec<FrameSlot, MAX_FRAMES_IN_FLIGHT>,
    frame_index: usize,
    generation: u64,
    dirty: bool,
}

impl Swapchain {
    /// The underlying swapchain is created lazily by the first `begin_frame`.
    pub fn new(device: &Device, options: SwapchainCreateOptions) -> Result<Self, SwapchainCreateError> {
        if device.instance.extensions.surface.is_none() {
            return Err(SwapchainCreateError::InstanceSurfaceSupportNotAvailable);
        }
        if device.shared.extensions.swapchain.is_none() {
            return Err(SwapchainCreateError::DeviceSwapchainSupportNotAvailable);
        }

        let mut overrided_options = options;
        overrided_options.frames_in_flight = options.frames_in_flight.clamp(1, MAX_FRAMES_IN_FLIGHT as u32);
        let slots = (0..options.frames_in_flight)
            .map(|_| FrameSlot::new(&device.shared))
            .collect();

        Ok(Self {
            device: device.shared.clone(),
            instance: device.instance.clone(),
            options: overrided_options,
            current: None,
            retired: ArrayVec::new(),
            slots,
            frame_index: 0,
            generation: 0,
            dirty: true,
        })
    }

    /// Syncs the current slot, acquires an image, rebuilding the swapchain if needed.
    ///
    /// Returns `Ok(None)` when there is nothing to render to right now
    /// (minimized, or the surface kept changing); just skip the frame.
    pub fn begin_frame(
        &mut self,
        timeline: &Timeline,
        window_extent: Extent2D,
    ) -> Result<Option<SwapchainFrame>, SwapchainError> {
        let old_extent = self.options.window_extent;
        if old_extent.width != window_extent.width || old_extent.height != window_extent.height {
            self.options.window_extent = window_extent;
            self.dirty = true;
        }

        self.sync_slot(timeline);
        self.tick_retired(timeline);

        for _ in 0..MAX_REBUILD_ATTEMPTS {
            if self.dirty || self.current.is_none() {
                match self.rebuild(timeline) {
                    Ok(()) => {}
                    Err(SwapchainCreateError::WindowMinimized) => return Ok(None),
                    Err(err) => return Err(err.into()),
                }
            }

            let extension = self.device.extensions.swapchain.as_ref().unwrap();
            let state = self.current.as_ref().unwrap();
            let slot = &self.slots[self.frame_index];

            let result = unsafe {
                extension.acquire_next_image(state.handle, u64::MAX, slot.image_available, vk::Fence::null())
            };
            match result {
                Ok((image_index, suboptimal)) => {
                    // Image and semaphore are valid: render this frame, rebuild at the next one.
                    if suboptimal {
                        self.dirty = true;
                    }
                    let i = image_index as usize;
                    return Ok(Some(SwapchainFrame {
                        image: state.images[i],
                        view: state.views[i],
                        extent: state.extent,
                        format: state.format,
                        image_index,
                        frame_index: self.frame_index,
                        wait_semaphore: slot.image_available,
                        signal_semaphore: slot.render_finished,
                        generation: state.generation,
                    }));
                }
                // The semaphore is not signaled in this case, so the slot can be reused as is.
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.dirty = true,
                Err(err) => return Err(err.into()),
            }
        }

        Ok(None)
    }

    /// `finish_value` is the timeline value signaled by the submit that rendered `frame`.
    pub fn present(
        &mut self,
        queue: &CmdQueue,
        frame: SwapchainFrame,
        finish_value: u64,
    ) -> Result<(), SwapchainError> {
        let queue = queue.handle;
        let slot_count = self.slots.len();
        let extension = self.device.extensions.swapchain.as_ref().unwrap();
        let state = self.current.as_mut().expect("present without an active swapchain");
        debug_assert_eq!(frame.generation, state.generation);
        let generation = state.generation;
        let slot = &mut self.slots[frame.frame_index];

        slot.finish_value = finish_value;
        state.last_use_value = state.last_use_value.max(finish_value);

        let waits = [slot.render_finished];
        let swapchains = [state.handle];
        let indices = [frame.image_index];
        let fences = [slot.present_fence];
        let mut fence_info = vk::SwapchainPresentFenceInfoKHR::default().fences(&fences);
        let info = vk::PresentInfoKHR::default()
            .push(&mut fence_info)
            .wait_semaphores(&waits)
            .swapchains(&swapchains)
            .image_indices(&indices);

        let result = unsafe { extension.queue_present(queue, &info) };
        self.frame_index = (frame.frame_index + 1) % slot_count;

        match result {
            Ok(suboptimal) => {
                slot.pending_present = Some(generation);
                self.dirty |= suboptimal;
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.dirty = true;
                self.reset_slot_after_failed_present(queue, frame.frame_index);
                Ok(())
            }
            Err(err) => Err(err.into()),
        }
    }

    pub fn wait_last_blocking(&self, timeline: &Timeline) {
        let last = self
            .slots
            .iter()
            .map(|s| s.finish_value)
            .chain(self.retired.iter().map(|r| r.last_use_value))
            .chain(self.current.iter().map(|s| s.last_use_value))
            .max()
            .unwrap_or(0);
        timeline.wait_blocking(last);
    }

    pub fn destroy(mut self) {
        let device = &*self.device;
        for slot in self.slots.iter_mut() {
            slot.wait_present(device);
        }
        for retired in self.retired.drain(..) {
            retired.destroy(device);
        }
        if let Some(state) = self.current.take() {
            state.into_retired().destroy(device);
        }
        for slot in self.slots.iter() {
            slot.destroy(device);
        }
    }

    fn sync_slot(&mut self, timeline: &Timeline) {
        let slot = &mut self.slots[self.frame_index];
        timeline.wait_blocking(slot.finish_value);
        slot.wait_present(&self.device);
    }

    fn tick_retired(&mut self, timeline: &Timeline) {
        let done = timeline.completed_value();
        let Self {
            retired, slots, device, ..
        } = self;
        retired.retain(|r| {
            let gpu_done = done >= r.last_use_value;
            let present_done = slots.iter().all(|slot| {
                slot.pending_present
                    .map_or(true, |generation| generation > r.generation)
            });
            if gpu_done && present_done {
                r.destroy(device);
                false
            } else {
                true
            }
        });
    }

    fn rebuild(&mut self, timeline: &Timeline) -> Result<(), SwapchainCreateError> {
        // Phase 1: query only. Failures here (minimized, unsupported) leave the old swapchain untouched.
        let params = self.query_params()?;

        // Phase 2: once `vkCreateSwapchainKHR` is called with `oldSwapchain`, the old one is retired
        // regardless of success.
        self.generation += 1;
        let old_handle = self.current.as_ref().map(|state| state.handle);
        let built = self.build(&params, old_handle);
        if let Some(old) = self.current.take() {
            self.retire(old, timeline);
        }
        self.current = Some(built?);
        self.dirty = false;
        Ok(())
    }

    fn retire(&mut self, old: SwapchainState, timeline: &Timeline) {
        if self.retired.is_full() {
            let oldest = self.retired.remove(0);
            timeline.wait_blocking(oldest.last_use_value);
            for slot in self.slots.iter_mut() {
                if slot
                    .pending_present
                    .is_some_and(|generation| generation <= oldest.generation)
                {
                    slot.wait_present(&self.device);
                }
            }
            oldest.destroy(&self.device);
        }
        self.retired.push(old.into_retired());
    }

    fn query_params(&self) -> Result<BuildParams, SwapchainCreateError> {
        let surface = self
            .instance
            .extensions
            .surface
            .as_ref()
            .ok_or(SwapchainCreateError::InstanceSurfaceSupportNotAvailable)?;
        let options = &self.options;

        let mut surface_present_mode = vk::SurfacePresentModeKHR::default().present_mode(options.present_mode.to_vk());
        let surface_info = vk::PhysicalDeviceSurfaceInfo2KHR::default()
            .push(&mut surface_present_mode)
            .surface(surface.handle);
        let mut capabilities2 = vk::SurfaceCapabilities2KHR::default();
        unsafe {
            surface
                .get_surface_capabilities2_extension
                .get_physical_device_surface_capabilities2(
                    self.device.physical_device,
                    &surface_info,
                    &mut capabilities2,
                )?
        };
        let capabilities = capabilities2.surface_capabilities;

        let extent = if capabilities.current_extent.width == u32::MAX || capabilities.current_extent.height == u32::MAX
        {
            vk::Extent2D {
                width: options
                    .window_extent
                    .width
                    .clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                height: options.window_extent.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };
        if extent.width == 0 || extent.height == 0 {
            return Err(SwapchainCreateError::WindowMinimized);
        }

        let desired = vk::SurfaceFormatKHR {
            format: options.format.to_vk(),
            color_space: options.color_space.to_vk(),
        };
        // Note: this query allocates a Vec; it only runs on (re)creation.
        let supported = unsafe {
            surface
                .extension
                .get_physical_device_surface_formats(self.device.physical_device, surface.handle)?
        };
        if !supported.contains(&desired) {
            return Err(SwapchainCreateError::UnsupportedSurfaceFormat(desired));
        }

        let mut image_count = options.image_count.max(capabilities.min_image_count);
        if capabilities.max_image_count != 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }
        image_count = image_count.min(MAX_SWAPCHAIN_IMAGES as u32);

        let composite_alpha = options.alpha_composite.to_vk();
        if !capabilities.supported_composite_alpha.contains(composite_alpha) {
            return Err(SwapchainCreateError::UnsupportedAlphaComposite(options.alpha_composite));
        }

        Ok(BuildParams {
            extent,
            image_count,
            composite_alpha,
            transform: capabilities.current_transform,
        })
    }

    fn build(
        &self,
        params: &BuildParams,
        old_handle: Option<vk::SwapchainKHR>,
    ) -> Result<SwapchainState, SwapchainCreateError> {
        let surface = self
            .instance
            .extensions
            .surface
            .as_ref()
            .ok_or(SwapchainCreateError::InstanceSurfaceSupportNotAvailable)?;
        let extension = self
            .device
            .extensions
            .swapchain
            .as_ref()
            .ok_or(SwapchainCreateError::DeviceSwapchainSupportNotAvailable)?;
        let options = &self.options;
        let format = options.format.to_vk();

        let present_modes = [options.present_mode.to_vk()];
        let mut present_modes_info = vk::SwapchainPresentModesCreateInfoKHR::default().present_modes(&present_modes);
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .push(&mut present_modes_info)
            .surface(surface.handle)
            .min_image_count(params.image_count)
            .image_format(format)
            .image_color_space(options.color_space.to_vk())
            .image_extent(params.extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(params.transform)
            .composite_alpha(params.composite_alpha)
            .present_mode(options.present_mode.to_vk())
            .clipped(true)
            .old_swapchain(old_handle.unwrap_or(vk::SwapchainKHR::null()));

        let handle = unsafe { extension.create_swapchain(&swapchain_info, None) }?;
        debug!(
            "created swapchain {handle:?}[generation-{}], {:?}, image count {}, format {format:?}, color space {:?}, present mode {:?}",
            self.generation, params.extent, params.image_count, options.color_space, options.present_mode
        );
        create_object_debug_name(&self.device, handle, format!("generation-{}", self.generation));

        let mut raw = [vk::Image::null(); MAX_SWAPCHAIN_IMAGES];
        let mut count = MAX_SWAPCHAIN_IMAGES as u32;
        let result = unsafe {
            (extension.fp().get_swapchain_images_khr)(extension.device(), handle, &mut count, raw.as_mut_ptr())
        };
        if result != vk::Result::SUCCESS {
            unsafe { extension.destroy_swapchain(handle, None) };
            return Err(if result == vk::Result::INCOMPLETE {
                SwapchainCreateError::TooManyImages
            } else {
                SwapchainCreateError::Vulkan(result)
            });
        }

        let mut images = ArrayVec::<vk::Image, MAX_SWAPCHAIN_IMAGES>::new();
        let mut views = ArrayVec::<vk::ImageView, MAX_SWAPCHAIN_IMAGES>::new();
        for &image in &raw[..count as usize] {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            let view = unsafe { self.device.handle.create_image_view(&view_info, None) }.check(&self.device);
            images.push(image);
            views.push(view);
        }

        Ok(SwapchainState {
            generation: self.generation,
            handle,
            extent: Extent2D {
                width: params.extent.width,
                height: params.extent.height,
            },
            format: options.format,
            images,
            views,
            last_use_value: 0,
        })
    }

    /// The semaphore / fence state after a failed present is murky; drain the queue and replace them.
    fn reset_slot_after_failed_present(&mut self, queue: vk::Queue, index: usize) {
        unsafe { self.device.handle.queue_wait_idle(queue) }.check(&self.device);
        let slot = &mut self.slots[index];
        let finish_value = slot.finish_value;
        slot.destroy(&self.device);
        *slot = FrameSlot::new(&self.device);
        slot.finish_value = finish_value;
    }
}

impl TargetView {
    pub fn from_frame(frame: &SwapchainFrame) -> Self {
        Self {
            handle: frame.view,
            format: frame.format,
            extent: frame.extent,
            aspect: TextureAspect::Color,
        }
    }
}

impl<'e> CmdEncoder<'e, state::Active> {
    pub fn swapchain_frame_acquire_barrier(&self, frame: &SwapchainFrame) {
        let barrier_infos = [vk::ImageMemoryBarrier2::default()
            .image(frame.image)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            )];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barrier_infos);

        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_pipeline_barrier2(buffer, &dependency_info) };
    }

    pub fn swapchain_frame_present_barrier(&self, frame: &SwapchainFrame) {
        let barrier_infos = [vk::ImageMemoryBarrier2::default()
            .image(frame.image)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            )];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barrier_infos);

        let device = &self.inner.queue.device.handle;
        let buffer = self.inner.buffer;
        unsafe { device.cmd_pipeline_barrier2(buffer, &dependency_info) };
    }
}
