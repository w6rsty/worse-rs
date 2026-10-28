use ash::vk::{self, TaggedStructure};
use std::{ptr::NonNull, range::Range, sync::Arc};
use thiserror::Error;
use tracing::debug;

use crate::core::{
    device::{Device, DeviceShared},
    types::*,
    utils::{VulkanCheck, align_up, create_object_debug_name, format_bytes, get_mip_extent_2d},
};

pub struct MemoryRange {
    pub mapped_base: NonNull<u8>,
    pub device_base: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryDomain {
    DeviceLocal,
    HostVisible,
    Readback,
}

/// vk::MemoryPropertyFlags combination
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryProperties {
    pub(crate) required: vk::MemoryPropertyFlags,
    pub(crate) preferred: vk::MemoryPropertyFlags,
    pub(crate) avoided: vk::MemoryPropertyFlags,
}

impl MemoryDomain {
    pub(crate) fn to_vk_memory_property(self) -> MemoryProperties {
        match self {
            Self::DeviceLocal => MemoryProperties {
                required: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                preferred: vk::MemoryPropertyFlags::empty(),
                avoided: vk::MemoryPropertyFlags::HOST_VISIBLE,
            },
            Self::HostVisible => MemoryProperties {
                required: vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                preferred: vk::MemoryPropertyFlags::empty(),
                avoided: vk::MemoryPropertyFlags::empty(),
            },
            Self::Readback => MemoryProperties {
                required: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_CACHED
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                preferred: vk::MemoryPropertyFlags::empty(),
                avoided: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceMemoryRequirements {
    pub device_size: u64,
    pub device_alignment: u64,
    pub memory_type_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MemoryBlockError {
    #[error("no capable memory type")]
    NoCapableMemoryType,
    #[error("the mapping call succeeded but returned a null pointer")]
    MapMemoryReturnedNullPointer,
}

#[derive(Debug)]
pub struct MemoryBlock {
    pub(crate) memory: vk::DeviceMemory,
    pub size: u64,
    pub memory_type_index: u32,
    pub mapped_ptr: Option<NonNull<u8>>,
}

impl Device {
    pub(crate) fn find_memory_type(
        &self,
        minimum_heap_size: u64,
        memory_type_bits: u32,
        properties: MemoryProperties,
    ) -> Result<u32, MemoryBlockError> {
        let memory_properties = &self.shared.memory_properties;
        // (type index, avoided flags matched, preferred flags matched)
        let mut best: Option<(u32, u32, u32)> = None;

        for index in 0..memory_properties.memory_type_count {
            if memory_type_bits & (1u32 << index) == 0 {
                continue;
            }

            let memory_type = &memory_properties.memory_types[index as usize];
            let flags = memory_type.property_flags;

            if !flags.contains(properties.required) {
                continue;
            }

            let heap = &memory_properties.memory_heaps[memory_type.heap_index as usize];
            if heap.size < minimum_heap_size {
                continue;
            }

            let avoided_count = (flags & properties.avoided).as_raw().count_ones();
            let preferred_count = (flags & properties.preferred).as_raw().count_ones();

            let is_better = match best {
                None => true,
                Some((_, best_avoided, best_preferred)) => {
                    avoided_count < best_avoided || (avoided_count == best_avoided && preferred_count > best_preferred)
                }
            };

            if is_better {
                best = Some((index, avoided_count, preferred_count));
            }
        }

        match best.map(|(index, _, _)| index) {
            Some(index) => Ok(index),
            None => Err(MemoryBlockError::NoCapableMemoryType),
        }
    }

    pub fn create_memory_block(
        &self,
        domain: MemoryDomain,
        size: u64,
        memory_type_bits: u32,
        name: impl AsRef<str>,
    ) -> Result<MemoryBlock, MemoryBlockError> {
        let device = &self.shared;

        let requirements = domain.to_vk_memory_property();
        let memory_type_index = self.find_memory_type(size, memory_type_bits, requirements)?;

        let mut allocate_flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .push(&mut allocate_flags)
            .allocation_size(size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { device.handle.allocate_memory(&allocate_info, None) }.check(device);
        debug!(
            "allocate memory {memory:?}[{}], size {}, domain {domain:?}, memory_type {memory_type_bits:b}",
            name.as_ref(),
            format_bytes(size),
        );
        create_object_debug_name(device, memory, name);

        let mapped_ptr = if requirements.required.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            let raw = unsafe {
                device
                    .handle
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            }
            .check(device);
            Some(NonNull::new(raw.cast::<u8>()).ok_or(MemoryBlockError::MapMemoryReturnedNullPointer)?)
        } else {
            None
        };

        Ok(MemoryBlock {
            memory,
            size,
            memory_type_index,
            mapped_ptr,
        })
    }

    pub fn destroy_memory_block(&self, block: MemoryBlock) {
        unsafe { self.shared.handle.free_memory(block.memory, None) };
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BufferError {
    #[error("memory block error: {0}")]
    MemoryBlock(#[from] MemoryBlockError),
    #[error("target buffer has no device address")]
    NoDeviceAddress,
    #[error("placement of buffer is out of bound")]
    PlacementOutOfBound,
    #[error("access of buffer is out of bound")]
    AccessOutOfBound,
    #[error("incapable memory")]
    IncapableMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsageFlags,
}

pub struct Buffer {
    pub(crate) handle: vk::Buffer,
    pub size: u64,
    pub device_address: u64,
    pub mapped_ptr: Option<NonNull<u8>>,
}

impl Device {
    fn convert_buffer_desc<'a>(&'a self, desc: &BufferDesc) -> vk::BufferCreateInfo<'a> {
        let indices = &self.shared.queue_indices.flat;
        let sharing_mode = if indices.len() > 1 && desc.usage.contains(BufferUsageFlags::SHARED) {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        vk::BufferCreateInfo::default()
            .size(desc.size)
            .usage(desc.usage.to_vk())
            .sharing_mode(sharing_mode)
            .queue_family_indices(&indices)
    }

    pub fn query_buffer_memory_requirements(&self, desc: &BufferDesc) -> ResourceMemoryRequirements {
        let buffer_info = self.convert_buffer_desc(&desc);
        let buffer_memory_info = vk::DeviceBufferMemoryRequirements::default().create_info(&buffer_info);
        let mut requirements_info = vk::MemoryRequirements2::default();
        unsafe {
            self.shared
                .handle
                .get_device_buffer_memory_requirements(&buffer_memory_info, &mut requirements_info)
        };
        let requirements = requirements_info.memory_requirements;
        ResourceMemoryRequirements {
            device_size: requirements.size,
            device_alignment: requirements.alignment,
            memory_type_bits: requirements.memory_type_bits,
        }
    }

    fn create_buffer_internal(&self, desc: BufferDesc, name: impl AsRef<str>) -> (vk::Buffer, vk::MemoryRequirements) {
        let device = &self.shared;
        let buffer_info = self.convert_buffer_desc(&desc);
        let buffer = unsafe { device.handle.create_buffer(&buffer_info, None) }.check(device);
        debug!("create buffer {buffer:?}[{}], {desc:?}", name.as_ref());
        create_object_debug_name(device, buffer, name);
        let requirements = unsafe { device.handle.get_buffer_memory_requirements(buffer) };
        (buffer, requirements)
    }

    pub fn create_buffer_dedicated(
        &self,
        domain: MemoryDomain,
        desc: BufferDesc,
        name: impl AsRef<str>,
    ) -> Result<(Buffer, MemoryBlock), BufferError> {
        let (buffer, memory_requirements) = self.create_buffer_internal(desc, &name);
        let allocation_size = memory_requirements.size;
        let block = self.create_memory_block(domain, allocation_size, memory_requirements.memory_type_bits, &name)?;

        let device = &self.shared;
        unsafe { device.handle.bind_buffer_memory(buffer, block.memory, 0) }.check(device);

        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe { device.handle.get_buffer_device_address(&address_info) };

        let buffer = Buffer {
            handle: buffer,
            size: desc.size,
            device_address,
            mapped_ptr: block.mapped_ptr.clone(),
        };

        Ok((buffer, block))
    }

    pub fn create_buffer_placed(
        &self,
        block: &MemoryBlock,
        offset: u64,
        desc: BufferDesc,
        name: impl AsRef<str>,
    ) -> Result<Buffer, BufferError> {
        let (buffer, memory_requirements) = self.create_buffer_internal(desc, name);

        if memory_requirements.memory_type_bits & (1u32 << block.memory_type_index) == 0 {
            return Err(BufferError::IncapableMemory);
        }

        let placed_size = memory_requirements.size;
        let placed_offset = align_up(offset, memory_requirements.alignment);

        if placed_offset + placed_size > block.size {
            return Err(BufferError::PlacementOutOfBound);
        }

        let device = &self.shared;
        unsafe { device.handle.bind_buffer_memory(buffer, block.memory, placed_offset) }.check(device);

        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe { device.handle.get_buffer_device_address(&address_info) };

        let mapped_ptr = block
            .mapped_ptr
            .and_then(|block_base| Some(unsafe { block_base.byte_add(usize::try_from(placed_offset).unwrap()) }));

        Ok(Buffer {
            handle: buffer,
            size: desc.size,
            device_address,
            mapped_ptr,
        })
    }

    pub fn destroy_buffer(&self, buffer: Buffer) {
        unsafe { self.shared.handle.destroy_buffer(buffer.handle, None) };
    }
}

impl Buffer {
    pub fn address_at(&self, offset: u64) -> u64 {
        self.device_address + offset
    }

    // /// SAFTY: Access of offset pointer can not exceed bound
    // unsafe fn ptr_at(&self, offset: u64) -> Result<NonNull<u8>, BufferError> {
    //     if offset > self.size {
    //         return Err(BufferError::AccessOutOfBound);
    //     }
    //     let base = self.mapped_ptr.ok_or(BufferError::NoDeviceAddress)?;
    //     Ok(unsafe { base.byte_add(usize::try_from(offset).unwrap()) })
    // }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TextureError {
    #[error("placement of buffer is out of bound")]
    PlacementOutOfBound,
    #[error("access of buffer is out of bound")]
    AccessOutOfBound,
    #[error("incapable memory")]
    IncapableMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    pub dimension: TextureDimension,
    pub format: PixelFormat,
    pub extent: Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: SampleCount,
    pub usage: TextureUsageFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Texture {
    pub(crate) handle: vk::Image,
    pub dimension: TextureDimension,
    pub format: PixelFormat,
    pub extent: Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: SampleCount,
    pub usage: TextureUsageFlags,
}

impl Device {
    fn convert_texture_desc<'a>(&'a self, desc: &TextureDesc) -> vk::ImageCreateInfo<'a> {
        let indices = &self.shared.queue_indices.flat;
        let sharing_mode = if indices.len() > 1 && desc.usage.contains(TextureUsageFlags::SHARED) {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let mut texture_flag = vk::ImageCreateFlags::empty();
        if desc.usage.contains(TextureUsageFlags::ALIAS) {
            texture_flag |= vk::ImageCreateFlags::ALIAS;
        }

        vk::ImageCreateInfo::default()
            .image_type(desc.dimension.to_vk_image_type())
            .format(desc.format.to_vk())
            .extent(vk::Extent3D {
                width: desc.extent.width,
                height: desc.extent.height,
                depth: desc.extent.depth,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(desc.array_layers)
            .samples(desc.samples.to_vk())
            .usage(desc.usage.to_vk())
            .sharing_mode(sharing_mode)
            .queue_family_indices(&indices)
            .flags(texture_flag)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    }

    pub fn query_texture_memory_requirements(&self, desc: &TextureDesc) -> ResourceMemoryRequirements {
        let texture_info = self.convert_texture_desc(desc);
        let texture_memory_info = vk::DeviceImageMemoryRequirements::default().create_info(&texture_info);
        let mut requirements_info = vk::MemoryRequirements2::default();
        unsafe {
            self.shared
                .handle
                .get_device_image_memory_requirements(&texture_memory_info, &mut requirements_info);
        };
        let requirements = requirements_info.memory_requirements;
        ResourceMemoryRequirements {
            device_size: requirements.size,
            device_alignment: requirements.alignment,
            memory_type_bits: requirements.memory_type_bits,
        }
    }

    fn create_texture_internal(&self, desc: TextureDesc, name: impl AsRef<str>) -> (vk::Image, vk::MemoryRequirements) {
        let device = &self.shared;
        let texture_info = self.convert_texture_desc(&desc);
        let texture = unsafe { device.handle.create_image(&texture_info, None) }.check(device);
        debug!("create texture {texture:?}[{}], {desc:?}", name.as_ref());
        create_object_debug_name(device, texture, name);
        let requirements = unsafe { device.handle.get_image_memory_requirements(texture) };
        (texture, requirements)
    }

    pub fn create_texture_placed(
        &self,
        block: &MemoryBlock,
        offset: u64,
        desc: TextureDesc,
        name: impl AsRef<str>,
    ) -> Result<Texture, TextureError> {
        let (texture, memory_requirements) = self.create_texture_internal(desc, name);

        if memory_requirements.memory_type_bits & (1u32 << block.memory_type_index) == 0 {
            return Err(TextureError::IncapableMemory);
        }

        let placed_size = memory_requirements.size;
        let placed_offset = align_up(offset, memory_requirements.alignment);

        if placed_offset + placed_size > block.size {
            return Err(TextureError::PlacementOutOfBound);
        }

        let device = &self.shared;
        unsafe { device.handle.bind_image_memory(texture, block.memory, offset) }.check(device);

        Ok(Texture {
            handle: texture,
            dimension: desc.dimension,
            format: desc.format,
            extent: desc.extent,
            mip_levels: desc.mip_levels,
            array_layers: desc.array_layers,
            samples: desc.samples,
            usage: desc.usage,
        })
    }

    pub fn destory_texture(&self, texture: Texture) {
        unsafe { self.shared.handle.destroy_image(texture.handle, None) };
    }
}

#[derive(Debug)]
pub struct TextureViewDesc {
    pub(crate) texture: vk::Image,
    pub format: PixelFormat,
    pub dimension: TextureDimension,
    pub mip_range: Range<u32>,
    pub layer_range: Range<u32>,
    pub aspect: TextureAspect,
}

impl TextureViewDesc {
    pub fn new(texture: &Texture, mip_range: Range<u32>, layer_range: Range<u32>, aspect: TextureAspect) -> Self {
        Self {
            texture: texture.handle,
            format: texture.format,
            dimension: texture.dimension,
            mip_range,
            layer_range,
            aspect,
        }
    }

    pub fn new_default_range(texture: &Texture, aspect: TextureAspect) -> Self {
        Self::new(texture, Range { start: 0, end: 1 }, Range { start: 0, end: 1 }, aspect)
    }
}

fn convert_texture_view_desc<'desc>(desc: &'desc TextureViewDesc) -> vk::ImageViewCreateInfo<'desc> {
    vk::ImageViewCreateInfo::default()
        .image(desc.texture)
        .view_type(desc.dimension.to_vk_image_view_type())
        .format(desc.format.to_vk())
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(desc.aspect.to_vk())
                .base_mip_level(desc.mip_range.start)
                .level_count(desc.mip_range.end - desc.mip_range.start)
                .base_array_layer(desc.layer_range.start)
                .layer_count(desc.layer_range.end - desc.layer_range.start),
        )
}

#[derive(Debug, Clone, Copy)]
pub struct TargetView {
    pub(crate) handle: vk::ImageView,
    pub format: PixelFormat,
    pub extent: Extent2D,
    pub aspect: TextureAspect,
}

impl Device {
    pub fn create_target_view(&self, texture: &Texture, view_desc: TextureViewDesc) -> TargetView {
        let device = &self.shared;
        let view_info = convert_texture_view_desc(&view_desc);
        let view = unsafe { device.handle.create_image_view(&view_info, None) }.check(device);
        let extent = get_mip_extent_2d(
            Extent2D {
                width: texture.extent.width,
                height: texture.extent.height,
            },
            view_desc.mip_range.start,
        );
        TargetView {
            handle: view,
            format: texture.format,
            extent,
            aspect: view_desc.aspect,
        }
    }

    pub fn destory_target_view(&self, view: TargetView) {
        unsafe { self.shared.handle.destroy_image_view(view.handle, None) };
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DescriptorHeapError {
    #[error("buffer creation error: {0}")]
    BufferCreationFailed(#[from] BufferError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorHeapKind {
    Texture,
    Sampler,
}

pub struct DescriptorHeap {
    device: Arc<DeviceShared>,
    pub capacity: u64,
    pub kind: DescriptorHeapKind,
    buffer: Buffer,
    memory: MemoryBlock,
    range: MemoryRange,
}

impl Device {
    pub fn create_descriptor_heap(
        &self,
        capacity: u64,
        kind: DescriptorHeapKind,
        name: impl AsRef<str>,
    ) -> Result<DescriptorHeap, DescriptorHeapError> {
        let heap_properties = &self.shared.properties.descriptor_heap;
        let is_resource_heap = kind == DescriptorHeapKind::Texture;

        let (reserved_aligment, heap_alignment, reserved_size, descriptor_size) = if is_resource_heap {
            (
                heap_properties.image_descriptor_alignment,
                heap_properties.resource_heap_alignment,
                heap_properties.min_resource_heap_reserved_range,
                capacity * heap_properties.image_descriptor_size,
            )
        } else {
            (
                heap_properties.sampler_descriptor_alignment,
                heap_properties.sampler_heap_alignment,
                heap_properties.min_sampler_heap_reserved_range,
                capacity * heap_properties.sampler_descriptor_size,
            )
        };
        let reserved_offset = align_up(descriptor_size, reserved_aligment);
        let payload_size = reserved_offset + reserved_size;
        let allocation_alignment = heap_alignment.max(16);
        let allocation_padding = allocation_alignment - 1;
        let allocation_size = payload_size + allocation_padding;

        let (buffer, memory) = self.create_buffer_dedicated(
            MemoryDomain::HostVisible,
            BufferDesc {
                size: allocation_size,
                usage: BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::DESCRIPTOR_HEAP,
            },
            &name,
        )?;

        let payload_base = align_up(buffer.device_address, allocation_alignment);
        let payload_offset = payload_base - buffer.device_address;

        let mapped_base = unsafe {
            memory
                .mapped_ptr
                .unwrap()
                .byte_add(usize::try_from(payload_offset).unwrap())
        };

        Ok(DescriptorHeap {
            device: self.shared.clone(),
            capacity,
            kind,
            buffer,
            memory,
            range: MemoryRange {
                mapped_base,
                device_base: payload_base,
                size: payload_size,
            },
        })
    }

    pub fn destroy_descriptor_heap(&self, heap: DescriptorHeap) {
        self.destroy_buffer(heap.buffer);
        self.destroy_memory_block(heap.memory);
    }
}

impl DescriptorHeap {
    pub fn write_texture_descriptor(&self, index: usize, view_desc: TextureViewDesc, kind: TextureDescriptorKind) {
        assert_eq!(self.kind, DescriptorHeapKind::Texture);

        let device = &self.device;
        let fp = &device.extensions.descriptor_heap;
        let descriptor_size = device.properties.descriptor_heap.image_descriptor_size as usize;

        let mut view_usage = vk::ImageViewUsageCreateInfo::default().usage(kind.to_vk_usage());
        let mut view_info = convert_texture_view_desc(&view_desc);
        view_info = view_info.push(&mut view_usage);
        let descriptor_info = [vk::ImageDescriptorInfoEXT::default()
            .view(&view_info)
            .layout(vk::ImageLayout::GENERAL)];

        let mut resourece_descriptor_info = vk::ResourceDescriptorDataEXT::default();
        resourece_descriptor_info.p_image = descriptor_info.as_ptr();
        let resources = [vk::ResourceDescriptorInfoEXT::default()
            .ty(kind.to_vk_descriptor_type())
            .data(resourece_descriptor_info)];

        let descriptor_range = unsafe {
            let shifted = self.range.mapped_base.as_ptr().add(index * descriptor_size);
            core::slice::from_raw_parts_mut(shifted, descriptor_size)
        };
        let descriptors = [vk::HostAddressRangeEXT::default().address(descriptor_range)];

        unsafe { fp.write_resource_descriptors(&resources, &descriptors) }.check(device);
    }
}

pub trait RootArgument {}

pub struct NullRoot {}
impl RootArgument for NullRoot {}

pub const NULL_ROOT: NullRoot = NullRoot {};
