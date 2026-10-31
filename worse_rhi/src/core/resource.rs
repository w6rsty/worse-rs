use ash::vk::{self, TaggedStructure};
use std::{marker::PhantomData, ptr::NonNull, range::Range, sync::Arc};
use thiserror::Error;
use tracing::debug;

use crate::core::{
    device::{Device, DeviceShared},
    types::*,
    utils::{VulkanCheck, align_up, create_object_debug_name, format_bytes, get_mip_extent_2d},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryDomain {
    DeviceLocal,
    HostVisible,
    Staging,
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
            Self::Staging => MemoryProperties {
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

#[derive(Debug, Clone, Copy)]
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
    pub size: usize,
    pub usage: BufferUsageFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Buffer {
    pub(crate) handle: vk::Buffer,
    pub size: usize,
    pub device_address: u64,
    pub ptr: Option<NonNull<u8>>,
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
            .size(desc.size as u64)
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
            ptr: block.mapped_ptr.clone(),
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
            ptr: mapped_ptr,
        })
    }

    pub fn destroy_buffer(&self, buffer: Buffer) {
        unsafe { self.shared.handle.destroy_buffer(buffer.handle, None) };
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferSlice<'a, T: bytemuck::Pod> {
    pub ptr: Option<NonNull<u8>>,
    pub device_address: u64,
    pub len: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T: bytemuck::Pod> BufferSlice<'a, T> {
    pub fn as_ref(&self) -> &T {
        let ptr = self.ptr.expect("not mapped");
        let type_size = std::mem::size_of::<T>();

        let current_bytes = self.len * type_size;
        assert!(
            current_bytes >= type_size,
            "BufferSlice size ({}) is smaller than type T size ({})",
            current_bytes,
            type_size
        );

        unsafe { &*(ptr.as_ptr().cast::<T>()) }
    }

    pub fn as_mut(&mut self) -> &mut T {
        let ptr = self.ptr.expect("not mapped");
        let type_size = std::mem::size_of::<T>();

        let current_bytes = self.len * type_size;
        assert!(
            current_bytes >= type_size,
            "BufferSlice size ({}) is smaller than type T size ({})",
            current_bytes,
            type_size
        );

        unsafe { &mut *(ptr.as_ptr().cast::<T>()) }
    }

    pub fn as_slice(&self) -> &[T] {
        let ptr = self.ptr.expect("not mapped");
        unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<T>(), self.len) }
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let ptr = self.ptr.expect("not mapped");
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().cast::<T>(), self.len) }
    }
}

impl Buffer {
    pub fn as_ref<T: bytemuck::Pod>(&self) -> &T {
        let ptr = self.ptr.expect("not mapped");
        let type_size = std::mem::size_of::<T>();
        let type_align = std::mem::align_of::<T>();

        assert!(
            self.size >= type_size,
            "Buffer size ({}) is smaller than type T size ({})",
            self.size,
            type_size
        );

        let base_addr = ptr.as_ptr() as usize;
        assert!(
            base_addr % type_align == 0,
            "Memory address {:#x} is not aligned for type T (required align: {})",
            base_addr,
            type_align
        );

        unsafe { &*(ptr.as_ptr().cast::<T>()) }
    }

    pub fn as_mut<T: bytemuck::Pod>(&mut self) -> &mut T {
        let ptr = self.ptr.expect("not mapped");
        let type_size = std::mem::size_of::<T>();
        let type_align = std::mem::align_of::<T>();

        assert!(
            self.size >= type_size,
            "Buffer size ({}) is smaller than type T size ({})",
            self.size,
            type_size
        );

        let base_addr = ptr.as_ptr() as usize;
        assert!(
            base_addr % type_align == 0,
            "Memory address {:#x} is not aligned for type T (required align: {})",
            base_addr,
            type_align
        );

        unsafe { &mut *(ptr.as_ptr().cast::<T>()) }
    }

    pub fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        let ptr = self.ptr.expect("not mapped");
        let count = self.size as usize / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<T>(), count) }
    }

    pub fn as_slice_mut<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        let ptr = self.ptr.expect("not mapped");
        let count = self.size as usize / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().cast::<T>(), count) }
    }

    pub fn slice<T: bytemuck::Pod>(&self, byte_offset: usize, elem_len: usize) -> BufferSlice<'_, T> {
        let elem_size = std::mem::size_of::<T>();
        let elem_alignment = std::mem::align_of::<T>();

        let required_bytes = byte_offset + (elem_size * elem_len);
        assert!(
            required_bytes <= self.size,
            "BufferSlice out of bounds: buffer size {}, required {}",
            self.size,
            required_bytes
        );

        let ptr = self.ptr.and_then(|ptr| {
            let base_addr = ptr.as_ptr() as usize;
            let offset_addr = base_addr + byte_offset;
            assert!(
                offset_addr % elem_alignment == 0,
                "Memory address is not aligned for type T (address: {:#x}, required align: {})",
                offset_addr,
                elem_alignment
            );
            Some(unsafe { ptr.byte_add(byte_offset) })
        });
        let device_address = self.device_address + byte_offset as u64;

        BufferSlice {
            ptr,
            device_address,
            len: elem_len,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TextureError {
    #[error("memory block error: {0}")]
    MemoryBlock(#[from] MemoryBlockError),
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
    pub flags: TextureCreateFlags,
}

impl TextureDesc {
    pub fn default() -> Self {
        Self {
            dimension: TextureDimension::Dimension2D,
            format: PixelFormat::Undefined,
            extent: Extent3D::default(),
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Count1,
            usage: TextureUsageFlags::empty(),
            flags: TextureCreateFlags::empty(),
        }
    }

    pub fn new_default_2d(width: u32, height: u32) -> Self {
        Self::default()
            .dimension(TextureDimension::Dimension2D)
            .extent(Extent3D {
                width,
                height,
                depth: 1,
            })
    }

    pub fn new_default_depth(width: u32, height: u32) -> Self {
        Self::new_default_2d(width, height)
            .format(PixelFormat::D32Float)
            .usage(TextureUsageFlags::DEPTH_STENCIL_ATTACHMENT)
    }

    pub fn dimension(mut self, dimension: TextureDimension) -> Self {
        self.dimension = dimension;
        self
    }
    pub fn format(mut self, format: PixelFormat) -> Self {
        self.format = format;
        self
    }
    pub fn extent(mut self, extent: Extent3D) -> Self {
        self.extent = extent;
        self
    }
    pub fn mip_levels(mut self, mip_levels: u32) -> Self {
        self.mip_levels = mip_levels;
        self
    }
    pub fn array_layers(mut self, array_layers: u32) -> Self {
        self.array_layers = array_layers;
        self
    }
    pub fn samples(mut self, samples: SampleCount) -> Self {
        self.samples = samples;
        self
    }
    pub fn usage(mut self, usage: TextureUsageFlags) -> Self {
        self.usage = usage;
        self
    }
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
        let sharing_mode = if indices.len() > 1 && desc.flags.contains(TextureCreateFlags::SHARED) {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let mut texture_flag = vk::ImageCreateFlags::empty();
        if desc.flags.contains(TextureCreateFlags::ALIAS) {
            texture_flag |= vk::ImageCreateFlags::ALIAS;
        }
        if desc.flags.contains(TextureCreateFlags::SPARSE) {
            texture_flag |= vk::ImageCreateFlags::SPARSE_BINDING | vk::ImageCreateFlags::SPARSE_RESIDENCY
        }
        if desc
            .flags
            .contains(TextureCreateFlags::ALIAS | TextureCreateFlags::SPARSE)
        {
            texture_flag |= vk::ImageCreateFlags::SPARSE_ALIASED;
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

    pub fn create_texture_dedicated(
        &self,
        desc: TextureDesc,
        name: impl AsRef<str>,
    ) -> Result<(Texture, MemoryBlock), TextureError> {
        let (texture, memory_requirements) = self.create_texture_internal(desc, &name);
        let allocation_size = memory_requirements.size;
        let block = self.create_memory_block(
            MemoryDomain::DeviceLocal,
            allocation_size,
            memory_requirements.memory_type_bits,
            &name,
        )?;

        let device = &self.shared;
        unsafe { device.handle.bind_image_memory(texture, block.memory, 0) }.check(device);

        Ok((
            Texture {
                handle: texture,
                dimension: desc.dimension,
                format: desc.format,
                extent: desc.extent,
                mip_levels: desc.mip_levels,
                array_layers: desc.array_layers,
                samples: desc.samples,
                usage: desc.usage,
            },
            block,
        ))
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
    pub capacity: usize,
    pub kind: DescriptorHeapKind,
    buffer: Buffer,
    memory: MemoryBlock,
    payload_offset: usize,
}

impl Device {
    pub fn create_descriptor_heap(
        &self,
        capacity: usize,
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
                capacity as u64 * heap_properties.image_descriptor_size,
            )
        } else {
            (
                heap_properties.sampler_descriptor_alignment,
                heap_properties.sampler_heap_alignment,
                heap_properties.min_sampler_heap_reserved_range,
                capacity as u64 * heap_properties.sampler_descriptor_size,
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
                size: allocation_size as usize,
                usage: BufferUsageFlags::SHADER_DEVICE_ADDRESS | BufferUsageFlags::DESCRIPTOR_HEAP,
            },
            &name,
        )?;

        let payload_base = align_up(buffer.device_address, allocation_alignment);
        let payload_offset = (payload_base - buffer.device_address) as usize;

        Ok(DescriptorHeap {
            device: self.shared.clone(),
            capacity,
            kind,
            buffer,
            memory,
            payload_offset,
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
            let ptr = self.buffer.ptr.unwrap().add(index * descriptor_size).as_ptr();
            core::slice::from_raw_parts_mut(ptr, descriptor_size)
        };
        let descriptors = [vk::HostAddressRangeEXT::default().address(descriptor_range)];

        unsafe { fp.write_resource_descriptors(&resources, &descriptors) }.check(device);
    }
}
