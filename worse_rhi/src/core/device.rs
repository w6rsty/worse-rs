use arrayvec::ArrayVec;
use ash::{
    ext, khr,
    vk::{self, TaggedStructure},
};
use std::sync::Arc;
use thiserror::Error;
use tracing::debug;

use crate::core::{
    command::{CmdQueue, QueueKind},
    instance::{Instance, InstanceShared, Surface},
    utils::{VulkanCheck, check_extension, parse_driver_version, parse_name, parse_uuid},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCreateOptions {
    /// Request dedicated compute queue
    pub dedicated_compute_queue: bool,
    /// Request dedictted transfer queue
    pub dedicated_transfer_queue: bool,
}

impl Default for DeviceCreateOptions {
    fn default() -> Self {
        Self {
            dedicated_compute_queue: false,
            dedicated_transfer_queue: false,
        }
    }
}

pub struct Device {
    pub(crate) instance: Arc<InstanceShared>,
    pub(crate) shared: Arc<DeviceShared>,
    pub(crate) queues: DeviceQueues,
}

impl Device {
    pub fn new(instance: &Instance, options: DeviceCreateOptions) -> Result<Self, DeviceCreateError> {
        let shared = Arc::new(DeviceShared::new(instance, &options)?);
        let queues = DeviceQueues::new(&shared, &options);

        debug!(
            device_name = parse_name(shared.properties.core.device_name_as_c_str()),
            driver_version =
                parse_driver_version(shared.properties.core.vendor_id, shared.properties.core.driver_version),
            pipeline_cache_uuid = parse_uuid(&shared.properties.core.pipeline_cache_uuid),
        );

        Ok(Self {
            instance: instance.shared.clone(),
            shared,
            queues,
        })
    }

    /// Device level wait idle (blocking)
    pub fn wait_idle_blocking(&self) {
        let device = &self.shared;
        unsafe { device.handle.device_wait_idle() }.check(device);
    }

    pub fn main_queue(&self) -> &Arc<CmdQueue> {
        &self.queues.main
    }

    pub fn dedicated_compute_queue(&self) -> Option<&Arc<CmdQueue>> {
        self.queues.dedicated_compute.as_ref()
    }

    pub fn dedciated_transfer_queue(&self) -> Option<&Arc<CmdQueue>> {
        self.queues.dedicated_transfer.as_ref()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.shared.handle.destroy_device(None) };
    }
}

pub(crate) struct DeviceShared {
    pub(crate) handle: ash::Device,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) properties: DeviceProperties,
    pub(crate) features: DeviceFeatures,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,

    pub(crate) queue_indices: QueueFamilyIndices,
    pub(crate) extensions: DeviceExtensions,
}

pub(crate) struct DeviceExtensions {
    pub(crate) debug_utils: Option<ext::debug_utils::Device>,
    pub(crate) swapchain: Option<khr::swapchain::Device>,
    pub(crate) device_fault: ext::device_fault::Device,
    pub(crate) descriptor_heap: ext::descriptor_heap::Device,
    pub(crate) device_address_commands: khr::device_address_commands::Device,
    pub(crate) mesh_shader: ext::mesh_shader::Device,
}

impl DeviceShared {
    pub fn new(instance: &Instance, options: &DeviceCreateOptions) -> Result<Self, DeviceCreateError> {
        let instance_extensions = &instance.shared.extensions;
        let ash_instance = &instance.shared.handle;

        let selected: CandidateDetails = unsafe { instance.shared.handle.enumerate_physical_devices()? }
            .iter()
            .find_map(
                |candidate| match Self::inspect_candidate(*candidate, &instance, &options) {
                    (candidate, Ok(())) => Some(candidate),
                    (candiate, error) => {
                        let device_name = candiate
                            .properties
                            .core
                            .device_name_as_c_str()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_else(|_| "unknown device".to_owned());
                        debug!("{device_name} is not capable: {error:?}");
                        None
                    }
                },
            )
            .ok_or(DeviceCreateError::NoCapableDevice)?;

        let physical_device = selected.physical_device;

        let enabled_presentation = instance_extensions.surface.is_some();
        let unified_image_layouts_available = selected.features.unified_image_layouts.unified_image_layouts == vk::TRUE;

        let mut enabled_features = DeviceFeatures::default();
        enabled_features.core = enabled_features
            .core
            .image_cube_array(selected.features.core.image_cube_array == vk::TRUE)
            .sampler_anisotropy(true)
            .shader_int16(true)
            .depth_bias_clamp(true)
            .independent_blend(true)
            .texture_compression_bc(selected.features.core.texture_compression_bc == vk::TRUE)
            .texture_compression_astc_ldr(selected.features.core.texture_compression_astc_ldr == vk::TRUE)
            .texture_compression_etc2(selected.features.core.texture_compression_etc2 == vk::TRUE)
            .fragment_stores_and_atomics(true)
            .vertex_pipeline_stores_and_atomics(true)
            .shader_storage_image_read_without_format(true)
            .shader_storage_image_write_without_format(true)
            .multi_draw_indirect(true)
            .draw_indirect_first_instance(true);
        enabled_features.vulkan11 = enabled_features
            .vulkan11
            .storage_buffer16_bit_access(true)
            .storage_push_constant16(true)
            .storage_input_output16(selected.features.vulkan11.storage_input_output16 == vk::TRUE)
            .shader_draw_parameters(true);
        enabled_features.vulkan12 = enabled_features
            .vulkan12
            .shader_float16(true)
            .scalar_block_layout(true)
            .timeline_semaphore(true)
            .buffer_device_address(true);
        enabled_features.vulkan13 = enabled_features
            .vulkan13
            .synchronization2(true)
            .dynamic_rendering(true)
            .maintenance4(true);
        enabled_features.vulkan14 = enabled_features.vulkan14.maintenance5(true);
        enabled_features.descriptor_heap = enabled_features.descriptor_heap.descriptor_heap(true);
        enabled_features.device_address_commands =
            enabled_features.device_address_commands.device_address_commands(true);
        enabled_features.shader_untyped_pointers =
            enabled_features.shader_untyped_pointers.shader_untyped_pointers(true);
        enabled_features.mesh_shader = enabled_features.mesh_shader.mesh_shader(true);
        enabled_features.unified_image_layouts = enabled_features
            .unified_image_layouts
            .unified_image_layouts(unified_image_layouts_available);
        enabled_features.swapchain_maintenance1 = enabled_features.swapchain_maintenance1.swapchain_maintenance1(true);

        let mut linked_features2 =
            enabled_features.linked_feature2(enabled_presentation, unified_image_layouts_available);

        let mut enabled_extensions = vec![
            ext::descriptor_heap::NAME.as_ptr(),
            khr::device_address_commands::NAME.as_ptr(),
            khr::shader_untyped_pointers::NAME.as_ptr(),
            ext::mesh_shader::NAME.as_ptr(),
            ext::device_fault::NAME.as_ptr(),
        ];
        if unified_image_layouts_available {
            enabled_extensions.push(khr::unified_image_layouts::NAME.as_ptr());
        }
        if enabled_presentation {
            enabled_extensions.push(khr::swapchain::NAME.as_ptr());
            enabled_extensions.push(
                if instance_extensions.khr_surface_maintentance_1_available {
                    khr::swapchain_maintenance1::NAME
                } else {
                    ext::swapchain_maintenance1::NAME
                }
                .as_ptr(),
            );
        }

        let queue_priorites = [1.0f32];
        let mut queue_infos = ArrayVec::<_, 3>::new();
        queue_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(selected.queue_indices.main)
                .queue_priorities(&queue_priorites),
        );
        let mut flat_indices = ArrayVec::<_, 3>::new();
        flat_indices.push(selected.queue_indices.main);

        if let Some(index) = selected.queue_indices.compute {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(index)
                    .queue_priorities(&queue_priorites),
            );
            flat_indices.push(index);
        }
        if let Some(index) = selected.queue_indices.transfer {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(index)
                    .queue_priorities(&queue_priorites),
            );
            flat_indices.push(index);
        }

        let device_info = unsafe { vk::DeviceCreateInfo::default().extend(&mut linked_features2) }
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_extensions);

        let device = unsafe { ash_instance.create_device(physical_device, &device_info, None)? };
        if cfg!(debug_assertions) {
            let ext_names: Vec<String> = enabled_extensions
                .iter()
                .map(|&ptr| unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
                .collect();

            debug!(
                "created device {:?}\nenabled extensions ({}) {:#?}",
                device.handle(),
                ext_names.len(),
                ext_names,
            );
        } else {
            debug!(
                "created device {:?}, extensions({})",
                device.handle(),
                enabled_extensions.len(),
            );
        }

        let extensions = DeviceExtensions {
            debug_utils: if instance_extensions.debug_utils.is_some() {
                Some(ext::debug_utils::Device::load(ash_instance, &device))
            } else {
                None
            },
            swapchain: if enabled_presentation {
                Some(khr::swapchain::Device::load(ash_instance, &device))
            } else {
                None
            },
            device_fault: ext::device_fault::Device::load(ash_instance, &device),
            descriptor_heap: ext::descriptor_heap::Device::load(ash_instance, &device),
            device_address_commands: khr::device_address_commands::Device::load(ash_instance, &device),
            mesh_shader: ext::mesh_shader::Device::load(ash_instance, &device),
        };

        Ok(Self {
            handle: device,
            physical_device,
            properties: selected.properties,
            features: selected.features,
            memory_properties: selected.memory_properties,
            queue_indices: selected.queue_indices,
            extensions,
        })
    }

    fn inspect_candidate<'surface>(
        physical_device: vk::PhysicalDevice,
        instance: &Instance,
        options: &DeviceCreateOptions,
    ) -> (CandidateDetails, Result<(), InspectDeviceError>) {
        let has_surface = instance.shared.extensions.surface.as_ref().is_some();
        let ash_instance = &instance.shared.handle;
        let instance_extensions = &instance.shared.extensions;

        let mut candidate = CandidateDetails {
            physical_device,
            ..Default::default()
        };

        // Check properties

        let mut properties = DeviceProperties::default();
        properties.query(&ash_instance, physical_device);
        candidate.properties = properties;

        let api_version = properties.core.api_version;
        if api_version < vk::API_VERSION_1_4 {
            return (
                candidate,
                Err(InspectDeviceError::UnsupportedVulkanDeviceVersion(
                    vk::api_version_major(api_version),
                    vk::api_version_minor(api_version),
                )),
            );
        }

        // Check memory properties
        let memory_properties = unsafe { ash_instance.get_physical_device_memory_properties(physical_device) };
        candidate.memory_properties = memory_properties;

        let has_capable_memory_type = memory_properties
            .memory_types
            .iter()
            .take(memory_properties.memory_type_count as usize)
            .any(|memory_type| {
                let required_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT;

                let forbidden_flags = vk::MemoryPropertyFlags::LAZILY_ALLOCATED
                    | vk::MemoryPropertyFlags::PROTECTED
                    | vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD
                    | vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD;

                memory_type.property_flags.contains(required_flags)
                    && !memory_type.property_flags.intersects(forbidden_flags)
            });

        let has_capable_memory_heap = memory_properties
            .memory_heaps
            .iter()
            .take(memory_properties.memory_heap_count as usize)
            .any(|heap| !heap.flags.contains(vk::MemoryHeapFlags::TILE_MEMORY_QCOM));

        if !has_capable_memory_type || !has_capable_memory_heap {
            return (candidate, Err(InspectDeviceError::NoCapableMemoryType));
        }

        // Check extensions

        let extensions = match unsafe { ash_instance.enumerate_device_extension_properties(physical_device) } {
            Ok(extesions) => extesions,
            Err(error) => {
                return (candidate, Err(InspectDeviceError::Vulkan(error)));
            }
        };
        let mut unsupported_extensions = Vec::new();
        let [
            khr_unified_image_layouts_available,
            ext_descriptor_heap_available,
            khr_device_address_commands_available,
            khr_shader_untyped_pointers_available,
            ext_mesh_shader_available,
            khr_swapchain_available,
            khr_swapchain_maintenance_1_available,
            ext_swapchain_maintenance1_available,
            ext_device_fault_available,
        ] = [
            khr::unified_image_layouts::NAME,
            ext::descriptor_heap::NAME,
            khr::device_address_commands::NAME,
            khr::shader_untyped_pointers::NAME,
            ext::mesh_shader::NAME,
            khr::swapchain::NAME,
            khr::swapchain_maintenance1::NAME,
            ext::swapchain_maintenance1::NAME,
            ext::device_fault::NAME,
        ]
        .map(|name| check_extension(&extensions, name, &mut unsupported_extensions));

        if !khr_unified_image_layouts_available
            || !ext_descriptor_heap_available
            || !khr_device_address_commands_available
            || !khr_shader_untyped_pointers_available
            || !ext_mesh_shader_available
            || !ext_device_fault_available
        {
            return (candidate, Err(InspectDeviceError::NoCapableExtension));
        }
        if !(!has_surface
            || (khr_swapchain_available
                && ((instance_extensions.khr_surface_maintentance_1_available
                    && khr_swapchain_maintenance_1_available)
                    || (instance_extensions.ext_surface_maintentance_1_available
                        && ext_swapchain_maintenance1_available))))
        {
            return (candidate, Err(InspectDeviceError::NoCapablePresentationExtension));
        }

        // Check features

        let mut features = DeviceFeatures::default();
        features.query(
            &ash_instance,
            physical_device,
            has_surface,
            khr_unified_image_layouts_available,
        );
        candidate.features = features;

        let supports_core_features = features.core.shader_int16 == vk::TRUE
            && features.core.sampler_anisotropy == vk::TRUE
            && features.core.depth_bias_clamp == vk::TRUE
            && features.core.independent_blend == vk::TRUE
            && features.core.fragment_stores_and_atomics == vk::TRUE
            && features.core.vertex_pipeline_stores_and_atomics == vk::TRUE
            && features.core.shader_storage_image_read_without_format == vk::TRUE
            && features.core.shader_storage_image_write_without_format == vk::TRUE
            && features.core.multi_draw_indirect == vk::TRUE
            && features.core.draw_indirect_first_instance == vk::TRUE
            && (features.core.texture_compression_bc == vk::TRUE
                || features.core.texture_compression_astc_ldr == vk::TRUE);

        let supports_vulkan11_features = features.vulkan11.storage_buffer16_bit_access == vk::TRUE
            && features.vulkan11.storage_push_constant16 == vk::TRUE
            && features.vulkan11.shader_draw_parameters == vk::TRUE;

        let supports_vulkan12_features = features.vulkan12.shader_float16 == vk::TRUE
            && features.vulkan12.scalar_block_layout == vk::TRUE
            && features.vulkan12.buffer_device_address == vk::TRUE
            && features.vulkan12.timeline_semaphore == vk::TRUE;

        let supports_vulkan13_features = features.vulkan13.synchronization2 == vk::TRUE
            && features.vulkan13.dynamic_rendering == vk::TRUE
            && features.vulkan13.maintenance4 == vk::TRUE;

        let supports_vulkan14_features = features.vulkan14.maintenance5 == vk::TRUE;

        let supports_extension_features = features.descriptor_heap.descriptor_heap == vk::TRUE
            && features.device_address_commands.device_address_commands == vk::TRUE
            && features.shader_untyped_pointers.shader_untyped_pointers == vk::TRUE
            && features.mesh_shader.mesh_shader == vk::TRUE
            && (!has_surface || features.swapchain_maintenance1.swapchain_maintenance1 == vk::TRUE);

        let supports_required_features = supports_core_features
            && supports_vulkan11_features
            && supports_vulkan12_features
            && supports_vulkan13_features
            && supports_vulkan14_features
            && supports_extension_features;

        if !supports_required_features {
            return (candidate, Err(InspectDeviceError::NotRequiredFeatures));
        }

        // Check queue support

        let queue_indices =
            match find_queue_families(&ash_instance, physical_device, &instance.shared.extensions.surface) {
                Ok(indices) => indices,
                Err(error) => return (candidate, Err(error)),
            };
        if options.dedicated_compute_queue && queue_indices.compute.is_none() {
            return (
                candidate,
                Err(InspectDeviceError::NoRequiredQueue(QueueKind::DedicatedCompute)),
            );
        }
        if options.dedicated_transfer_queue && queue_indices.transfer.is_none() {
            return (
                candidate,
                Err(InspectDeviceError::NoRequiredQueue(QueueKind::DedicatedTransfer)),
            );
        }
        candidate.queue_indices = queue_indices;

        (candidate, Ok(()))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DeviceProperties {
    pub core: vk::PhysicalDeviceProperties,
    pub vulkan12: vk::PhysicalDeviceVulkan12Properties<'static>,
    pub descriptor_heap: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        Self {
            core: vk::PhysicalDeviceProperties::default(),
            vulkan12: vk::PhysicalDeviceVulkan12Properties::default(),
            descriptor_heap: vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default(),
        }
    }
}

impl DeviceProperties {
    pub fn query(&mut self, instance: &ash::Instance, physical_device: vk::PhysicalDevice) {
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push(&mut self.vulkan12)
            .push(&mut self.descriptor_heap);

        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
        self.core = properties2.properties;
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DeviceFeatures {
    pub core: vk::PhysicalDeviceFeatures,
    pub vulkan11: vk::PhysicalDeviceVulkan11Features<'static>,
    pub vulkan12: vk::PhysicalDeviceVulkan12Features<'static>,
    pub vulkan13: vk::PhysicalDeviceVulkan13Features<'static>,
    pub vulkan14: vk::PhysicalDeviceVulkan14Features<'static>,
    pub descriptor_heap: vk::PhysicalDeviceDescriptorHeapFeaturesEXT<'static>,
    pub device_address_commands: vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR<'static>,
    pub shader_untyped_pointers: vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR<'static>,
    pub mesh_shader: vk::PhysicalDeviceMeshShaderFeaturesEXT<'static>,
    pub unified_image_layouts: vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR<'static>,
    pub swapchain_maintenance1: vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR<'static>,
}

impl Default for DeviceFeatures {
    fn default() -> Self {
        Self {
            core: vk::PhysicalDeviceFeatures::default(),
            vulkan11: vk::PhysicalDeviceVulkan11Features::default(),
            vulkan12: vk::PhysicalDeviceVulkan12Features::default(),
            vulkan13: vk::PhysicalDeviceVulkan13Features::default(),
            vulkan14: vk::PhysicalDeviceVulkan14Features::default(),
            descriptor_heap: vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default(),
            device_address_commands: vk::PhysicalDeviceDeviceAddressCommandsFeaturesKHR::default(),
            shader_untyped_pointers: vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR::default(),
            mesh_shader: vk::PhysicalDeviceMeshShaderFeaturesEXT::default(),
            unified_image_layouts: vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default(),
            swapchain_maintenance1: vk::PhysicalDeviceSwapchainMaintenance1FeaturesKHR::default(),
        }
    }
}

impl DeviceFeatures {
    pub fn linked_feature2(
        &mut self,
        has_surface: bool,
        unified_image_layout_available: bool,
    ) -> vk::PhysicalDeviceFeatures2<'_> {
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push(&mut self.vulkan11)
            .push(&mut self.vulkan12)
            .push(&mut self.vulkan13)
            .push(&mut self.vulkan14)
            .push(&mut self.descriptor_heap)
            .push(&mut self.device_address_commands)
            .push(&mut self.shader_untyped_pointers)
            .push(&mut self.mesh_shader);

        if has_surface {
            features2 = features2.push(&mut self.swapchain_maintenance1);
        }

        if unified_image_layout_available {
            features2 = features2.push(&mut self.unified_image_layouts);
        }

        features2
    }

    pub fn query(
        &mut self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        has_surface: bool,
        unified_image_layout_available: bool,
    ) {
        let mut features2 = self.linked_feature2(has_surface, unified_image_layout_available);
        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
        self.core = features2.features;
    }
}

#[derive(Debug)]
struct CandidateDetails {
    physical_device: vk::PhysicalDevice,
    properties: DeviceProperties,
    features: DeviceFeatures,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_indices: QueueFamilyIndices,
}

impl Default for CandidateDetails {
    fn default() -> Self {
        Self {
            physical_device: vk::PhysicalDevice::null(),
            properties: DeviceProperties::default(),
            features: DeviceFeatures::default(),
            memory_properties: vk::PhysicalDeviceMemoryProperties::default(),
            queue_indices: QueueFamilyIndices::default(),
        }
    }
}

#[derive(Debug, Error)]
pub enum DeviceCreateError {
    #[error("does not have capable device")]
    NoCapableDevice,

    #[error("vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

#[derive(Debug, Clone, Copy, Error)]
pub enum InspectDeviceError {
    #[error("unsupported vulkan device version {0}.{1}")]
    UnsupportedVulkanDeviceVersion(u32, u32),

    #[error("no capable extensions")]
    NoCapableExtension,

    #[error("no capable presentation extension")]
    NoCapablePresentationExtension,

    #[error("no capable memory type")]
    NoCapableMemoryType,

    #[error("no required features")]
    NotRequiredFeatures,

    #[error("no required queue")]
    NoRequiredQueue(QueueKind),

    #[error("vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

#[derive(Debug)]
pub struct QueueFamilyIndices {
    // raster + compute + (presentation)
    pub main: u32,
    pub compute: Option<u32>,
    pub transfer: Option<u32>,
    pub flat: ArrayVec<u32, 3>,
}

impl Default for QueueFamilyIndices {
    fn default() -> Self {
        Self {
            main: 0,
            compute: None,
            transfer: None,
            flat: ArrayVec::new(),
        }
    }
}

fn find_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: &Option<Surface>,
) -> Result<QueueFamilyIndices, InspectDeviceError> {
    let queues = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let required_queue_flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;

    let mut main_candidates = Vec::new();
    let mut compute_candidates = Vec::new();
    let mut transfer_candidates = Vec::new();

    for (index, properties) in queues.iter().enumerate() {
        let index = index as u32;

        if properties.queue_count == 0 || properties.timestamp_valid_bits != 64 {
            continue;
        }

        let flags = properties.queue_flags;

        let supports_presentation = surface.as_ref().map_or(true, |surface| unsafe {
            surface
                .extension
                .get_physical_device_surface_support(physical_device, index, surface.handle)
                .unwrap_or(false)
        });

        if flags.contains(required_queue_flags) && supports_presentation {
            main_candidates.push(index);
        }

        if flags.contains(vk::QueueFlags::COMPUTE) && !flags.contains(vk::QueueFlags::GRAPHICS) {
            compute_candidates.push(index);
        }

        if flags.contains(vk::QueueFlags::TRANSFER) && !flags.intersects(required_queue_flags) {
            transfer_candidates.push(index);
        }
    }

    let main = main_candidates
        .into_iter()
        .next()
        .ok_or(InspectDeviceError::NoRequiredQueue(QueueKind::Main))?;

    let compute = compute_candidates.into_iter().next();
    let transfer = transfer_candidates.into_iter().next();

    Ok(QueueFamilyIndices {
        main,
        compute,
        transfer,
        flat: ArrayVec::new(),
    })
}

pub(crate) struct DeviceQueues {
    pub(crate) main: Arc<CmdQueue>,
    pub(crate) dedicated_compute: Option<Arc<CmdQueue>>,
    pub(crate) dedicated_transfer: Option<Arc<CmdQueue>>,
}

impl DeviceQueues {
    pub(crate) fn new(device: &Arc<DeviceShared>, options: &DeviceCreateOptions) -> Self {
        let indices = &device.queue_indices;

        let dedicated_compute = indices.compute.and_then(|index| {
            if options.dedicated_compute_queue {
                Some(Arc::new(CmdQueue::new(device, QueueKind::DedicatedCompute, index, 0)))
            } else {
                None
            }
        });
        let dedicated_transfer = indices.transfer.and_then(|index| {
            if options.dedicated_transfer_queue {
                Some(Arc::new(CmdQueue::new(device, QueueKind::DedicatedTransfer, index, 0)))
            } else {
                None
            }
        });

        Self {
            main: Arc::new(CmdQueue::new(device, QueueKind::Main, indices.main, 0)),
            dedicated_compute,
            dedicated_transfer,
        }
    }
}
