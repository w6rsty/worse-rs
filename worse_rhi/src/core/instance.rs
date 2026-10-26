use ash::{ext, khr, vk};
use raw_window_handle::{HasWindowHandle, RawDisplayHandle, WindowsDisplayHandle};
use std::{
    ffi::{CStr, c_void},
    sync::Arc,
};
use thiserror::Error;
use tracing::{debug, error, info, warn};

use crate::core::utils::{check_extension, check_layer};

pub struct InstanceCreateOptions {
    pub enable_debug: bool,
    pub window: Option<Box<dyn HasWindowHandle>>,
}

impl Default for InstanceCreateOptions {
    fn default() -> Self {
        Self {
            enable_debug: false,
            window: None,
        }
    }
}

impl InstanceCreateOptions {
    pub fn enable_debug(mut self, enable: bool) -> Self {
        self.enable_debug = enable;
        self
    }

    pub fn window(mut self, window: Box<dyn HasWindowHandle>) -> Self {
        self.window = Some(window);
        self
    }
}

pub struct Instance {
    pub(crate) shared: Arc<InstanceShared>,
}

impl Instance {
    pub fn new(options: InstanceCreateOptions) -> Result<Self, InstanceCreateError> {
        Ok(Self {
            shared: Arc::new(InstanceShared::new(options)?),
        })
    }

    pub fn get_surface(&self) -> &Option<Surface> {
        &self.shared.extensions.surface
    }
}

pub(crate) struct InstanceShared {
    pub(crate) _entry: ash::Entry,
    pub(crate) handle: ash::Instance,

    pub(crate) extensions: InstanceExtension,
}

pub(crate) struct InstanceExtension {
    pub(crate) debug_utils: Option<DebugUtils>,
    pub(crate) surface: Option<Surface>,
    pub(crate) khr_surface_maintentance_1_available: bool,
    pub(crate) ext_surface_maintentance_1_available: bool,
}

impl InstanceShared {
    pub fn new(options: InstanceCreateOptions) -> Result<Self, InstanceCreateError> {
        let entry = unsafe { ash::Entry::load().map_err(|e| InstanceCreateError::Entry(e))? };

        let instance_version = match unsafe { entry.try_enumerate_instance_version() }? {
            Some(version) => version,
            None => vk::API_VERSION_1_0,
        };
        let (instance_version_major, instance_version_minor) = (
            vk::api_version_major(instance_version),
            vk::api_version_minor(instance_version),
        );
        if instance_version < vk::API_VERSION_1_4 {
            return Err(InstanceCreateError::UnsupportedVulkanInstanceVersion(
                instance_version_major,
                instance_version_minor,
            ));
        }
        debug!("loaded vulkan entry, api version {instance_version_major}.{instance_version_minor}",);

        // Check instance extensions

        let instance_extensions = unsafe { entry.enumerate_instance_extension_properties(None)? };
        let mut unsupported_extensions = Vec::new();

        let [
            khr_surface_maintentance_1_available,
            ext_surface_maintentance_1_available,
            khr_get_surface_capability_2_available,
            ext_debug_utils_available,
        ] = [
            khr::surface_maintenance1::NAME,
            ext::surface_maintenance1::NAME,
            khr::get_surface_capabilities2::NAME,
            ext::debug_utils::NAME,
        ]
        .map(|name| check_extension(&instance_extensions, name, &mut unsupported_extensions));

        let display_handle = RawDisplayHandle::Windows(WindowsDisplayHandle::new());
        let mut surface_extensions = Vec::new();
        let enable_presentation = options.window.is_some();

        if enable_presentation {
            surface_extensions.extend_from_slice(ash_window::enumerate_required_extensions(display_handle)?);
            let all_surface_extensions_available =
                check_all_surface_extensions(&instance_extensions, &surface_extensions, &mut unsupported_extensions);

            if !all_surface_extensions_available
                || !khr_get_surface_capability_2_available
                || (!khr_surface_maintentance_1_available && !ext_surface_maintentance_1_available)
            {
                return Err(InstanceCreateError::UnsupportedExtension(unsupported_extensions));
            }
        }

        // Check instance layers

        let instance_layers = unsafe { entry.enumerate_instance_layer_properties()? };
        let mut unsupported_layers = Vec::new();

        let validation_available = check_layer(
            &instance_layers,
            c"VK_LAYER_KHRONOS_validation",
            &mut unsupported_layers,
        );

        if !unsupported_layers.is_empty() {
            warn!("unsupport instance layers: {unsupported_layers:?}");
        }

        // Create instance

        let mut enabled_extensions = Vec::new();
        if ext_debug_utils_available {
            enabled_extensions.push(ext::debug_utils::NAME.as_ptr())
        }
        if enable_presentation {
            enabled_extensions.extend_from_slice(&surface_extensions);
            enabled_extensions.push(khr::surface_maintenance1::NAME.as_ptr());
            enabled_extensions.push(ext::surface_maintenance1::NAME.as_ptr());
            enabled_extensions.push(khr::get_surface_capabilities2::NAME.as_ptr());
        }

        let mut enabled_layers = Vec::new();
        if options.enable_debug && validation_available {
            enabled_layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        }

        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_4);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&enabled_extensions)
            .enabled_layer_names(&enabled_layers);

        let instance = unsafe { entry.create_instance(&instance_info, None)? };
        if cfg!(debug_assertions) {
            let ext_names: Vec<String> = enabled_extensions
                .iter()
                .map(|&ptr| unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
                .collect();

            let layer_names: Vec<String> = enabled_layers
                .iter()
                .map(|&ptr| unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
                .collect();

            debug!(
                "created instance {:?}\nenabled extensions ({}) {:#?}\nenabled layers ({}): {:#?}",
                instance.handle(),
                ext_names.len(),
                ext_names,
                layer_names.len(),
                layer_names
            );
        } else {
            debug!(
                "created instance {:?}, extensions({}) layers({})",
                instance.handle(),
                enabled_extensions.len(),
                enabled_layers.len()
            );
        }

        // Debug utils
        let debug_utils = if ext_debug_utils_available {
            let debug_utils_extension = ext::debug_utils::Instance::load(&entry, &instance);
            let debug_tils_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                )
                .pfn_user_callback(Some(debug_utils_messenger_callback));
            let debug_utils_messenger =
                unsafe { debug_utils_extension.create_debug_utils_messenger(&debug_tils_info, None)? };

            debug!("created debug utils messenger {debug_utils_messenger:?}");

            Some(DebugUtils {
                extension: debug_utils_extension,
                messenger: debug_utils_messenger,
            })
        } else {
            None
        };

        let surface = if enable_presentation {
            let display_handle = RawDisplayHandle::Windows(WindowsDisplayHandle::new());
            let factory = ash_window::SurfaceFactory::new(&entry, &instance, display_handle)?;
            let window_handle = options.window.as_ref().unwrap().window_handle()?.as_raw();

            let handle = unsafe { factory.create_surface(window_handle, None)? };
            debug!("created surface {handle:?}");

            Some(Surface {
                extension: khr::surface::Instance::load(&entry, &instance),
                handle,
                get_surface_capabilities2_extension: khr::get_surface_capabilities2::Instance::load(&entry, &instance),
            })
        } else {
            None
        };

        Ok(Self {
            _entry: entry,
            handle: instance,
            extensions: InstanceExtension {
                debug_utils,
                surface,
                khr_surface_maintentance_1_available,
                ext_surface_maintentance_1_available,
            },
        })
    }
}

impl Drop for InstanceShared {
    fn drop(&mut self) {
        if let Some(surface) = &self.extensions.surface {
            unsafe {
                surface.extension.destroy_surface(surface.handle, None);
            }
        }

        if let Some(debug_utils) = &self.extensions.debug_utils {
            unsafe {
                debug_utils
                    .extension
                    .destroy_debug_utils_messenger(debug_utils.messenger, None)
            };
        }

        unsafe { self.handle.destroy_instance(None) };
    }
}

#[derive(Debug, Error)]
pub enum InstanceCreateError {
    #[error("failed to load entry: {0}")]
    Entry(#[from] ash::LoadingError),

    #[error("unsupport vulkan instance version {0}.{1}, must be 1.4")]
    UnsupportedVulkanInstanceVersion(u32, u32),

    #[error("lack of necessary extensions, count: {0:#?}")]
    UnsupportedExtension(Vec<String>),

    #[error("window handle error: {0}")]
    WindowHandle(#[from] raw_window_handle::HandleError),

    #[error("vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

pub struct DebugUtils {
    pub(crate) extension: ext::debug_utils::Instance,
    pub(crate) messenger: vk::DebugUtilsMessengerEXT,
}

pub struct Surface {
    pub(crate) extension: khr::surface::Instance,
    pub(crate) handle: vk::SurfaceKHR,
    pub(crate) get_surface_capabilities2_extension: khr::get_surface_capabilities2::Instance,
}

pub extern "system" fn debug_utils_messenger_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    if p_callback_data.is_null() {
        return vk::FALSE;
    }

    let callback_data = unsafe { &*p_callback_data };

    let message = if !callback_data.p_message.is_null() {
        unsafe { CStr::from_ptr(callback_data.p_message).to_string_lossy() }
    } else {
        "No message".into()
    };

    let message_id_name = if !callback_data.p_message_id_name.is_null() {
        unsafe { CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
    } else {
        "Unknown".into()
    };

    let message_id_number = callback_data.message_id_number;

    let object_count = callback_data.object_count;

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!(
                target: "vulkan::validation",
                type = ?message_type,
                id_name = %message_id_name,
                id_num = message_id_number,
                objects = object_count,
                "{message}"
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!(
                target: "vulkan::validation",
                type = ?message_type,
                id_name = %message_id_name,
                id_num = message_id_number,
                objects = object_count,
                "{message}"
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!(
                target: "vulkan::validation",
                type = ?message_type,
                id_name = %message_id_name,
                id_num = message_id_number,
                "{message}"
            );
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            debug!(
                target: "vulkan::validation",
                type = ?message_type,
                id_name = %message_id_name,
                id_num = message_id_number,
                "{message}"
            );
        }
        _ => {
            debug!(
                target: "vulkan::validation",
                type = ?message_type,
                "{message}"
            );
        }
    }

    vk::FALSE
}

fn check_all_surface_extensions(
    extensions: &[vk::ExtensionProperties],
    targets: &[*const i8],
    unsupported_extensions: &mut Vec<String>,
) -> bool {
    let mut all_found = true;

    for &target_ptr in targets {
        let target_cstr = unsafe { CStr::from_ptr(target_ptr) };

        let found = extensions.iter().any(|prop| {
            let prop_cstr = unsafe { CStr::from_ptr(prop.extension_name.as_ptr()) };
            prop_cstr == target_cstr
        });

        if !found {
            all_found = false;
            unsupported_extensions.push(target_cstr.to_string_lossy().into_owned());
        }
    }

    all_found
}
