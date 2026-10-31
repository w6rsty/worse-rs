use arrayvec::ArrayString;
use ash::vk;
use core::ops::{Add, BitAnd, Not, Sub};
use std::{
    ffi::{CStr, CString},
    fmt::Write,
    panic::Location,
    sync::atomic::{AtomicBool, Ordering},
};
use tracing::error;

use crate::core::{device::DeviceShared, types::*};
use worse_core::critical;

pub(crate) fn format_bytes(bytes: u64) -> ArrayString<32> {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];

    let mut size = bytes as f64;
    let mut unit = 0;

    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }

    let mut out = ArrayString::<32>::new();

    if unit == 0 {
        write!(&mut out, "{} B", bytes).unwrap();
    } else {
        write!(&mut out, "{:.2} {}", size, UNITS[unit]).unwrap();
    }

    out
}

pub(crate) fn cstr_to_string(s: &CStr) -> Result<String, std::str::Utf8Error> {
    s.to_str().map(ToOwned::to_owned)
}

pub(crate) fn cstr_to_string_lossy(s: &CStr) -> String {
    s.to_string_lossy().into_owned()
}

pub(crate) fn string_to_cstring(s: &str) -> Result<CString, std::ffi::NulError> {
    CString::new(s)
}

pub(crate) fn string_to_cstring_lossy(s: &str) -> CString {
    CString::new(s.replace('\0', "")).unwrap()
}

/// Return parsed result, or empty if invalid, c_str must be null-terminated UTF-8 strings
pub(crate) fn parse_name<E>(c_str: Result<&CStr, E>) -> String {
    c_str
        .ok()
        .and_then(|c| c.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_default()
}

/// Return parsed result, or empty if invalid, c_str can be invalid
pub(crate) fn parse_name_lossy<E>(c_str: Result<&CStr, E>) -> String {
    c_str.map(|c| c.to_string_lossy().into_owned()).unwrap_or_default()
}

pub(crate) fn parse_driver_version(vendor_id: u32, driver_version: u32) -> String {
    match vendor_id {
        // NVIDIA
        0x10DE => {
            let major = (driver_version >> 22) & 0x3FF;
            let minor = (driver_version >> 14) & 0xFF;
            let patch = (driver_version >> 6) & 0xFF;
            let build = driver_version & 0x3F;
            format!("{}.{}.{}.{}", major, minor, patch, build)
        }
        // Intel
        0x8086 => {
            let major = (driver_version >> 14) & 0xFFFF;
            let minor = driver_version & 0x3FFF;
            format!("{}.{}", major, minor)
        }
        // AMD or other
        _ => format!("0x{:08X}", driver_version),
    }
}

pub(crate) fn parse_uuid(uuid: &[u8; 16]) -> String {
    format!(
        "{:02X}{:02X}{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
        uuid[0],
        uuid[1],
        uuid[2],
        uuid[3],
        uuid[4],
        uuid[5],
        uuid[6],
        uuid[7],
        uuid[8],
        uuid[9],
        uuid[10],
        uuid[11],
        uuid[12],
        uuid[13],
        uuid[14],
        uuid[15]
    )
}

pub(crate) fn create_object_debug_name<T: vk::Handle + std::fmt::Debug + Copy>(
    device: &DeviceShared,
    object: T,
    name: impl AsRef<str>,
) {
    if let Some(debug_utils) = &device.extensions.debug_utils {
        let name_cstr = CString::new(name.as_ref()).unwrap_or_default();

        let object_name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(object)
            .object_name(&name_cstr);

        if let Err(err) = unsafe { debug_utils.set_debug_utils_object_name(&object_name_info) } {
            error!("failed to set debug name for object {object:?}: {err}");
        }
    }
}

pub trait VulkanCheck<T> {
    fn check(self, device: &DeviceShared) -> T;
}

impl<T> VulkanCheck<T> for Result<T, vk::Result> {
    #[track_caller]
    fn check(self, device: &DeviceShared) -> T {
        match self {
            Ok(v) => v,
            Err(vk::Result::ERROR_DEVICE_LOST) => on_device_lost(&device),
            Err(err) => {
                let location = Location::caller();
                critical!("vulkan error at  {}:{}: {err}", location.file(), location.line());
                panic!();
            }
        }
    }
}

#[cold]
#[track_caller]
fn on_device_lost(device: &DeviceShared) -> ! {
    static REPORTED: AtomicBool = AtomicBool::new(false);
    if !REPORTED.swap(true, Ordering::SeqCst) {
        dump_device_fault(device);
    }
    panic!("device lost")
}

fn dump_device_fault(device: &DeviceShared) {
    let get_device_fault_info_fn = device.extensions.device_fault.fp().get_device_fault_info_ext;

    let mut counts = vk::DeviceFaultCountsEXT::default();
    let r = unsafe { get_device_fault_info_fn(device.handle.handle(), &mut counts, std::ptr::null_mut()) };
    if r != vk::Result::SUCCESS {
        error!("get device fault counts failed: {r}");
        return;
    }

    let mut addrs = vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
    let mut vendors = vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
    let mut binary = vec![0u8; counts.vendor_binary_size as usize];

    let mut info = vk::DeviceFaultInfoEXT::default();
    info.p_address_infos = addrs.as_mut_ptr();
    info.p_vendor_infos = vendors.as_mut_ptr();
    info.p_vendor_binary_data = binary.as_mut_ptr().cast();

    let result = unsafe { get_device_fault_info_fn(device.handle.handle(), &mut counts, &mut info) };
    if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
        error!("get device fault infos failed: {result}");
        return;
    }

    let desc = unsafe { std::ffi::CStr::from_ptr(info.description.as_ptr()) }.to_string_lossy();
    error!("device fault: {desc}");
    for addr in addrs.iter().take(counts.address_info_count as usize) {
        error!(
            "  address {:?} @ {:#x} (precision {:#x})",
            addr.address_type, addr.reported_address, addr.address_precision
        );
    }
    for v in vendors.iter().take(counts.vendor_info_count as usize) {
        let d = unsafe { std::ffi::CStr::from_ptr(v.description.as_ptr()) }.to_string_lossy();
        error!(
            "  vendor: {d} code={:#x} data={:#x}",
            v.vendor_fault_code, v.vendor_fault_data
        );
    }
    if counts.vendor_binary_size > 0 {
        let n = (counts.vendor_binary_size as usize).min(binary.len());
        match std::fs::write("device_fault.bin", &binary[..n]) {
            Ok(()) => error!("vendor binary ({n} bytes) written to device_fault.bin"),
            Err(e) => error!("failed to write vendor binary: {e}"),
        }
    }
}

pub(crate) fn check_extension(
    extensions: &[vk::ExtensionProperties],
    target: &CStr,
    unsupported_extensions: &mut Vec<String>,
) -> bool {
    let found = extensions.iter().any(|prop| {
        let c_str = unsafe { CStr::from_ptr(prop.extension_name.as_ptr()) };
        c_str == target
    });

    if !found {
        unsupported_extensions.push(target.to_string_lossy().into_owned());
    }

    found
}

pub(crate) fn check_layer(layers: &[vk::LayerProperties], target: &CStr, unsupported_layers: &mut Vec<String>) -> bool {
    let found = layers.iter().any(|prop| {
        let c_str = unsafe { CStr::from_ptr(prop.layer_name.as_ptr()) };
        c_str == target
    });

    if !found {
        unsupported_layers.push(target.to_string_lossy().into_owned());
    }

    found
}

pub(crate) fn align_up<T>(value: T, align: T) -> T
where
    T: Copy + PartialEq + From<u8> + Add<Output = T> + Sub<Output = T> + BitAnd<Output = T> + Not<Output = T>,
{
    let zero = T::from(0u8);
    let mask = align - T::from(1u8);
    debug_assert!(align != zero && (align & mask) == zero, "align must be a power of two");
    (value + mask) & !mask
}

pub fn get_mip_extent_2d(extent0: Extent2D, mip_level: u32) -> Extent2D {
    Extent2D {
        width: 1.max(extent0.width >> mip_level),
        height: 1.max(extent0.height >> mip_level),
    }
}

pub fn get_mip_extent_3d(extent0: Extent3D, mip_level: u32) -> Extent3D {
    Extent3D {
        width: 1.max(extent0.width >> mip_level),
        height: 1.max(extent0.height >> mip_level),
        depth: 1.max(extent0.depth >> mip_level),
    }
}
