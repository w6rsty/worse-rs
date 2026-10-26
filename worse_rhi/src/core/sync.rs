use ash::vk::{self, TaggedStructure};
use std::{sync::Arc, time::Duration};
use tracing::debug;

use crate::core::{
    StageFlags,
    device::{Device, DeviceShared},
    utils::{VulkanCheck, create_object_debug_name},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncPoint {
    pub semaphore: vk::Semaphore,
    pub value: u64,
    pub stage: vk::PipelineStageFlags2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitResult {
    Completed,
    Timeout,
}

impl SyncPoint {
    pub(crate) fn binary(semaphore: vk::Semaphore, stage: vk::PipelineStageFlags2) -> Self {
        Self {
            semaphore,
            value: 0,
            stage,
        }
    }

    pub(crate) fn to_vk<'a>(self) -> vk::SemaphoreSubmitInfo<'a> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(self.value)
            .stage_mask(self.stage)
    }
}

pub(crate) fn wait_blocking(device: &DeviceShared, point: SyncPoint) {
    // Drive may replace u64::MAX with a smaller timeout value
    loop {
        match wait_ns(device, point, u64::MAX) {
            Err(vk::Result::TIMEOUT) => continue,
            result => {
                result.check(device);
                return;
            }
        }
    }
}

pub(crate) fn wait_blocking_timeout(device: &DeviceShared, point: SyncPoint, timeout: Duration) -> WaitResult {
    let ns = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
    match wait_ns(device, point, ns) {
        Ok(()) => WaitResult::Completed,
        Err(vk::Result::TIMEOUT) => WaitResult::Timeout,
        result => {
            let _ = result.check(device);
            unreachable!();
        }
    }
}

pub(crate) fn wait_ns(device: &DeviceShared, point: SyncPoint, timeout_ns: u64) -> Result<(), vk::Result> {
    let semaphores = [point.semaphore];
    let values = [point.value];
    let wait_info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
    unsafe { device.handle.wait_semaphores(&wait_info, timeout_ns) }
}

pub struct Timeline {
    device: Arc<DeviceShared>,
    pub(crate) semaphore: vk::Semaphore,
}

impl Timeline {
    pub fn new(device: &Device, initial_value: u64, name: impl AsRef<str>) -> Self {
        let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let timeline_info = vk::SemaphoreCreateInfo::default().push(&mut timeline_type);
        let timeline = unsafe { device.shared.handle.create_semaphore(&timeline_info, None) }.check(&device.shared);
        debug!(
            "create timeline {timeline:?}[{}], initial value: {}",
            name.as_ref(),
            initial_value
        );
        create_object_debug_name(&device.shared, timeline, name);
        Self {
            device: device.shared.clone(),
            semaphore: timeline,
        }
    }

    pub fn sync_point(&self, value: u64, stage: StageFlags) -> SyncPoint {
        SyncPoint {
            semaphore: self.semaphore,
            value,
            stage: stage.to_vk(),
        }
    }

    pub fn signal(&self, value: u64) {
        let signal_info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(value);
        unsafe { self.device.handle.signal_semaphore(&signal_info) }.check(&self.device);
    }

    pub fn completed_value(&self) -> u64 {
        unsafe { self.device.handle.get_semaphore_counter_value(self.semaphore) }.check(&self.device)
    }

    pub fn is_completed(&self, value: u64) -> bool {
        self.completed_value() >= value
    }

    pub fn wait_blocking(&self, value: u64) {
        // Drive may replace u64::MAX with a smaller timeout value
        loop {
            match self.wait_ns(value, u64::MAX) {
                Err(vk::Result::TIMEOUT) => continue,
                result => {
                    result.check(&self.device);
                    return;
                }
            }
        }
    }

    pub fn wait_blocking_timeout(&self, value: u64, timeout: Duration) -> WaitResult {
        let ns = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        match self.wait_ns(value, ns) {
            Ok(()) => WaitResult::Completed,
            Err(vk::Result::TIMEOUT) => WaitResult::Timeout,
            result => {
                let _ = result.check(&self.device);
                unreachable!();
            }
        }
    }

    fn wait_ns(&self, value: u64, timeout_ns: u64) -> Result<(), vk::Result> {
        let semaphores = [self.semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
        unsafe { self.device.handle.wait_semaphores(&wait_info, timeout_ns) }
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        unsafe { self.device.handle.destroy_semaphore(self.semaphore, None) };
    }
}
