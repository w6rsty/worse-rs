use std::{
    cell::{Ref, RefCell},
    fs, io,
    path::Path,
};
use worse_core::log::{self};

fn main() {
    use worse_core::log::*;
    use worse_core::window::*;

    let _log_guard = log::init({
        let mut cfg = log::LogConfig::cli_preset();
        cfg.filter = "debug".to_string();
        cfg
    })
    .unwrap();

    let window = Window::new(
        &WindowDesc::default()
            .title("worse")
            .initial_size(800, 600)
            .resizable(false)
            .theme(WindowTheme::Light),
    )
    .unwrap();

    let names = ["toay", "seos", "iiyno"];
    let mut index = 0;

    let mut exit = false;
    while !exit {
        for event in window.pump_events() {
            info!("{event:?}");

            match event {
                Event::CloseRequested => exit = true,
                Event::Char(char) if char == 'c' => {
                    window.set_title(names[index]);
                    index = (index + 1) % names.len();
                }
                Event::Key { scancode, .. } if scancode == 1 => exit = true,
                _ => {}
            }
        }
    }
    window.close();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{ops::Add, range::Range, sync::Arc, time::Duration};
    use worse_core::window::{Window, WindowDesc, WindowTheme};
    use worse_rhi::core::{self as rhi, PixelFormat, TextureUsageFlags, TimelinePoint};

    const ENABLE_LOGGING: bool = true;

    static RHI_TEST_LOGGER: std::sync::OnceLock<log::LoggerGuard> = std::sync::OnceLock::new();

    #[cfg(test)]
    static RHI_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const SIMULATION_CS_PATH: &'static str = "examples/assets/simulation.spv";
    const PRE_DEPTH_VS_PATH: &'static str = "examples/assets/pre_depth_vs.spv";
    const PRE_DEPTH_FS_PATH: &'static str = "examples/assets/pre_depth_fs.spv";

    fn load_spirv(path: impl AsRef<Path>) -> io::Result<Vec<u32>> {
        let bytes = fs::read(path)?;

        if bytes.len() % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "SPIR-V size is not aligned to 4 bytes",
            ));
        }

        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    fn align_up(value: u64, align: u64) -> u64 {
        let zero = u64::from(0u8);
        let mask = align - u64::from(1u8);
        debug_assert!(align != zero && (align & mask) == zero, "align must be a power of two");
        (value + mask) & !mask
    }

    fn init_rhi_test_logger() {
        if ENABLE_LOGGING {
            RHI_TEST_LOGGER.get_or_init(|| {
                log::init({
                    let mut cfg = log::LogConfig::cli_preset();
                    cfg.filter = "debug".to_string();
                    cfg
                })
                .expect("failed to initialize RHI test logger")
            });
        }
    }

    fn rhi_test<F>(test: F)
    where
        F: FnOnce(&rhi::Device),
    {
        #[cfg(test)]
        let _test_guard = RHI_TEST_LOCK.lock().expect("RHI test lock poisoned");

        init_rhi_test_logger();

        let instance = rhi::Instance::new(rhi::InstanceCreateOptions::default()).expect("failed to create instance");
        let device = rhi::Device::new(&instance, rhi::DeviceCreateOptions::default()).expect("failed to create device");
        test(&device);
    }

    fn rhi_test_with_window<F>(test: F)
    where
        F: FnOnce(&rhi::Device, &rhi::Surface, u32, u32),
    {
        #[cfg(test)]
        let _test_guard = RHI_TEST_LOCK.lock().expect("RHI test lock poisoned");

        init_rhi_test_logger();

        const WIDTH: u32 = 720;
        const HEIGHT: u32 = 480;

        let window = Arc::new(
            Window::new(
                &WindowDesc::default()
                    .title("worse")
                    .initial_size(WIDTH, HEIGHT)
                    .theme(WindowTheme::Dark),
            )
            .unwrap(),
        );
        window.hide();

        let instance = rhi::Instance::new(rhi::InstanceCreateOptions::default().window(Box::new(window.clone())))
            .expect("failed to create instance");
        let device = rhi::Device::new(&instance, rhi::DeviceCreateOptions::default()).expect("failed to create device");
        test(&device, instance.get_surface().as_ref().unwrap(), WIDTH, HEIGHT);
    }

    #[test]
    fn test_compute_pipeline() {
        rhi_test(|device| {
            let spirv = load_spirv(Path::new(env!("CARGO_MANIFEST_DIR")).join(SIMULATION_CS_PATH)).unwrap();
            let compute = rhi::ShaderStage::new(&spirv, "main");

            let simulation_pipeline = { device.create_compute_pipeline(compute).unwrap() };
            device.destroy_pipeline(simulation_pipeline);
        })
    }

    #[test]
    fn test_raster_pipeline() {
        rhi_test(|device| {
            let vertex = load_spirv(Path::new(env!("CARGO_MANIFEST_DIR")).join(PRE_DEPTH_VS_PATH)).unwrap();
            let fragment = load_spirv(Path::new(env!("CARGO_MANIFEST_DIR")).join(PRE_DEPTH_FS_PATH)).unwrap();

            let color_targets = [];
            let desc = rhi::RasterPipelineDesc {
                primitive: rhi::PrimitiveStages::Vertex {
                    vertex: rhi::ShaderStage::new(&vertex, "main"),
                },
                fragment: rhi::ShaderStage::new(&fragment, "main"),
                rasterizer: rhi::RasterizerState::default(),
                color_targets: &color_targets,
                depth_format: rhi::PixelFormat::D32Float,
                stencil_format: rhi::PixelFormat::Undefined,
            };

            let pre_depth_pipeline = { device.create_raster_pipeline(desc).unwrap() };

            device.destroy_pipeline(pre_depth_pipeline);
        });
    }

    #[test]
    fn test_swapchain_creation() {
        rhi_test_with_window(|device, surface, width, height| {
            let swapchain = rhi::Swapchain::new(
                &device,
                rhi::SwapchainCreateOptions {
                    format: rhi::PixelFormat::RGBA8Srgb,
                    color_space: rhi::ColorSpace::SrgbNonLinear,
                    image_count: 2,
                    frames_in_flight: 2,
                    present_mode: rhi::PresentMode::Mailbox,
                    alpha_composite: rhi::SwapchainAlphaComposite::Opaque,
                    window_extent: rhi::Extent2D { width, height },
                },
            )
            .expect("failed to create swapchain");
            swapchain.destroy();
        });
    }

    #[test]
    fn test_buffer_dedicated() {
        rhi_test(|device| {
            const HEAP_SIZE: u64 = 256 * 1024;

            let (buffer, block) = device
                .create_buffer_dedicated(
                    rhi::MemoryDomain::HostVisible,
                    rhi::BufferDesc {
                        size: HEAP_SIZE,
                        usage: rhi::BufferUsageFlags::universal_bits(),
                    },
                    "test",
                )
                .unwrap();

            device.destroy_buffer(buffer);
            device.destroy_memory_block(block);
        });
    }

    #[test]
    fn test_buffer_placed() {
        rhi_test(|device| {
            const HEAP_SIZE: u64 = 256 * 1024;

            let heap = device
                .create_memory_block(rhi::MemoryDomain::HostVisible, HEAP_SIZE, 0b11111, "heap")
                .unwrap();

            let device_address = rhi::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

            let mut offset = 0;
            let cb = device
                .create_buffer_placed(
                    &heap,
                    offset,
                    rhi::BufferDesc {
                        size: 256,
                        usage: rhi::BufferUsageFlags::UNIFORM | device_address,
                    },
                    "cb",
                )
                .unwrap();
            offset += cb.size;
            let vertex = device
                .create_buffer_placed(
                    &heap,
                    offset,
                    rhi::BufferDesc {
                        size: 256,
                        usage: rhi::BufferUsageFlags::VERTEX | device_address,
                    },
                    "vertex",
                )
                .unwrap();
            offset += vertex.size;

            device.destroy_buffer(cb);
            device.destroy_buffer(vertex);
            device.destroy_memory_block(heap);
        });
    }

    #[test]
    fn test_timeline() {
        rhi_test(|device| {
            let timeline = rhi::Timeline::new(device, 0, "timeline");

            let event = 1;
            timeline.signal(event);
            assert_eq!(timeline.completed_value(), 1);
            timeline.wait_blocking(event);
            assert!(timeline.is_completed(event));

            let wait_failed = timeline.wait_blocking_timeout(2, Duration::from_micros(1));
            assert_eq!(wait_failed, rhi::timeline::WaitResult::Timeout);
        });
    }

    #[test]
    fn test_texture() {
        rhi_test(|device| {
            const HEAP_SIZE: u64 = 10 * 1024 * 1024;

            let d0 = rhi::TextureDesc {
                dimension: rhi::TextureDimension::Dimension2D,
                format: rhi::PixelFormat::RGBA8Unorm,
                extent: rhi::Extent3D {
                    width: 256,
                    height: 256,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: rhi::SampleCount::Count1,
                usage: rhi::TextureUsageFlags::COLOR_ATTACHMENT,
            };
            let d1 = rhi::TextureDesc {
                format: rhi::PixelFormat::RGBA8Srgb,
                mip_levels: 4,
                usage: rhi::TextureUsageFlags::SAMPLED | rhi::TextureUsageFlags::TRANSFER_DST,
                ..d0
            };
            let d2 = rhi::TextureDesc {
                format: rhi::PixelFormat::R8UInt,
                usage: rhi::TextureUsageFlags::STORAGE
                    | rhi::TextureUsageFlags::TRANSFER_SRC
                    | rhi::TextureUsageFlags::TRANSFER_DST,
                ..d0
            };

            let r0 = device.query_texture_memory_requirements(&d0);
            let r1 = device.query_texture_memory_requirements(&d1);
            let r2 = device.query_texture_memory_requirements(&d2);

            let memory_type = r0.memory_type_bits | r1.memory_type_bits | r2.memory_type_bits;

            let block = device
                .create_memory_block(rhi::MemoryDomain::DeviceLocal, HEAP_SIZE, memory_type, "heap")
                .unwrap();

            let mut offset = 0;

            offset = align_up(offset, r0.device_alignment);
            let t0 = device.create_texture_placed(&block, offset, d0, "ColorTarget").unwrap();
            offset = align_up(offset, r1.device_alignment);
            offset += r0.device_size;
            let t1 = device.create_texture_placed(&block, offset, d1, "SRV").unwrap();
            offset = align_up(offset, r2.device_alignment);
            offset += r1.device_size;
            let t2 = device.create_texture_placed(&block, offset, d2, "UAV").unwrap();

            device.destory_texture(t0);
            device.destory_texture(t1);
            device.destory_texture(t2);
            device.destroy_memory_block(block);
        });
    }

    #[test]
    fn test_descriptor_heap() {
        rhi_test(|device| {
            let texture_desc = rhi::TextureDesc {
                dimension: rhi::TextureDimension::Dimension2D,
                format: rhi::PixelFormat::RGBA8Unorm,
                extent: rhi::Extent3D {
                    width: 256,
                    height: 256,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: rhi::SampleCount::Count1,
                usage: rhi::TextureUsageFlags::SAMPLED | rhi::TextureUsageFlags::TRANSFER_DST,
            };
            let requirements = device.query_texture_memory_requirements(&texture_desc);

            let block = device
                .create_memory_block(
                    rhi::MemoryDomain::DeviceLocal,
                    1024 * 1024,
                    requirements.memory_type_bits,
                    "texture_heap",
                )
                .unwrap();
            let texture = device.create_texture_placed(&block, 0, texture_desc, "SRV").unwrap();

            let texture_descriptor_heap = device
                .create_descriptor_heap(128, rhi::DescriptorHeapKind::Texture, "descriptor_heap")
                .unwrap();

            texture_descriptor_heap.write_texture_descriptor(
                0,
                rhi::TextureViewDesc::new_default_range(&texture, rhi::TextureAspect::Color),
                rhi::TextureDescriptorKind::Sampled,
            );

            device.destory_texture(texture);
            device.destroy_descriptor_heap(texture_descriptor_heap);
            device.destroy_memory_block(block);
        });
    }
}
