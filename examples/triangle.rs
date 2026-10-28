use arrayvec::ArrayVec;
use std::{fs, io, ops::Range, path::Path, sync::Arc};
use worse_core::{
    log::{self, LogConfig, error},
    window::{Event, Window, WindowDesc},
};
use worse_rhi::core as rhi;

const WIDTH: u32 = 720;
const HEIGHT: u32 = 480;
const SPWAPCHAIN_IMAGE_COUNT: u32 = 2;
const FRAMES_IN_FLIGHT: u32 = 2;

fn main() {
    let mut log_cfg = LogConfig::engine_preset();
    log_cfg.filter = "debug".to_string();
    let _log_guard = log::init(log_cfg).unwrap();
    let window = Arc::new(Window::new(&WindowDesc::default().initial_size(WIDTH, HEIGHT)).unwrap());

    let instance = rhi::Instance::new(
        rhi::InstanceCreateOptions::default()
            .enable_debug(true)
            .window(Box::new(window.clone())),
    )
    .expect("failed to create instance");
    let device = rhi::Device::new(&instance, rhi::DeviceCreateOptions::default()).expect("failed to create device");

    let pipeline = {
        let vs = load_spirv(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/triangle_vs.spv")).unwrap();
        let fs = load_spirv(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/triangle_fs.spv")).unwrap();
        device
            .create_raster_pipeline(rhi::RasterPipelineDesc {
                primitive: rhi::PrimitiveStages::Vertex {
                    vertex: rhi::ShaderStage::new(&vs, "vertex_main"),
                },
                fragment: rhi::ShaderStage::new(&fs, "fragment_main"),
                rasterizer: rhi::RasterizerState::default(),
                color_targets: &[rhi::ColorTargetDesc::default_with_format(rhi::PixelFormat::RGBA8Srgb)],
                depth_format: rhi::PixelFormat::D32Float,
                stencil_format: rhi::PixelFormat::Undefined,
            })
            .expect("failed to create pipeline")
    };

    let mut swapchain = rhi::Swapchain::new(
        &device,
        rhi::SwapchainCreateOptions::default_with_count(SPWAPCHAIN_IMAGE_COUNT, FRAMES_IN_FLIGHT),
    )
    .expect("failed to create swapchain");
    let depth_desc = rhi::TextureDesc {
        dimension: rhi::TextureDimension::Dimension2D,
        format: rhi::PixelFormat::D32Float,
        extent: rhi::Extent3D {
            width: WIDTH,
            height: HEIGHT,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: rhi::SampleCount::Count1,
        usage: rhi::TextureUsageFlags::DEPTH_STENCIL_ATTACHMENT,
    };
    let depth_requirements = device.query_texture_memory_requirements(&depth_desc);
    let memory_block = device
        .create_memory_block(
            rhi::MemoryDomain::DeviceLocal,
            align_up(depth_requirements.device_size, depth_requirements.device_alignment),
            depth_requirements.memory_type_bits,
            "texture_heap",
        )
        .unwrap();
    let depth = device
        .create_texture_placed(&memory_block, 0, depth_desc, "depth")
        .unwrap();
    let depth_view = device.create_target_view(
        &depth,
        rhi::TextureViewDesc::new_default_range(&depth, rhi::TextureAspect::Depth),
    );

    let cmd_queue = device.main_queue();
    let mut cmd_frames = ArrayVec::from([
        Some(rhi::CmdFrame::new(&cmd_queue, "cmd_frame_0")),
        Some(rhi::CmdFrame::new(&cmd_queue, "cmd_frame_1")),
    ]);

    let timeline = rhi::Timeline::new(&device, 0, "main_timeline");
    let mut last_finish_value = 0;

    {
        let prepare_frame = cmd_frames[0].take().unwrap().wait_and_reset().begin();
        let enc = prepare_frame.bump_encoder();
        enc.texture_init_barrier(
            &depth,
            rhi::TextureAspect::Depth,
            Range { start: 0, end: 1 },
            Range { start: 0, end: 1 },
        );

        last_finish_value += 1;
        let prepare_finish_point = timeline.sync_point(last_finish_value, rhi::StageFlags::ALL_COMMANDS);
        prepare_frame.batch([enc.finish()], [], [prepare_finish_point]);
        cmd_frames[0] = Some(prepare_frame.end_and_submit(prepare_finish_point));
    }

    let mut running = true;
    while running {
        for event in window.pump_events() {
            match event {
                Event::CloseRequested => running = false,
                _ => {}
            }
        }

        let viewport = rhi::Rect2D {
            extent: rhi::Extent2D {
                width: window.width(),
                height: window.height(),
            },
            ..Default::default()
        };

        let swapchain_frame = match swapchain.begin_frame(&timeline, viewport.extent) {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                continue; // Nothing to render
            }
            Err(err) => {
                error!("failed to begin frame {err}");
                continue;
            }
        };

        let solt_index = (last_finish_value % FRAMES_IN_FLIGHT as u64) as usize;
        let cmd_frame = cmd_frames[solt_index].take().unwrap().wait_and_reset().begin();

        let enc = cmd_frame.bump_encoder();
        enc.swapchain_frame_acquire_barrier(&swapchain_frame);
        enc.begin_rendering(rhi::RenderingDesc {
            area: viewport,
            colors: &[rhi::ColorTarget::new(
                rhi::TargetView::from_frame(&swapchain_frame),
                rhi::LoadOp::Clear,
                rhi::StoreOp::Store,
                rhi::ClearColor::EMPTY,
            )],
            depth: Some(rhi::DepthTarget {
                view: depth_view,
                load: rhi::LoadOp::Clear,
                store: rhi::StoreOp::Store,
                clear: 0.0,
            }),
            stencil: None,
        });
        enc.bind_pipeline(&pipeline);
        enc.set_viewport(rhi::Viewport {
            width: viewport.extent.width as f32,
            height: viewport.extent.height as f32,
            min_depth: 1.0,
            max_depth: 0.0,
            ..Default::default()
        });
        enc.set_scissor(viewport);
        enc.set_stencil(rhi::DepthStencilState {
            depth: rhi::DepthState {
                test: true,
                write: true,
                compare: rhi::CompareOp::Always,
                ..Default::default()
            },
            ..Default::default()
        });
        enc.draw(3, 1, 0, 0);
        enc.end_rendering();
        enc.swapchain_frame_present_barrier(&swapchain_frame);

        last_finish_value += 1;
        let frame_end_point = timeline.sync_point(last_finish_value, rhi::StageFlags::ALL_COMMANDS);
        cmd_frame.batch(
            [enc.finish()],
            [swapchain_frame.acquisition_wait()],
            [swapchain_frame.presentation_signal(), frame_end_point],
        );
        cmd_frames[solt_index] = Some(cmd_frame.end_and_submit(frame_end_point));

        swapchain
            .present(&cmd_queue, swapchain_frame, last_finish_value)
            .expect("presentation failed");
    }

    for mut cmd_frame in cmd_frames {
        let _ = cmd_frame.take().unwrap().wait_and_reset();
    }

    swapchain.wait_last_blocking(&timeline);
    device.wait_idle_blocking();

    device.destory_target_view(depth_view);
    device.destory_texture(depth);
    device.destroy_memory_block(memory_block);

    swapchain.destroy();
    device.destroy_pipeline(pipeline);
}

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
