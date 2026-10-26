use arrayvec::ArrayVec;
use std::{fs, io, path::Path, sync::Arc};
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
                depth_format: rhi::PixelFormat::Undefined,
                stencil_format: rhi::PixelFormat::Undefined,
            })
            .expect("failed to create pipeline")
    };

    let mut swapchain = rhi::Swapchain::new(
        &device,
        rhi::SwapchainCreateOptions::default_with_count(SPWAPCHAIN_IMAGE_COUNT, FRAMES_IN_FLIGHT),
    )
    .expect("failed to create swapchain");

    let cmd_queue = device.main_queue();
    let mut cmd_frames = ArrayVec::from([
        Some(rhi::CmdFrame::new(&cmd_queue, "cmd_frame_0")),
        Some(rhi::CmdFrame::new(&cmd_queue, "cmd_frame_1")),
    ]);

    let timeline = rhi::Timeline::new(&device, 0, "main_timeline");
    let mut last_finish_value = 0;

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
            ..Default::default()
        });
        enc.bind_pipeline(&pipeline);
        enc.set_viewport(rhi::Viewport {
            width: viewport.extent.width as f32,
            height: viewport.extent.height as f32,
            ..Default::default()
        });
        enc.set_scissor(viewport);
        enc.set_stencil(rhi::DepthStencilState::default());
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
