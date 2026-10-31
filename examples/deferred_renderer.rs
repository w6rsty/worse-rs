use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use rand::Rng;
use std::{cell::RefCell, fs, io, path::Path, sync::Arc, time};
use worse_core::{
    log::{self, LogConfig, error},
    window::{Event, Window, WindowDesc},
};
use worse_rhi::core::{self as rhi, BufferSlice};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const SPWAPCHAIN_IMAGE_COUNT: u32 = 2;
const FRAMES_IN_FLIGHT: u32 = 1;
const PARTICLE_COUNTS: usize = 1024;

fn main() {
    let mut log_cfg = LogConfig::engine_preset();
    log_cfg.filter = "debug".to_string();
    let _log_guard = log::init(log_cfg).unwrap();
    let window = Arc::new(Window::new(&WindowDesc::default().initial_size(WIDTH, HEIGHT).resizable(false)).unwrap());

    let (_instance, device) = make_context(window.clone());

    let root_allocator = RootBumpAllocator::new(512, &device);

    let simulation_buffer = make_simulation_buffer(&device);
    let simulation_pipeline = make_simulation_pipeline(&device);

    let base_pass_pipeline = make_base_pass_pipeline(&device);

    let depth_texture = device
        .create_texture_dedicated(rhi::TextureDesc::new_default_depth(WIDTH, HEIGHT), "depth")
        .unwrap();
    let depth_texture_view = device.create_target_view(
        &depth_texture.0,
        rhi::TextureViewDesc::new_default_range(&depth_texture.0, rhi::TextureAspect::Depth),
    );

    let mut swapchain = rhi::Swapchain::new(
        &device,
        rhi::SwapchainCreateOptions::default_with_count(SPWAPCHAIN_IMAGE_COUNT, FRAMES_IN_FLIGHT),
    )
    .expect("failed to create swapchain");

    let main_queue = device.main_queue();
    let mut cmd_frame = rhi::CmdFrame::new(&main_queue, "cmd_frame");

    let mut previous_time = time::Instant::now();
    let mut last_finish_value: u64 = 0;
    let timeline = rhi::Timeline::new(&device, last_finish_value, "timeline");

    {
        let prepare_frame = cmd_frame.wait_and_reset().begin();
        let enc = prepare_frame.bump_encoder();
        enc.unifiy_texture_layout(&depth_texture.0, rhi::TextureAspect::Depth);

        last_finish_value += 1;
        let prepare_finish_point = timeline.sync_point(last_finish_value, rhi::StageFlags::ALL_COMMANDS);
        prepare_frame.batch([enc.finish()], [], [prepare_finish_point]);
        cmd_frame = prepare_frame.end_and_submit(prepare_finish_point);
    }

    let mut running = true;
    while running {
        for event in window.pump_events() {
            match event {
                Event::CloseRequested => running = false,
                _ => {}
            }
        }

        let window_extent = rhi::Extent2D {
            width: window.width(),
            height: window.height(),
        };

        let swapchain_frame = match swapchain.begin_frame(&timeline, window_extent) {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                continue; // Nothing to render
            }
            Err(err) => {
                error!("failed to begin frame: {err}");
                continue;
            }
        };

        root_allocator.reset();

        let current_time = time::Instant::now();
        let delta_seconds = current_time.duration_since(previous_time).as_secs_f32();
        previous_time = current_time;

        let active_cmd_frame = cmd_frame.wait_and_reset().begin();
        let enc = active_cmd_frame.bump_encoder();

        enc.swapchain_frame_acquire_barrier(&swapchain_frame);
        enc.barrier(
            rhi::StageFlags::COMPUTE | rhi::StageFlags::MESH,
            rhi::AccessFlags::SHADER_READ | rhi::AccessFlags::SHADER_WRITE,
            rhi::StageFlags::COMPUTE,
            rhi::AccessFlags::SHADER_READ | rhi::AccessFlags::SHADER_WRITE,
        );
        simulation(
            &enc,
            &simulation_pipeline,
            &root_allocator,
            &simulation_buffer.0,
            delta_seconds,
        );
        enc.barrier(
            rhi::StageFlags::COMPUTE,
            rhi::AccessFlags::SHADER_WRITE,
            rhi::StageFlags::MESH,
            rhi::AccessFlags::SHADER_READ,
        );
        base_pass(
            &enc,
            &base_pass_pipeline,
            window_extent,
            rhi::TargetView::from_frame(&swapchain_frame),
            depth_texture_view,
            &root_allocator,
            &simulation_buffer.0,
        );
        enc.swapchain_frame_present_barrier(&swapchain_frame);

        last_finish_value += 1;
        let frame_finish_point = timeline.sync_point(last_finish_value, rhi::StageFlags::ALL_COMMANDS);
        active_cmd_frame.batch(
            [enc.finish()],
            [swapchain_frame.acquisition_wait()],
            [swapchain_frame.presentation_signal(), frame_finish_point],
        );
        cmd_frame = active_cmd_frame.end_and_submit(frame_finish_point);
        swapchain
            .present(&main_queue, swapchain_frame, last_finish_value)
            .unwrap();
    }

    let _ = cmd_frame.wait_and_reset();

    swapchain.wait_last_blocking(&timeline);
    device.wait_idle_blocking();
    device.destroy_pipeline(base_pass_pipeline);
    device.destroy_pipeline(simulation_pipeline);
    device.destroy_buffer(simulation_buffer.0);
    device.destroy_memory_block(simulation_buffer.1);
    device.destory_target_view(depth_texture_view);
    device.destory_texture(depth_texture.0);
    device.destroy_memory_block(depth_texture.1);
    swapchain.destroy();
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

fn make_context(window: Arc<Window>) -> (rhi::Instance, rhi::Device) {
    let instance = rhi::Instance::new(
        rhi::InstanceCreateOptions::default()
            .enable_debug(true)
            .window(Box::new(window)),
    )
    .expect("failed to create instance");
    let device = rhi::Device::new(&instance, rhi::DeviceCreateOptions::default()).expect("failed to create device");

    (instance, device)
}

fn make_simulation_pipeline(device: &rhi::Device) -> rhi::Pipeline {
    let cs = load_spirv(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/simulation_cs.spv")).unwrap();
    device
        .create_compute_pipeline(rhi::ShaderStage::new(&cs, "compute_main"))
        .unwrap()
}

fn make_base_pass_pipeline(device: &rhi::Device) -> rhi::Pipeline {
    let ms = load_spirv(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/base_pass_ms.spv")).unwrap();
    let fs = load_spirv(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/base_pass_fs.spv")).unwrap();
    device
        .create_raster_pipeline(rhi::RasterPipelineDesc {
            primitive: rhi::PrimitiveStages::Mesh {
                task: None,
                mesh: rhi::ShaderStage::new(&ms, "mesh_main"),
            },
            fragment: rhi::ShaderStage::new(&fs, "fragment_main"),
            rasterizer: rhi::RasterizerState { ..Default::default() },
            color_targets: &[rhi::ColorTargetDesc::default_with_format(rhi::PixelFormat::RGBA8Srgb)],
            depth_format: rhi::PixelFormat::D32Float,
            stencil_format: rhi::PixelFormat::Undefined,
        })
        .expect("failed to create pipeline")
}

fn simulation(
    enc: &rhi::CmdEncoder<'_, rhi::state::Active>,
    pipeline: &rhi::Pipeline,
    root_allocator: &RootBumpAllocator,
    buffer: &rhi::Buffer,
    delta_seconds: f32,
) {
    enc.bind_pipeline(pipeline);
    let mut root = root_allocator.allocate();
    *root.as_mut() = SimulationRoot {
        particles: buffer.device_address,
        delta_seconds,
        _pad: 0.0,
    };
    enc.emit_root(&root);
    enc.dispatch(1, 1, 1);
}

fn base_pass(
    enc: &rhi::CmdEncoder<'_, rhi::state::Active>,
    pipeline: &rhi::Pipeline,
    extent: rhi::Extent2D,
    color: rhi::TargetView,
    depth: rhi::TargetView,
    root_allocator: &RootBumpAllocator,
    simulation_buffer: &rhi::Buffer,
) {
    enc.begin_rendering(rhi::RenderingDesc {
        area: rhi::Rect2D {
            extent,
            offset: rhi::Offset2D::default(),
        },
        colors: &[rhi::ColorTarget::new(
            color,
            rhi::LoadOp::Clear,
            rhi::StoreOp::Store,
            rhi::ClearColor::EMPTY,
        )],
        depth: Some(rhi::DepthTarget {
            view: depth,
            load: rhi::LoadOp::Clear,
            store: rhi::StoreOp::Store,
            clear: 0.0,
        }),
        stencil: None,
    });
    enc.bind_pipeline(&pipeline);
    let aspect_ratio = extent.width.max(1) as f32 / extent.height.max(1) as f32;
    let view = glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, Vec3::Y);
    let projection = glam::camera::rh::proj::vulkan::perspective(60.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);
    let orientation = Quat::IDENTITY;
    let view_projection = (projection * view).to_cols_array_2d();
    let mut root = root_allocator.allocate();
    *root.as_mut() = BasePassRoot {
        view_projection,
        orientation: [orientation.x, orientation.y, orientation.z, orientation.w],
        particles: simulation_buffer.device_address,
        _padding: [0; 2],
    };
    enc.emit_root(&root);
    enc.set_viewport(rhi::Viewport {
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
        ..Default::default()
    });
    enc.set_scissor(rhi::Rect2D {
        extent,
        offset: rhi::Offset2D::default(),
    });
    enc.set_stencil(rhi::DepthStencilState {
        depth: rhi::DepthState {
            test: true,
            write: true,
            compare: rhi::CompareOp::Always,
            ..Default::default()
        },
        ..Default::default()
    });
    enc.draw_mesh(PARTICLE_COUNTS as u32, 1, 1);
    enc.end_rendering();
}

fn make_simulation_buffer(device: &rhi::Device) -> (rhi::Buffer, rhi::MemoryBlock) {
    let simulation_buffer_desc = rhi::BufferDesc {
        size: size_of::<Particle>() * PARTICLE_COUNTS,
        usage: rhi::BufferUsageFlags::SHADER_DEVICE_ADDRESS | rhi::BufferUsageFlags::STORAGE,
    };
    let mut simulation_buffer = device
        .create_buffer_dedicated(
            rhi::MemoryDomain::HostVisible,
            simulation_buffer_desc,
            "simulation_buffer",
        )
        .unwrap();

    let mut rng = rand::rng();
    for particle in simulation_buffer.0.as_slice_mut::<Particle>() {
        *particle = Particle {
            position: [
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(-0.75..0.75),
            ],
            velocity: [
                rng.random_range(-0.1..0.1),
                rng.random_range(-0.1..0.1),
                rng.random_range(-0.1..0.1) * rng.random_range(-2.0..2.0),
            ],
        };
    }

    simulation_buffer
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Particle {
    position: [f32; 3],
    velocity: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SimulationRoot {
    particles: u64,
    delta_seconds: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BasePassRoot {
    view_projection: [[f32; 4]; 4],
    orientation: [f32; 4],
    particles: u64,
    _padding: [u32; 2],
}

fn align_up(value: usize, align: usize) -> usize {
    let zero = usize::from(0u8);
    let mask = align - usize::from(1u8);
    debug_assert!(align != zero && (align & mask) == zero, "align must be a power of two");
    (value + mask) & !mask
}

fn aligned_size<T: bytemuck::Pod>() -> usize {
    align_up(std::mem::size_of::<T>(), std::mem::align_of::<T>())
}

struct RootBumpAllocator<'a> {
    device: &'a rhi::Device,
    state: RefCell<RootBumpAllocatorState>,
    buffer: rhi::Buffer,
    memory: rhi::MemoryBlock,
}

#[derive(Clone, Copy)]
struct RootBumpAllocatorState {
    cursor: usize,
    capacity: usize,
}

impl<'a> RootBumpAllocator<'a> {
    pub fn new(capacity: usize, device: &'a rhi::Device) -> Self {
        let (buffer, memory) = device
            .create_buffer_dedicated(
                rhi::MemoryDomain::HostVisible,
                rhi::BufferDesc {
                    size: capacity,
                    usage: rhi::BufferUsageFlags::UNIFORM | rhi::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                },
                "root_bump_allocator_underlying",
            )
            .unwrap();

        Self {
            device,
            state: RefCell::new(RootBumpAllocatorState { cursor: 0, capacity }),
            buffer,
            memory,
        }
    }

    pub fn allocate<T: bytemuck::Pod>(&'a self) -> BufferSlice<'a, T> {
        let state = self.state.borrow().clone();
        let required_bytes = aligned_size::<T>();
        assert!(state.cursor + required_bytes <= state.capacity);
        let slice = self.buffer.slice(state.cursor, 1);
        self.state.borrow_mut().cursor += required_bytes;
        slice
    }

    pub fn reset(&self) {
        self.state.borrow_mut().cursor = 0;
    }
}

impl Drop for RootBumpAllocator<'_> {
    fn drop(&mut self) {
        self.device.destroy_buffer(self.buffer);
        self.device.destroy_memory_block(self.memory);
    }
}
