use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use rand::Rng;
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    sync::Arc,
    time,
};
use worse_core::{
    log::{self, LogConfig, error, info},
    window::{Event, Window, WindowDesc},
};
use worse_rhi::core::{self as rhi, BufferSlice, TargetView};
use worse_shader_compiler as compile;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const SPWAPCHAIN_IMAGE_COUNT: u32 = 2;
const FRAMES_IN_FLIGHT: u32 = 1;
const PARTICLE_COUNTS: usize = 1024;
const CAMERA_MOVE_STEP: f32 = 0.25;
const CAMERA_FOV_STEP: f32 = 5.0_f32.to_radians();
const CAMERA_CLIP_STEP: f32 = 0.1;

fn main() {
    let mut log_cfg = LogConfig::engine_preset();
    log_cfg.filter = "info".to_string();
    let _log_guard = log::init(log_cfg).unwrap();
    let window = Arc::new(Window::new(&WindowDesc::default().initial_size(WIDTH, HEIGHT)).unwrap());

    let (_instance, device) = make_context(window.clone());

    let root_allocator = RootBumpAllocator::new(512, &device);
    let texture_descriptor_heap = device
        .create_descriptor_heap(32, rhi::DescriptorHeapKind::Texture, "texture_descriptor_heap")
        .unwrap();
    let texture_allocator = TextureAllocator::new(256 * 1024 * 1024, &device);

    let compile_service = compile::CompileService::new().unwrap();
    let simulation_pipeline = make_simulation_pipeline(&device, &compile_service);
    let gbuffer_pipeline = make_gbuffer_pipeline(&device, &compile_service);
    let shading_pipeline = make_shading_pipeline(&device, &compile_service);

    let simulation_buffer = make_simulation_buffer(&device);

    let mut gbuffer: Option<GBuffer> = None;

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
    let mut camera = CameraSettings::default();
    info!("Controls: WASD move, Q/E move vertically, +/- adjust FOV, J/K adjust near, N/M adjust far, R reset");

    let mut running = true;
    while running {
        for event in window.pump_events() {
            match event {
                Event::CloseRequested => running = false,
                Event::Char(character) => camera.handle_char(character),
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

        let current_time = time::Instant::now();
        let delta_seconds = current_time.duration_since(previous_time).as_secs_f32();
        previous_time = current_time;

        let active_cmd_frame = cmd_frame.wait_and_reset().begin();

        root_allocator.reset();
        texture_allocator.reset();

        let enc = active_cmd_frame.bump_encoder();

        if let Some(gbuffer) = gbuffer {
            gbuffer.destroy(&device);
        }
        gbuffer = Some(GBuffer::create(
            &enc,
            window_extent,
            &texture_allocator,
            &texture_descriptor_heap,
        ));
        enc.bind_descriptor_heap(&texture_descriptor_heap);

        enc.barrier(
            rhi::StageFlags::MESH,
            rhi::AccessFlags::SHADER_READ,
            rhi::StageFlags::COMPUTE,
            rhi::AccessFlags::SHADER_WRITE,
        );
        simulation_pass(
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
        gbuffer_pass(
            &enc,
            &gbuffer_pipeline,
            gbuffer.as_ref().unwrap(),
            &root_allocator,
            &simulation_buffer.0,
            &camera,
        );
        enc.swapchain_frame_acquire_barrier(&swapchain_frame);
        shading_pass(
            &enc,
            &shading_pipeline,
            window_extent,
            &swapchain_frame,
            &root_allocator,
            &camera,
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
    if let Some(gbuffer) = gbuffer {
        gbuffer.destroy(&device);
    }

    swapchain.wait_last_blocking(&timeline);
    device.wait_idle_blocking();
    device.destroy_pipeline(gbuffer_pipeline);
    device.destroy_pipeline(simulation_pipeline);
    device.destroy_pipeline(shading_pipeline);
    device.destroy_buffer(simulation_buffer.0);
    device.destroy_memory_block(simulation_buffer.1);
    device.destroy_descriptor_heap(texture_descriptor_heap);
    swapchain.destroy();
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

fn make_simulation_pipeline(device: &rhi::Device, compile_service: &compile::CompileService) -> rhi::Pipeline {
    let cs = compile_program(
        compile_service,
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/simulation.slang"),
        "compute_main",
        compile::Stage::Compute,
    );
    device
        .create_compute_pipeline(rhi::ShaderStage::new(&cs, "compute_main"))
        .unwrap()
}

fn make_gbuffer_pipeline(device: &rhi::Device, compile_service: &compile::CompileService) -> rhi::Pipeline {
    let ms = compile_program(
        compile_service,
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/gbuffer.slang"),
        "mesh_main",
        compile::Stage::Mesh,
    );
    let fs = compile_program(
        compile_service,
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/gbuffer.slang"),
        "fragment_main",
        compile::Stage::Fragment,
    );
    device
        .create_raster_pipeline(rhi::RasterPipelineDesc {
            primitive: rhi::PrimitiveStages::Mesh {
                task: None,
                mesh: rhi::ShaderStage::new(&ms, "mesh_main"),
            },
            fragment: rhi::ShaderStage::new(&fs, "fragment_main"),
            rasterizer: rhi::RasterizerState {
                cull_mode: rhi::CullMode::Back,
                front_face: rhi::FrontFace::CounterClockwise,
                ..Default::default()
            },
            color_targets: &[
                rhi::ColorTargetDesc::default_with_format(rhi::PixelFormat::RGBA8Unorm),
                rhi::ColorTargetDesc::default_with_format(rhi::PixelFormat::RGBA16Float),
            ],
            depth_format: rhi::PixelFormat::D32Float,
            stencil_format: rhi::PixelFormat::Undefined,
        })
        .expect("failed to create pipeline")
}

fn make_shading_pipeline(device: &rhi::Device, compile_service: &compile::CompileService) -> rhi::Pipeline {
    let vs = compile_program(
        compile_service,
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/deferred_shading.slang"),
        "vertex_main",
        compile::Stage::Vertex,
    );
    let fs = compile_program(
        compile_service,
        Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/assets/deferred_shading.slang"),
        "fragment_main",
        compile::Stage::Fragment,
    );
    device
        .create_raster_pipeline(rhi::RasterPipelineDesc {
            primitive: rhi::PrimitiveStages::Vertex {
                vertex: rhi::ShaderStage::new(&vs, "vertex_main"),
            },
            fragment: rhi::ShaderStage::new(&fs, "fragment_main"),
            rasterizer: rhi::RasterizerState { ..Default::default() },
            color_targets: &[rhi::ColorTargetDesc::default_with_format(rhi::PixelFormat::RGBA8Srgb)],
            depth_format: rhi::PixelFormat::Undefined,
            stencil_format: rhi::PixelFormat::Undefined,
        })
        .expect("failed to create pipeline")
}

fn simulation_pass(
    enc: &rhi::CmdEncoder<'_, rhi::state::Active>,
    pipeline: &rhi::Pipeline,
    root_allocator: &RootBumpAllocator,
    buffer: &rhi::Buffer,
    delta_seconds: f32,
) {
    let _label = rhi::LabelScope::new(enc, "simulation", rhi::LabelColor::BLUE);
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

fn gbuffer_pass(
    enc: &rhi::CmdEncoder<'_, rhi::state::Active>,
    pipeline: &rhi::Pipeline,
    gbuffer: &GBuffer,
    root_allocator: &RootBumpAllocator,
    simulation_buffer: &rhi::Buffer,
    camera: &CameraSettings,
) {
    let _label = rhi::LabelScope::new(enc, "gbuffer", rhi::LabelColor::CYAN);
    enc.begin_rendering(rhi::RenderingDesc {
        area: rhi::Rect2D {
            extent: gbuffer.extent,
            offset: rhi::Offset2D::default(),
        },
        colors: &[
            rhi::ColorTarget {
                view: gbuffer.albedo_view,
                load: rhi::LoadOp::Clear,
                store: rhi::StoreOp::Store,
                clear: rhi::ClearColor::EMPTY,
            },
            rhi::ColorTarget {
                view: gbuffer.normal_roughness_view,
                load: rhi::LoadOp::Clear,
                store: rhi::StoreOp::Store,
                clear: rhi::ClearColor::EMPTY,
            },
        ],
        depth: Some(rhi::DepthTarget {
            view: gbuffer.depth_view,
            load: rhi::LoadOp::Clear,
            store: rhi::StoreOp::Store,
            clear: 1.0,
        }),
        stencil: None,
    });
    enc.bind_pipeline(&pipeline);
    let aspect_ratio = gbuffer.extent.width.max(1) as f32 / gbuffer.extent.height.max(1) as f32;
    let view = glam::camera::rh::view::look_at_mat4(camera.position, camera.target, camera.up);
    let projection =
        glam::camera::rh::proj::vulkan::perspective(camera.vertical_fov, aspect_ratio, camera.near, camera.far);
    let orientation = Quat::IDENTITY;
    let view_projection = (projection * view).to_cols_array_2d();
    let mut root = root_allocator.allocate();
    *root.as_mut() = GBufferRoot {
        view_projection,
        orientation: [orientation.x, orientation.y, orientation.z, orientation.w],
        particles: simulation_buffer.device_address,
        _padding: [0; 2],
    };
    enc.emit_root(&root);
    enc.set_viewport(rhi::Viewport {
        width: gbuffer.extent.width as f32,
        height: gbuffer.extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
        ..Default::default()
    });
    enc.set_scissor(rhi::Rect2D {
        extent: gbuffer.extent,
        offset: rhi::Offset2D::default(),
    });
    enc.set_stencil(rhi::DepthStencilState {
        depth: rhi::DepthState {
            test: true,
            write: true,
            compare: rhi::CompareOp::Less,
            ..Default::default()
        },
        ..Default::default()
    });
    enc.draw_mesh(PARTICLE_COUNTS as u32, 1, 1);
    enc.end_rendering();
}

fn shading_pass(
    enc: &rhi::CmdEncoder<'_, rhi::state::Active>,
    pipeline: &rhi::Pipeline,
    extent: rhi::Extent2D,
    swapchain_frame: &rhi::SwapchainFrame,
    root_allocator: &RootBumpAllocator,
    camera: &CameraSettings,
) {
    let _label = rhi::LabelScope::new(enc, "shading", rhi::LabelColor::GREEN);
    enc.begin_rendering(rhi::RenderingDesc {
        area: rhi::Rect2D {
            extent,
            offset: rhi::Offset2D::default(),
        },
        colors: &[rhi::ColorTarget {
            view: TargetView::from_swapchain_frame(swapchain_frame),
            load: rhi::LoadOp::Clear,
            store: rhi::StoreOp::Store,
            clear: rhi::ClearColor::EMPTY,
        }],
        depth: None,
        stencil: None,
    });
    enc.bind_pipeline(&pipeline);
    let mut root = root_allocator.allocate();
    let aspect_ratio = extent.width.max(1) as f32 / extent.height.max(1) as f32;
    let tan_half_fov = (camera.vertical_fov * 0.5).tan();
    let forward = (camera.target - camera.position).normalize();
    let right = forward.cross(camera.up).normalize();
    let camera_up = right.cross(forward).normalize();
    *root.as_mut() = ShadingRoot {
        camera_position: [camera.position.x, camera.position.y, camera.position.z, 1.0],
        ray_center: [forward.x, forward.y, forward.z, 0.0],
        ray_horizontal: [
            right.x * tan_half_fov * aspect_ratio,
            right.y * tan_half_fov * aspect_ratio,
            right.z * tan_half_fov * aspect_ratio,
            0.0,
        ],
        ray_vertical: [
            camera_up.x * tan_half_fov,
            camera_up.y * tan_half_fov,
            camera_up.z * tan_half_fov,
            0.0,
        ],
        depth_linearize: [
            camera.near * camera.far / (camera.far - camera.near),
            camera.far / (camera.far - camera.near),
        ],
        gbuffer_pixel_scale: [1.0, 1.0],
    };
    enc.emit_root(&root);
    enc.set_viewport(rhi::Viewport {
        width: extent.width as f32,
        height: extent.height as f32,
        ..Default::default()
    });
    enc.set_scissor(rhi::Rect2D {
        extent,
        offset: rhi::Offset2D::default(),
    });
    enc.set_stencil(rhi::DepthStencilState::default());
    enc.draw(3, 1, 0, 0);
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
struct GBufferRoot {
    view_projection: [[f32; 4]; 4],
    orientation: [f32; 4],
    particles: u64,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ShadingRoot {
    camera_position: [f32; 4],
    ray_center: [f32; 4],
    ray_horizontal: [f32; 4],
    ray_vertical: [f32; 4],
    depth_linearize: [f32; 2],
    gbuffer_pixel_scale: [f32; 2],
}

struct CameraSettings {
    position: Vec3,
    target: Vec3,
    up: Vec3,
    vertical_fov: f32,
    near: f32,
    far: f32,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 4.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            vertical_fov: 60.0_f32.to_radians(),
            near: 0.1,
            far: 100.0,
        }
    }
}

impl CameraSettings {
    fn handle_char(&mut self, character: char) {
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(self.up).normalize();

        match character.to_ascii_lowercase() {
            'w' => self.translate(forward * CAMERA_MOVE_STEP),
            's' => self.translate(-forward * CAMERA_MOVE_STEP),
            'a' => self.translate(-right * CAMERA_MOVE_STEP),
            'd' => self.translate(right * CAMERA_MOVE_STEP),
            'q' => self.translate(-self.up * CAMERA_MOVE_STEP),
            'e' => self.translate(self.up * CAMERA_MOVE_STEP),
            '+' | '=' => {
                self.vertical_fov = (self.vertical_fov + CAMERA_FOV_STEP).min(120.0_f32.to_radians());
            }
            '-' => {
                self.vertical_fov = (self.vertical_fov - CAMERA_FOV_STEP).max(20.0_f32.to_radians());
            }
            'j' => self.near = (self.near - CAMERA_CLIP_STEP).max(0.05),
            'k' => self.near = (self.near + CAMERA_CLIP_STEP).min(self.far - 1.0),
            'n' => self.far = (self.far - 1.0).max(self.near + 1.0),
            'm' => self.far += 1.0,
            'r' => *self = Self::default(),
            _ => {}
        }
    }

    fn translate(&mut self, offset: Vec3) {
        self.position += offset;
        self.target += offset;
    }
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

struct TextureAllocator<'a> {
    device: &'a rhi::Device,
    memory: rhi::MemoryBlock,
    state: RefCell<TextureAllocatorState>,
}

#[derive(Clone, Copy)]
struct TextureAllocatorState {
    cursor: usize,
    capacity: usize,
}

impl<'a> TextureAllocator<'a> {
    pub fn new(capacity: usize, device: &'a rhi::Device) -> Self {
        let memory_type = device
            .query_texture_memory_requirements(
                &rhi::TextureDesc::new_default_2d(1, 1)
                    .format(rhi::PixelFormat::RGBA8Unorm)
                    .usage(rhi::TextureUsageFlags::SAMPLED | rhi::TextureUsageFlags::COLOR_ATTACHMENT),
            )
            .memory_type_bits
            | device
                .query_texture_memory_requirements(
                    &rhi::TextureDesc::new_default_2d(1, 1)
                        .format(rhi::PixelFormat::RGBA16Float)
                        .usage(rhi::TextureUsageFlags::SAMPLED | rhi::TextureUsageFlags::COLOR_ATTACHMENT),
                )
                .memory_type_bits
            | device
                .query_texture_memory_requirements(&rhi::TextureDesc::new_default_depth(1, 1))
                .memory_type_bits;

        let memory = device
            .create_memory_block(
                rhi::MemoryDomain::DeviceLocal,
                capacity,
                memory_type,
                "texture_allocator_back_memory",
            )
            .unwrap();

        Self {
            device,
            memory,
            state: RefCell::new(TextureAllocatorState { cursor: 0, capacity }),
        }
    }

    pub fn allocate(&self, desc: &rhi::TextureDesc, name: impl AsRef<str>) -> rhi::Texture {
        let state = self.state.borrow().clone();

        let requirements = self.device.query_texture_memory_requirements(desc);

        let start = align_up(state.cursor, requirements.device_alignment as usize);
        let end = start + requirements.device_size as usize;
        assert!(end <= state.capacity);

        let texture = self
            .device
            .create_texture_placed(&self.memory, start, desc, name)
            .unwrap();
        self.state.borrow_mut().cursor = end;
        texture
    }

    pub fn reset(&self) {
        self.state.borrow_mut().cursor = 0;
    }
}

impl Drop for TextureAllocator<'_> {
    fn drop(&mut self) {
        self.device.destroy_memory_block(self.memory);
    }
}

struct GBuffer {
    extent: rhi::Extent2D,
    albedo: rhi::Texture,
    normal_roughness: rhi::Texture,
    depth: rhi::Texture,
    albedo_view: rhi::TargetView,
    normal_roughness_view: rhi::TargetView,
    depth_view: rhi::TargetView,
}

impl GBuffer {
    pub fn create(
        enc: &rhi::CmdEncoder<'_, rhi::state::Active>,
        extent: rhi::Extent2D,
        allocator: &TextureAllocator<'_>,
        descriptor_heap: &rhi::DescriptorHeap,
    ) -> Self {
        let albedo_desc = rhi::TextureDesc::new_default_2d(extent.width, extent.height)
            .format(rhi::PixelFormat::RGBA8Unorm)
            .usage(rhi::TextureUsageFlags::COLOR_ATTACHMENT | rhi::TextureUsageFlags::SAMPLED);
        let normal_roughenss_desc = rhi::TextureDesc::new_default_2d(extent.width, extent.height)
            .format(rhi::PixelFormat::RGBA16Float)
            .usage(rhi::TextureUsageFlags::COLOR_ATTACHMENT | rhi::TextureUsageFlags::SAMPLED);
        let depth_desc = rhi::TextureDesc::new_default_2d(extent.width, extent.height)
            .format(rhi::PixelFormat::D32Float)
            .usage(rhi::TextureUsageFlags::DEPTH_STENCIL_ATTACHMENT | rhi::TextureUsageFlags::SAMPLED);

        let albedo = allocator.allocate(&albedo_desc, "albedo");
        let normal_roughness = allocator.allocate(&normal_roughenss_desc, "normal_roughness");
        let depth = allocator.allocate(&depth_desc, "depth");

        enc.unifiy_texture_layout(&albedo, rhi::TextureAspect::Color);
        enc.unifiy_texture_layout(&normal_roughness, rhi::TextureAspect::Color);
        enc.unifiy_texture_layout(&depth, rhi::TextureAspect::Depth);

        let albedo_view_desc = rhi::TextureViewDesc::new_default_range(&albedo, rhi::TextureAspect::Color);
        let normal_roughness_view_desc =
            rhi::TextureViewDesc::new_default_range(&normal_roughness, rhi::TextureAspect::Color);
        let depth_view_desc = rhi::TextureViewDesc::new_default_range(&depth, rhi::TextureAspect::Depth);

        let device = allocator.device;
        let albedo_view = device.create_target_view(&albedo, &albedo_view_desc);
        let normal_roughness_view = device.create_target_view(&normal_roughness, &normal_roughness_view_desc);
        let depth_view = device.create_target_view(&depth, &depth_view_desc);

        descriptor_heap
            .write_texture_descriptor(0, &albedo_view_desc, rhi::TextureDescriptorKind::Sampled)
            .unwrap();
        descriptor_heap
            .write_texture_descriptor(1, &normal_roughness_view_desc, rhi::TextureDescriptorKind::Sampled)
            .unwrap();
        descriptor_heap
            .write_texture_descriptor(2, &depth_view_desc, rhi::TextureDescriptorKind::Sampled)
            .unwrap();

        Self {
            extent,
            albedo,
            normal_roughness,
            depth,
            albedo_view,
            normal_roughness_view,
            depth_view,
        }
    }

    pub fn destroy(self, device: &rhi::Device) {
        device.destory_target_view(self.albedo_view);
        device.destory_target_view(self.normal_roughness_view);
        device.destory_target_view(self.depth_view);
        device.destory_texture(self.albedo);
        device.destory_texture(self.normal_roughness);
        device.destory_texture(self.depth);
    }
}

fn compile_program(
    service: &compile::CompileService,
    path: PathBuf,
    entry_point: &str,
    stage: compile::Stage,
) -> Vec<u32> {
    let desc = compile::CompileDesc {
        source: compile::FileSource::File(path),
        entry: compile::EntryPoint {
            name: entry_point.to_string(),
            stage,
        },
        matrix_layout: compile::MatrixLayout::ColumnMajor,
        optimization: compile::OptimizationLevel::Default,
        debug_info: None,
        specialization: None,
    };

    let codes = service.compile(desc).unwrap();
    codes[0].to_spirv_byte_code().unwrap()
}
