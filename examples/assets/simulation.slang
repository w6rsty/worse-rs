struct Particle {
    float3 position;
    float3 velocity;
};

struct SimulationRoot {
    Particle* particles;
    float delta_seconds;
};

static const uint PARTICLE_COUNTS = 1024;

[[vk::binding(0, 0)]]
ConstantBuffer<SimulationRoot> root;

[shader("compute")]
[numthreads(PARTICLE_COUNTS, 1, 1)]
void compute_main(uint3 dtid: SV_DispatchThreadID) {
    uint object_index = dtid.x;

    Particle entity = root.particles[object_index];
    entity.position += entity.velocity * root.delta_seconds;

    root.particles[object_index] = entity;
}
