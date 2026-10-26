struct Entity {
    float3 position;
    float3 velocity;
};

struct SimulationRoot {
    Entity* entities;
    float delta_seconds;
};

struct PushRoot {
    uint64_t frame;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SimulationRoot> root;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 dtid: SV_DispatchThreadID) {
    uint object_index = dtid.x;

    Entity entity = root.entities[object_index];
    entity.position += entity.velocity * root.delta_seconds;

    root.entities[object_index] = entity;
}
