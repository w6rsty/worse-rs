struct Particle {
    float3 position;
    float3 velocity;
};

struct BasePassRoot {
    float4x4 view_projection;
    float4 orientation;
    Particle* particles;
    uint2 padding;
};

[[vk::binding(0, 0)]]
ConstantBuffer<BasePassRoot> root;

float3 rotate_by_quaternion(float3 value, float4 orientation) {
    float3 axis = orientation.xyz;
    return value + 2.0f * cross(axis, cross(axis, value) + orientation.w * value);
}

struct VertexOutput {
    float4 position : SV_Position;
    nointerpolation float velocity;
};

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(24, 1, 1)]
void mesh_main(
    uint3 group_id: SV_GroupID,
    uint thread_index: SV_GroupIndex,
    out vertices VertexOutput vertices[24],
    out indices uint3 triangles[12]
) {
    SetMeshOutputCounts(24, 12);

    Particle particle = root.particles[group_id.x];

    if (thread_index < 24) {
        float3 cube_position = float3(
            float((0x6fc6c0u >> thread_index) & 1u),
            float((0xc3f06cu >> thread_index) & 1u),
            float((0xf66c06u >> thread_index) & 1u)) * 0.02f - 0.01f;
        float3 world_position = rotate_by_quaternion(particle.position + cube_position, root.orientation);
        vertices[thread_index].position = mul(root.view_projection, float4(world_position, 1.0f));
        vertices[thread_index].velocity = length(particle.velocity);
    }

    if (thread_index < 12) {
        uint base_vertex = (thread_index >> 1u) * 4u;
        triangles[thread_index] = (thread_index & 1u) == 0u
            ? uint3(base_vertex, base_vertex + 1u, base_vertex + 2u)
            : uint3(base_vertex + 2u, base_vertex + 3u, base_vertex);
    }
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target0 {
    float wavelength = lerp(380.0f, 780.0f, saturate(input.velocity / 0.15f));
    float3 color;

    if (wavelength < 440.0f) {
        color = float3(-(wavelength - 440.0f) / 60.0f, 0.0f, 1.0f);
    } else if (wavelength < 490.0f) {
        color = float3(0.0f, (wavelength - 440.0f) / 50.0f, 1.0f);
    } else if (wavelength < 510.0f) {
        color = float3(0.0f, 1.0f, -(wavelength - 510.0f) / 20.0f);
    } else if (wavelength < 580.0f) {
        color = float3((wavelength - 510.0f) / 70.0f, 1.0f, 0.0f);
    } else if (wavelength < 645.0f) {
        color = float3(1.0f, -(wavelength - 645.0f) / 65.0f, 0.0f);
    } else {
        color = float3(1.0f, 0.0f, 0.0f);
    }

    float intensity = 1.0f;
    if (wavelength < 420.0f) {
        intensity = 0.3f + 0.7f * (wavelength - 380.0f) / 40.0f;
    } else if (wavelength > 700.0f) {
        intensity = 0.3f + 0.7f * (780.0f - wavelength) / 80.0f;
    }

    return float4(color * intensity, 1.0f);
}
