static const float2 positions[3] = {
    float2(0.0f, -0.5f),
    float2(0.5f, 0.5f),
    float2(-0.5f, 0.5f),
};

static const float3 colors[3] = {
    float3(1.0f, 0.0f, 0.0f),
    float3(0.0f, 1.0f, 0.0f),
    float3(0.0f, 0.0f, 1.0f),
};

struct VertexOutput {
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vs_main(uint vertex_index: SV_VertexID) {
    VertexOutput output;
    output.position = float4(positions[vertex_index], 0.0f, 1.0f);
    return output;
}

[shader("fragment")]
void fs_main(VertexOutput input) {
    // depth only
}
