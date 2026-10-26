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

static const float3 barycentrics[3] = {
    float3(1.0f, 0.0f, 0.0f),
    float3(0.0f, 1.0f, 0.0f),
    float3(0.0f, 0.0f, 1.0f),
};

struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
    float3 barycentric : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertex_main(uint vertex_index: SV_VertexID) {
    VertexOutput output;
    output.position    = float4(positions[vertex_index], 0.0f, 1.0f);
    output.color       = colors[vertex_index];
    output.barycentric = barycentrics[vertex_index];
    
    return output;
}

static float line_width  = 5.0f;
static float4 line_color = float4(1.0f, 1.0f, 0.0f, 1.0f);

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target0 {
    float3 d          = fwidth(input.barycentric);
    float3 a3         = smoothstep(float3(0.0f), d * line_width, input.barycentric);
    float edge_factor = min(min(a3.x, a3.y), a3.z);

    float3 final_color = lerp(line_color.rgb, input.color, edge_factor);

    float final_alpha = lerp(line_color.a, 1.0f, edge_factor);

    return float4(final_color, final_alpha);
}
