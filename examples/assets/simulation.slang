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
    uint object_index     = dtid.x;
    float dt              = min(root.delta_seconds, 1.0 / 30.0);
                  
    const float R         = 6.0;
    const float WALL_K    = 2.0;
    const float MAX_SPEED = 10.0;

    Particle entity = root.particles[object_index];
    float3 pos      = entity.position;
    float3 vel      = entity.velocity;

    float id[6];
    uint h = object_index * 747796405u + 2891336453u;
    [unroll]
    for (int i = 0; i < 6; i++) {
        h      = h * 747796405u + 2891336453u;
        uint w = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
        w      = (w >> 22u) ^ w;
        id[i]  = float(w) * (1.0 / 4294967295.0);
    }

    float shell       = R * (0.2 + 0.7 * id[0]);
     
    float orbit_speed = 1.0 + 6.0 * id[1] * id[1];
     
    float response    = 0.8 + 2.5 * id[2];
     
    float spin_dir    = (id[4] > 0.5) ? 1.0 : -1.0;
    
    float3 axis       = normalize(float3(id[3] * 2.0 - 1.0, 0.4 + id[4], id[5] * 2.0 - 1.0) + 1e-4);
    float phase       = id[5] * 6.2831853;

    uint s = object_index * 9781u + 6271u;
    s ^= asuint(pos.x) * 0x9E3779B1u;
    s ^= asuint(pos.y) * 0x85EBCA6Bu;
    s ^= asuint(pos.z) * 0xC2B2AE35u;
    float r[4];
    [unroll]
    for (int j = 0; j < 4; j++) {
        s      = s * 747796405u + 2891336453u;
        uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
        w      = (w >> 22u) ^ w;
        r[j]   = float(w) * (1.0 / 4294967295.0);
    }
    float3 rand_vec = float3(r[0], r[1], r[2]) * 2.0 - 1.0;

    float dist    = max(length(pos), 1e-3);
    float3 radial = pos / dist;

    float3 tangent = cross(axis, radial);
    tangent /= max(length(tangent), 1e-3);

    float3 desired = tangent * (orbit_speed * spin_dir);
    desired += radial * ((shell - dist) * 0.8);
    

    float3 turb = float3(
                      sin(pos.y * 0.6 + phase),
                      sin(pos.z * 0.5 + phase * 1.7),
                      sin(pos.x * 0.7 + phase * 0.6)) *
                  (0.8 + 1.5 * id[2]);
    desired += turb;

    if (dist > R) {
        desired -= radial * ((dist - R) * WALL_K);
    }

    vel += (desired - vel) * min(response * dt, 1.0);
    vel += rand_vec * (1.2 * dt * orbit_speed);

    if (r[3] < 0.0015) {
        vel += normalize(rand_vec + 1e-5) * (orbit_speed * 2.0);
    }

    float speed = length(vel);
    if (speed > MAX_SPEED) {
        vel *= MAX_SPEED / speed;
    }

    pos += vel * dt;

    float hard_r   = R * 1.15;
    float new_dist = length(pos);
    if (new_dist > hard_r) {
        float3 n = pos / new_dist;
        pos      = n * hard_r;
        vel -= n * max(dot(vel, n), 0.0) * 1.5;
    }

    entity.position              = pos;
    entity.velocity              = vel;
    root.particles[object_index] = entity;
}
