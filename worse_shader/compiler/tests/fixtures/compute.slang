#include "shared.slang"
RWStructuredBuffer<float> output;
[shader("compute")]
[numthreads(4, 1, 1)]
void dispatch_main(uint3 id : SV_DispatchThreadID) { output[id.x] = scaleValue(); }
