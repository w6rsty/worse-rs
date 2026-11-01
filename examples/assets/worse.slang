#define ROOT_ARGUMENT(type, name) \
    [[vk::binding(0, 0)]]         \
    ConstantBuffer<type> name;

T get_texture<T : IOpaqueDescriptor>(uint index) {
    return getDescriptorFromHandle(DescriptorHandle<T>(uint2(index, 0)));
}

T get_sampler<T : IOpaqueDescriptor = SamplerState>(uint index) {
    return getDescriptorFromHandle(DescriptorHandle<T>(uint2(index, 0)));
}
