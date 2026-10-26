T get_texture<T : IOpaqueDescriptor>(uint index) {
    return getDescriptorFromHandle(DescriptorHandle<T>(uint2(index, 0)));
}

T get_sampler<T : IOpaqueDescriptor = SamplerState>(uint index) {
    return getDescriptorFromHandle(DescriptorHandle<T>(uint2(index, 0)));
}
