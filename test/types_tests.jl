@test begin
    params = RDEParam{Float32}(;N=16)
    d = RDE.create_dealiasing_vector(params)
    typeof(d) == Vector{Float32}
end

@test begin
    params = RDEParam{Float32}(;N=16)
    ik, k2 = RDE.create_spectral_derivative_arrays(params)
    @debug typeof(ik), typeof(k2)
    typeof(ik) == Vector{ComplexF32} && typeof(k2) == Vector{Float32}
end

@test begin
    params = RDEParam{Float32}(;N=16)
    cache = RDE.PseudospectralRDECache{Float32}(params, dealias=true)
    typeof(cache) == RDE.PseudospectralRDECache{Float32}
end

@test begin
    params = RDEParam{Float32}(;N=16)
    cache = RDE.FDRDECache{Float32}(params, Float32(2π/16.0))
    typeof(cache) == RDE.FDRDECache{Float32}
end
