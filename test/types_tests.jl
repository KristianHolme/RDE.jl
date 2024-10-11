@test begin
    d = RDE.create_dealiasing_vector(16, Float32)
    typeof(d) == Vector{Float32}
end

@test begin
    ik, k2 = RDE.create_spectral_derivative_arrays(16, Float32)
    typeof(ik) == Vector{ComplexF32} && typeof(k2) == Vector{Float32}
end

@test begin
    env = RDEEnv{Float32}()
    true
end