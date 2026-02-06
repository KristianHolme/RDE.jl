using Test
using RDE

@testitem "spatial kernel normalization" begin
    kernel = RDE.build_spatial_kernel(8, Float32)
    @test length(kernel) == 9
    @test isapprox(sum(kernel), 1.0f0; atol = 1.0f-6)
end

@testitem "spatial smoothing reduces boundary jump" begin
    N = 64
    raw = vcat(fill(0.0f0, N ÷ 2), fill(1.0f0, N ÷ 2))
    kernel = RDE.build_spatial_kernel(9, Float32)
    half = (length(kernel) - 1) ÷ 2
    scratch = zeros(Float32, N + 2 * half)
    smoothed = copy(raw)
    RDE.smooth_spatial!(smoothed, scratch, kernel)

    boundary_index = N ÷ 2 + 1
    jump_before = abs(raw[boundary_index] - raw[boundary_index - 1])
    jump_after = abs(smoothed[boundary_index] - smoothed[boundary_index - 1])
    @test jump_after < jump_before
end

@testitem "apply_spatial_smoothing! reduces boundary jump in place" begin
    params = RDEParam{Float32}(N = 64, tmax = 1.0f0)
    prob = RDEProblem(params; control_shift_strategy = ZeroControlShift())
    cache = prob.method.cache

    RDE.set_spatial_control_smoothing!(cache, 9)
    cache.u_p_current .= vcat(fill(0.0f0, params.N ÷ 2), fill(1.0f0, params.N ÷ 2))

    boundary_index = params.N ÷ 2 + 1
    jump_before = abs(cache.u_p_current[boundary_index] - cache.u_p_current[boundary_index - 1])

    RDE.apply_spatial_smoothing!(cache.u_p_current, cache)

    jump_after = abs(cache.u_p_current[boundary_index] - cache.u_p_current[boundary_index - 1])
    @test jump_after < jump_before
end

@testitem "apply_spatial_smoothing! is no-op when spatial_kernel_width is 0" begin
    params = RDEParam{Float32}(N = 32, tmax = 1.0f0)
    prob = RDEProblem(params; control_shift_strategy = ZeroControlShift())
    cache = prob.method.cache
    @test cache.spatial_kernel_width == 0

    cache.u_p_current .= vcat(fill(0.0f0, params.N ÷ 2), fill(1.0f0, params.N ÷ 2))
    expected = copy(cache.u_p_current)

    RDE.apply_spatial_smoothing!(cache.u_p_current, cache)

    @test cache.u_p_current == expected
end
