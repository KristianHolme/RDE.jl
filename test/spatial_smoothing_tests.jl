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
    N = 64
    v = vcat(fill(0.0f0, N ÷ 2), fill(1.0f0, N ÷ 2))
    kernel = RDE.build_spatial_kernel(9, Float32)
    half = (length(kernel) - 1) ÷ 2
    scratch = zeros(Float32, N + 2 * half)
    boundary_index = N ÷ 2 + 1
    jump_before = abs(v[boundary_index] - v[boundary_index - 1])
    RDE.apply_spatial_smoothing!(v, kernel, scratch)
    jump_after = abs(v[boundary_index] - v[boundary_index - 1])
    @test jump_after < jump_before
end

@testitem "apply_spatial_smoothing! is no-op when kernel is empty" begin
    N = 32
    v = vcat(fill(0.0f0, N ÷ 2), fill(1.0f0, N ÷ 2))
    expected = copy(v)
    RDE.apply_spatial_smoothing!(v, Float32[], Float32[])
    @test v == expected
end
