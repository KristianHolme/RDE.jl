using Test
using RDE
import JLD2: @load

@testitem "periodic_simpsons_rule" begin
    @test RDE.periodic_simpsons_rule(zeros(16), 0.5) ≈ 0.0
end

@testitem "energy_balance" begin
    @test RDE.energy_balance(vcat(zeros(128), ones(128)), RDEParam{Float64}()) ≈ 0.0
end

@testitem "get_dx" begin
    prob = RDEProblem(RDEParam(N = 512, L = 2π))
    @test RDE.get_dx(prob) ≈ 2π / 512
end

@testitem "count_shocks" begin
    import JLD2: @load
    @load "test_data/shocks.jld2" u1 u2 u3 u4
    dx = Float32(2π / 512)
    @test RDE.count_shocks(u1, dx) == 1
    @test RDE.count_shocks(u2, dx) == 2
    @test RDE.count_shocks(u3, dx) == 3
    @test RDE.count_shocks(u4, dx) == 4
end

@testitem "shock_indices" begin
    u = zeros(512)
    u[100] = 1.0

    calc_inds = RDE.shock_indices(u, 1.0)
    @test calc_inds == [100]
end

@testitem "turbo_maximum" begin
    arr = [1, 5, 2, 8, 3]
    @test RDE.turbo_maximum(arr) == maximum(arr)
    @test RDE.turbo_maximum(arr) == 8


    u = rand(512)
    @test RDE.turbo_maximum(u) == maximum(u)
    @test RDE.turbo_maximum(u) == maximum(u)

    u = rand(2^16)
    @test RDE.turbo_maximum(u) == maximum(u)
    @test RDE.turbo_maximum(u) == maximum(u)

    u = rand(2^16 - 1)
    @test RDE.turbo_maximum(u) == maximum(u)
    @test RDE.turbo_maximum(u) == maximum(u)
end

@testitem "turbo_minimum" begin
    arr = [1, 5, 2, 8, 3]
    @test RDE.turbo_minimum(arr) == minimum(arr)
    @test RDE.turbo_minimum(arr) == 1

    u = rand(512)
    @test RDE.turbo_minimum(u) == minimum(u)
    @test RDE.turbo_minimum(u) == minimum(u)

    u = rand(2^16)
    @test RDE.turbo_minimum(u) == minimum(u)
    @test RDE.turbo_minimum(u) == minimum(u)
end

@testitem "turbo_maximum_abs" begin
    arr = [-3, 1, -7, 4]
    @test RDE.turbo_maximum_abs(arr) == maximum(abs, arr)
    @test RDE.turbo_maximum_abs(arr) == 7

    u = rand(512)
    @test RDE.turbo_maximum_abs(u) == maximum(abs, u)
    @test RDE.turbo_maximum_abs(u) == maximum(abs, u)

    u = rand(2^16)
    @test RDE.turbo_maximum_abs(u) == maximum(abs, u)
    @test RDE.turbo_maximum_abs(u) == maximum(abs, u)

    u = rand(2^16 - 1)
    @test RDE.turbo_maximum_abs(u) == maximum(abs, u)
    @test RDE.turbo_maximum_abs(u) == maximum(abs, u)
end

@testitem "turbo_max_sum" begin
    u = [1, 2, 3]
    v = [4, 5, 6]
    @test RDE.turbo_max_sum(u, v) == maximum(u .+ v)
    @test RDE.turbo_max_sum(u, v) == 9

    u = rand(512)
    v = rand(512)
    @test RDE.turbo_max_sum(u, v) == maximum(u .+ v)

    u = rand(2^16)
    v = rand(2^16)
    @test RDE.turbo_max_sum(u, v) == maximum(u .+ v)

    u = rand(2^16 - 1)
    v = rand(2^16 - 1)
    @test RDE.turbo_max_sum(u, v) == maximum(u .+ v)
end

@testitem "turbo functions broadcast over vector-of-vectors" begin
    # Same-length inner vectors
    arrs = [rand(128) for _ in 1:8]
    @test all(RDE.turbo_maximum.(arrs) .== maximum.(arrs))
    @test all(RDE.turbo_minimum.(arrs) .== minimum.(arrs))
    @test all(RDE.turbo_maximum_abs.(arrs) .== map(a -> maximum(abs, a), arrs))
    @test all(RDE.turbo_extrema.(arrs) .== extrema.(arrs))

    us = [rand(128) for _ in 1:8]
    vs = [rand(128) for _ in 1:8]
    @test all(RDE.turbo_max_sum.(us, vs) .== map((a, b) -> maximum(a .+ b), us, vs))

    # Varying-length inner vectors
    arrs2 = [rand(17), rand(64), rand(65), rand(1023)]
    @test all(RDE.turbo_maximum.(arrs2) .== maximum.(arrs2))
    @test all(RDE.turbo_minimum.(arrs2) .== minimum.(arrs2))
    @test all(RDE.turbo_maximum_abs.(arrs2) .== map(a -> maximum(abs, a), arrs2))
    @test all(RDE.turbo_extrema.(arrs2) .== extrema.(arrs2))

    us2 = [rand(31), rand(255), rand(1024)]
    vs2 = [rand(31), rand(255), rand(1024)]
    @test all(RDE.turbo_max_sum.(us2, vs2) .== map((a, b) -> maximum(a .+ b), us2, vs2))
end
