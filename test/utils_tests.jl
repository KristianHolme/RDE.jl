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
    dx = 2π / 512
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
