@test RDE.periodic_simpsons_rule(zeros(16), 0.5) ≈ 0.0

@test RDE.energy_balance(vcat(zeros(128), ones(128)), RDEParam{Float64}()) ≈ 0.0

import JLD2: @load
@load "test_data/shocks.jld2" u1 u2 u3 u4

dx = 2π/512
@test RDE.count_shocks(u1,dx) == 1
@test RDE.count_shocks(u2,dx) == 2
@test RDE.count_shocks(u3,dx) == 3
@test RDE.count_shocks(u4,dx) == 4

@test begin
    u = zeros(512)
    u[100] = 1.0

    calc_inds = RDE.shock_indices(u, 1.0)
    findall(calc_inds) == [100]
end

@testset "apply_periodic_shift!" begin
    source = [1, 2, 3, 4, 5]
    target = similar(source)
    
    # Test positive shift
    RDE.apply_periodic_shift!(target, source, 2)
    @test target == [3, 4, 5, 1, 2]
    
    # Test negative shift
    RDE.apply_periodic_shift!(target, source, -1)
    @test target == [5, 1, 2, 3, 4]
    
    # Test zero shift
    RDE.apply_periodic_shift!(target, source, 0)
    @test target == source
    
    # Test shift larger than array length
    RDE.apply_periodic_shift!(target, source, 7)
    @test target == [3, 4, 5, 1, 2]  # Same as shift by 2
end