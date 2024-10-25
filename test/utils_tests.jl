@test RDE.periodic_simpsons_rule(zeros(16), 0.5) ≈ 0.0

@test RDE.energy_balance(vcat(zeros(128), ones(128)), RDEParam{Float64}()) ≈ 0.0

@load "test_data/shocks.jld2" u1 u2 u3 u4

@test count_shocks(u1,dx) == 1
@test count_shocks(u2,dx) == 2
@test count_shocks(u3,dx) == 3
@test count_shocks(u4,dx) == 4

@test begin
    u = zeros(512)
    u[100] = 1.0

    calc_inds = RDE.shock_indices(u, 1.0)
    findall(calc_inds) == [100]
end