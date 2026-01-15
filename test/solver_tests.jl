@testitem "Basic Solver" begin
    using OrdinaryDiffEq
    for T in [Float32, Float64]
        prob = RDEProblem(RDEParam{T}(N = 512, tmax = 0.01))
        solve_pde!(prob)
        @test prob.sol.retcode == OrdinaryDiffEq.ReturnCode.Success
    end
end

@testitem "Core Functions" begin
    @test RDE.ω(1.0, 0.0, 1.0) ≈ exp(1.0)
    @test RDE.ω(1.0, 1.0, 1.0) ≈ 1.0
    @test RDE.ω(0.0, 1.0, 0.5) ≈ exp(-2.0)

    @test RDE.ξ(1.0, 0.0, 1.0) ≈ -1.0
    @test RDE.ξ(0.0, 1.0, 1.0) ≈ 0.0
    @test RDE.ξ(2.0, 3.0, 2) ≈ 4.0

    @test RDE.β(1.0, 1.0, 0.0, 1.0) ≈ 0.0
    @test RDE.β(1.0, 1.0, 1.0, 1.0) ≈ 0.5
    @test RDE.β(2.0, 3.5, 0.56, 5.0) ≈ 0.0014622165143
end

@testitem "Control Smoothing" begin
    # Test smooth transition
    t = 0.5
    control_t = 0.0
    current = [2.0]
    previous = [1.0]
    τ_smooth = 1.0
    c = [0.0]
    RDE.smooth_control!(c, t, control_t, current, previous, τ_smooth)
    @test c ≈ [1.5]

    # Test after transition period
    t = 2.0
    RDE.smooth_control!(c, t, control_t, current, previous, τ_smooth)
    @test c ≈ current

    # Test at start of transition
    t = 0.0
    RDE.smooth_control!(c, t, control_t, current, previous, τ_smooth)
    @test c ≈ previous
end

@testitem "Long Integration" begin
    using OrdinaryDiffEq
    for T in [Float32, Float64]
        prob = RDEProblem(RDEParam{T}(N = 512, tmax = 5.0))
        solve_pde!(prob)
        @test prob.sol.retcode == OrdinaryDiffEq.ReturnCode.Success
    end
end

@testitem "FV shock speed (Burgers limit)" begin
    using OrdinaryDiffEq
    params = RDEParam{Float32}(
        N = 256,
        L = 2π,
        q_0 = 0.0f0,
        ϵ = 0.0f0,
        ν_1 = 0.0f0,
        ν_2 = 0.0f0,
        u_c = 100.0f0,
        α = 1.0f0,
        s = 0.0f0,
        tmax = 0.1f0
    )
    prob = RDEProblem(params)
    x = prob.x
    x0 = Float32(π)

    uL = 2.0f0
    uR = 1.0f0
    prob.u0 .= ifelse.(x .< x0, uL, uR)
    prob.λ0 .= 0.5f0

    solve_pde!(prob; saveframes = 10, alg = OrdinaryDiffEq.SSPRK33(), adaptive = false)
    u_final, = RDE.split_sol(prob.sol.u[end])

    dx = RDE.get_dx(prob)
    shock_inds = RDE.shock_indices(u_final, dx)
    @test !isempty(shock_inds)

    expected_speed = 0.5f0 * (uL + uR)
    expected_pos = mod(x0 + expected_speed * params.tmax, params.L)
    est_pos = x[shock_inds[1]]

    dist = abs(mod(est_pos - expected_pos + params.L / 2, params.L) - params.L / 2)
    @test dist ≤ 4 * dx
end

@testitem "FV solution bounds" begin
    using OrdinaryDiffEq
    params = RDEParam{Float32}(N = 128, tmax = 0.5f0)
    prob = RDEProblem(params)
    solve_pde!(prob; saveframes = 5, alg = OrdinaryDiffEq.SSPRK33(), adaptive = false)

    u_final, λ_final = RDE.split_sol(prob.sol.u[end])
    @test all(u_final .>= 0.0f0)
    @test all((0.0f0 .<= λ_final) .& (λ_final .<= 1.0f0))
end

@testitem "Solver Options" begin
    using OrdinaryDiffEq
    params = RDEParam{Float32}(N = 512, tmax = 1.0)
    prob = RDEProblem(params)

    solve_pde!(prob)
    @test prob.sol.retcode == OrdinaryDiffEq.ReturnCode.Success

    solve_pde!(prob; alg = OrdinaryDiffEq.SSPRK33(), adaptive = false)
    @test prob.sol.retcode == OrdinaryDiffEq.ReturnCode.Success

    # Test saveframes option
    solve_pde!(prob, saveframes = 100)
    @test length(prob.sol.t) == 101  # Including initial condition
end
