using Test
using RDE
using OrdinaryDiffEq

@testitem "Basic Solver" begin
    using OrdinaryDiffEq
    for T in [Float32, Float64]
        prob = RDEProblem(RDEParam{T}(N = 512, tmax = 0.01))
        solve_pde!(prob)
        @test prob.sol.retcode == OrdinaryDiffEq.ReturnCode.Success
    end
end

@testitem "FFT Plans" begin
    for T in [Float32, Float64]
        prob = RDEProblem(
            RDEParam{T}(N = 128, tmax = 5.0),
            method = PseudospectralMethod{T}()
        )
        cache = prob.method.cache
        u0 = prob.u0
        u0_hat = cache.fft_plan * u0
        u0_hat_hat = cache.ifft_plan * u0_hat
        @test u0_hat_hat ≈ u0
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
        # Test finite difference method
        prob_fd = RDEProblem(RDEParam{T}(N = 512, tmax = 5.0))
        solve_pde!(prob_fd)
        @test prob_fd.sol.retcode == OrdinaryDiffEq.ReturnCode.Success

        # Test pseudospectral method
        prob_ps = RDEProblem(
            RDEParam{T}(N = 1024, tmax = 5.0),
            method = PseudospectralMethod{T}()
        )
        solve_pde!(prob_ps)
        @test prob_ps.sol.retcode == OrdinaryDiffEq.ReturnCode.Success
    end
end

@testitem "Solver Options" begin
    using OrdinaryDiffEq
    params = RDEParam{Float32}(N = 512, tmax = 1.0)
    prob = RDEProblem(params)

    # Test different solvers
    solve_pde!(prob, solver = Tsit5())
    @test prob.sol.retcode == OrdinaryDiffEq.ReturnCode.Success

    solve_pde!(prob, solver = Rodas4(autodiff = false))
    @test prob.sol.retcode == OrdinaryDiffEq.ReturnCode.Success

    # Test saveframes option
    solve_pde!(prob, saveframes = 100)
    @test length(prob.sol.t) == 101  # Including initial condition
end
