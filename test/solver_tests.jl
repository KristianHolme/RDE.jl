@test begin
    for T in [Float32, Float64]
        prob = RDEProblem(RDEParam{T}(;N=32, tmax = 0.01));
        solve_pde!(prob);
    end
    true
end

@test begin
    success = true
    for T in [Float32, Float64]
        rde = RDEProblem(RDEParam{T}(;N=128, tmax = 5.0), method = :pseudospectral);
        fft_plan = rde.cache.fft_plan
        ifft_plan = rde.cache.ifft_plan
        u0 = rde.u0
        u0_hat = fft_plan*u0
        u0_hat_hat = ifft_plan*u0_hat
        success = success && (u0_hat_hat ≈ u0)
    end
    success
end

@test RDE.ω(1.0, 0.0, 1.0) ≈ exp(1.0)
@test RDE.ω(1.0, 1.0, 1.0) ≈ 1.0
@test RDE.ω(0.0, 1.0, 0.5) ≈ exp(-2.0)

@test RDE.ξ(1.0, 0.0, 1.0) ≈ -1.0
@test RDE.ξ(0.0, 1.0, 1.0) ≈ 0.0
@test RDE.ξ(2.0, 3.0, 2) ≈ 4.0


@test RDE.β(1.0, 1.0, 0.0, 1.0) ≈ 0.0
@test RDE.β(1.0, 1.0, 1.0, 1.0) ≈ 0.5
@test RDE.β(2.0, 3.5, 0.56, 5.0) ≈ 0.0014622165143

@test begin
    # Test smooth transition
    t = 0.5
    control_t = 0.0
    current = [2.0]
    previous = [1.0]
    τ_smooth = 1.0
    c = [0.0]
    RDE.smooth_control!(c, t, control_t, current, previous, τ_smooth)
    c ≈ [1.5]
end

@test begin
    # Test after transition period
    t = 2.0
    control_t = 0.0
    current = [2.0]
    previous = [1.0]
    τ_smooth = 1.0
    c = [0.0]
    
    RDE.smooth_control!(c, t, control_t, current, previous, τ_smooth)
    c ≈ current
end

@test begin
    # Test at start of transition
    t = 0.0
    control_t = 0.0
    current = [2.0]
    previous = [1.0]
    τ_smooth = 1.0
    c = [0.0]
    RDE.smooth_control!(c, t, control_t, current, previous, τ_smooth)
    c ≈ previous
end

@testset "ODE Solver Tests" begin
    @testset "Basic Solver Success" begin
        for T in [Float32, Float64]
            prob = RDEProblem(RDEParam{T}(;N=512, tmax=0.01))
            solve_pde!(prob)
            @test Symbol(prob.sol.retcode) == :Success
        end
    end

    @testset "Longer Integration Success" begin
        for T in [Float32, Float64]
            prob = RDEProblem(RDEParam{T}(;N=512, tmax=5.0))
            solve_pde!(prob)
            @test Symbol(prob.sol.retcode) == :Success
        end
    end
end





