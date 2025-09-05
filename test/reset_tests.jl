using Test
using RDE

@testset "Reset Strategies" begin
    @testset "Default Reset" begin
        prob = RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = Default())
        @test prob.u0 ≈ (3.0f0 / 2.0f0) .* sech.(prob.x .- 1.0f0) .^ 20.0f0
        @test prob.λ0 ≈ 0.5f0 .* ones(Float32, prob.params.N)
        @test prob.params.u_p ≈ 0.5f0
    end

    @testset "NShock Reset" begin
        for n in 1:4
            prob = RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = NShock(n))
            @test length(prob.u0) == prob.params.N
            @test length(prob.λ0) == prob.params.N
            @test all(isfinite.(prob.u0))
            @test all(isfinite.(prob.λ0))
            @test all(0 .≤ prob.λ0 .≤ 1)
            @test prob.params.u_p ≈ RDE.SHOCK_PRESSURES[n]
        end

        # Test invalid n values
        @test_throws ArgumentError RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = NShock(0))
        @test_throws ArgumentError RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = NShock(5))
    end

    @testset "RandomCombination Reset" begin
        prob = RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = RandomCombination())
        @test length(prob.u0) == prob.params.N
        @test length(prob.λ0) == prob.params.N
        @test all(isfinite.(prob.u0))
        @test all(isfinite.(prob.λ0))
        @test all(0 .≤ prob.λ0 .≤ 1)
        @test RDE.SHOCK_PRESSURES[1] ≤ prob.params.u_p ≤ RDE.SHOCK_PRESSURES[4]  # Reasonable range for pressure

        # Test temperature parameter
        prob_hot = RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = RandomCombination(temp = 1.0))
        prob_cold = RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = RandomCombination(temp = 0.1))
        @test all(isfinite.(prob_hot.u0))
        @test all(isfinite.(prob_cold.u0))
    end

    @testset "RandomShockOrCombination Reset" begin
        prob = RDEProblem(
            RDEParam{Float32}(N = 32),
            reset_strategy = RandomShockOrCombination()
        )
        @test length(prob.u0) == prob.params.N
        @test length(prob.λ0) == prob.params.N
        @test all(isfinite.(prob.u0))
        @test all(isfinite.(prob.λ0))
        @test all(0 .≤ prob.λ0 .≤ 1)

        # Test probability parameter
        n_shocks = 0
        n_trials = 100
        prob_shock = RandomShockOrCombination(shock_prob = 0.8)
        for _ in 1:n_trials
            prob = RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = prob_shock)
            # Check if the u_p matches one of the shock pressures
            if any(abs(prob.params.u_p - p) < 1.0e-6 for p in RDE.SHOCK_PRESSURES)
                n_shocks += 1
            end
        end
        # Should be roughly 80% shocks
        @test 60 ≤ n_shocks ≤ 95  # Allow for some random variation
    end
end 
