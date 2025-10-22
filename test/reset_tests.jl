using Test
using RDE

@testitem "Default Reset" begin
    prob = RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = Default())
    @test prob.u0 ≈ (3.0f0 / 2.0f0) .* (sech.(prob.x .- 1.0f0) .^ 20.0f0)
    @test prob.λ0 ≈ 0.5f0 .* ones(Float32, prob.params.N)
    @test prob.params.u_p ≈ 0.5f0
end

@testitem "NShock Reset" begin
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

@testitem "RandomCombination Reset" begin
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

@testitem "RandomShockOrCombination Reset" begin
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
    let n_shocks = 0, n_trials = 100
        prob_shock = RandomShockOrCombination(shock_prob = 0.8)
        for i in 1:n_trials
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

@testitem "EvalCycleShockReset Initialization" begin
    reset_strat = EvalCycleShockReset(4)
    @test reset_strat.repetitions_per_config == 4
    @test reset_strat.current_config == 0 #initially set to bc we reset once at problem construction
    @test length(reset_strat.init_shocks) == 48  # 4 goals × 3 non-goal shocks × 4 repetitions

    # Verify the pattern: [2,2,2,2,3,3,3,3,4,4,4,4,1,1,1,1,3,3,3,3,4,4,4,4,1,1,1,1,2,2,2,2,4,4,4,4,1,1,1,1,2,2,2,2,3,3,3,3]
    expected_pattern = vcat(
        [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],  # Goal 1: non-goal shocks repeated
        [1, 1, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4],  # Goal 2: non-goal shocks repeated
        [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4],  # Goal 3: non-goal shocks repeated
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],   # Goal 4: non-goal shocks repeated
    )
    @test reset_strat.init_shocks == expected_pattern
end

@testitem "EvalCycleShockReset Custom Repetitions" begin
    reset_strat = EvalCycleShockReset(2)
    @test length(reset_strat.init_shocks) == 24  # 4 goals × 3 non-goal shocks × 2 repetitions

    # Verify the pattern: [2,2,3,3,4,4,1,1,3,3,4,4,1,1,2,2,4,4,1,1,2,2,3,3]
    expected_pattern = vcat(
        [2, 2, 3, 3, 4, 4],  # Goal 1: non-goal shocks repeated
        [1, 1, 3, 3, 4, 4],  # Goal 2: non-goal shocks repeated
        [1, 1, 2, 2, 4, 4],  # Goal 3: non-goal shocks repeated
        [1, 1, 2, 2, 3, 3],   # Goal 4: non-goal shocks repeated
    )
    @test reset_strat.init_shocks == expected_pattern
end

@testitem "EvalCycleShockReset Wrap Around" begin
    reset_strat = EvalCycleShockReset(2)
    prob = RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = reset_strat)

    # Go through all 24 configurations using actual reset function
    for i in 1:24
        RDE.reset_state_and_pressure!(prob, reset_strat)
    end

    # Should have wrapped around to config 1
    @test reset_strat.current_config == 1
end

@testitem "EvalCycleShockReset State Loading" begin
    reset_strat = EvalCycleShockReset(4)
    prob = RDEProblem(RDEParam{Float32}(N = 32), reset_strategy = reset_strat)

    # Test that reset_state_and_pressure! works correctly
    # This should load the first shock (2) from the first goal phase
    RDE.reset_state_and_pressure!(prob, reset_strat)
    @test reset_strat.current_config == 2  # Should increment after reset

    # Test a few more resets to verify cycling
    for i in 1:5
        RDE.reset_state_and_pressure!(prob, reset_strat)
    end
    @test reset_strat.current_config == 7  # Should be at 7th config

    # Verify the pressure matches expected shock pressure
    expected_shock = reset_strat.init_shocks[6]  # Current config - 1
    @test prob.params.u_p ≈ RDE.SHOCK_PRESSURES[expected_shock]
end
