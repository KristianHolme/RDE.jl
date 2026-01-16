using Test
using RDE

@testitem "resample_data - Identity" begin
    # Test that resampling to same size returns copy
    for T in [Float32, Float64]
        data = T[1.0, 2.0, 3.0, 4.0, 5.0]
        result = RDE.resample_data(data, length(data))
        @test result == data
        @test result !== data  # Should be a copy
    end
end

@testitem "resample_data - Downsampling" begin
    # Test downsampling from larger to smaller size
    for T in [Float32, Float64]
        data = T[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = RDE.resample_data(data, 4)
        @test length(result) == 4
        @test all(isfinite.(result))
        @test typeof(result) == Vector{T}
    end
end

@testitem "resample_data - Upsampling" begin
    # Test upsampling from smaller to larger size
    for T in [Float32, Float64]
        data = T[1.0, 2.0, 3.0, 4.0]
        result = RDE.resample_data(data, 8)
        @test length(result) == 8
        @test all(isfinite.(result))
        @test typeof(result) == Vector{T}
        
        # First element should be close to original first element
        @test abs(result[1] - data[1]) < 1.0e-5
    end
end

@testitem "resample_data - Periodic Boundary" begin
    # Test that periodic boundary is preserved
    for T in [Float32, Float64]
        # Create periodic data: [1, 2, 3, 4, 1, 2, 3, 4]
        data = T[1.0, 2.0, 3.0, 4.0]
        
        # Upsample to 8 points - last point should interpolate between 4 and 1 (wrapped)
        result = RDE.resample_data(data, 8)
        
        # The last point should be close to interpolating between data[4] and data[1]
        # Since we're at position 8 mapping to ~4.5 in periodic [1, 5), it should be
        # between data[4] and wrapped data[1]
        @test result[end] ≈ T(0.5) * (data[4] + data[1]) atol = T(1.0e-5)
        
        # First element should match
        @test result[1] ≈ data[1] atol = T(1.0e-5)
    end
end

@testitem "resample_data - Periodic Function" begin
    # Test with a known periodic function (sine wave)
    for T in [Float32, Float64]
        L = 16
        x = range(T(0), T(2π), length = L)
        data = sin.(x)  # Periodic function
        
        # Upsample
        result_up = RDE.resample_data(data, 2 * L)
        @test length(result_up) == 2 * L
        @test all(isfinite.(result_up))
        
        # Downsample
        result_down = RDE.resample_data(data, L ÷ 2)
        @test length(result_down) == L ÷ 2
        @test all(isfinite.(result_down))
        
        # Verify periodicity: first and last should match for complete period
        # (For a full period of sine, sin(0) ≈ sin(2π))
        @test abs(result_up[1] - result_up[end]) < T(1.0e-3)
    end
end

@testitem "resample_data - Edge Cases" begin
    # Test edge cases
    for T in [Float32, Float64]
        # Single element
        data1 = T[5.0]
        result1 = RDE.resample_data(data1, 1)
        @test result1 == data1
        
        # Single element upsampled
        result1_up = RDE.resample_data(data1, 4)
        @test length(result1_up) == 4
        @test all(result1_up .≈ T(5.0))
        
        # Two elements
        data2 = T[1.0, 2.0]
        result2 = RDE.resample_data(data2, 4)
        @test length(result2) == 4
        @test all(isfinite.(result2))
        
        # Large upsampling
        data_small = T[1.0, 2.0, 3.0]
        result_large = RDE.resample_data(data_small, 100)
        @test length(result_large) == 100
        @test all(isfinite.(result_large))
    end
end

@testitem "resample_data - Monotonicity Preservation" begin
    # Test that monotonic data remains approximately monotonic after resampling
    for T in [Float32, Float64]
        # Strictly increasing data
        data_increasing = T[i for i in 1:10]
        
        # Downsample
        result_down = RDE.resample_data(data_increasing, 5)
        @test result_down[1] < result_down[end]  # Should still be increasing overall
        
        # Upsample
        result_up = RDE.resample_data(data_increasing, 20)
        @test result_up[1] < result_up[end]  # Should still be increasing overall
    end
end

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
