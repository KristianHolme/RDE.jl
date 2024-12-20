@testset "Policy Execution" begin
    @testset "run_policy with different saves_per_action" begin
        # Setup a small environment for testing
        env = RDEEnv(;
            dt=0.01,
            smax=4.0,
            u_pmax=1.2,
            params=RDEParam(;N=32, tmax=0.05),
            τ_smooth=0.001,  # Small smoothing time to avoid discontinuities
            momentum=0.0,    # No momentum for simpler testing
            observation_strategy=FourierObservation(8),
            action_type=ScalarPressureAction()
        )
        policy = RandomRDEPolicy(env)

        # Test with different saves_per_action values
        for saves_per_action in [0, 1, 2, 5]
            @testset "saves_per_action = $saves_per_action" begin
                data = run_policy(policy, env, saves_per_action=saves_per_action)
                
                # Basic structure tests
                @test data isa PolicyRunData
                @test length(data.action_ts) > 0
                @test length(data.ss) == length(data.action_ts)
                @test length(data.u_ps) == length(data.action_ts)
                @test length(data.rewards) == length(data.action_ts)
                
                # Test state data length
                if saves_per_action == 0
                    # For saves_per_action = 0, we should only have states at action points
                    @test length(data.states) == length(data.action_ts)
                else
                    # For saves_per_action > 0, we should have more state points
                    @test length(data.states) ≥ length(data.action_ts) * saves_per_action
                end
                
                # Test time consistency
                @test issorted(data.action_ts)
                @test issorted(data.state_ts)
                @test data.action_ts[1] ≤ env.dt
                @test data.state_ts[1] ≤ env.dt
                @test data.action_ts[end] ≤ env.prob.params.tmax + env.dt
                @test data.state_ts[end] ≤ env.prob.params.tmax + env.dt
                
                # Test data consistency
                @test length(data.energy_bal) == length(data.states)
                @test length(data.chamber_p) == length(data.states)
                @test length(data.state_ts) == length(data.states)
                
                # Test state vector dimensions
                N = env.prob.params.N
                @test all(length(state) == 2N for state in data.states)
                
                # Test value ranges
                @test all(0.0 ≤ s ≤ env.smax for s in data.ss)
                @test all(0.0 ≤ u_p ≤ env.u_pmax for u_p in data.u_ps)
                @test all(isfinite, data.rewards)
                @test all(isfinite, data.energy_bal)
                @test all(isfinite, data.chamber_p)
            end
        end
    end
end 