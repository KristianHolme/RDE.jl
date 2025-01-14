using Test
using RDE
using CommonRLInterface
using POMDPs

@testset "Reward Interface Tests" begin
    @testset "ShockSpanReward" begin
        # Create environment with ShockSpanReward
        env = RDEEnv(;
            dt=1.0,
            reward_type=ShockSpanReward(
                target_shock_count=3,
                span_scale=4.0f0,
                shock_weight=5.0f0
            )
        )
        
        # Test initial reward
        CommonRLInterface.reset!(env)
        @test env.reward isa Float32
        @test !isnan(env.reward)
        @test !isinf(env.reward)
        
        # Test reward with no shocks
        env.state .= 1.0  # Constant state = no shocks
        set_reward!(env, env.reward_type)
        @test env.reward < 0  # Should be negative with no shocks
        
        # Test reward with target number of shocks
        # Create a state with 3 shocks
        N = env.prob.params.N
        env.state[1:N] .= 1.0
        env.state[N÷4] = 2.0  # First shock
        env.state[N÷2] = 2.0  # Second shock
        env.state[3N÷4] = 2.0  # Third shock
        set_reward!(env, env.reward_type)
        @test env.reward > 0  # Should be positive with target shocks
    end

    @testset "ShockPreservingReward" begin
        # Create environment with ShockPreservingReward
        params = RDEParam(tmax=1.0)
        env = RDEEnv(params;
            dt=0.1,
            reward_type=ShockPreservingReward(
                target_shock_count=3, 
                abscence_limit=0.01f0
            )
        )
        
        # Test initial reward
        CommonRLInterface.reset!(env)
        @test env.reward isa Float32
        @test !isnan(env.reward)
        @test !isinf(env.reward)
        
        # Test truncation with wrong number of shocks
        env.state .= 1.0  # Constant state = no shocks
        set_reward!(env, env.reward_type)
        CommonRLInterface.act!(env, 0.0f0)
        CommonRLInterface.act!(env, 0.0f0) #act twice to outrun abscence limit

        @test env.terminated  # Should be terminated with wrong shock count
        @test env.reward ≈ -2.0  # Should get penalty reward
        
        # Test reward with correct number of shocks
        env.terminated = false  # Reset termination flag
        N = env.prob.params.N
        env.state[1:N] .= 1.0
        # Create 3 evenly spaced shocks
        env.state[N÷3] = 2.0
        env.state[2N÷3] = 2.0
        env.state[N] = 2.0
        set_reward!(env, env.reward_type)
        @test !env.terminated  # Should not be terminated
        @test env.reward > 0  # Should get positive reward for evenly spaced shocks
        @test env.reward ≤ 1.0  # Reward should be normalized
    end

    @testset "Policy Reward Behavior" begin
        # Test parameters
        tmax = 1.0
        dt = 0.1
        n_steps = Int(tmax/dt)

        @testset "Constant Policy with ShockSpanReward" begin
            env = RDEEnv(;
                dt=dt,
                params=RDEParam(tmax=tmax),
                reward_type=ShockSpanReward(
                    target_shock_count=3,
                    span_scale=4.0f0,
                    shock_weight=5.0f0
                )
            )
            policy = ConstantRDEPolicy(env)
            
            # Run policy and collect rewards
            CommonRLInterface.reset!(env)
            rewards = Float32[]
            for _ in 1:n_steps
                action = POMDPs.action(policy, nothing)
                push!(rewards, CommonRLInterface.act!(env, action))
                @test !isnan(env.reward)
                @test !isinf(env.reward)
            end
            @test length(rewards) == n_steps
            @test !any(isnan.(rewards))
            @test !any(isinf.(rewards))
        end

        @testset "Random Policy with ShockPreservingReward" begin
            env = RDEEnv(;
                dt=dt,
                params=RDEParam(tmax=tmax),
                reward_type=ShockPreservingReward(
                    target_shock_count=2
                )
            )
            policy = RandomRDEPolicy(env)
            
            # Run policy and collect rewards
            CommonRLInterface.reset!(env)
            rewards = Float32[]
            truncations = Int[]
            for step in 1:n_steps
                if env.done
                    break
                end
                action = POMDPs.action(policy, nothing)
                push!(rewards, CommonRLInterface.act!(env, action))
                if env.truncated
                    push!(truncations, step)
                end
                @test !isnan(env.reward)
                @test !isinf(env.reward)
            end
            
            # Test reward properties
            @test !isempty(rewards)
            @test !any(isnan.(rewards))
            @test !any(isinf.(rewards))
            
            # Test truncation behavior
            if !isempty(truncations)
                @test all(rewards[truncations] .≈ -2.0)  # All truncated steps should have penalty reward
            end
        end

        @testset "Policy Run Data Collection" begin
            env = RDEEnv(;
                dt=dt,
                params=RDEParam(tmax=tmax),
                reward_type=ShockSpanReward(
                    target_shock_count=3,
                    span_scale=4.0f0,
                    shock_weight=5.0f0
                )
            )
            policy = ConstantRDEPolicy(env)
            
            # Run policy and collect data
            data = run_policy(policy, env)
            
            # Test data properties
            @test !isempty(data.rewards)
            @test !any(isnan.(data.rewards))
            @test !any(isinf.(data.rewards))
            @test length(data.action_ts) == length(data.rewards)
            @test all(diff(data.action_ts) .≈ dt)  # Time steps should be consistent
        end
    end
end 