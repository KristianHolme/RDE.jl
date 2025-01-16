using Test
using RLBridge
using RDE
using Random
using CommonRLInterface

@testset "VecEnv Tests" begin
    @testset "Construction" begin
        # Test vector constructor
        envs = [RDEEnv(dt=0.1, τ_smooth=0.01) for _ in 1:3]
        vec_env = RDEVecEnv(envs)
        @test vec_env.n_envs == 3
        @test length(vec_env.envs) == 3
        @test size(vec_env.observations, 2) == 3
    end
    
    @testset "Reset and Seed" begin
        envs = [RDEEnv(dt=0.1, τ_smooth=0.01, u_init=random_shock_combination_init_func) for _ in 1:4]
        vec_env = RDEVecEnv(envs)
        
        # Test initial reset
        reset!(vec_env)
        obs1 = observe(vec_env)
        @test size(obs1) == (length(observe(envs[1])), 4)
        @test !any(isnan, obs1)
        
        # Test seeding
        seed!(vec_env, 42)
        reset!(vec_env)
        obs2 = observe(vec_env)
        
        seed!(vec_env, 42)
        reset!(vec_env)
        obs3 = observe(vec_env)
        
        @test obs2 == obs3  # Same seed should give same results
        
        seed!(vec_env, 43)
        reset!(vec_env)
        obs4 = observe(vec_env)
        
        @test obs2 != obs4  # Different seeds should give different results
    end
    
    @testset "Step" begin
        envs = [RDEEnv(dt=0.1, τ_smooth=0.01) for _ in 1:4]
        vec_env = RDEVecEnv(envs)
        reset!(vec_env)
        
        # Create test actions
        action_dim = RDE.n_actions(envs[1].action_type)
        actions = zeros(Float32, action_dim, 4)
        actions[:, 1] .= 0.5  # Different action for first env
        
        # Test act!
        rewards = act!(vec_env, actions)
        obs = observe(vec_env)
        dones = terminated(vec_env)
        
        @test size(obs) == (length(CommonRLInterface.observe(envs[1])), 4)
        @test length(rewards) == 4
        @test length(dones) == 4
        @test !any(isnan, obs)
        @test !any(isnan, rewards)
        
        # Test different actions give different results
        @test obs[:, 1] != obs[:, 2]  # First env should differ due to different action
    end
    
    @testset "Termination" begin
        # Create envs with short time limit for testing termination
        envs = [RDEEnv(dt=0.1, τ_smooth=0.01, params=RDEParam(tmax=0.2)) for _ in 1:4]
        vec_env = RDEVecEnv(envs)
        reset!(vec_env)
        
        # Run until termination
        action_dim = RDE.n_actions(envs[1].action_type)
        actions = zeros(Float32, action_dim, 4)
        is_terminated = false
        n_steps = 0
        
        while !is_terminated && n_steps < 100
            act!(vec_env, actions)
            dones = terminated(vec_env)
            is_terminated = any(dones)
            n_steps += 1
        end
        
        @test is_terminated
        @test n_steps < 100  # Should terminate due to time limit
        # @test any(haskey(info, "TimeLimit.truncated") for info in vec_env.infos)
        
        # Check auto-reset
        @test !any(isnan, vec_env.observations)  # Environments should be reset
    end
    
    @testset "Thread Safety" begin
        envs = [RDEEnv(dt=0.1, τ_smooth=0.01) for _ in 1:8]
        vec_env = RDEVecEnv(envs)  # Use more envs to test threading
        reset!(vec_env)
        
        action_dim = RDE.n_actions(envs[1].action_type)
        actions = zeros(Float32, action_dim, 8)
        
        # Run multiple steps in parallel
        CommonRLInterface.act!(vec_env, actions)
        
        # If we got here without errors, threading worked
        @test true
    end
end 