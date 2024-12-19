using CommonRLInterface
using POMDPs
@testset "RDEEnv Initialization" begin
    @test begin
        params = RDEParam(;N=16, tmax = 0.01)
        prob = RDEProblem(params)
        sum(abs.(prob.u0)) > 0.01
    end

    @test begin
        mdp = convert(POMDPs.MDP, RDEEnv())
        pomdp = convert(POMDPs.POMDP, RDEEnv())
        true
    end
end

@testset "RDEEnv Policies" begin
    @test begin
        ConstPolicy = ConstantRDEPolicy()
        data = run_policy(ConstPolicy, RDEEnv(RDEParam(;N=16, tmax = 0.1)))
        data isa PolicyRunData
    end
end

@testset "Observation Strategies" begin
    N = 16  # Number of spatial points for testing
    params = RDEParam(;N=N, tmax=0.1)
    
    @testset "Fourier Observation" begin
        fft_terms = 4
        env = RDEEnv(params=params, observation_strategy=FourierObservation(fft_terms))
        
        # Test initialization
        @test length(env.observation) == 2fft_terms + 1  # 2 * fft_terms + time
        
        # Test observation
        obs = CommonRLInterface.observe(env)
        @test length(obs) == 2fft_terms + 1
        @test all(0 .<= obs[1:end-1] .<= 1)  # FFT coefficients should be normalized
        @test 0 <= obs[end] <= 1  # Normalized time
    end
    
    @testset "State Observation" begin
        env = RDEEnv(params=params, observation_strategy=StateObservation())
        
        # Test initialization
        @test length(env.observation) == 2N + 1  # u and λ states + time
        
        # Test observation
        obs = CommonRLInterface.observe(env)
        @test length(obs) == 2N + 1
        @test all(-1 .<= obs[1:end-1] .<= 1)  # State components should be normalized
        @test 0 <= obs[end] <= 1  # Normalized time
        
        # Test that first N components are normalized u and next N are normalized λ
        u_obs = obs[1:N]
        λ_obs = obs[N+1:2N]
        @test length(u_obs) == length(λ_obs) == N
    end
    
    @testset "Sampled Observation" begin
        n_samples = 8
        env = RDEEnv(params=params, observation_strategy=SampledStateObservation(n_samples))
        
        # Test initialization
        @test length(env.observation) == 2n_samples + 1  # sampled u and λ + time
        
        # Test observation
        obs = CommonRLInterface.observe(env)
        @test length(obs) == 2n_samples + 1
        @test all(-1 .<= obs[1:end-1] .<= 1)  # Sampled values should be normalized
        @test 0 <= obs[end] <= 1  # Normalized time
        
        # Test sampling points
        u_samples = obs[1:n_samples]
        λ_samples = obs[n_samples+1:2n_samples]
        @test length(u_samples) == length(λ_samples) == n_samples
    end
    
    @testset "Observation Consistency" begin
        # Test that observations remain consistent after reset
        for strategy in [
            FourierObservation(4),
            StateObservation(),
            SampledStateObservation(8)
        ]
            env = RDEEnv(params=params, observation_strategy=strategy)
            obs1 = CommonRLInterface.observe(env)
            CommonRLInterface.reset!(env)
            obs2 = CommonRLInterface.observe(env)
            @test length(obs1) == length(obs2)
            @test all(isfinite.(obs1))
            @test all(isfinite.(obs2))
        end
    end

    @testset "Observation Type Consistency" begin
        # Test for Float32
        @test begin
            env = RDEEnv(params=RDEParam{Float32}())
            obs = CommonRLInterface.observe(env)
            eltype(obs) == Float32
        end

        # Test for Float64
        @test begin
            env = RDEEnv(params=RDEParam{Float64}())
            obs = CommonRLInterface.observe(env)
            eltype(obs) == Float64
        end

        # Test type consistency across different observation strategies
        for T in [Float32, Float64]
            for strategy in [
                FourierObservation(4),
                StateObservation(),
                SampledStateObservation(8)
            ]
                @test begin
                    env = RDEEnv(RDEParam{T}(), observation_strategy=strategy)
                    obs = CommonRLInterface.observe(env)
                    eltype(obs) == T
                end
            end
        end
    end
end
