using CommonRLInterface
using RDE

"""
    RDEVecEnv{E<:AbstractEnv}

A vectorized environment that runs multiple RDE environments in parallel.
Implements the CommonRLInterface and provides a Stable Baselines compatible step! function.
"""
mutable struct RDEVecEnv{E<:AbstractEnv} <: AbstractEnv
    envs::Vector{E}
    n_envs::Int
    observations::Matrix{Float32}  # Pre-allocated for efficiency
    rewards::Vector{Float32}
    dones::Vector{Bool}
    infos::Vector{Dict{String,Any}}
    reset_infos::Vector{Dict{String,Any}}
end

"""
    RDEVecEnv(envs::Vector{<:AbstractEnv})

Create a vectorized environment from a vector of environments.
"""
function RDEVecEnv(envs::Vector{E}) where {E<:AbstractEnv}
    n_envs = length(envs)
    obs_dim = length(CommonRLInterface.observe(envs[1]))
    observations = Matrix{Float32}(undef, obs_dim, n_envs)
    rewards = zeros(Float32, n_envs)
    dones = fill(false, n_envs)
    infos = [Dict{String,Any}() for _ in 1:n_envs]
    reset_infos = [Dict{String,Any}() for _ in 1:n_envs]
    
    RDEVecEnv{E}(envs, n_envs, observations, rewards, dones, infos, reset_infos)
end

# CommonRLInterface implementations
function CommonRLInterface.reset!(env::RDEVecEnv)
    Threads.@threads for i in 1:env.n_envs
        CommonRLInterface.reset!(env.envs[i])
        env.observations[:, i] .= CommonRLInterface.observe(env.envs[i])
        env.reset_infos[i] = Dict{String,Any}()
    end
    nothing
end

function CommonRLInterface.observe(env::RDEVecEnv)
    copy(env.observations)
end

function CommonRLInterface.act!(env::RDEVecEnv, actions::AbstractMatrix)
    @debug "VecEnv act! starting threaded loop, actions size: $(size(actions))"
    Threads.@threads for i in 1:env.n_envs
        # Step environment
        env.rewards[i] = CommonRLInterface.act!(env.envs[i], @view actions[:, i])
        @debug "VecEnv act! done with env $i, starting termination check"
        # Check termination
        if CommonRLInterface.terminated(env.envs[i])
            env.dones[i] = true
            env.infos[i]["terminal_observation"] = CommonRLInterface.observe(env.envs[i])
            if env.envs[i].truncated  # TODO: Add truncated to CommonRLInterface?
                env.infos[i]["TimeLimit.truncated"] = true
            end
            CommonRLInterface.reset!(env.envs[i])
        else
            env.dones[i] = false
            empty!(env.infos[i])
        end
        
        # Update observation
        @debug "VecEnv act! done with env $i, starting observation update"
        env.observations[:, i] .= CommonRLInterface.observe(env.envs[i])
        @debug "VecEnv act! done with env $i, observation update done"
    end
    
    copy(env.rewards)
end

function CommonRLInterface.terminated(env::RDEVecEnv) #this is SB done, not terminated
    copy(env.dones)
end

"""
    seed!(env::RDEVecEnv, seed::Int)

Set the random seed for the environment.
"""
function seed!(env::RDEVecEnv, seed::Int)
    Random.seed!(seed)
end

"""
    step!(env::RDEVecEnv, actions::AbstractMatrix)

Step all environments in parallel with the given actions.
Returns (observations, rewards, dones, infos) matching the Stable Baselines API.
"""
function step!(env::RDEVecEnv, actions::AbstractMatrix)
    @debug "VecEnv act!, actions size: $(size(actions))"
    CommonRLInterface.act!(env, actions)
    @debug "VecEnv act! done, returning stuff"
    return (
        CommonRLInterface.observe(env),
        copy(env.rewards),
        CommonRLInterface.terminated(env),
        copy(env.infos)
    )
end 