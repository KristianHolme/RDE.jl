using CommonRLInterface
using RDE

"""
    MultiAgentVecEnv{E<:AbstractEnv}

A vectorized environment that runs multiple RDE environments in parallel. 
Each environment has multiple agents, and the observations are concatenated for each agent.
Implements the CommonRLInterface and provides a Stable Baselines compatible step! function.
"""
mutable struct MultiAgentVecEnv{E<:AbstractEnv} <: AbstractEnv
    envs::Vector{E}
    n_envs::Int
    n_agents_per_env::Int
    observations::Matrix{Float32}
    rewards::Vector{Float32}
    dones::Vector{Bool}
    infos::Vector{Dict{String,Any}}
    reset_infos::Vector{Dict{String,Any}}
end

"""
    MultiAgentVecEnv(envs::Vector{<:AbstractEnv})

Create a vectorized environment from a vector of environments.
"""
function MultiAgentVecEnv(envs::Vector{E}) where {E<:AbstractEnv}
    n_envs = length(envs)
    obs = CommonRLInterface.observe(envs[1])
    obs_dim = size(obs, 1)
    n_agents_per_env = size(obs, 2)
    observations = Matrix{Float32}(undef, obs_dim, n_envs*n_agents_per_env)
    rewards = zeros(Float32, n_envs*n_agents_per_env)
    dones = fill(false, n_envs*n_agents_per_env)
    infos = [Dict{String,Any}() for _ in 1:n_envs*n_agents_per_env]
    reset_infos = [Dict{String,Any}() for _ in 1:n_envs*n_agents_per_env]
    
    MultiAgentVecEnv{E}(envs, n_envs, n_agents_per_env, observations, rewards, dones, infos, reset_infos)
end
function env_indices(i::Int, n_agents_per_env::Int)
    return (1+(i-1)*n_agents_per_env):(i*n_agents_per_env)
end

# CommonRLInterface implementations
function CommonRLInterface.reset!(env::MultiAgentVecEnv)
    num_agents = env.n_agents_per_env
    num_envs = env.n_envs
    Threads.@threads for i in 1:num_envs
        CommonRLInterface.reset!(env.envs[i])
        env.observations[:, env_indices(i, num_agents)] .= CommonRLInterface.observe(env.envs[i])
        env.reset_infos[i] = Dict{String,Any}()
    end
    nothing
end

function CommonRLInterface.observe(env::MultiAgentVecEnv)
    copy(env.observations)
end

function CommonRLInterface.act!(env::MultiAgentVecEnv, actions::AbstractMatrix)
    num_agents = env.n_agents_per_env
    num_envs = env.n_envs
    @debug "VecEnv act! starting threaded loop, actions size: $(size(actions))"
    Threads.@threads for i in 1:num_envs
        # Step environment
        env.rewards[env_indices(i, num_agents)] = CommonRLInterface.act!(env.envs[i], @view actions[:, env_indices(i, num_agents)])
        @debug "VecEnv act! done with env $i, starting termination check"
        # Check termination
        if CommonRLInterface.terminated(env.envs[i])
            env.dones[env_indices(i, num_agents)] = true
            env_inds = env_indices(i, num_agents)
            for agent_i in 1:num_agents
                env.infos[env_inds[agent_i]]["terminal_observation"] = CommonRLInterface.observe(env.envs[i])[:, agent_i]
                if env.envs[i].truncated  # TODO: Add truncated to CommonRLInterface?
                    env.infos[env_inds[agent_i]]["TimeLimit.truncated"] = true
                end
            end
            CommonRLInterface.reset!(env.envs[i])
        else
            env.dones[env_inds] = false
            empty!.(env.infos[env_inds])
        end
        
        # Update observation
        @debug "VecEnv act! done with env $i, starting observation update"
        env.observations[:, env_indices(i, num_agents)] .= CommonRLInterface.observe(env.envs[i])
        @debug "VecEnv act! done with env $i, observation update done"
    end
    
    copy(env.rewards)
end

function CommonRLInterface.terminated(env::MultiAgentVecEnv) #this is SB done, not terminated
    copy(env.dones)
end


"""
    step!(env::MultiAgentVecEnv, actions::AbstractMatrix)

Step all environments in parallel with the given actions.
Returns (observations, rewards, dones, infos) matching the Stable Baselines API.
"""
function step!(env::MultiAgentVecEnv, actions::AbstractMatrix)
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