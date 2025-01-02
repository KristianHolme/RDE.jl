"""
    AbstractObservationStrategy

Abstract type for different strategies to observe the RDE system state.
"""
abstract type AbstractObservationStrategy end

"""
    FourierObservation <: AbstractObservationStrategy

Observation strategy using Fourier coefficients of state differences.

# Fields
- `fft_terms::Int`: Number of Fourier terms to use in observation
"""
struct FourierObservation <: AbstractObservationStrategy
    fft_terms::Int
end

"""
    StateObservation <: AbstractObservationStrategy

Observation strategy using the full normalized state vector.
"""
struct StateObservation <: AbstractObservationStrategy end

"""
    SampledStateObservation <: AbstractObservationStrategy

Observation strategy using sampled points from the state vector.

# Fields
- `n_samples::Int`: Number of points to sample from state vector
"""
struct SampledStateObservation <: AbstractObservationStrategy 
    n_samples::Int
end

abstract type AbstractRDEReward end

struct ShockSpanReward <: AbstractRDEReward 
    target_shock_count::Int
    span_scale::Float32
    shock_weight::Float32
    function ShockSpanReward(;target_shock_count::Int=3, span_scale::Float32=4.0f0, shock_weight::Float32=0.8f0)
        return new(target_shock_count, span_scale, shock_weight)
    end
end

mutable struct ShockPreservingReward <: AbstractRDEReward 
    target_shock_count::Int
    span_scale::Float32
    shock_weight::Float32
    abscence_limit::Float32
    abscence_start::Union{Float32, Nothing}
    function ShockPreservingReward(;target_shock_count::Int=3,
                                    span_scale::Float32=4.0f0, 
                                    shock_weight::Float32=0.8f0,
                                    abscence_limit::Float32=5.0f0)
        return new(target_shock_count, span_scale, shock_weight, abscence_limit, nothing)
    end
end

mutable struct ShockPreservingSymmetryReward <: AbstractRDEReward 
    target_shock_count::Int
    cache::Vector{Float32}  
    function ShockPreservingSymmetryReward(;target_shock_count::Int=4,
                                    N::Int = 512)
        return new(target_shock_count, zeros(Float32, N))
    end
end


"""
    RDEEnvCache{T<:AbstractFloat}

Cache for RDE environment computations and state tracking.

# Fields
- `circ_u::CircularVector{T}`: Circular buffer for velocity field
- `circ_λ::CircularVector{T}`: Circular buffer for reaction progress
- `prev_u::Vector{T}`: Previous velocity field
- `prev_λ::Vector{T}`: Previous reaction progress
"""
mutable struct RDEEnvCache{T<:AbstractFloat}
    circ_u::CircularVector{T, Vector{T}}
    circ_λ::CircularVector{T, Vector{T}}
    prev_u::Vector{T}  # Previous step's u values
    prev_λ::Vector{T}  # Previous step's λ values
    
    function RDEEnvCache{T}(N::Int) where {T<:AbstractFloat}
        # Initialize all arrays with zeros instead of undefined values
        circ_u = CircularArray(zeros(T, N))
        circ_λ = CircularArray(zeros(T, N))
        prev_u = zeros(T, N)
        prev_λ = zeros(T, N)
        
        return new{T}(circ_u, circ_λ, prev_u, prev_λ)
    end
end

"""
    RDEEnv{T<:AbstractFloat} <: AbstractEnv

Reinforcement learning environment for the RDE system.

# Fields
- `prob::RDEProblem{T}`: Underlying RDE problem
- `state::Vector{T}`: Current system state
- `observation::Vector{T}`: Current observation vector
- `dt::T`: Time step
- `t::T`: Current time
- `done::Bool`: Episode termination flag
- `reward::T`: Current reward
- `smax::T`: Maximum value for s parameter
- `u_pmax::T`: Maximum value for u_p parameter
- `α::T`: Action momentum parameter
- `τ_smooth::T`: Control smoothing time constant
- `cache::RDEEnvCache{T}`: Environment cache
- `action_type::AbstractActionType`: Type of control actions
- `reward_type::AbstractRDEReward`: Type of reward function
- `verbose::Bool`: Control solver output
# Constructor
```julia
RDEEnv{T}(;
    dt=10.0,
    smax=4.0,
    u_pmax=1.2,
    params::RDEParam{T}=RDEParam{T}(),
    momentum=0.5,
    τ_smooth=1.25,
    fft_terms::Int=32,
    observation_strategy::AbstractObservationStrategy=FourierObservation(fft_terms),
    action_type::AbstractActionType=ScalarPressureAction(),
    reward_type::AbstractRDEReward=ShockSpanReward(target_shock_count=3),
    verbose::Bool=true,
    kwargs...
) where {T<:AbstractFloat}
```

# Example
```julia
env = RDEEnv(dt=5.0, smax=3.0)
```
"""
mutable struct RDEEnv{T<:AbstractFloat} <: AbstractEnv
    prob::RDEProblem{T}                  # RDE problem
    state::Vector{T}
    observation::Vector{T}
    dt::T                       # time step
    t::T                        # Current time
    done::Bool                        # Termination flag
    truncated::Bool
    reward::T
    smax::T
    u_pmax::T
    α::T #action momentum
    τ_smooth::T #smoothing time constant
    cache::RDEEnvCache{T}
    action_type::AbstractActionType
    observation_strategy::AbstractObservationStrategy
    reward_type::AbstractRDEReward
    verbose::Bool               # Control solver output
    function RDEEnv{T}(;
        dt=1.0,
        smax=4.0,
        u_pmax=1.2,
        params::RDEParam{T}=RDEParam{T}(),
        momentum=0.0,
        τ_smooth=0.1,
        fft_terms::Int=16,
        observation_strategy::AbstractObservationStrategy=FourierObservation(fft_terms),
        action_type::AbstractActionType=ScalarPressureAction(),
        reward_type::AbstractRDEReward=ShockSpanReward(target_shock_count=3),
        verbose::Bool=true,
        kwargs...) where {T<:AbstractFloat}

        if τ_smooth > dt
            @warn "τ_smooth > dt, this will cause discontinuities in the control signal"
            @info "Setting τ_smooth = $(dt/8)"
            τ_smooth = dt/8
        end

        prob = RDEProblem(params; kwargs...)
        prob.cache.τ_smooth = τ_smooth

        # Set N in action_type
        set_N!(action_type, params.N)

        fft_terms = min(fft_terms, params.N ÷ 2)

        initial_state = vcat(prob.u0, prob.λ0)
        init_observation = init_observation_vector(observation_strategy, params.N)
        cache = RDEEnvCache{T}(params.N)
        return new{T}(prob, initial_state, init_observation,
                      dt, 0.0, false, false, 0.0, smax, u_pmax,
                      momentum, τ_smooth, cache,
                      action_type, observation_strategy, 
                      reward_type, verbose)
    end
end

RDEEnv(; kwargs...) = RDEEnv{Float32}(; kwargs...)
RDEEnv(params::RDEParam{T}; kwargs...) where {T<:AbstractFloat} = RDEEnv{T}(; params=params, kwargs...)

"""
    compute_observation(env::RDEEnv{T}, strategy::AbstractObservationStrategy) where {T}

Compute observation given an observation strategy.

# Arguments
- `env::RDEEnv{T}`: RDE environment
- `strategy::AbstractObservationStrategy`: Observation strategy

# Returns
- Vector containing observation
"""
function compute_observation(env::RDEEnv{T}, strategy::AbstractObservationStrategy) where {T}
    @error "compute_observation not implemented for strategy $(typeof(strategy))"
end

function compute_observation(env::RDEEnv{T}, strategy::FourierObservation) where {T}
    N = env.prob.params.N
    
    current_u = @view env.state[1:N]
    current_λ = @view env.state[N+1:end]
    
    env.cache.circ_u[:] .= current_u
    env.cache.circ_λ[:] .= current_λ
    
    fft_u = abs.(fft(env.cache.circ_u))
    fft_λ = abs.(fft(env.cache.circ_λ))
    
    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    
    # Take relevant FFT terms
    u_terms = fft_u[1:n_terms]
    λ_terms = fft_λ[1:n_terms]
    
    # Min-max normalization to [0,1] range
    u_obs = (u_terms .- minimum(u_terms)) ./ (maximum(u_terms) - minimum(u_terms) + T(1e-8))
    λ_obs = (λ_terms .- minimum(λ_terms)) ./ (maximum(λ_terms) - minimum(λ_terms) + T(1e-8))
    
    # Scale to [-1,1] range if desired
    u_obs = 2 .* u_obs .- 1
    λ_obs = 2 .* λ_obs .- 1
    
    # Control parameters already naturally bounded
    s_scaled = mean(env.prob.cache.s_current) / env.smax
    u_p_scaled = mean(env.prob.cache.u_p_current) / env.u_pmax
    
    return vcat(u_obs, λ_obs, s_scaled, u_p_scaled)
end


function compute_observation(env::RDEEnv, rt::StateObservation)
    N = length(env.state) ÷ 2
    u = @view env.state[1:N]
    λ = @view env.state[N+1:end]
    
    ϵ = 1e-8
    u_max = max(maximum(abs.(u)), ϵ)
    λ_max = max(maximum(abs.(λ)), ϵ)
    
    normalized_state = similar(env.state)
    normalized_state[1:N] = u ./ u_max 
    normalized_state[N+1:end] = λ ./ λ_max
    s_scaled = mean(env.prob.cache.s_current) / env.smax
    u_p_scaled = mean(env.prob.cache.u_p_current) / env.u_pmax
    return vcat(normalized_state, s_scaled, u_p_scaled)
end


function compute_observation(env::RDEEnv, strategy::SampledStateObservation)
    N = env.prob.params.N
    n = strategy.n_samples
    
    u = @view env.state[1:N]
    λ = @view env.state[N+1:end]
    
    indices = round.(Int, range(1, N, length=n))
    sampled_u = u[indices]
    sampled_λ = λ[indices]
    
    normalized_time = env.t / env.prob.params.tmax
    ϵ = 1e-8
    u_max = max(maximum(abs.(sampled_u)), ϵ)
    λ_max = max(maximum(abs.(sampled_λ)), ϵ)
    
    sampled_u ./= u_max
    sampled_λ ./= λ_max
    return vcat(sampled_u, sampled_λ, normalized_time)
end

function set_reward!(env::RDEEnv, rt::AbstractRDEReward)
    @error "No reward set for type $(typeof(rt))"
end

"""
    CommonRLInterface.act!(env::RDEEnv{T}, action; saves_per_action::Int=0) where {T<:AbstractFloat}

Take an action in the environment.

# Arguments
- `env::RDEEnv{T}`: RDE environment
- `action`: Control action to take
- `saves_per_action::Int=0`: Number of intermediate saves per action

# Returns
- Current reward

# Notes
- Updates environment state and reward
- Handles smooth control transitions
- Supports multiple action types
"""
function CommonRLInterface.act!(env::RDEEnv{T}, action; saves_per_action::Int=0) where {T<:AbstractFloat}
    # Store current state before taking action
    N = env.prob.params.N
    env.cache.prev_u .= @view env.state[1:N]
    env.cache.prev_λ .= @view env.state[N+1:end]

    t_span = (env.t, env.t + env.dt)
    env.prob.cache.control_time = env.t

    prev_controls = [env.prob.cache.s_current, env.prob.cache.u_p_current]
    c = [env.prob.cache.s_current, env.prob.cache.u_p_current]
    c_max = [env.smax, env.u_pmax]

    normalized_standard_actions = get_standard_normalized_actions(env.action_type, action)
    
    for i in 1:2
        a = normalized_standard_actions[i]
        if any(abs.(a) .> 1)
            @warn "action $a out of bounds [-1,1]"
        end
        c_prev = c[i]
        c_hat = @. ifelse(a < 0, c_prev .* (a .+ 1), c_prev .+ (c_max[i] .- c_prev) .* a)
        c[i] = env.α .* c_prev .+ (1 - env.α) .* c_hat
    end

    env.prob.cache.s_previous = env.prob.cache.s_current
    env.prob.cache.u_p_previous = env.prob.cache.u_p_current
    env.prob.cache.s_current = c[1]
    env.prob.cache.u_p_current = c[2]

    @debug "taking action $action at time $(env.t), controls: $(mean.(prev_controls)) to $(mean.(c))"

    prob_ode = ODEProblem(RDE_RHS!, env.state, t_span, env.prob)
    
    if saves_per_action == 0
        sol = OrdinaryDiffEq.solve(prob_ode, Tsit5(), save_on=false, isoutofdomain=outofdomain, verbose=env.verbose)
    else
        saveat = env.dt / saves_per_action
        sol = OrdinaryDiffEq.solve(prob_ode, Tsit5(), saveat=saveat, isoutofdomain=outofdomain, verbose=env.verbose)
    end

    
    set_reward!(env, env.reward_type)
    if env.t ≥ env.prob.params.tmax
        env.done = true
        @debug "tmax reached, t=$(env.t)"
    end
    if sol.retcode != :Success || any(isnan.(sol.u[end]))
        if any(isnan.(sol.u[end]))
            @warn "NaN state detected"
        end
        @debug "ODE solver failed, controls: $(mean(prev_controls)) to $(mean(c))"
        env.truncated = true
        env.done = true
        env.reward = -2.0
    elseif env.truncated
        env.reward = -2.0
        env.done = true;
        @debug "truncated"
    else
        env.prob.sol = sol
        env.t = sol.t[end]
        env.state = sol.u[end]
    end
    @debug "reward: $(env.reward)"

    return env.reward
end



# CommonRLInterface implementations
CommonRLInterface.state(env::RDEEnv) = vcat(env.state, env.t)
CommonRLInterface.terminated(env::RDEEnv) = env.done
function CommonRLInterface.observe(env::RDEEnv)
    return compute_observation(env, env.observation_strategy)
end

function CommonRLInterface.actions(env::RDEEnv)
    n = n_actions(env.action_type)
    return [(-1 .. 1) for _ in 1:n]
end

function CommonRLInterface.clone(env::RDEEnv)
    env2 = deepcopy(env)
    @debug "env is copied!"
    return env2
end

function CommonRLInterface.setstate!(env::RDEEnv, s)
    env.state = s[1:end-1]
    env.t = s[end]
end

function POMDPs.initialobs(RLEnvPOMDP, s)
    return [CommonRLInterface.observe(RLEnvPOMDP.env)]
end

"""
    CommonRLInterface.reset!(env::RDEEnv)

Reset the environment to its initial state.

# Arguments
- `env::RDEEnv`: Environment to reset

# Effects
- Resets time to 0
- Resets state to initial conditions
- Resets reward to 0
- Resets control parameters to initial values
- Initializes previous state tracking
"""
function CommonRLInterface.reset!(env::RDEEnv)
    env.t = 0
    set_init_state!(env.prob)
    env.state = vcat(env.prob.u0, env.prob.λ0)
    env.reward = 0.0
    env.done = false
    env.truncated = false
    set_reward!(env, env.reward_type)

    env.prob.cache.τ_smooth = env.τ_smooth
    env.prob.cache.u_p_previous = fill(env.prob.params.u_p, env.prob.params.N)
    env.prob.cache.u_p_current = fill(env.prob.params.u_p, env.prob.params.N)
    env.prob.cache.s_previous = fill(env.prob.params.s, env.prob.params.N)
    env.prob.cache.s_current = fill(env.prob.params.s, env.prob.params.N)

    # Initialize previous state
    N = env.prob.params.N
    env.cache.prev_u .= @view env.state[1:N]
    env.cache.prev_λ .= @view env.state[N+1:end]
    
    nothing
end

"""
    PolicyRunData{T<:AbstractFloat}

Container for data collected during policy execution.

# Fields
- `action_ts::Vector{T}`: Time points for actions
- `ss::Vector{T}`: Control parameter s at each action
- `u_ps::Vector{T}`: Control parameter u_p at each action
- `rewards::Vector{T}`: Rewards at each action
- `energy_bal::Vector{T}`: Energy balance at each state
- `chamber_p::Vector{T}`: Chamber pressure at each state
- `state_ts::Vector{T}`: Time points for states
- `states::Vector{Vector{T}}`: States at each time point
"""
struct PolicyRunData{T<:AbstractFloat}
    action_ts::Vector{T} #time points for actions
    ss::Vector{T} #control parameter s at each action
    u_ps::Vector{T} #control parameter u_p at each action
    rewards::Vector{T} #rewards at each action
    energy_bal::Vector{T} #energy balance at each state
    chamber_p::Vector{T} #chamber pressure at each state
    state_ts::Vector{T} #time points for states
    states::Vector{Vector{T}} #states at each time point
end

"""
    run_policy(π::Policy, env::RDEEnv{T}; saves_per_action=1) where {T}

Run a policy `π` on the RDE environment and collect trajectory data.

# Arguments
- `π::Policy`: Policy to execute
- `env::RDEEnv{T}`: RDE environment to run the policy in
- `saves_per_action=1`: Save full state every `saves_per_action` steps

# Returns
`PolicyRunData` containing:
- `ts`: Time points for each action
- `ss`: Control parameter s values
- `u_ps`: Control parameter u_p values  
- `rewards`: Rewards received
- `energy_bal`: Energy balance at each step
- `chamber_p`: Chamber pressure at each step
- `state_ts`: Time points for each state
- `states`: Full system state at each time point

# Example
```julia
env = RDEEnv()
policy = ConstantRDEPolicy(env)
data = run_policy(policy, env, saves_per_action=10)
```
"""
function run_policy(π::Policy, env::RDEEnv{T}; saves_per_action=1) where {T}
    reset!(env)
    dt = env.dt
    max_steps = ceil(env.prob.params.tmax / dt) + 1 |> Int
    
    # Initialize vectors for action data
    ts = Vector{T}(undef, max_steps)
    ss = Vector{T}(undef, max_steps)
    u_ps = Vector{T}(undef, max_steps)
    rewards = Vector{T}(undef, max_steps)
    
    # For saves_per_action > 0, we need more space for state data
    max_state_points = if saves_per_action == 0
        max_steps  # Only save at action points
    else
        max_steps * (saves_per_action + 1)  # +1 to account for potential extra points
    end
    
    energy_bal = Vector{T}(undef, max_state_points)
    chamber_p = Vector{T}(undef, max_state_points)
    states = Vector{Vector{T}}(undef, max_state_points)
    state_ts = Vector{T}(undef, max_state_points)
    
    step = 0
    total_state_steps = 0
    
    function log!(step)
        ts[step] = env.t
        ss[step] = mean(env.prob.cache.s_current)
        u_ps[step] = mean(env.prob.cache.u_p_current)
        rewards[step] = env.reward

        step_states = env.prob.sol.u[2:end]
        step_ts = env.prob.sol.t[2:end]
        n_states = length(step_states)

        # Calculate indices for this step's data
        start_idx = total_state_steps + 1
        end_idx = total_state_steps + n_states
        
        # Ensure we have enough space
        if end_idx > max_state_points
            # Extend arrays if needed
            new_size = end_idx + max_steps * (saves_per_action + 1)
            resize!(energy_bal, new_size)
            resize!(chamber_p, new_size)
            resize!(states, new_size)
            resize!(state_ts, new_size)
            max_state_points = new_size
        end
        
        # Save states and timestamps
        state_ts[start_idx:end_idx] = step_ts
        states[start_idx:end_idx] = step_states
        
        # Save energy balance and chamber pressure
        energy_bal[start_idx:end_idx] = energy_balance.(step_states, Ref(env.prob.params))
        chamber_p[start_idx:end_idx] = chamber_pressure.(step_states, Ref(env.prob.params))
        
        total_state_steps += n_states
    end

    while !env.done && step < max_steps
        step += 1
        action = POMDPs.action(π, observe(env))
        act!(env, action, saves_per_action=saves_per_action)
        log!(step)
    end
    
    # Trim arrays to actual size
    ts = ts[1:step]
    ss = ss[1:step]
    u_ps = u_ps[1:step]
    rewards = rewards[1:step]
    energy_bal = energy_bal[1:total_state_steps]
    chamber_p = chamber_p[1:total_state_steps]
    state_ts = state_ts[1:total_state_steps]
    states = states[1:total_state_steps]

    return PolicyRunData{T}(ts, ss, u_ps, rewards, energy_bal, chamber_p, state_ts, states)
end

"""
    ConstantRDEPolicy <: Policy

Policy that maintains constant control values.

# Fields
- `env::RDEEnv`: RDE environment

# Notes
Returns [0.0, 0.0] for ScalarAreaScalarPressureAction
Returns 0.0 for ScalarPressureAction
"""
struct ConstantRDEPolicy <: Policy
    env::RDEEnv
    ConstantRDEPolicy(env::RDEEnv=RDEEnv()) = new(env)
end

function POMDPs.action(π::ConstantRDEPolicy, s)
    if π.env.action_type isa ScalarAreaScalarPressureAction
        return [0.0, 0.0]
    elseif π.env.action_type isa ScalarPressureAction
        return 0.0
    else
        @error "Unknown action type $(typeof(π.env.action_type)) for ConstantRDEPolicy"
    end
end

"""
    SinusoidalRDEPolicy{T<:AbstractFloat} <: Policy

Policy that applies sinusoidal control signals.

# Fields
- `env::RDEEnv{T}`: RDE environment
- `w_1::T`: Phase speed parameter for first action
- `w_2::T`: Phase speed parameter for second action

# Constructor
```julia
SinusoidalRDEPolicy(env::RDEEnv{T}; w_1::T=1.0, w_2::T=2.0) where {T<:AbstractFloat}
```
"""
struct SinusoidalRDEPolicy{T<:AbstractFloat} <: Policy
    env::RDEEnv{T}
    w_1::T  # Phase speed parameter for first action
    w_2::T  # Phase speed parameter for second action

    function SinusoidalRDEPolicy(env::RDEEnv{T}; w_1::T=1.0, w_2::T=2.0) where {T<:AbstractFloat}
        new{T}(env, w_1, w_2)
    end
end

function POMDPs.action(π::SinusoidalRDEPolicy, s)
    t = s[end]
    action1 = sin(π.w_1 * t)
    action2 = sin(π.w_2 * t)
    if π.env.action_type isa ScalarAreaScalarPressureAction 
        return [action1, action2]
    elseif π.env.action_type isa ScalarPressureAction
        return action2
    else
        @error "Unknown action type $(typeof(π.env.action_type)) for SinusoidalRDEPolicy"
    end
end

"""
    StepwiseRDEPolicy{T<:AbstractFloat} <: Policy

Policy that applies predefined control values at specified times.

# Fields
- `env::RDEEnv{T}`: RDE environment
- `ts::Vector{T}`: Vector of time steps
- `c::Vector{Vector{T}}`: Vector of control actions

# Notes
- Only supports ScalarAreaScalarPressureAction
- Requires sorted time steps
- Each control action must have 2 elements
"""
struct StepwiseRDEPolicy{T<:AbstractFloat} <: Policy
    env::RDEEnv{T}
    ts::Vector{T}  # Vector of time steps
    c::Vector{Vector{T}}  # Vector of control actions

    function StepwiseRDEPolicy(env::RDEEnv{T}, ts::Vector{T}, c::Vector{Vector{T}}) where {T<:AbstractFloat}
        @assert length(ts) == length(c) "Length of time steps and control actions must be equal"
        @assert all(length(action) == 2 for action in c) "Each control action must have 2 elements"
        @assert issorted(ts) "Time steps must be in ascending order"
        @assert env.action_type isa ScalarAreaScalarPressureAction "StepwiseRDEPolicy only supports ScalarAreaScalarPressureAction"
        env.α = 0.0 #to assure that get_scaled_control works
        new{T}(env, ts, c)
    end
end

function POMDPs.action(π::StepwiseRDEPolicy, s)
    t = π.env.t
    cache = π.env.prob.cache
    past = π.ts .≤ t    
    idx = findlast(past)
    if isnothing(idx)
        return [0.0, 0.0]
    end
    return get_scaled_control.([cache.s_current[1], cache.u_p_current[1]], [π.env.smax, π.env.u_pmax], π.c[idx])
end

"""
    get_scaled_control(current, max_val, target)

Scale control value to [-1, 1] range based on current value and target.

# Arguments
- `current`: Current control value
- `max_val`: Maximum allowed value
- `target`: Target control value

# Returns
Scaled control value in [-1, 1]

# Notes
Assumes zero momentum (env.α = 0)
"""
function get_scaled_control(current, max_val, target)
    if target < current
        return target / current - 1.0
    else
        return (target - current) / (max_val - current)
    end
end

"""
    RandomRDEPolicy{T<:AbstractFloat} <: Policy

Policy that applies random control values.

# Fields
- `env::RDEEnv{T}`: RDE environment

# Notes
Generates random values in [-1, 1] for each control dimension
"""
struct RandomRDEPolicy{T<:AbstractFloat} <: Policy
    env::RDEEnv{T}
end

function POMDPs.action(π::RandomRDEPolicy, state)
    action1 = 2 * rand() - 1
    action2 = 2 * rand() - 1
    if π.env.action_type isa ScalarAreaScalarPressureAction
        return [action1, action2]
    elseif π.env.action_type isa ScalarPressureAction
        return action2
    else
        @error "Unknown action type $(typeof(π.env.action_type)) for RandomRDEPolicy"
    end
end

"""
    init_observation_vector(strategy::AbstractObservationStrategy, N::Int)

Initialize observation vector for given strategy.

# Arguments
- `strategy`: Observation strategy
- `N`: Number of grid points

# Returns
Preallocated vector for observations
"""
function init_observation_vector(strategy::FourierObservation, N::Int)
    n_terms = min(strategy.fft_terms, N ÷ 2 + 1)
    return Vector{Float32}(undef, n_terms * 2 + 2)
end

function init_observation_vector(::StateObservation, N::Int)
    return Vector{Float32}(undef, 2N + 2)
end

function init_observation_vector(strategy::SampledStateObservation, N::Int)
    return Vector{Float32}(undef, 2 * strategy.n_samples + 1)
end

