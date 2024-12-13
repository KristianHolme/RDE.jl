mutable struct RDEEnvCache{T<:AbstractFloat}
    circ_u::CircularVector{T, Vector{T}}
    circ_λ::CircularVector{T, Vector{T}}
    prev_u::Vector{T}  # Previous step's u values
    prev_λ::Vector{T}  # Previous step's λ values
    fft_terms::Int  # Number of Fourier terms to keep
    function RDEEnvCache{T}(N::Int; fft_terms::Int=8) where {T<:AbstractFloat}
        @assert fft_terms <= N "Number of FFT terms cannot exceed spatial resolution"
        return new{T}(
            CircularArray(Vector{T}(undef, N)), 
            CircularArray(Vector{T}(undef, N)),
            Vector{T}(undef, N),
            Vector{T}(undef, N),
            fft_terms
        )
    end
end

mutable struct RDEEnv{T<:AbstractFloat} <: AbstractEnv
    prob::RDEProblem{T}                  # RDE problem
    state::Vector{T}
    observation::Vector{T}
    dt::T                       # time step
    t::T                        # Current time
    done::Bool                        # Termination flag
    reward::T
    smax::T
    u_pmax::T
    observation_samples::Int64
    reward_func::Function
    α::T #action momentum
    τ_smooth::T #smoothing time constant
    cache::RDEEnvCache{T}
    action_type::AbstractActionType
    function RDEEnv{T}(;
        dt=10.0,
        smax=4.0,
        u_pmax=1.2,
        observation_samples::Int64=-1,
        params::RDEParam{T}=RDEParam{T}(),
        reward_func::Function=RDE_reward_combined!,
        momentum=0.5,
        c=0.0,
        τ_smooth=1.25,
        fft_terms::Int=32,
        action_type::AbstractActionType=ScalarPressureAction(),
        kwargs...) where {T<:AbstractFloat}

        if τ_smooth > dt
            @warn "τ_smooth > dt, this will cause discontinuities in the control signal"
            @info "Setting τ_smooth = $(dt/8)"
            τ_smooth = dt/8
        end

        if observation_samples == -1
            observation_samples = params.N
        end
        @assert observation_samples == params.N "observation_samples must be equal to params.N"

        prob = RDEProblem(params; kwargs...)
        prob.cache.τ_smooth = τ_smooth

        # Set N in action_type
        set_N!(action_type, params.N)

        fft_terms = min(fft_terms, params.N ÷ 2)

        initial_state = vcat(prob.u0, prob.λ0)
        init_observation = Vector{T}(undef, fft_terms * 2 + 1)
        cache = RDEEnvCache{T}(observation_samples; fft_terms=fft_terms)
        return new{T}(prob, initial_state, init_observation, dt, 0.0, false, 0.0,
            smax, u_pmax, observation_samples, reward_func, momentum, τ_smooth, cache, action_type)
    end
end


RDEEnv(; kwargs...) = RDEEnv{Float32}(; kwargs...)
RDEEnv(params::RDEParam{T}) where {T<:AbstractFloat} = RDEEnv{T}(params=params)

function interpolate_state(env::RDEEnv)
    N = env.prob.params.N
    n = env.observation_samples
    L = env.prob.params.L
    dx = n / L
    xs_sample = LinRange(0.0, L, n + 1)[1:end-1]
    u = env.state[1:N]
    λ = env.state[N+1:end]

    itp_u = LinearInterpolation(env.prob.x, u)
    itp_λ = LinearInterpolation(env.prob.x, λ)

    env.observation[1:n] = itp_u(xs_sample)
    env.observation[n+1:end] = itp_λ(xs_sample)
    return env.observation
end

# CommonRLInterface.reward(env::RDEEnv) = env.reward
CommonRLInterface.state(env::RDEEnv) = vcat(env.state, env.t)
CommonRLInterface.terminated(env::RDEEnv) = env.done
function CommonRLInterface.observe(env::RDEEnv)
    N = env.prob.params.N
    
    # Get current state components
    current_u = @view env.state[1:N]
    current_λ = @view env.state[N+1:end]
    
    # Calculate state differences
    env.cache.circ_u[:] .= current_u .- env.cache.prev_u
    env.cache.circ_λ[:] .= current_λ .- env.cache.prev_λ
    
    # Compute FFT and get magnitudes (shift-invariant)
    fft_u = abs.(fft(env.cache.circ_u))
    fft_λ = abs.(fft(env.cache.circ_λ))
    
    # Keep only first n terms (up to Nyquist frequency)
    n_terms = min(env.cache.fft_terms, N ÷ 2 + 1)
    
    # Normalize FFT coefficients
    # We can use the DC component (first coefficient) as normalization factor
    # Adding a small epsilon to avoid division by zero
    ϵ = 1e-8
    norm_factor_u = max(maximum(fft_u), ϵ)
    norm_factor_λ = max(maximum(fft_λ), ϵ)
    
    u_obs = fft_u[1:n_terms] ./ norm_factor_u
    λ_obs = fft_λ[1:n_terms] ./ norm_factor_λ
    
    # Add normalized time to observation
    normalized_time = env.t / env.prob.params.tmax
    
    # Return concatenated FFT magnitudes and normalized time
    return vcat(u_obs, λ_obs, normalized_time)
end

function CommonRLInterface.actions(env::RDEEnv)
    return [(-1 .. 1)]
end

#TODO test that this works
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


function CommonRLInterface.reset!(env::RDEEnv)
    # if env.t > 0 && env.t < 40
    #     error("resetting early?")
    # end
    env.t = 0
    set_init_state!(env.prob)
    env.state = vcat(env.prob.u0, env.prob.λ0)
    env.reward = 0.0
    env.done = false
    env.reward_func(env)

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
    # if !isa(action, AbstractArray) 
    #     action = [T(0.0), action] #only control u_p
    #     # action = [3.5/env.smax*2 - 1, action]
    # elseif length(action) == 1
    #     action = [T(0.0), action[1]] #only control u_p
    #     # action = [3.5/env.smax*2 - 1, action[1]]
    # end
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

    @debug "taking action $action at time $(env.t), controls: $prev_controls to $c"

    prob_ode = ODEProblem(RDE_RHS!, env.state, t_span, env.prob)
    
    # Determine solver settings based on extra_saves_per_step
    if saves_per_action == 0
        sol = OrdinaryDiffEq.solve(prob_ode, Tsit5(), save_on=false, isoutofdomain=outofdomain)
    else
        saveat = env.dt / saves_per_action
        sol = OrdinaryDiffEq.solve(prob_ode, Tsit5(), saveat=saveat, isoutofdomain=outofdomain)
    end

    env.prob.sol = sol
    env.t = sol.t[end]
    env.state = sol.u[end]

    env.reward_func(env)
    if env.t ≥ env.prob.params.tmax
        env.done = true
    end
    if sol.retcode != :Success
        @debug "ODE solver failed, controls: $prev_controls to $c"
        env.done = true
        env.reward = -2.0
    end

    return env.reward
end

function RDE_reward_max!(env::RDEEnv) #just to have something
    prob = env.prob
    N = prob.params.N
    u = env.state[1:N]
    env.reward = maximum(u)
    nothing
end

function RDE_reward_energy_balance!(env::RDEEnv)
    env.reward = -1 * energy_balance(env.state, env.prob.params)
    nothing
end

function RDE_reward_combined!(env::RDEEnv)
    prob = env.prob
    params = prob.params

    # Calculate energy balance
    energy_bal = energy_balance(env.state, params)

    # Calculate chamber pressure using the function from utils.jl
    pressure = chamber_pressure(env.state, params)

    # Combine rewards
    # We want to maximize chamber pressure and minimize energy imbalance
    # The negative sign for energy_bal is because we want to minimize it
    # The weights can be adjusted based on the relative importance of each component
    weight_energy = 0.6
    weight_pressure = 0.4
    weight_span = 1.0
    u, = split_sol(env.state)
    amplitude_ratio = (maximum(u) - minimum(u)) / maximum(u)

    env.reward = weight_pressure * pressure
    -weight_energy * abs(energy_bal)
    +weight_span * amplitude_ratio

    nothing
end


"""
    run_policy(π::Policy, env::RDEEnv{T}; sparse_skip=1, tmax=26.0, overacting=1) where {T}

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
´´´julia
env = RDEEnv()
policy = ConstantRDEPolicy(env)
data = run_policy(policy, env, saves_per_action=10)
´´´
"""
function run_policy(π::Policy, env::RDEEnv{T}; saves_per_action=1) where {T}
    reset!(env)
    dt = env.dt
    max_steps = ceil(env.prob.params.tmax / dt) + 1 |> Int
    
    # Initialize vectors with maximum possible size
    ts = Vector{T}(undef, max_steps)
    ss = Vector{T}(undef, max_steps)
    u_ps = Vector{T}(undef, max_steps)
    energy_bal = Vector{T}(undef, max_steps*saves_per_action)
    chamber_p = Vector{T}(undef, max_steps*saves_per_action)
    states = Vector{Vector{T}}(undef, max_steps*saves_per_action)
    state_ts = Vector{T}(undef, max_steps*saves_per_action)
    rewards = Vector{T}(undef, max_steps)
    
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
        
        # Save states and timestamps
        state_ts[start_idx:end_idx] = step_ts
        states[start_idx:end_idx] = step_states
        
        # Save energy balance and chamber pressure
        energy_bal[start_idx:end_idx] = energy_balance(step_states, env.prob.params)
        chamber_p[start_idx:end_idx] = chamber_pressure(step_states, env.prob.params)
        
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

function POMDPs.action(π::StepwiseRDEPolicy, state)
    t = state[end]
    controls = π.c
    s = π.env.prob.cache.s_current
    u_p = π.env.prob.cache.u_p_current
    
    idx = searchsortedlast(π.ts, t)
    @debug "t = $t, s = $s, u_p = $u_p, idx = $idx"

    if idx == 0
        return [0.0, 0.0]  # Default action before the first time step
    else
        a = zeros(2)
        a[1] = get_scaled_control(s, π.env.smax, controls[idx][1])
        a[2] = get_scaled_control(u_p, π.env.u_pmax, controls[idx][2])
        return a
    end
end


function get_scaled_control(current, max_val, target)
    #assumes env.momentum == 0
    if target < current
        return target / current - 1.0
    else
        return (target - current) / (max_val - current)
    end
end


struct RandomRDEPolicy{T<:AbstractFloat} <: Policy
    env::RDEEnv{T}
end

function POMDPs.action(π::RandomRDEPolicy, state)
    # Generate two random numbers between -1 and 1
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

