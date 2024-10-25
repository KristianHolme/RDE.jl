# using CommonRLInterface
# using Interpolations
# using IntervalSets
# using POMDPs

mutable struct RDEEnv{T <: AbstractFloat} <: AbstractEnv
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

    function RDEEnv{T}(;
        dt = 0.5,
        smax = 4.0,
        u_pmax = 1.5,
        observation_samples::Int64 = 9,
        params::RDEParam{T} = RDEParam{T}(),
        reward_func::Function = RDE_reward_combined!,
        momentum = 0.5,
        kwargs...) where T <:AbstractFloat

        prob = RDEProblem(params; kwargs...)
        initial_state = vcat(prob.u0, prob.λ0)
        init_observation = Vector{T}(undef, observation_samples*2)
        
        return new{T}(prob, initial_state, init_observation, dt, 0.0, false, 0.0, 
                   smax, u_pmax, observation_samples, reward_func, momentum)
    end
end


RDEEnv(;kwargs...) = RDEEnv{Float32}(;kwargs...) 
RDEEnv(params::RDEParam{T}) where T <: AbstractFloat = RDEEnv{T}(params=params)

function interpolate_state(env::RDEEnv)
    N = env.prob.params.N
    n = env.observation_samples
    L = env.prob.params.L
    dx = n/L
    xs_sample = LinRange(0.0, L, n+1)[1:end-1]
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
CommonRLInterface.observe(env::RDEEnv) = vcat(interpolate_state(env), env.t)

function CommonRLInterface.actions(env::RDEEnv)
    return [(-1..1) for i in 1:2]
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


function CommonRLInterface.reset!(env::RDEEnv)
    # if env.t > 0 && env.t < 40
    #     error("resetting early?")
    # end
    env.t = 0
    set_init_state!(env.prob)
    env.state = vcat(env.prob.u0, env.prob.λ0)
    env.reward = 0.0
    env.done = false
    nothing
end

function CommonRLInterface.act!(env::RDEEnv, action)
    t_span = (env.t, env.t + env.dt)

    c = [env.prob.params.s, env.prob.params.u_p]
    c_max = [env.smax, env.u_pmax]


    for i in 1:2
        
        a = action[i]
        if abs(a) > 1
            @warn "action $a out of bounds [-1,1]"
        end
        c_prev = c[i]
        c_hat = a<0 ? c_prev*(a+1) : c_prev + (c_max[i]-c_prev)*a
        if c_hat> c_max[i] || c_hat < 0
            @warn "c_hat $c_hat out of bounds $c_max[$i]"
        end
        c[i] = env.α*c_prev + (1-env.α)*c_hat
    end
    env.prob.params.s = c[1]
    env.prob.params.u_p = c[2]
    
    prob_ode = ODEProblem(RDE_RHS!, env.state, t_span, env.prob)
    sol = OrdinaryDiffEq.solve(prob_ode, Tsit5(), save_on=true)
    env.prob.sol = sol
    env.t = sol.t[end]
    env.state = sol.u[end]

    if env.t ≥ env.prob.params.tmax || sol.retcode != :Success
        env.done = true
    end
    env.reward_func(env)
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
    env.reward = -1*energy_balance(env.state, env.prob.params)
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



function run_policy(π::Policy, env::RDEEnv{T}; sparse_skip=1, tmax=26.0) where T
    reset!(env)
    env.prob.params.tmax = tmax
    dt = env.dt
    N = ceil(tmax/dt)+1 |> Int

    ts = Vector{T}(undef,N)
    ss = Vector{T}(undef,N)
    u_ps = Vector{T}(undef,N)
    energy_bal = Vector{T}(undef,N)
    chamber_p = Vector{T}(undef,N)
    sparse_N = Int(ceil(N/sparse_skip))
    sparse_logged = 0
    sparse_states = Vector{Vector{T}}(undef, sparse_N)
    sparse_ts = Vector{T}(undef, sparse_N)

    function log!(step)
        ts[step] = env.t
        ss[step] = env.prob.params.s
        u_ps[step] = env.prob.params.u_p
        state = CommonRLInterface.state(env)[1:end-1]
        energy_bal[step] = energy_balance(state, env.prob.params)
        chamber_p[step] = chamber_pressure(state, env.prob.params)
        if (step-1)%sparse_skip == 0
            sparse_step = (step-1) ÷ sparse_skip + 1
            sparse_states[sparse_step] = state
            sparse_ts[sparse_step] = env.t
            sparse_logged += 1
        elseif step == N
            sparse_logged += 1
            sparse_states[end] = state
            sparse_ts[end] = env.t
        end
    end

    for step = 1:N
        log!(step)
        action = POMDPs.action(π, state(env))
        act!(env, action)
    end

    if sparse_logged != sparse_N
        @warn "sparse logged= $(sparse_logged) not equal to sparse N= $(sparse_N)"
        pop!(sparse_ts)
        pop!(sparse_states)
    end

    return PolicyRunData(ts, ss, u_ps, energy_bal, chamber_p, sparse_ts, sparse_states)
end

struct PolicyRunData
    ts
    ss
    u_ps
    energy_bal
    chamber_p
    sparse_ts
    sparse_states
end

struct ConstantRDEPolicy <: Policy
    env::RDEEnv
    ConstantRDEPolicy(env::RDEEnv = RDEEnv()) = new(env)
end

function POMDPs.action(π::ConstantRDEPolicy, s)
    return [0,0]
end

struct SinusoidalRDEPolicy{T <: AbstractFloat} <: Policy
    env::RDEEnv{T}
    w_1::T  # Phase speed parameter for first action
    w_2::T  # Phase speed parameter for second action
    
    function SinusoidalRDEPolicy(env::RDEEnv{T}; w_1::T = 1.0, w_2::T = 2.0) where T <: AbstractFloat
        new{T}(env, w_1, w_2)
    end
end

function POMDPs.action(π::SinusoidalRDEPolicy, s)
    t = s[end]
    action1 = sin(π.w_1 * t)
    action2 = sin(π.w_2 * t)
    return [action1, action2]
end

struct StepwiseRDEPolicy{T <: AbstractFloat} <: Policy
    env::RDEEnv{T}
    ts::Vector{T}  # Vector of time steps
    c::Vector{Vector{T}}  # Vector of control actions

    function StepwiseRDEPolicy(env::RDEEnv{T}, ts::Vector{T}, c::Vector{Vector{T}}) where T <: AbstractFloat
        @assert length(ts) == length(c) "Length of time steps and control actions must be equal"
        @assert all(length(action) == 2 for action in c) "Each control action must have 2 elements"
        @assert issorted(ts) "Time steps must be in ascending order"
        env.α = 0.0 #to assure that get_scaled_control works
        new{T}(env, ts, c)
    end
end

function POMDPs.action(π::StepwiseRDEPolicy, state)
    t = state[end]
    controls = π.c
    s = π.env.prob.params.s
    u_p = π.env.prob.params.u_p
    
    idx = searchsortedlast(π.ts, t)
    
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
        return target/current - 1.0
    else
        return (target - current)/(max_val - current)
    end
end

