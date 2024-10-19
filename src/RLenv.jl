using CommonRLInterface
using Interpolations
# using IntervalSets
# using DomainSets
using POMDPs

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
    max_state_val::T
    reward_func::Function
    action_num::Int64

    function RDEEnv{T}(;
        dt = 0.5,
        smax = 5.0,
        u_pmax = 3.0,
        observation_samples::Int64 = 9,
        params::RDEParam{T} = RDEParam{T}(),
        reward_func::Function = RDE_reward_energy_balance!,
        action_num::Int64 = 10,
        kwargs...) where T <:AbstractFloat

        prob = RDEProblem(params; kwargs...)
        initial_state = vcat(prob.u0, prob.λ0)
        init_observation = Vector{T}(undef, observation_samples*2)
        
        return new{T}(prob, initial_state, init_observation, dt, 0.0, false, 0.0, 
                   smax, u_pmax, observation_samples, 100.0, reward_func, action_num)
    end
end

#default to float32 to be gpu compatible
RDEEnv(;kwargs...) = RDEEnv{Float64}(;kwargs...) 

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
CommonRLInterface.state(env::RDEEnv) = env.state
CommonRLInterface.terminated(env::RDEEnv) = env.done
CommonRLInterface.observe(env::RDEEnv) = interpolate_state(env)

function CommonRLInterface.actions(env::RDEEnv)
    single_actions = LinRange(0, 1, env.action_num)
    collect(Iterators.product(single_actions, single_actions))[:]
end

#TODO test that this works
function CommonRLInterface.clone(env::RDEEnv)
    env2 = deepcopy(env)
    @debug "env is copied!"
    return env2
end

function CommonRLInterface.setstate!(env::RDEEnv, s)
    env.state = s
end

# function CommonRLInterface.state_space(env::RDEEnv)
    # (0.0 .. env.max_state_val)^env.observation_samples × (0.0 .. 1.1)^env.observation_samples 
# end

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

    env.prob.params.s = action[1]*env.smax #actions between 0 and 1
    env.prob.params.u_p = action[2]*env.u_pmax


    prob_ode = ODEProblem(RDE_RHS!, env.state, t_span, env.prob)
    sol = DifferentialEquations.solve(prob_ode)
    env.prob.sol = sol
    env.t = sol.t[end]
    env.state = sol.u[end]

    if env.t ≥ env.prob.params.tmax || sol.retcode != :Success || maximum(abs.(env.state)) > 100
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

function run_policy(π::P, env::RDEEnv; sparse_skip=5, tmax=26.0, T=Float64) where P <: Policy
    reset!(env)
    env.prob.params.tmax = tmax
    dt = env.dt
    N = ceil(tmax/dt)+1 |> Int

    ts = Vector{T}(undef,N)
    ss = Vector{T}(undef,N)
    u_ps = Vector{T}(undef,N)
    energy_bal = Vector{T}(undef,N)
    sparse_N = Int(ceil(N/sparse_skip))
    sparse_logged = 0
    sparse_states = Vector{Vector{T}}(undef, sparse_N)
    sparse_ts = Vector{T}(undef, sparse_N)

    function log!(step)
        ts[step] = env.t
        ss[step] = env.prob.params.s
        u_ps[step] = env.prob.params.u_p
        energy_bal[step] = energy_balance(state(env), env.prob.params)
        if (step-1)%sparse_skip == 0
            sparse_step = (step-1) ÷ sparse_skip + 1
            sparse_states[sparse_step] = state(env)
            sparse_ts[sparse_step] = env.t
            sparse_logged += 1
        elseif step == N
            sparse_logged += 1
            sparse_states[end] = state(env)
            sparse_ts[end] = env.t
        end
    end

    for step = 1:N
        log!(step)
        action = POMDPs.action(π, state(env))[1]
        act!(env, action)
    end

    if sparse_logged != sparse_N
        @warn "sparse logged $(sparse_logged) not equal to sparse N $(sparse_N)"
        pop!(sparse_ts)
        pop!(sparse_states)
    end

    return PolicyRunData(ts, ss, u_ps, energy_bal, sparse_ts, sparse_states)
end

struct PolicyRunData
    ts
    ss
    u_ps
    energy_bal
    sparse_ts
    sparse_states
end

struct ConstantRDEPolicy <: Policy
    env::RDEEnv
    ConstantRDEPolicy(env::RDEEnv = RDEEnv()) = new(env)
end

function POMDPs.action(π::ConstantRDEPolicy, s)
    return [(π.env.prob.params.s/π.env.smax, π.env.prob.params.u_p/π.env.u_pmax)]
end
