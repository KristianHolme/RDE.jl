function set_reward!(env::RDEEnv{T}, rt::ShockSpanReward) where T<:AbstractFloat
    target_shock_count = rt.target_shock_count
    max_span = rt.span_scale
    λ = rt.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.params.L/env.prob.params.N
    shocks = Float32(RDE.count_shocks(u, dx))
    span = maximum(u) - minimum(u)
    span_reward = span/max_span
    if shocks >= target_shock_count
        shock_reward = one(T)
    elseif shocks > 0
        shock_reward = shocks/(2*target_shock_count)
    else
        shock_reward = T(-1.0)
    end
    
    env.reward = λ*shock_reward + (1-λ)*span_reward
    nothing
end

"""
    set_reward!(env::RDEEnv{T}, rt::ShockPreservingReward) where T

Reward for preserving a given number of shocks. terminated/truncated if the number of shocks is not preserved.
    penalize reward if shocks are not evenly spaced, reward for large span.
"""
function set_reward!(env::RDEEnv{T}, rt::ShockPreservingReward) where T<:AbstractFloat
    target_shock_count = rt.target_shock_count
    max_span = rt.span_scale
    λ = rt.shock_weight

    u, = RDE.split_sol_view(env.state)
    dx = env.prob.x[2] - env.prob.x[1]
    N = env.prob.params.N
    L = env.prob.params.L
    shock_inds = shock_indices(u, dx)

    span = maximum(u) - minimum(u)
    span_reward = span/max_span

    if length(shock_inds) != target_shock_count
        env.truncated = true
        env.reward = T(-2.0)
    else
        optimal_spacing = L/target_shock_count
        shock_spacing = mod.(periodic_diff(shock_inds), N) .* dx
        shock_reward = T(1.0) - mean(abs.((shock_spacing .- optimal_spacing)./optimal_spacing))
        env.reward = λ*shock_reward + (1-λ)*span_reward
    end
    nothing
end
