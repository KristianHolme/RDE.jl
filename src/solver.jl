"""
    ω(u, u_c, α)

Compute the reaction rate function ω(u).

# Arguments
- `u`: Velocity field
- `u_c`: Critical velocity
- `α`: Rate parameter

# Returns
- Reaction rate value: exp((u - u_c) / α)
"""
ω(u, u_c, α) = exp((u - u_c) / α)

"""
    ξ(u, u_0, n)

Compute the velocity damping function ξ(u).

# Arguments
- `u`: Velocity field
- `u_0`: Reference velocity
- `n`: Power law exponent

# Returns
- Damping value: (u_0 - u) * u^n
"""
ξ(u, u_0, n) = (u_0 - u) * u^n

"""
    β(u, s, u_p, k)

Compute the refueling function β(u).

# Arguments
- `u`: Velocity field (scalar)
- `s`: Quenching strength parameter
- `u_p`: Pressure parameter
- `k`: Steepness parameter

# Returns
- Refueling rate: s * u_p / (1 + exp(k * (u - u_p)))
"""
β(u, s, u_p, k) = s * u_p / (1 + exp(k * (u - u_p)))

"""
    write_advection!(du, u, cache)

Write the advective contribution into `du` using dispatch over cache type.
For conservative FV cache, uses `cache.adv`. Otherwise uses `-u .* u_x`.
"""
function write_advection!(du::AbstractVector{T}, u::AbstractVector{T}, cache::FVCache{T}) where {T}
    @turbo for i in eachindex(du)
        du[i] = cache.adv[i]
    end
    return nothing
end

function write_advection!(du::AbstractVector{T}, u::AbstractVector{T}, cache::AbstractRDECache{T}) where {T}
    @turbo for i in eachindex(du)
        du[i] = -u[i] * cache.u_x[i]
    end
    return nothing
end

"""
    RDE_RHS!(duλ, uλ, prob::RDEProblem, t)

Compute the right-hand side of the RDE system for the ODE solver.

# Arguments
- `duλ`: Output array for derivatives [du/dt; dλ/dt]
- `uλ`: Current state [u; λ]
- `prob`: RDE problem containing parameters and cache
- `t`: Current time

# System of equations
The RDE system consists of coupled PDEs for velocity (u) and reaction progress (λ):

```math
\\frac{∂u}{∂t} = -u\\frac{∂u}{∂x} + (1-λ)ω(u)q_0 + ν_1\\frac{∂^2u}{∂x^2} + ϵξ(u)
```
```math
\\frac{∂λ}{∂t} = (1-λ)ω(u) - β(u)λ + ν_2\\frac{∂^2λ}{∂x^2}
```

# Implementation Notes
- Uses in-place operations for efficiency
- Handles smooth control transitions
- Supports both pseudospectral and finite difference methods
- Includes periodic boundary conditions
"""
function RDE_RHS!(duλ, uλ, prob::RDEProblem, t)
    N = prob.params.N
    ν_1 = prob.params.ν_1
    ν_2 = prob.params.ν_2
    q_0 = prob.params.q_0
    ϵ = prob.params.ϵ
    u_c = prob.params.u_c
    α = prob.params.α
    u_0 = prob.params.u_0
    n = prob.params.n
    k_param = prob.params.k_param

    cache = prob.method.cache       # Get cache from method

    # Extract physical components
    u = @view uλ[1:N]
    λ = @view uλ[(N + 1):end]
    du = @view duλ[1:N]
    dλ = @view duλ[(N + 1):end]

    # Calculate derivatives using the method's cache
    calc_derivatives!(u, λ, prob.method)

    # For method-dependent fields below, not all caches have u_x; we branch later
    u_xx = cache.u_xx
    λ_xx = cache.λ_xx
    ωu = cache.ωu                  # Real array of size N
    ξu = cache.ξu                  # Real array of size N
    βu = cache.βu                  # Real array of size N
    u_p_t = cache.u_p_t
    s_t = cache.s_t

    # Update control values with smooth transition
    smooth_control!(u_p_t, t, cache.control_time, cache.u_p_current, cache.u_p_previous, cache.τ_smooth)
    smooth_control!(s_t, t, cache.control_time, cache.s_current, cache.s_previous, cache.τ_smooth)

    # Calculate shift based on current time
    dx = prob.params.L / prob.params.N
    shift = Int(round(get_control_shift(prob.control_shift_strategy, u, t) / dx))

    # Apply shifts
    if shift != 0
        circshift!(cache.u_p_t_shifted, cache.u_p_t, -shift)
        circshift!(cache.s_t_shifted, cache.s_t, -shift)
    else
        cache.u_p_t_shifted .= cache.u_p_t
        cache.s_t_shifted .= cache.s_t
    end

    # @logmsg LogLevel(-10000) "RHS:u_p_t $(cache.u_p_t_shifted), s_t $(cache.s_t_shifted) at time $t"
    # @logmsg LogLevel(-10000) "RHS:u_p_current $(cache.u_p_current), s_current $(cache.s_current)"

    @turbo @. ωu = ω(u, u_c, α)

    @turbo @. ξu = ξ(u, u_0, n)

    @turbo @. βu = β(u, cache.s_t_shifted, cache.u_p_t_shifted, k_param)

    write_advection!(du, u, cache)
    @turbo @. du = du + (1.0f0 - λ) * ωu * q_0 + ν_1 * u_xx + ϵ * ξu

    @turbo @. dλ = (1.0f0 - λ) * ωu - βu * λ + ν_2 * λ_xx

    return nothing
end


"""
    solve_pde!(prob::RDEProblem; solver=Tsit5(), kwargs...)

Solve the RDE system using the specified ODE solver.

# Arguments
- `prob`: RDE problem containing initial conditions and parameters

# Keywords
- `solver=Tsit5()`: ODE solver to use (default: Tsit5)
- `kwargs...`: Additional arguments passed to OrdinaryDiffEq.solve

# Implementation Notes
- Uses OrdinaryDiffEq.jl for time integration
- Checks for unphysical solutions using outofdomain
- Saves solution at specified intervals
- Stores solution in prob.sol

# Example
```julia
prob = RDEProblem(params)
solve_pde!(prob)
solve_pde!(prob, solver=Rodas4())  # Use a different solver
```
"""
function solve_pde!(prob::RDEProblem; solver = Tsit5(), saveframes = 75, kwargs...)
    uλ_0 = vcat(prob.u0, prob.λ0)
    tspan = (zero(typeof(prob.params.tmax)), prob.params.tmax)

    saveat = prob.params.tmax / saveframes

    prob_ode = ODEProblem(RDE_RHS!, uλ_0, tspan, prob)

    sol = OrdinaryDiffEq.solve(prob_ode, solver; saveat = saveat, isoutofdomain = outofdomain, kwargs...)
    if sol.retcode != :Success
        @warn "failed to solve PDE"
    end
    # Store the solution in the struct
    return prob.sol = sol
end

"""
    solve_pde!(prob::RDEProblem{T, FiniteVolumeMethod, R, C}; solver=SSPRK33(), saveat=nothing, dtmax=nothing, safety=0.9, kwargs...)

Multiple-dispatch overload for FiniteVolumeMethod: uses adaptive stepping with an
SSP RK by default, with a dynamic dtmax enforced via a DiscreteCallback based on
`cfl_dtmax`. If `saveat` is a vector, the solver will hit those exact times.
"""
function solve_pde!(prob::RDEProblem{T, M, R, C}; solver = SSPRK33(), saveat = nothing, dtmax = nothing, safety = 0.9, kwargs...) where {T <: AbstractFloat, M <: FiniteVolumeMethod, R <: AbstractReset, C <: AbstractControlShift}
    uλ_0 = vcat(prob.u0, prob.λ0)
    tspan = (zero(typeof(prob.params.tmax)), prob.params.tmax)
    default_saveat = isnothing(saveat) ? (prob.params.tmax / 75) : saveat

    prob_ode = ODEProblem(RDE_RHS!, uλ_0, tspan, prob)

    # Dynamic dtmax via callback to respect CFL while allowing adaptive substeps to hit saveat
    function cfl_affect!(integrator)
        u = view(integrator.u, 1:prob.params.N)
        integrator.opts.dtmax = isnothing(dtmax) ? cfl_dtmax(prob.params, u; safety = safety) : dtmax
    end
    cfl_condition(u, t, integrator) = true
    cfl_cb = DiscreteCallback(cfl_condition, cfl_affect!; save_positions = (false, false))

    # Enforce initial dtmax from initial condition as well
    dtmax0 = isnothing(dtmax) ? cfl_dtmax(prob.params, prob.u0; safety = safety) : dtmax

    sol = OrdinaryDiffEq.solve(prob_ode, solver; adaptive = true, dtmax = dtmax0, saveat = default_saveat, isoutofdomain = outofdomain, callback = cfl_cb, kwargs...)
    if sol.retcode != :Success
        @warn "failed to solve PDE"
    end
    return prob.sol = sol
end

"""
    solve_pde!(prob::RDEProblem{T, UpwindMethod, R, C}; solver=SSPRK33(), saveat=nothing, dtmax=nothing, safety=0.9, kwargs...)

Multiple-dispatch overload for UpwindMethod, same behavior as FiniteVolumeMethod.
"""
function solve_pde!(prob::RDEProblem{T, M, R, C}; solver = SSPRK33(), saveat = nothing, dtmax = nothing, safety = 0.9, kwargs...) where {T <: AbstractFloat, M <: UpwindMethod, R <: AbstractReset, C <: AbstractControlShift}
    uλ_0 = vcat(prob.u0, prob.λ0)
    tspan = (zero(typeof(prob.params.tmax)), prob.params.tmax)
    default_saveat = isnothing(saveat) ? (prob.params.tmax / 75) : saveat

    prob_ode = ODEProblem(RDE_RHS!, uλ_0, tspan, prob)

    function cfl_affect!(integrator)
        u = view(integrator.u, 1:prob.params.N)
        integrator.opts.dtmax = isnothing(dtmax) ? cfl_dtmax(prob.params, u; safety = safety) : dtmax
    end
    cfl_condition(u, t, integrator) = true
    cfl_cb = DiscreteCallback(cfl_condition, cfl_affect!; save_positions = (false, false))

    dtmax0 = isnothing(dtmax) ? cfl_dtmax(prob.params, prob.u0; safety = safety) : dtmax

    sol = OrdinaryDiffEq.solve(prob_ode, solver; adaptive = true, dtmax = dtmax0, saveat = default_saveat, isoutofdomain = outofdomain, callback = cfl_cb, kwargs...)
    if sol.retcode != :Success
        @warn "failed to solve PDE"
    end
    return prob.sol = sol
end
