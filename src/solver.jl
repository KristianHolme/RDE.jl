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
    write_advection!(du, cache)

Write the conservative advective contribution from `cache.adv` into `du`.
"""
function write_advection!(du::AbstractVector{T}, cache::FVCache{T}) where {T}
    @turbo for i in eachindex(du)
        du[i] = cache.adv[i]
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
"""
function RDE_RHS!(duλ, uλ, prob::RDEProblem{T, M, R, C}, t) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
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
    update_control_shifted!(cache, prob.control_shift_strategy, u, t)

    # @logmsg LogLevel(-10000) "RHS:u_p_t $(cache.u_p_t_shifted), s_t $(cache.s_t_shifted) at time $t"
    # @logmsg LogLevel(-10000) "RHS:u_p_current $(cache.u_p_current), s_current $(cache.s_current)"

    @turbo @. ωu = ω(u, u_c, α)

    @turbo @. ξu = ξ(u, u_0, n)

    @turbo @. βu = β(u, cache.s_t_shifted, cache.u_p_t_shifted, k_param)

    write_advection!(du, cache)
    @turbo @. du = du + (one(T) - λ) * ωu * q_0 + ν_1 * u_xx + ϵ * ξu

    @turbo @. dλ = (one(T) - λ) * ωu - βu * λ + ν_2 * λ_xx

    return nothing
end


"""
    solve_pde!(prob::RDEProblem; kwargs...)

Solve the RDE system using the specified ODE solver.

# Arguments
- `prob`: RDE problem containing initial conditions and parameters

# Keywords
- `alg=SSPRK33()`: ODE solver to use (default: SSPRK33)
- `adaptive=false`: Whether to use adaptive time stepping
- `dt=nothing`: Fixed time step to use when `adaptive=false`
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
solve_pde!(prob; alg=OrdinaryDiffEq.SSPRK33())  # Use a different solver
```
"""
function solve_pde!(
        prob::RDEProblem;
        saveframes = 75,
        alg = SSPRK33(),
        adaptive = false,
        dt = nothing,
        callback = nothing,
        kwargs...
    )
    tspan = (zero(typeof(prob.params.tmax)), prob.params.tmax)
    saveat = prob.params.tmax / saveframes

    uλ_0 = vcat(prob.u0, prob.λ0)
    prob_ode = ODEProblem(RDE_RHS!, uλ_0, tspan, prob)
    sol = solve_pde_step(
        prob,
        prob_ode;
        alg = alg,
        adaptive = adaptive,
        dt = dt,
        callback = callback,
        saveat = saveat,
        kwargs...
    )
    if sol.retcode != :Success
        @warn "failed to solve PDE"
    end
    # Store the solution in the struct
    prob.sol = sol
    return prob
end

function solve_pde_step(
        rde_problem::RDEProblem{T, M, R, C},
        ode_problem::ODEProblem;
        alg,
        adaptive,
        dt,
        callback,
        kwargs...
    ) where {T <: AbstractFloat, M <: FiniteVolumeMethod, R <: AbstractReset, C <: AbstractControlShift}
    cfl_cb = StepsizeLimiter(
        cfl_dtFE;
        safety_factor = T(0.62),
        max_step = true,
        cached_dtcache = zero(T)
    )

    if callback === nothing && adaptive == false
        callback = cfl_cb
    end

    dt0 = if adaptive == false && dt === nothing
        cfl_dtFE(ode_problem.u0, rde_problem, zero(T))
    else
        dt
    end

    sol = OrdinaryDiffEq.solve(
        ode_problem,
        alg;
        adaptive = adaptive,
        dt = dt0,
        isoutofdomain = outofdomain,
        callback = callback,
        kwargs...
    )
    return sol
end
