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
- `u`: Velocity field
- `s`: Quenching strength parameter
- `u_p`: Pressure parameter
- `k`: Steepness parameter

# Returns
- Refueling rate: s * u_p / (1 + exp(k * (u - u_p)))
"""
β(u, s, u_p, k) = s .* u_p ./ (1 .+ exp.(k .* (u .- u_p)))

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
    
    cache = prob.cache       # Preallocated arrays

    # Extract physical components
    u = @view uλ[1:N]
    λ = @view uλ[N+1:end]
    du = @view duλ[1:N]
    dλ = @view duλ[N+1:end]

    calc_derivatives!(u, λ, prob.cache)

    u_x = cache.u_x
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
    shift = Int(round(prob.control_shift_func(u, t) / dx))
    
    # Apply shifts
    apply_periodic_shift!(cache.u_p_t_shifted, cache.u_p_t, shift)
    apply_periodic_shift!(cache.s_t_shifted, cache.s_t, shift)
    
    @debug "RHS:u_p_t $(cache.u_p_t_shifted), s_t $(cache.s_t_shifted) at time $t"
    @debug "RHS:u_p_current $(cache.u_p_current), s_current $(cache.s_current)"
    
    # Use shifted controls in calculations
    @turbo @. ωu = ω(u, u_c, α)
    @turbo @. ξu = ξ(u, u_0, n)
    @turbo @. βu = β(u, cache.s_t_shifted, cache.u_p_t_shifted, k_param)

    # Combine both loops to reduce overhead
    @turbo @. du = -u * u_x + (1 - λ) * ωu * q_0 + ν_1 * u_xx + ϵ * ξu
    @turbo @. dλ = (1 - λ) * ωu - βu * λ + ν_2 * λ_xx
    nothing
end

"""
    calc_derivatives!(u::T, λ::T, cache::PseudospectralRDECache) where T <:AbstractArray

Calculate spatial derivatives using pseudospectral method with FFT.

# Arguments
- `u`: Velocity field
- `λ`: Reaction progress
- `cache`: Pseudospectral cache containing FFT plans and workspace arrays

# Implementation Notes
- Uses in-place FFT operations
- Applies dealiasing filter in spectral space
- Computes first and second derivatives for u
- Computes second derivative for λ
- Handles periodic boundary conditions automatically
"""
function calc_derivatives!(u::T, λ::T, cache::PseudospectralRDECache) where T <:AbstractArray
    ik = cache.ik             # Complex array of size N÷2+1
    k2 = cache.k2             # Real array of size N÷2+1
    dealias_filter = cache.dealias_filter  # Real array of size N÷2+1
    # Preallocated arrays
    u_hat = cache.u_hat            # Complex array of size N÷2+1
    u_x_hat = cache.u_x_hat        # Complex array of size N÷2+1
    u_xx_hat = cache.u_xx_hat      # Complex array of size N÷2+1
    u_x = cache.u_x                # Real array of size N
    u_xx = cache.u_xx              # Real array of size N
    λ_hat = cache.λ_hat            # Complex array of size N÷2+1
    λ_xx_hat = cache.λ_xx_hat      # Complex array of size N÷2+1
    λ_xx = cache.λ_xx
    
    # Transform to spectral space (in-place)
    mul!(u_hat, cache.fft_plan, u)  # Apply fft_plan to u, store in u_hat
    mul!(λ_hat, cache.fft_plan, λ)

    # Compute derivatives in spectral space (with dealiasing)
    @. u_x_hat = ik * u_hat * dealias_filter
    @. u_xx_hat = -k2 * u_hat * dealias_filter
    @. λ_xx_hat = -k2 * λ_hat * dealias_filter

    # Inverse FFT to get derivatives in physical space (in-place)
    mul!(u_x, cache.ifft_plan, u_x_hat)
    mul!(u_xx, cache.ifft_plan, u_xx_hat)
    mul!(λ_xx, cache.ifft_plan, λ_xx_hat)
    nothing
end

"""
    calc_derivatives!(u::T, λ::T, cache::FDRDECache) where T <: AbstractArray

Calculate spatial derivatives using finite difference method.

# Arguments
- `u`: Velocity field
- `λ`: Reaction progress
- `cache`: Finite difference cache containing grid parameters and workspace arrays

# Implementation Notes
- Uses second-order central differences
- Handles periodic boundary conditions explicitly
- Computes first and second derivatives for u
- Computes second derivative for λ
- Optimized with @turbo macro for performance
"""
function calc_derivatives!(u::T, λ::T, cache::FDRDECache) where T <: AbstractArray
    dx = cache.dx
    N = cache.N
    inv_2dx = 1 / (2 * dx)
    inv_dx2 = 1 / dx^2
    
    # Preallocated arrays
    u_x = cache.u_x               
    u_xx = cache.u_xx              
    λ_xx = cache.λ_xx
    
    # Compute u_x using central differences with periodic boundary conditions
    u_x[1] = (u[2] - u[N]) * inv_2dx
    @turbo for i in 2:N-1
        u_x[i] = (u[i+1] - u[i-1]) * inv_2dx
    end
    u_x[N] = (u[1] - u[N-1]) * inv_2dx
    
    # Compute u_xx using central differences with periodic boundary conditions
    u_xx[1] = (u[2] - 2 * u[1] + u[N]) * inv_dx2
    @turbo for i in 2:N-1
        u_xx[i] = (u[i+1] - 2 * u[i] + u[i-1]) * inv_dx2
    end
    u_xx[N] = (u[1] - 2 * u[N] + u[N-1]) * inv_dx2
    
    # Compute λ_xx using central differences with periodic boundary conditions
    λ_xx[1] = (λ[2] - 2 * λ[1] + λ[N]) * inv_dx2
    @turbo for i in 2:N-1
        λ_xx[i] = (λ[i+1] - 2 * λ[i] + λ[i-1]) * inv_dx2
    end
    λ_xx[N] = (λ[1] - 2 * λ[N] + λ[N-1]) * inv_dx2

    nothing
end

"""
    outofdomain(uλ, prob, t)

Check if the solution has left the physical domain.

# Arguments
- `uλ`: Current state [u; λ]
- `prob`: RDE problem
- `t`: Current time

# Returns
- `true` if solution is unphysical (u < 0 or λ ∉ [0,1])
- `false` otherwise
"""
function outofdomain(uλ, prob, t)
    N = prob.params.N
    u = @view uλ[1:N]
    λ = @view uλ[N+1:end]
    u_out = any(u .< 0.0)
    λ_out = any((λ .< 0.0) .| (λ .> 1.0))
    return u_out || λ_out
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
function solve_pde!(prob::RDEProblem; solver=Tsit5(), kwargs...)
    uλ_0 = vcat(prob.u0, prob.λ0)
    tspan = (zero(typeof(prob.params.tmax)), prob.params.tmax)

    saveat = prob.params.tmax / prob.params.saveframes

    prob_ode = ODEProblem(RDE_RHS!, uλ_0, tspan, prob)

    sol = OrdinaryDiffEq.solve(prob_ode, solver; saveat=saveat, isoutofdomain=outofdomain, kwargs...)
    if sol.retcode != :Success
        @warn "failed to solve PDE"
    end
    # Store the solution in the struct
    prob.sol = sol
end
