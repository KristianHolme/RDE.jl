
ω(u, u_c, α) = exp((u - u_c) / α)
ξ(u, u_0, n) = (u_0 - u) * u^n 
β(u, s, u_p, k) = s * u_p / (1 + exp(k * (u - u_p)))


# RHS function for the ODE solver using in-place operations
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
    u_p = prob.params.u_p
    s = prob.params.s
    
    cache = prob.cache       # Preallocated arrays

    # Extract physical components
    u = @view uλ[1:N]
    λ = @view uλ[N+1:end]
    du = @view duλ[1:N]
    dλ = @view duλ[N+1:end]

    prob.calc_derivatives(u, λ, prob)

    u_x = cache.u_x
    u_xx = cache.u_xx
    λ_xx = cache.λ_xx
    ωu = cache.ωu                  # Real array of size N
    ξu = cache.ξu                  # Real array of size N
    βu = cache.βu                  # Real array of size N

    # Compute nonlinear terms using fused broadcasting
    @turbo @. ωu = ω(u, u_c, α)
    @turbo @. ξu = ξ(u, u_0, n)
    @turbo @. βu = β(u, s, u_p, k_param)

    # RHS for u_t
    @turbo @. du = -u * u_x + (1 - λ) * ωu * q_0 + ν_1 * u_xx + ϵ * ξu

    # RHS for λ_t
    @turbo @. dλ = (1 - λ) * ωu - βu * λ + ν_2 * λ_xx
end

function pseudospectral_derivatives!(u::T, λ::T, prob::RDEProblem) where T <:AbstractArray
    cache = prob.cache
    ik = prob.ik             # Complex array of size N÷2+1
    k2 = prob.k2             # Real array of size N÷2+1
    dealiasing = prob.dealiasing  # Real array of size N÷2+1
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
    mul!(u_hat, prob.fft_plan, u)  # Apply fft_plan to u, store in u_hat
    mul!(λ_hat, prob.fft_plan, λ)

    # Compute derivatives in spectral space (with dealiasing)
    @. u_x_hat = ik * u_hat * dealiasing
    @. u_xx_hat = -k2 * u_hat * dealiasing

    @. λ_xx_hat = -k2 * λ_hat * dealiasing

    # Inverse FFT to get derivatives in physical space (in-place)
    mul!(u_x, prob.ifft_plan, u_x_hat)
    mul!(u_xx, prob.ifft_plan, u_xx_hat)

    mul!(λ_xx, prob.ifft_plan, λ_xx_hat)
    nothing
end

function fd_derivatives!(u::T, λ::T, prob::RDEProblem) where T <: AbstractArray
    cache = prob.cache
    dx = prob.dx
    N = prob.params.N
    inv_2dx = 1 / (2 * dx)
    inv_dx2 = 1 / dx^2
    
    # Preallocated arrays
    u_x = cache.u_x                # Real array of size N
    u_xx = cache.u_xx              # Real array of size N
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


# Solve the PDE with an optional solver argument
function solve_pde!(prob::RDEProblem; solver=nothing, kwargs...)
    uλ_0 = vcat(prob.u0, prob.λ0)
    tspan = (zero(typeof(prob.params.tmax)), prob.params.tmax)

    saveat = prob.params.tmax / prob.params.saveframes

    prob_ode = ODEProblem(RDE_RHS!, uλ_0, tspan, prob)

    sol = DifferentialEquations.solve(prob_ode, solver; saveat=saveat, kwargs...)
    if sol.retcode != :Success
        @warn "failed to solve PDE"
    end
    # Store the solution in the struct
    prob.sol = sol
end

"""
Calculate
Ė₍domain₎ = ∫₀ᴸ (q(1-λ)ω(u) - ϵξ(u))dx
"""
function energy_balance(u::Vector{T}, λ::Vector{T}, params::RDEParam) where T <: Real
    q_0 = params.q_0
    u_c = params.u_c
    α = params.α
    ϵ = params.ϵ
    u_0 = params.u_0
    n = params.n
    L = params.L
    N = params.N
    dx = L / N
    

    integrand = q_0 * (1 .- λ) .* ω.(u, u_c, α) .- ϵ * ξ.(u, u_0, n)

    # trapezoidal_rule_integral = dx * sum(integrand)

    simpsons_rule_integral = periodic_simpsons_rule(integrand, dx)

    return simpsons_rule_integral
end

function periodic_simpsons_rule(u::Vector{T}, dx::T) where {T<:Real}
    dx / 3 * sum((2 * u[1:2:end] + 4 * u[2:2:end]))
end

function energy_balance(uλ::Vector{T}, params::RDEParam) where T <: Real
    u, λ = split_sol(uλ)
    energy_balance(u, λ, params)
end
function energy_balance(uλs::Vector{Vector{T}}, params::RDEParam) where {T<:Real}
    [energy_balance(uλ, params) for uλ in uλs]
end

function chamber_pressure(uλ::Vector{T}, params::RDEParam;) where T <: Real
    if length(uλ) != params.N
        @assert length(uλ) == 2 * params.N
        u,  = split_sol(uλ)
    else
        u = uλ
    end
    L = params.L
    dx = L / params.N
    mean_pressure = periodic_simpsons_rule(u, dx)/L
    return mean_pressure
end

function chamber_pressure(uλs::Vector{Vector{T}}, params::RDEParam) where T <: Real
    [chamber_pressure(uλ, params) for uλ in uλs]
end

