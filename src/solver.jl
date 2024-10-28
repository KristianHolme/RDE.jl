
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

    calc_derivatives!(u, λ, prob.cache)

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

    @turbo for i in eachindex(u)
        du[i] = -u[i] * u_x[i] + (1 - λ[i]) * ωu[i] * q_0 + ν_1 * u_xx[i] + ϵ * ξu[i]
    end

    @turbo for i in eachindex(λ)
        dλ[i] = (1 - λ[i]) * ωu[i] - βu[i] * λ[i] + ν_2 * λ_xx[i]
    end
end

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

function outofdomain(uλ, params, t)
    u, λ = split_sol(uλ)
    u_out = any(u .< 0.0)
    λ_out = any((λ .< 0.0) .| (λ .> 1.0))
    return u_out || λ_out
end

# Solve the PDE with an optional solver argument
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
