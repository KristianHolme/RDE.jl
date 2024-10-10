# using FFTW
# using DifferentialEquations
# # using ProgressLogging
# using LinearAlgebra
# using LoopVectorization


Base.@kwdef mutable struct RDEParam
    N::Int = 256             # Number of spatial points
    L::Float64 = 2π          # Domain length
    ν_1::Float64 = 0.1       # Viscosity coefficient
    ν_2::Float64 = 0.1
    u_c::Float64 = 1.1       # Parameter in ω(u)
    α::Float64 = 0.3         # Parameter in ω(u)
    q_0::Float64 = 1.0       # Source term coefficient
    u_0::Float64 = 0.0       # Parameter in ξ(u, u_0)
    n::Int = 1               # Exponent in ξ(u, u_0)
    k_param::Float64 = 5.0   # Parameter in β(u, s)
    u_p::Float64 = 0.5       # Parameter in β(u, s)
    s::Float64 = 3.5         # Parameter in β(u, s)
    ϵ::Float64 = 0.15        # Small parameter in ξ(u)
    tmax::Float64 = 50.0     # Maximum simulation time
    x0::Float64 = 1          # Initial position
    saveframes::Int64 = 75    # Number of time steps to 
end
Base.length(params::RDEParam) = 1
# Define the RDEProblem structure with default values and explanations
mutable struct RDECache
    u_hat::Vector{ComplexF64}      # Complex array of size N÷2+1
    u_x_hat::Vector{ComplexF64}    # Complex array of size N÷2+1
    u_x::Vector{Float64}           # Real array of size N
    u_xx_hat::Vector{ComplexF64}   # Complex array of size N÷2+1
    u_xx::Vector{Float64}          # Real array of size N

    λ_hat::Vector{ComplexF64}      # Complex array of size N÷2+1
    λ_xx_hat::Vector{ComplexF64}   # Complex array of size N÷2+1
    λ_xx::Vector{Float64}          # Real array of size N

    ωu::Vector{Float64}            # Real array of size N
    ξu::Vector{Float64}            # Real array of size N
    βu::Vector{Float64}            # Real array of size N

    function RDECache(N::Int)
        N_complex = div(N, 2) + 1          # Size for complex arrays in rfft
        u_hat = Vector{ComplexF64}(undef, N_complex)
        u_x_hat = Vector{ComplexF64}(undef, N_complex)
        u_xx_hat = Vector{ComplexF64}(undef, N_complex)
        u_x = Vector{Float64}(undef, N)
        u_xx = Vector{Float64}(undef, N)

        λ_hat = Vector{ComplexF64}(undef, N_complex)
        λ_xx_hat = Vector{ComplexF64}(undef, N_complex)
        λ_xx = Vector{Float64}(undef, N)

        ωu = Vector{Float64}(undef, N)
        ξu = Vector{Float64}(undef, N)
        βu = Vector{Float64}(undef, N)
        return new(u_hat, u_x_hat, u_x, u_xx_hat, u_xx, λ_hat, λ_xx_hat, λ_xx, ωu, ξu, βu)
    end
end

mutable struct RDEProblem
    # Parameters with defaults and explanations
    params::RDEParam
    # Precomputed variables (initialized in init!)
    dx::Float64              # Spatial resolution
    x::Vector{Float64}       # Spatial grid
    k::Vector{Float64}       # Wavenumbers
    ik::Vector{ComplexF64}   # Spectral derivative operator (i*k)
    k2::Vector{Float64}      # Square of wavenumbers for Laplacian
    u0::Vector{Float64}      # Initial condition for u(x, 0)
    u_init::Function
    λ0::Vector{Float64}      # Initial condition for λ(x, 0)
    λ_init::Function
    sol::Union{Nothing,ODESolution}  # Solution (initially nothing)
    dealiasing::Vector{Float64}
    cache::RDECache
    fft_plan::FFTW.rFFTWPlan{Float64}
    ifft_plan::FFTW.ScaledPlan

    # Constructor accepting keyword arguments to override defaults
    function RDEProblem(params::RDEParam; 
            u_init= (x, x0) -> (3 / 2) * (sech(x - x0)) ^ (20),
            λ_init= x->0.5, dealias=true)
        prob = new()
        prob.params = params
        prob.dx = prob.params.L / prob.params.N
        prob.x = prob.dx * (0:prob.params.N-1)
        prob.ik, prob.k2 = create_spectral_derivative_arrays(params.N)
        prob.dealiasing = create_dealiasing_vector(params.N)
        if !dealias
            prob.dealiasing = ones(Float64, length(prob.dealiasing))
        end

        prob.u_init = u_init
        prob.λ_init = λ_init
        set_init_state!(prob)
    
        prob.λ0 = λ_init.(prob.x)
        prob.sol = nothing
        prob.cache = RDECache(params.N)
        prob.fft_plan = plan_rfft(prob.u0; flags=FFTW.MEASURE)
        prob.ifft_plan = plan_irfft(prob.cache.u_hat, length(prob.u0); flags=FFTW.MEASURE)
        set_init_state!(prob) #as u0 may have been wiped while creating fft plans
        return prob
    end
end

function create_dealiasing_vector(N::Int)
    N_complex = div(N, 2) + 1
    k = collect(0:N_complex-1)
    k_cutoff = div(N, 3)

    # Construct the dealiasing vector using broadcasting
    dealiasing = @. ifelse(k <= k_cutoff, 1.0, 0.0)

    return dealiasing
end

function create_spectral_derivative_arrays(N::Int)
    N_complex = div(N, 2) + 1
    k = collect(0:N_complex-1)
    ik = 1im .* k
    k2 = k .^ 2
    return ik, k2
end

function set_init_state!(prob::RDEProblem)
    prob.u0 = prob.u_init.(prob.x, prob.params.x0)
    prob.λ0 = prob.λ_init.(prob.x)
end

ω(u, u_c, α) = exp((u-u_c)/α)
ξ(u, u_0, n) = (u_0 - u)*u^n
β(u, s, u_p, k) = s*u_p/(1 + exp(k * (u - u_p)))
# β(u, s, u_p, k) = s/(1 + exp(k * (u - u_p)))


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
    ik = prob.ik             # Complex array of size N÷2+1
    k2 = prob.k2             # Real array of size N÷2+1
    dealiasing = prob.dealiasing  # Real array of size N÷2+1
    cache = prob.cache       # Preallocated arrays

    # Extract physical components
    u = @view uλ[1:N]
    λ = @view uλ[N+1:end]
    du = @view duλ[1:N]
    dλ = @view duλ[N+1:end]


    # Preallocated arrays
    u_hat = cache.u_hat            # Complex array of size N÷2+1
    u_x_hat = cache.u_x_hat        # Complex array of size N÷2+1
    u_xx_hat = cache.u_xx_hat      # Complex array of size N÷2+1
    u_x = cache.u_x                # Real array of size N
    u_xx = cache.u_xx              # Real array of size N
    λ_hat = cache.λ_hat            # Complex array of size N÷2+1
    λ_xx_hat = cache.λ_xx_hat      # Complex array of size N÷2+1
    λ_xx = cache.λ_xx
    ωu = cache.ωu                  # Real array of size N
    ξu = cache.ξu                  # Real array of size N
    βu = cache.βu                  # Real array of size N

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

    # Compute nonlinear terms using fused broadcasting
    @turbo @. ωu = ω(u, u_c, α)
    @turbo @. ξu = ξ(u, u_0, n)
    @turbo @. βu = β(u, s, u_p, k_param)

    # RHS for u_t
    @turbo @. du = -u * u_x + (1 - λ) * ωu * q_0 + ν_1 * u_xx + ϵ * ξu

    # RHS for λ_t
    @turbo @. dλ = (1 - λ) * ωu - βu * λ + ν_2*λ_xx
end



# Solve the PDE with an optional solver argument
function solve_pde!(prob::RDEProblem; solver=nothing, kwargs...)
    uλ_0 = vcat(prob.u0, prob.λ0)
    tspan = (0.0, prob.params.tmax)

    saveat = prob.params.tmax / prob.params.saveframes

    prob_ode = ODEProblem(RDE_RHS!, uλ_0, tspan, prob)

    sol = DifferentialEquations.solve(prob_ode, solver; saveat=saveat, kwargs...)

    # Store the solution in the struct
    prob.sol = sol
end

"""
Calculate
Ė₍domain₎ = ∫₀ᴸ (q(1-λ)ω(u) - ϵξ(u))dx
"""
function energy_balance(u::Vector{Float64}, λ::Vector{Float64}, params::RDEParam)
    q_0 = params.q_0
    u_c = params.u_c
    α = params.α
    ϵ = params.ϵ
    u_0 = params.u_0
    n = params.n
    L = params.L
    N = params.N
    dx = L/N

    integrand = q_0*(1 .- λ) .* ω.(u, u_c, α) .- ϵ*ξ.(u, u_0, n)

    # trapezoidal_rule_integral = dx * sum(integrand)

    simpsons_rule_integral = periodic_simpsons_rule(integrand, dx)

    return simpsons_rule_integral
end

function periodic_simpsons_rule(u::Vector{T}, dx::T) where T <: Real
    dx/3*sum((2*u[1:2:end] + 4*u[2:2:end]))
end

function energy_balance(uλ::Vector{Float64}, params::RDEParam)
    u, λ = split_sol(uλ)
    energy_balance(u, λ, params)
end
function energy_balance(uλs::Vector{Vector{T}}, params::RDEParam) where T <: Real
    [energy_balance(uλ, params) for uλ in uλs]
end