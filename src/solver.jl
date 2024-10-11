mutable struct RDEParam{T<:AbstractFloat}
    N::Int               # Number of spatial points
    L::T              # Domain length
    ν_1::T           # Viscosity coefficient
    ν_2::T
    u_c::T            # Parameter in ω(u)
    α::T            # Parameter in ω(u)
    q_0::T            # Source term coefficient
    u_0::T            # Parameter in ξ(u, u_0)
    n::Int                 # Exponent in ξ(u, u_0)
    k_param::T        # Parameter in β(u, s)
    u_p::T            # Parameter in β(u, s)
    s::T              # Parameter in β(u, s)
    ϵ::T             # Small parameter in ξ(u)
    tmax::T          # Maximum simulation time
    x0::T             # Initial position
    saveframes::Int       # Number of time steps to save
end
Base.length(params::RDEParam) = 1

function RDEParam{T}(;
    N=256,
    L=2π,
    ν_1=0.1,
    ν_2=0.1,
    u_c=1.1,
    α=0.3,
    q_0=1.0,
    u_0=0.0,
    n=1,
    k_param=5.0,
    u_p=0.5,
    s=3.5,
    ϵ=0.15,
    tmax=26.0,
    x0=1.0,
    saveframes=75) where {T<:AbstractFloat}
    RDEParam{T}(N, L, ν_1, ν_2, u_c, α, q_0, u_0, n, k_param, u_p, s, ϵ, tmax, x0, saveframes)
end

RDEParam(;kwargs...) = RDEParam{Float64}(;kwargs...)

"""Cache for computed values in the solver. To avoid allocations during simulation."""
mutable struct RDECache{T<:AbstractFloat}
    u_hat::Vector{Complex{T}}      # Complex array of size N÷2+1
    u_x_hat::Vector{Complex{T}}    # Complex array of size N÷2+1
    u_x::Vector{T}           # Real array of size N
    u_xx_hat::Vector{Complex{T}}   # Complex array of size N÷2+1
    u_xx::Vector{T}          # Real array of size N

    λ_hat::Vector{Complex{T}}      # Complex array of size N÷2+1
    λ_xx_hat::Vector{Complex{T}}   # Complex array of size N÷2+1
    λ_xx::Vector{T}          # Real array of size N

    ωu::Vector{T}            # Real array of size N
    ξu::Vector{T}            # Real array of size N
    βu::Vector{T}            # Real array of size N

    function RDECache{T}(N::Int) where {T<:AbstractFloat}
        N_complex = div(N, 2) + 1          # Size for complex arrays in rfft
        u_hat = Vector{Complex{T}}(undef, N_complex)
        u_x_hat = Vector{Complex{T}}(undef, N_complex)
        u_xx_hat = Vector{Complex{T}}(undef, N_complex)
        u_x = Vector{T}(undef, N)
        u_xx = Vector{T}(undef, N)

        λ_hat = Vector{Complex{T}}(undef, N_complex)
        λ_xx_hat = Vector{Complex{T}}(undef, N_complex)
        λ_xx = Vector{T}(undef, N)

        ωu = Vector{T}(undef, N)
        ξu = Vector{T}(undef, N)
        βu = Vector{T}(undef, N)
        return new{T}(u_hat, u_x_hat, u_x, u_xx_hat, u_xx, λ_hat, λ_xx_hat, λ_xx, ωu, ξu, βu)
    end
end

mutable struct RDEProblem{T<:AbstractFloat}
    # Parameters with defaults and explanations
    params::RDEParam{T}
    # Precomputed variables (initialized in init!)
    dx::T              # Spatial resolution
    x::Vector{T}       # Spatial grid
    k::Vector{T}       # Wavenumbers
    ik::Vector{Complex{T}}   # Spectral derivative operator (i*k)
    k2::Vector{T}      # Square of wavenumbers for Laplacian
    u0::Vector{T}      # Initial condition for u(x, 0)
    u_init::Function
    λ0::Vector{T}      # Initial condition for λ(x, 0)
    λ_init::Function
    sol::Union{Nothing,ODESolution}  # Solution (initially nothing)
    dealiasing::Vector{T}
    cache::RDECache{T}
    fft_plan::FFTW.rFFTWPlan{T}
    ifft_plan::FFTW.ScaledPlan

    # Constructor accepting keyword arguments to override defaults
    function RDEProblem{T}(params::RDEParam{T};
        u_init= (x, x0) -> (3 / 2) * (sech(x - x0))^(20),
        λ_init=x -> 0.5, dealias=true) where {T<:AbstractFloat}

        prob = new{T}()
        prob.params = params
        prob.dx = prob.params.L / prob.params.N
        prob.x = prob.dx * (0:prob.params.N-1)
        prob.ik, prob.k2 = create_spectral_derivative_arrays(params.N)
        prob.dealiasing = create_dealiasing_vector(params.N, T)
        if !dealias
            prob.dealiasing = ones(T, length(prob.dealiasing))
        end

        prob.u_init = u_init
        prob.λ_init = λ_init
        set_init_state!(prob)

        prob.λ0 = λ_init.(prob.x)
        prob.sol = nothing
        prob.cache = RDECache{T}(params.N)
        prob.fft_plan = plan_rfft(prob.u0; flags=FFTW.MEASURE)
        prob.ifft_plan = plan_irfft(prob.cache.u_hat, length(prob.u0); flags=FFTW.MEASURE)
        set_init_state!(prob) #as u0 may have been wiped while creating fft plans
        return prob
    end
end

RDEProblem(params::RDEParam{T}; kwargs...) where {T<:AbstractFloat} = RDEProblem{T}(params; kwargs...)

function create_dealiasing_vector(N::Int, T::Type=Float64)
    N_complex = div(N, 2) + 1
    k = collect(0:N_complex-1)
    k_cutoff = div(N, 3)

    # Construct the dealiasing vector using broadcasting
    dealiasing = @. ifelse(k <= k_cutoff, one(T), zero(T))

    return dealiasing
end

function create_spectral_derivative_arrays(N::Int, T::Type=Float64)
    N_complex = div(N, 2) + 1
    k = collect(T, 0:N_complex-1)
    ik = 1im .* k
    k2 = k .^ 2
    return ik, k2
end

function set_init_state!(prob::RDEProblem)
    prob.u0 = prob.u_init.(prob.x, prob.params.x0)
    prob.λ0 = prob.λ_init.(prob.x)
end

ω(u, u_c, α) = exp((u - u_c) / α)
ξ(u, u_0, n) = (u_0 - u) * u^n
β(u, s, u_p, k) = s * u_p / (1 + exp(k * (u - u_p)))
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
    @turbo @. dλ = (1 - λ) * ωu - βu * λ + ν_2 * λ_xx
end



# Solve the PDE with an optional solver argument
function solve_pde!(prob::RDEProblem; solver=nothing, kwargs...)
    uλ_0 = vcat(prob.u0, prob.λ0)
    tspan = (zero(typeof(prob.params.tmax)), prob.params.tmax)

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