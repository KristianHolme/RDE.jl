"""
    RDEParam{T<:AbstractFloat}

Parameters for the rotating detonation engine (RDE) model.

# Fields
- `N::Int`: Number of spatial points
- `L::T`: Domain length
- `ν_1::T`: Viscosity coefficient for velocity field
- `ν_2::T`: Viscosity coefficient for reaction progress
- `u_c::T`: Parameter in ω(u)
- `α::T`: Parameter in ω(u)
- `q_0::T`: Source term coefficient
- `u_0::T`: Parameter in ξ(u, u_0)
- `n::Int`: Exponent in ξ(u, u_0)
- `k_param::T`: Parameter in β(u, s)
- `u_p::T`: Parameter in β(u, s)
- `s::T`: Parameter in β(u, s)
- `ϵ::T`: Small parameter in ξ(u)
- `tmax::T`: Maximum simulation time
- `x0::T`: Initial position
- `saveframes::Int`: Number of time steps to save
"""
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

"""
    RDEParam{T}(; kwargs...) where {T<:AbstractFloat}

Construct RDE parameters with specified type T.

# Keywords
- `N::Int=512`: Number of spatial points
- `L::T=2π`: Domain length
- `ν_1::T=0.0075`: Viscosity coefficient for velocity
- `ν_2::T=0.0075`: Viscosity coefficient for reaction
- `u_c::T=1.1`: Parameter in ω(u)
- `α::T=0.3`: Parameter in ω(u)
- `q_0::T=1.0`: Source term coefficient
- `u_0::T=0.0`: Parameter in ξ(u, u_0)
- `n::Int=1`: Exponent in ξ(u, u_0)
- `k_param::T=5.0`: Parameter in β(u, s)
- `u_p::T=0.5`: Parameter in β(u, s)
- `s::T=3.5`: Parameter in β(u, s)
- `ϵ::T=0.15`: Small parameter in ξ(u)
- `tmax::T=50.0`: Maximum simulation time
- `x0::T=1.0`: Initial position
- `saveframes::Int=75`: Number of time steps to save

# Example
```julia
params = RDEParam{Float64}(N=1024, tmax=100.0)
```
"""
function RDEParam{T}(;
    N=512,
    L=2π,
    ν_1=0.0075,
    ν_2=0.0075,
    u_c=1.1,
    α=0.3,
    q_0=1.0,
    u_0=0.0,
    n=1,
    k_param=5.0,
    u_p=0.5,
    s=3.5,
    ϵ=0.15,
    tmax=50.0,
    x0=1.0,
    saveframes=75) where {T<:AbstractFloat}
    RDEParam{T}(N, L, ν_1, ν_2, u_c, α, q_0, u_0, n, k_param, u_p, s, ϵ, tmax, x0, saveframes)
end

"""
    RDEParam(; kwargs...)

Construct RDE parameters with default type Float32. See [`RDEParam{T}`](@ref) for available keywords.
"""
RDEParam(; kwargs...) = RDEParam{Float32}(; kwargs...)



abstract type AbstractRDECache{T<:AbstractFloat} end

"""
    PseudospectralRDECache{T<:AbstractFloat} <: AbstractRDECache{T}

Cache for pseudospectral method computations in RDE solver.

# Fields
## Physical Space Arrays
- `u_x, u_xx`: Velocity derivatives
- `λ_xx`: Reaction progress derivatives
- `ωu, ξu, βu`: Nonlinear terms

## Spectral Space Arrays
- `u_hat, u_x_hat, u_xx_hat`: Velocity Fourier transforms
- `λ_hat, λ_xx_hat`: Reaction progress Fourier transforms

## FFT Plans
- `fft_plan`: Forward FFT plan
- `ifft_plan`: Inverse FFT plan

## Spectral Operations
- `dealias_filter`: Dealiasing filter
- `ik`: Wavenumbers for first derivative
- `k2`: Wavenumbers for second derivative

## Control Parameters
- `u_p_current, u_p_previous`: Current and previous pressure values
- `s_current, s_previous`: Current and previous s parameter values
- `τ_smooth`: Smoothing time scale
- `control_time`: Time of last control update
"""
mutable struct PseudospectralRDECache{T<:AbstractFloat} <: AbstractRDECache{T}
    u_x::Vector{T}
    u_xx::Vector{T}
    λ_xx::Vector{T}
    ωu::Vector{T}
    ξu::Vector{T}
    βu::Vector{T}
    
    u_hat::Vector{Complex{T}}
    u_x_hat::Vector{Complex{T}}
    u_xx_hat::Vector{Complex{T}}
    λ_hat::Vector{Complex{T}}
    λ_xx_hat::Vector{Complex{T}}

    fft_plan::FFTW.rFFTWPlan{T}
    ifft_plan::FFTW.ScaledPlan

    dealias_filter::Vector{T}
    ik::Vector{Complex{T}}
    k2::Vector{T}
    u_p_current::Vector{T}
    u_p_previous::Vector{T}
    τ_smooth::T
    s_previous::Vector{T}
    s_current::Vector{T}
    control_time::T
    u_p_t::Vector{T}
    s_t::Vector{T}
    u_p_t_shifted::Vector{T}
    s_t_shifted::Vector{T}
end

"""
    PseudospectralRDECache{T}(params::RDEParam{T}; dealias=true) where {T<:AbstractFloat}

Construct a cache for pseudospectral method computations.

# Arguments
- `params::RDEParam{T}`: RDE parameters
- `dealias::Bool=true`: Whether to apply dealiasing filter

# Returns
- `PseudospectralRDECache{T}`: Initialized cache for computations
"""
function PseudospectralRDECache{T}(params::RDEParam{T}; 
        dealias=true) where {T<:AbstractFloat}
    N = params.N
    N_complex = div(N, 2) + 1
    
    dealias_filter = if dealias
        create_dealiasing_vector(params)
    else
        ones(T, N_complex)
    end

    ik, k2 = create_spectral_derivative_arrays(params)

    PseudospectralRDECache{T}(
        [Vector{T}(undef, N) for _ in 1:6]...,
        [Vector{Complex{T}}(undef, N_complex) for _ in 1:5]...,
        plan_rfft(Vector{T}(undef, N), flags=FFTW.MEASURE),
        plan_irfft(Vector{Complex{T}}(undef, N_complex), N, flags=FFTW.MEASURE),
        dealias_filter,
        ik, k2,
        fill(params.u_p, N), fill(params.u_p, N), T(1),
        fill(params.s, N), fill(params.s, N),
        T(0),
        fill(params.u_p, N),  # u_p_t
        fill(params.s, N),    # s_t
        fill(params.u_p, N),  # u_p_t_shifted
        fill(params.s, N),    # s_t_shifted
    )
end

"""
    FDRDECache{T<:AbstractFloat} <: AbstractRDECache{T}

Cache for finite difference method computations in RDE solver.

# Fields
## Spatial Arrays
- `u_x, u_xx`: Velocity derivatives
- `λ_xx`: Reaction progress derivatives
- `ωu, ξu, βu`: Nonlinear terms

## Grid Parameters
- `dx`: Grid spacing
- `N`: Number of grid points

## Control Parameters
- `u_p_current, u_p_previous`: Current and previous pressure values
- `s_current, s_previous`: Current and previous s parameter values
- `τ_smooth`: Smoothing time scale
- `control_time`: Time of last control update
"""
mutable struct FDRDECache{T<:AbstractFloat} <: AbstractRDECache{T}
    u_x::Vector{T}
    u_xx::Vector{T}
    λ_xx::Vector{T}
    ωu::Vector{T}
    ξu::Vector{T}
    βu::Vector{T}
    dx::T
    N::Int
    u_p_current::Vector{T}
    u_p_previous::Vector{T}
    τ_smooth::T
    s_previous::Vector{T}
    s_current::Vector{T}
    control_time::T
    u_p_t::Vector{T}
    s_t::Vector{T}
    u_p_t_shifted::Vector{T}
    s_t_shifted::Vector{T}
end

"""
    FDRDECache{T}(params::RDEParam{T}, dx::T) where {T<:AbstractFloat}

Construct a cache for finite difference method computations.

# Arguments
- `params::RDEParam{T}`: RDE parameters
- `dx::T`: Grid spacing

# Returns
- `FDRDECache{T}`: Initialized cache for computations
"""
function FDRDECache{T}(params::RDEParam{T}, dx::T) where {T<:AbstractFloat}
    N = params.N
    FDRDECache{T}(
        Vector{T}(undef, N),  # u_x
        Vector{T}(undef, N),  # u_xx
        Vector{T}(undef, N),  # λ_xx
        Vector{T}(undef, N),  # ωu
        Vector{T}(undef, N),  # ξu
        Vector{T}(undef, N),  # βu
        dx,                   # dx
        N,                    # N
        fill(params.u_p, N),  # u_p_current
        fill(params.u_p, N),  # u_p_previous
        T(1),                 # τ_smooth
        fill(params.s, N),    # s_previous
        fill(params.s, N),    # s_current
        T(0),                 # control_time
        fill(params.u_p, N),  # u_p_t
        fill(params.s, N),    # s_t
        fill(params.u_p, N),  # u_p_t_shifted
        fill(params.s, N),    # s_t_shifted
    )
end

"""
    RDEProblem{T<:AbstractFloat}

Main problem type for the rotating detonation engine solver.

# Fields
- `params::RDEParam{T}`: Model parameters
- `u0::Vector{T}`: Initial velocity field
- `λ0::Vector{T}`: Initial reaction progress
- `x::Vector{T}`: Spatial grid points
- `u_init::Function`: Velocity initialization function
- `λ_init::Function`: Reaction progress initialization function
- `sol::Union{Nothing, Any}`: Solution (if computed)
- `cache::AbstractRDECache{T}`: Computation cache
- `control_shift_func::Function`: Control shift function
"""
mutable struct RDEProblem{T<:AbstractFloat}
    params::RDEParam{T}
    u0::Vector{T}
    λ0::Vector{T}
    x::Vector{T}
    u_init::Function
    λ_init::Function
    sol::Union{Nothing, Any}
    cache::AbstractRDECache{T}
    control_shift_func::Function
end

"""
    RDEProblem(params::RDEParam{T}; kwargs...) where {T<:AbstractFloat}

Construct an RDE problem with given parameters.

# Arguments
- `params::RDEParam{T}`: Model parameters

# Keywords
- `u_init::Function = x -> (3/2) .* sech.(x .- 1).^20`: Initial velocity field function
- `λ_init::Function = x -> 0.5 .* ones(length(x))`: Initial reaction progress function
- `dealias::Bool = true`: Whether to apply dealiasing (pseudospectral only)
- `method::Symbol = :fd`: Numerical method (`:fd` or `:pseudospectral`)
- `control_shift_func::Function = (u, t) -> zero(T)`: Control shift function

# Returns
- `RDEProblem{T}`: Initialized problem
"""
function RDEProblem(params::RDEParam{T};
    u_init = x -> (3 / 2) .* sech.(x .- 1).^20,
    λ_init = x -> 0.5 .* ones(length(x)),
    dealias = true,
    method = :fd,
    control_shift_func = (u, t) -> zero(T)) where {T<:AbstractFloat}

    x = range(0, params.L, length=params.N+1)[1:end-1]
    dx = x[2] - x[1]

    cache = if method == :pseudospectral
        PseudospectralRDECache{T}(params, dealias=dealias)
    elseif method == :fd
        FDRDECache{T}(params, dx)
    else
        throw(ArgumentError("method must be :pseudospectral or :fd"))
    end

    prob = RDEProblem{T}(params, Vector{T}(undef, params.N), Vector{T}(undef, params.N), 
                         x, u_init, λ_init, nothing, cache, control_shift_func)
    set_init_state!(prob) #state may have been erased when creating fft plans in pseudospectral cache
    return prob
end

"""
    set_init_state!(prob::RDEProblem)

Initialize the state vectors of an RDE problem using the initialization functions.
"""
function set_init_state!(prob::RDEProblem)
    prob.u0 = prob.u_init(prob.x)
    prob.λ0 = prob.λ_init(prob.x)
end

"""
    create_dealiasing_vector(params::RDEParam{T}) where {T<:AbstractFloat}

Create a 2/3 rule dealiasing filter for pseudospectral computations.
"""
function create_dealiasing_vector(params::RDEParam{T}) where {T<:AbstractFloat}
    N = params.N
    N_complex = div(N, 2) + 1
    k = collect(0:N_complex-1)
    k_cutoff = div(N, 3)
    dealiasing = @. ifelse(k <= k_cutoff, one(T), zero(T))
    return dealiasing
end

"""
    create_spectral_derivative_arrays(params::RDEParam{T}) where {T<:AbstractFloat}

Create arrays for spectral derivatives: wavenumbers for first (ik) and second (k2) derivatives.
"""
function create_spectral_derivative_arrays(params::RDEParam{T}) where {T<:AbstractFloat}
    N = params.N
    L = params.L
    N_complex = div(N, 2) + 1
    k = T(2π / L) .* Vector(0:N_complex-1)
    ik = 1im .* k
    k2 = k .^ 2
    return ik, k2
end
