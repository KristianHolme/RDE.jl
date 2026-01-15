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
"""
@kwdef mutable struct RDEParam{T <: AbstractFloat}
    N::Int = 512               # Number of spatial points
    L::T = 2.0f0π                  # Domain length
    ν_1::T = 0.0075f0           # Viscosity coefficient
    ν_2::T = 0.0075f0
    u_c::T = 1.1f0              # Parameter in ω(u)
    α::T = 0.3f0                # Parameter in ω(u)
    q_0::T = 1.0f0              # Source term coefficient
    u_0::T = 0.0f0              # Parameter in ξ(u, u_0)
    n::Int = 1                  # Exponent in ξ(u, u_0)
    k_param::T = 5.0f0          # Parameter in β(u, s)
    u_p::T = 0.5f0              # Parameter in β(u, s)
    s::T = 3.5f0                # Parameter in β(u, s)
    ϵ::T = 0.15f0               # Small parameter in ξ(u)
    tmax::T = 500.0f0            # Maximum simulation time #TODO: move tmax to env?
    x0::T = 1.0f0               # Initial position
end
RDEParam(args...; kwargs...) = RDEParam{Float32}(args...; kwargs...)

# Method caches
abstract type AbstractRDECache{T <: AbstractFloat} end

abstract type AbstractMethod end

"""
    FVCache{T<:AbstractFloat} <: AbstractRDECache{T}

Cache for finite-volume method (conservative) with MUSCL reconstruction and
Rusanov (Lax–Friedrichs) flux.

# Fields
## Spatial Arrays
- `u_xx`: Velocity second derivative (for diffusion)
- `λ_xx`: Reaction progress second derivative (for diffusion)
- `ωu, ξu, βu`: Nonlinear/source terms
- `σ`: Limited slope per cell
- `UL, UR`: Left/right reconstructed interface states (size N, for i+1/2)
- `F̂`: Numerical flux at interfaces (size N, for i+1/2)
- `adv`: Conservative advective residual per cell, −(F̂_{i+1/2}−F̂_{i−1/2})/dx

## Grid Parameters
- `dx`: Grid spacing
- `N`: Number of grid points

## Control Parameters
- `u_p_current, u_p_previous`: Current and previous pressure values
- `s_current, s_previous`: Current and previous s parameter values
- `τ_smooth`: Smoothing time scale
- `control_time`: Time of last control update
"""
mutable struct FVCache{T <: AbstractFloat} <: AbstractRDECache{T}
    u_xx::Vector{T}
    λ_xx::Vector{T}
    ωu::Vector{T}
    ξu::Vector{T}
    βu::Vector{T}
    σ::Vector{T}
    UL::Vector{T}
    UR::Vector{T}
    F̂::Vector{T}
    adv::Vector{T}
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
    FVCache{T}(params::RDEParam{T}) where {T<:AbstractFloat}

Construct a cache for finite-volume method computations.
"""
function FVCache{T}(params::RDEParam{T}) where {T <: AbstractFloat}
    N = params.N
    L = params.L
    dx = L / N
    return FVCache{T}(
        zeros(T, N),          # u_xx
        zeros(T, N),          # λ_xx
        zeros(T, N),          # ωu
        zeros(T, N),          # ξu
        zeros(T, N),          # βu
        zeros(T, N),          # σ
        zeros(T, N),          # UL (i+1/2)
        zeros(T, N),          # UR (i+1/2)
        zeros(T, N),          # F̂ (i+1/2)
        zeros(T, N),          # adv residual
        dx,                   # dx
        N,
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

#TODO fix placemant of docstring
"""
    FiniteVolumeMethod{T<:AbstractFloat} <: AbstractMethod

Conservative finite-volume method with MUSCL reconstruction and Rusanov flux.

# Fields
- `limiter::L` where `L <: AbstractLimiter`: Slope limiter (e.g., `MinmodLimiter()`, `MCLimiter()`)
- `cache::Union{Nothing, FVCache{T}}`: Computation cache (initialized later)
"""
abstract type AbstractLimiter end

struct MinmodLimiter <: AbstractLimiter end
struct MCLimiter <: AbstractLimiter end

mutable struct FiniteVolumeMethod{T <: AbstractFloat, L <: AbstractLimiter} <: AbstractMethod
    limiter::L
    cache::Union{Nothing, FVCache{T}}
end

function Base.show(io::IO, method::FiniteVolumeMethod{T, L}) where {T, L}
    cache_status = isnothing(method.cache) ? "uninitialized" : "initialized"
    return print(io, "FiniteVolumeMethod{$T,$(L)} (cache $cache_status)")
end

"""
    FiniteVolumeMethod{T}(; limiter::AbstractLimiter=MCLimiter()) where {T<:AbstractFloat}
"""
FiniteVolumeMethod{T}(; limiter::AbstractLimiter = MCLimiter()) where {T <: AbstractFloat} =
    FiniteVolumeMethod{T, typeof(limiter)}(limiter, nothing)

"""
    FiniteVolumeMethod(; limiter::AbstractLimiter=MCLimiter())
"""
FiniteVolumeMethod(; limiter::AbstractLimiter = MCLimiter()) = FiniteVolumeMethod{Float32, typeof(limiter)}(limiter, nothing)

## Reset types
abstract type AbstractReset end

struct Default <: AbstractReset end

struct NShock <: AbstractReset
    n::Int
end

@kwdef struct RandomCombination <: AbstractReset
    temp::Real = 0.2f0
end

@kwdef struct RandomShockOrCombination <: AbstractReset
    shock_prob::Real = 0.5f0
    temp::Real = 0.2f0
end

struct RandomShock <: AbstractReset end

struct ShiftReset{R <: AbstractReset} <: AbstractReset
    reset_strategy::R
end

struct CustomPressureReset <: AbstractReset
    f::Function
end

## Problem type
"""
    RDEProblem{T<:AbstractFloat, M<:AbstractMethod, R<:AbstractReset, C<:AbstractControlShift}

Main problem type for the rotating detonation engine solver.

# Fields
- `params::RDEParam{T}`: Model parameters
- `u0::Vector{T}`: Initial velocity field
- `λ0::Vector{T}`: Initial reaction progress
- `x::Vector{T}`: Spatial grid points
- `reset_strategy::AbstractReset`: Reset strategy
- `sol::Union{Nothing, Any}`: Solution (if computed)
- `method::AbstractMethod`: Numerical method
- `control_shift_strategy::AbstractControlShift`: Control shift strategy
"""
mutable struct RDEProblem{T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    params::RDEParam{T}
    u0::Vector{T}
    λ0::Vector{T}
    x::Vector{T}
    reset_strategy::R
    sol::Union{Nothing, SciMLBase.ODESolution}
    method::M
    control_shift_strategy::C
end
