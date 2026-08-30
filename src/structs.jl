# Display methods for RDEParam
function Base.show(io::IO, params::RDEParam{T}) where {T}
    println(io, "RDEParam{$T}:")
    println(io, "  N: $(params.N)")
    println(io, "  L: $(params.L)")
    println(io, "  ν_1: $(params.ν_1)")
    println(io, "  ν_2: $(params.ν_2)")
    println(io, "  u_c: $(params.u_c)")
    println(io, "  α: $(params.α)")
    println(io, "  q_0: $(params.q_0)")
    println(io, "  u_0: $(params.u_0)")
    println(io, "  n: $(params.n)")
    println(io, "  k_param: $(params.k_param)")
    println(io, "  u_p: $(params.u_p)")
    println(io, "  s: $(params.s)")
    println(io, "  ϵ: $(params.ϵ)")
    println(io, "  tmax: $(params.tmax)")
    return println(io, "  x0: $(params.x0)")
end

Base.length(params::RDEParam) = 1

# Display methods for RDEProblem
function Base.show(io::IO, prob::RDEProblem{T, M, R, P}) where {T, M, R, P}
    println(io, "RDEProblem{$T, $M, $R, $P}:")
    println(io, "  params: $(prob.params)")
    println(io, "  u0: $(typeof(prob.u0))")
    println(io, "  λ0: $(typeof(prob.λ0))")
    println(io, "  x: $(typeof(prob.x))")
    println(io, "  reset_strategy: $(prob.reset_strategy)")
    println(io, "  sol: $(typeof(prob.sol))")
    println(io, "  method: $(prob.method)")
    return println(io, "  injection: $(typeof(prob.injection))")
end

function get_RDE_grid(L::T, N::Int) where {T <: AbstractFloat}
    x = range(0, L, length = N + 1)[1:(end - 1)]
    dx = x[2] - x[1]
    return x, dx
end

"""
    RDEProblem(params::RDEParam{T}; kwargs...) where {T<:AbstractFloat}

Construct an RDE problem with given parameters.

# Arguments
- `params::RDEParam{T}`: Model parameters

# Keywords
- `reset_strategy::AbstractReset = DefaultReset()`: Reset strategy
- `method::AbstractMethod = FiniteVolumeMethod{T}()`: Numerical method
- `control_shift_strategy::AbstractControlShift = ZeroControlShift()`: Optional shift wrapper
- `injection`: Prebuilt profile. If omitted, a uniform multi-step hold at `params.u_p`
- `τ_smooth`: C∞ step-then-hold time for the default profile
- `spatial_kernel_width`: Commit-time kernel width for a default spatial profile

# Returns
- `RDEProblem{T, M, R, P}`: Initialized problem

# Examples
```julia
# Using finite-volume method (default)
prob = RDEProblem(params)
```
"""
function RDEProblem(
        params::RDEParam{T};
        reset_strategy::R = Default(),
        method::M = FiniteVolumeMethod{T}(),
        control_shift_strategy::AbstractControlShift = ZeroControlShift(),
        injection = nothing,
        τ_smooth::T = one(T),
        spatial_kernel_width::Int = 0,
    ) where {T <: AbstractFloat, R <: AbstractReset, M <: AbstractMethod}

    x, dx = get_RDE_grid(params.L, params.N)

    init_cache!(method, params, dx)

    inner = if injection === nothing
        if spatial_kernel_width > 0
            default_spatial_injection(params, τ_smooth, spatial_kernel_width)
        else
            default_uniform_injection(params, τ_smooth)
        end
    else
        injection
    end
    inj = if inner isa MovingFrameInjectionProfile || control_shift_strategy isa ZeroControlShift
        inner
    else
        wrap_injection(inner, control_shift_strategy, params)
    end

    P = typeof(inj)
    prob = RDEProblem{T, M, R, P}(
        params, Vector{T}(undef, params.N), Vector{T}(undef, params.N),
        x, reset_strategy, nothing, method, inj
    )
    set_init_state!(prob)
    return prob
end

function wrap_injection(
        inner::I,
        shift::S,
        params::RDEParam{T},
    ) where {T <: AbstractFloat, I <: AbstractInjectionProfile, S <: AbstractControlShift}
    if shift isa ZeroControlShift
        return inner
    end
    dx = params.L / T(params.N)
    return MovingFrameInjectionProfile(inner, shift, dx, params.N)
end

function default_uniform_injection(params::RDEParam{T}, τ::T) where {T <: AbstractFloat}
    return UniformMultiStepPressureProfile(
        T[zero(T), one(T)],
        T[params.u_p, params.u_p],
        params.s,
        CinftySmoother(τ),
    )
end

function default_spatial_injection(
        params::RDEParam{T},
        τ::T,
        kernel_width::Int,
    ) where {T <: AbstractFloat}
    N = params.N
    profile = SpatialMultiStepPressureProfile(
        T[zero(T), one(T)],
        [fill(params.u_p, N), fill(params.u_p, N)],
        params.s,
        CinftySmoother(τ),
    )
    set_spatial_kernel!(profile, kernel_width)
    return profile
end

function commit_reset_injection!(prob::RDEProblem)
    fill_constant_schedule!(prob.injection, prob.params.u_p, prob.params.s)
    return nothing
end

control_shift(prob::RDEProblem) = control_shift(prob.injection)
is_uniform_injection(prob::RDEProblem) = is_uniform_injection(prob.injection)

"""
    set_init_state!(prob::RDEProblem)

Initialize the state vectors of an RDE problem using the initialization functions.
"""
function set_init_state!(prob::RDEProblem)
    reset_state_and_pressure!(prob, prob.reset_strategy)
    commit_reset_injection!(prob)
    @assert all(isfinite.(prob.u0)) "NaN or Inf values detected in u0"
    return @assert all(isfinite.(prob.λ0)) "NaN or Inf values detected in λ0"
end
