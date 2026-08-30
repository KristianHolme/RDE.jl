"""
    AbstractInjectionProfile

Time-dependent injection `u_p(x, t)` and `s(x, t)`. The RHS evaluates with
`update_injection!`. Actions install a new knot schedule with `commit_schedule!`.
"""
abstract type AbstractInjectionProfile end

"""
    update_injection!(u_p, s, profile, t)

Evaluate the injection profile at time `t` into preallocated `u_p` and `s`.
"""
function update_injection! end

"""
    commit_schedule!(profile, t0, target_times, u_p_targets; s = nothing)

Install a new knot schedule. Knot 1 is the live field at `t0` (previous basis).
Later knots are `target_times` / `u_p_targets`. Spatial profiles may smooth new
knots only. Previous is never re-smoothed.
"""
function commit_schedule! end

abstract type AbstractMultiStepSmoother end

struct CinftySmoother{T <: AbstractFloat} <: AbstractMultiStepSmoother
    τ::T
end

struct LinearSmoother <: AbstractMultiStepSmoother end

@inline function _segment_progress(
        t::T,
        t_prev::T,
        t_next::T,
        smoother::CinftySmoother{T},
    ) where {T <: AbstractFloat}
    τ = smoother.τ
    if τ <= zero(T)
        return one(T)
    end
    return clamp((t - t_prev) / τ, zero(T), one(T))
end

@inline function _segment_progress(
        t::T,
        t_prev::T,
        t_next::T,
        ::LinearSmoother,
    ) where {T <: AbstractFloat}
    dt_seg = t_next - t_prev
    if dt_seg <= zero(T)
        return one(T)
    end
    return clamp((t - t_prev) / dt_seg, zero(T), one(T))
end

@inline function _blend(
        progress::T,
        previous_value::T,
        next_value::T,
        ::CinftySmoother,
    ) where {T <: AbstractFloat}
    g = smooth_g(progress)
    return previous_value + (next_value - previous_value) * g
end

@inline function _blend(
        progress::T,
        previous_value::T,
        next_value::T,
        ::LinearSmoother,
    ) where {T <: AbstractFloat}
    return previous_value + (next_value - previous_value) * progress
end

@inline function _multistep_segment(times::Vector{T}, t::T) where {T <: AbstractFloat}
    n = length(times)
    i = searchsortedlast(times, t)
    return clamp(i, 1, n - 1)
end

# -----------------------------------------------------------------------------
# Constant
# -----------------------------------------------------------------------------

struct ConstantInjectionProfile{T <: AbstractFloat} <: AbstractInjectionProfile
    u_p::T
    s::T
end

function update_injection!(
        u_p::Vector{T},
        s::Vector{T},
        profile::ConstantInjectionProfile{T},
        ::T,
    ) where {T <: AbstractFloat}
    fill!(u_p, profile.u_p)
    fill!(s, profile.s)
    return u_p, s
end

function commit_schedule!(
        profile::ConstantInjectionProfile{T},
        ::T,
        ::AbstractVector{T},
        u_p_targets::AbstractVector{T};
        s = nothing,
    ) where {T <: AbstractFloat}
    return ConstantInjectionProfile{T}(
        u_p_targets[end],
        s === nothing ? profile.s : T(s),
    )
end

# -----------------------------------------------------------------------------
# Uniform multi-step
# -----------------------------------------------------------------------------

mutable struct UniformMultiStepPressureProfile{T <: AbstractFloat, S <: AbstractMultiStepSmoother} <:
    AbstractInjectionProfile
    times::Vector{T}
    u_p_values::Vector{T}
    s_value::T
    smoother::S
    function UniformMultiStepPressureProfile{T, S}(
            times::Vector{T},
            u_p_values::Vector{T},
            s_value::T,
            smoother::S,
        ) where {T <: AbstractFloat, S <: AbstractMultiStepSmoother}
        length(times) == length(u_p_values) ||
            throw(ArgumentError("times and u_p_values must have the same length"))
        length(times) >= 2 ||
            throw(ArgumentError("need at least two knots (previous + one target)"))
        return new{T, S}(times, u_p_values, s_value, smoother)
    end
end

function UniformMultiStepPressureProfile(
        times::Vector{T},
        u_p_values::Vector{T},
        s_value::T,
        smoother::S,
    ) where {T <: AbstractFloat, S <: AbstractMultiStepSmoother}
    return UniformMultiStepPressureProfile{T, S}(times, u_p_values, s_value, smoother)
end

@inline function _eval_uniform_u_p(
        profile::UniformMultiStepPressureProfile{T, S},
        t::T,
    ) where {T <: AbstractFloat, S <: AbstractMultiStepSmoother}
    idx = _multistep_segment(profile.times, t)
    t_prev = profile.times[idx]
    t_next = profile.times[idx + 1]
    progress = _segment_progress(t, t_prev, t_next, profile.smoother)
    return _blend(progress, profile.u_p_values[idx], profile.u_p_values[idx + 1], profile.smoother)
end

function update_injection!(
        u_p::Vector{T},
        s::Vector{T},
        profile::UniformMultiStepPressureProfile{T, S},
        t::T,
    ) where {T <: AbstractFloat, S <: AbstractMultiStepSmoother}
    fill!(u_p, _eval_uniform_u_p(profile, t))
    fill!(s, profile.s_value)
    return u_p, s
end

function commit_schedule!(
        profile::UniformMultiStepPressureProfile{T, S},
        t0::T,
        target_times::AbstractVector{T},
        u_p_targets::AbstractVector{T};
        s = nothing,
    ) where {T <: AbstractFloat, S <: AbstractMultiStepSmoother}
    length(target_times) == length(u_p_targets) ||
        throw(ArgumentError("target_times and u_p_targets must have the same length"))
    length(target_times) >= 1 || throw(ArgumentError("need at least one target knot"))
    prev = _eval_uniform_u_p(profile, t0)
    n = length(u_p_targets)
    resize!(profile.times, n + 1)
    resize!(profile.u_p_values, n + 1)
    profile.times[1] = t0
    profile.u_p_values[1] = prev
    copyto!(view(profile.times, 2:(n + 1)), target_times)
    copyto!(view(profile.u_p_values, 2:(n + 1)), u_p_targets)
    if s !== nothing
        profile.s_value = T(s)
    end
    return profile
end

function commit_schedule!(
        profile::UniformMultiStepPressureProfile{T, S},
        t0::T,
        t1::T,
        u_p_target::T;
        s = nothing,
    ) where {T <: AbstractFloat, S <: AbstractMultiStepSmoother}
    return commit_schedule!(profile, t0, T[t1], T[u_p_target]; s = s)
end

# -----------------------------------------------------------------------------
# Spatial multi-step
# -----------------------------------------------------------------------------

mutable struct SpatialMultiStepPressureProfile{T <: AbstractFloat, S <: AbstractMultiStepSmoother} <:
    AbstractInjectionProfile
    times::Vector{T}
    u_p_values::Vector{Vector{T}}
    s_value::T
    smoother::S
    kernel::Vector{T}
    scratch::Vector{T}
    eval_u_p::Vector{T}
    eval_s::Vector{T}
end

function SpatialMultiStepPressureProfile(
        times::Vector{T},
        u_p_values::Vector{Vector{T}},
        s_value::T,
        smoother::S;
        kernel::Vector{T} = T[],
        scratch::Vector{T} = T[],
    ) where {T <: AbstractFloat, S <: AbstractMultiStepSmoother}
    length(times) == length(u_p_values) ||
        throw(ArgumentError("times and u_p_values must have the same length"))
    length(times) >= 2 ||
        throw(ArgumentError("need at least two knots (previous + one target)"))
    N = length(u_p_values[1])
    for v in u_p_values
        length(v) == N || throw(ArgumentError("all spatial knots must have length N"))
    end
    return SpatialMultiStepPressureProfile{T, S}(
        times,
        u_p_values,
        s_value,
        smoother,
        kernel,
        scratch,
        zeros(T, N),
        zeros(T, N),
    )
end

function set_spatial_kernel!(
        profile::SpatialMultiStepPressureProfile{T},
        width_points::Int,
    ) where {T <: AbstractFloat}
    width_points = normalize_width_points(width_points)
    if width_points == 0
        profile.kernel = T[]
        profile.scratch = T[]
        return profile
    end
    profile.kernel = build_spatial_kernel(width_points, T)
    half = (length(profile.kernel) - 1) ÷ 2
    profile.scratch = zeros(T, length(profile.eval_u_p) + 2 * half)
    return profile
end

function _maybe_smooth_new_knot!(
        dest::Vector{T},
        profile::SpatialMultiStepPressureProfile{T},
    ) where {T <: AbstractFloat}
    if !isempty(profile.kernel)
        smooth_spatial!(dest, profile.scratch, profile.kernel)
    end
    return dest
end

function update_injection!(
        u_p::Vector{T},
        s::Vector{T},
        profile::SpatialMultiStepPressureProfile{T, S},
        t::T,
    ) where {T <: AbstractFloat, S <: AbstractMultiStepSmoother}
    idx = _multistep_segment(profile.times, t)
    t_prev = profile.times[idx]
    t_next = profile.times[idx + 1]
    progress = _segment_progress(t, t_prev, t_next, profile.smoother)
    prev = profile.u_p_values[idx]
    next = profile.u_p_values[idx + 1]
    if profile.smoother isa CinftySmoother
        g = smooth_g(progress)
        @. u_p = prev + (next - prev) * g
    else
        @. u_p = prev + (next - prev) * progress
    end
    fill!(s, profile.s_value)
    return u_p, s
end

function commit_schedule!(
        profile::SpatialMultiStepPressureProfile{T, S},
        t0::T,
        target_times::AbstractVector{T},
        u_p_targets::AbstractVector{<:AbstractVector{T}};
        s = nothing,
    ) where {T <: AbstractFloat, S <: AbstractMultiStepSmoother}
    length(target_times) == length(u_p_targets) ||
        throw(ArgumentError("target_times and u_p_targets must have the same length"))
    length(target_times) >= 1 || throw(ArgumentError("need at least one target knot"))
    update_injection!(profile.eval_u_p, profile.eval_s, profile, t0)
    n = length(u_p_targets)
    N = length(profile.eval_u_p)
    prev_knot = isassigned(profile.u_p_values, 1) ? profile.u_p_values[1] : zeros(T, N)
    copyto!(prev_knot, profile.eval_u_p)
    new_knots = Vector{Vector{T}}(undef, n + 1)
    new_knots[1] = prev_knot
    for i in 1:n
        target = u_p_targets[i]
        length(target) == N || throw(ArgumentError("spatial target $i must have length $N"))
        dest = isassigned(profile.u_p_values, i + 1) ? profile.u_p_values[i + 1] : similar(target)
        copyto!(dest, target)
        _maybe_smooth_new_knot!(dest, profile)
        new_knots[i + 1] = dest
    end
    resize!(profile.times, n + 1)
    profile.times[1] = t0
    copyto!(view(profile.times, 2:(n + 1)), target_times)
    profile.u_p_values = new_knots
    if s !== nothing
        profile.s_value = T(s)
    end
    return profile
end

function commit_schedule!(
        profile::SpatialMultiStepPressureProfile{T, S},
        t0::T,
        t1::T,
        u_p_target::AbstractVector{T};
        s = nothing,
    ) where {T <: AbstractFloat, S <: AbstractMultiStepSmoother}
    return commit_schedule!(profile, t0, T[t1], [u_p_target]; s = s)
end

# -----------------------------------------------------------------------------
# Moving-frame wrapper
# -----------------------------------------------------------------------------

mutable struct MovingFrameInjectionProfile{
        T <: AbstractFloat,
        I <: AbstractInjectionProfile,
        S <: AbstractControlShift,
    } <: AbstractInjectionProfile
    inner::I
    shift::S
    dx::T
    u_p_shift::Vector{T}
    s_shift::Vector{T}
end

function MovingFrameInjectionProfile(
        inner::I,
        shift::S,
        dx::T,
        N::Integer,
    ) where {T <: AbstractFloat, I <: AbstractInjectionProfile, S <: AbstractControlShift}
    return MovingFrameInjectionProfile{T, I, S}(
        inner,
        shift,
        dx,
        zeros(T, N),
        zeros(T, N),
    )
end

function update_injection!(
        u_p::Vector{T},
        s::Vector{T},
        profile::MovingFrameInjectionProfile{T, I, S},
        t::T,
    ) where {T <: AbstractFloat, I <: AbstractInjectionProfile, S <: AbstractControlShift}
    update_injection!(profile.u_p_shift, profile.s_shift, profile.inner, t)
    shift_pos = get_control_shift(profile.shift, profile.u_p_shift, t)
    shift = Int(round(shift_pos / profile.dx))
    if shift != 0
        circshift!(u_p, profile.u_p_shift, shift)
        circshift!(s, profile.s_shift, shift)
    else
        copyto!(u_p, profile.u_p_shift)
        copyto!(s, profile.s_shift)
    end
    return u_p, s
end

function commit_schedule!(
        profile::MovingFrameInjectionProfile,
        args...;
        kwargs...,
    )
    return commit_schedule!(profile.inner, args...; kwargs...)
end

function set_spatial_kernel!(
        profile::MovingFrameInjectionProfile,
        width_points::Int,
    )
    return set_spatial_kernel!(profile.inner, width_points)
end

function set_spatial_kernel!(::AbstractInjectionProfile, ::Int)
    return nothing
end

# -----------------------------------------------------------------------------
# Accessors
# -----------------------------------------------------------------------------

inner_profile(profile::MovingFrameInjectionProfile) = profile.inner
inner_profile(profile::AbstractInjectionProfile) = profile

control_shift(profile::MovingFrameInjectionProfile) = profile.shift
control_shift(::AbstractInjectionProfile) = ZeroControlShift()

is_uniform_injection(::UniformMultiStepPressureProfile) = true
is_uniform_injection(::ConstantInjectionProfile) = true
is_uniform_injection(::SpatialMultiStepPressureProfile) = false
is_uniform_injection(profile::MovingFrameInjectionProfile) = is_uniform_injection(profile.inner)

current_u_p(profile::UniformMultiStepPressureProfile) = profile.u_p_values[end]
previous_u_p(profile::UniformMultiStepPressureProfile) = profile.u_p_values[1]
current_s(profile::UniformMultiStepPressureProfile) = profile.s_value
previous_s(profile::UniformMultiStepPressureProfile) = profile.s_value

current_u_p(profile::SpatialMultiStepPressureProfile) = profile.u_p_values[end]
previous_u_p(profile::SpatialMultiStepPressureProfile) = profile.u_p_values[1]
current_s(profile::SpatialMultiStepPressureProfile) = profile.s_value
previous_s(profile::SpatialMultiStepPressureProfile) = profile.s_value

current_u_p(profile::ConstantInjectionProfile) = profile.u_p
previous_u_p(profile::ConstantInjectionProfile) = profile.u_p
current_s(profile::ConstantInjectionProfile) = profile.s
previous_s(profile::ConstantInjectionProfile) = profile.s

current_u_p(profile::MovingFrameInjectionProfile) = current_u_p(profile.inner)
previous_u_p(profile::MovingFrameInjectionProfile) = previous_u_p(profile.inner)
current_s(profile::MovingFrameInjectionProfile) = current_s(profile.inner)
previous_s(profile::MovingFrameInjectionProfile) = previous_s(profile.inner)

function mean_u_p(profile::AbstractInjectionProfile)
    u = current_u_p(profile)
    return u isa AbstractVector ? sum(u) / length(u) : u
end

function committed_pressure_delta(profile::AbstractInjectionProfile)
    cur = current_u_p(profile)
    prev = previous_u_p(profile)
    if cur isa AbstractVector
        return turbo_maximum_abs_diff(cur, prev)
    end
    return abs(cur - prev)
end

function fill_constant_schedule!(
        profile::UniformMultiStepPressureProfile{T},
        u_p::T,
        s::T,
    ) where {T <: AbstractFloat}
    fill!(profile.u_p_values, u_p)
    profile.s_value = s
    profile.times[1] = zero(T)
    if length(profile.times) >= 2
        profile.times[2] = one(T)
    end
    return profile
end

function fill_constant_schedule!(
        profile::SpatialMultiStepPressureProfile{T},
        u_p::T,
        s::T,
    ) where {T <: AbstractFloat}
    for v in profile.u_p_values
        fill!(v, u_p)
    end
    profile.s_value = s
    profile.times[1] = zero(T)
    if length(profile.times) >= 2
        profile.times[2] = one(T)
    end
    return profile
end

function fill_constant_schedule!(
        profile::ConstantInjectionProfile{T},
        u_p::T,
        s::T,
    ) where {T <: AbstractFloat}
    return ConstantInjectionProfile{T}(u_p, s)
end

function fill_constant_schedule!(
        profile::MovingFrameInjectionProfile,
        u_p,
        s,
    )
    fill_constant_schedule!(profile.inner, u_p, s)
    return profile
end
