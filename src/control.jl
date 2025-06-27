abstract type AbstractControlShift end

struct ZeroControlShift <: AbstractControlShift end

struct LinearControlShift <: AbstractControlShift
    c::Real
end

function get_control_shift(control_shift_strategy::AbstractControlShift, u::AbstractVector, t::Real)
    throw(ErrorException("get_control_shift not implemented for control shift strategy $(typeof(control_shift_strategy))"))
end

function get_control_shift(control_shift_strategy::LinearControlShift, u::AbstractVector, t::Real)
    return control_shift_strategy.c * t
end

function get_control_shift(control_shift_strategy::ZeroControlShift, u::AbstractVector{T}, t::Real) where T <: AbstractFloat
    return zero(T)
end 


"""
    smooth_f(x::Real) -> Real

Helper function for smooth control transitions. Returns exp(-1/x) for x > 0, 0 otherwise.
"""
smooth_f(x::Real) = x > zero(x) ? exp(-1/x) : zero(x)

"""
    smooth_g(x::Real) -> Real

Helper function for smooth control transitions. Returns normalized smooth_f.
"""
smooth_g(x::Real) = smooth_f(x)/(smooth_f(x)+smooth_f(1-x))

"""
    smooth_control!(target, t, control_t, current_value, previous_value, τ_smooth::T) where T <: AbstractFloat

Apply smooth transition between control values.

# Arguments
- `target`: Array to store result
- `t`: Current time
- `control_t`: Time of control change
- `current_value`: Target control value
- `previous_value`: Previous control value
- `τ_smooth::T`: Smoothing time scale
"""
function smooth_control!(target, t, control_t, current_value, previous_value, τ_smooth::T) where T <: AbstractFloat
    progress = smooth_g((t - control_t)/τ_smooth)
    @turbo @. target = previous_value + (current_value - previous_value) * progress
end