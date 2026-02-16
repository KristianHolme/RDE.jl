abstract type AbstractControlShift end

struct ZeroControlShift <: AbstractControlShift end

struct LinearControlShift{T <: Real} <: AbstractControlShift
    c::T
end

"""
    get_control_shift(control_shift_strategy::AbstractControlShift, u::AbstractVector, t::Real) -> Real

    Get the control shift for the given control shift strategy, state, and time. shift is in x-units,
        e.g. π means shift for half the length of the domain. Shift should be positive if the u_p is shifted in the flow direction, negative otherwise.
"""
function get_control_shift end

function get_control_shift(control_shift_strategy::AbstractControlShift, u::AbstractVector, t::Real)
    throw(ErrorException("get_control_shift not implemented for control shift strategy $(typeof(control_shift_strategy))"))
end

function get_control_shift(control_shift_strategy::LinearControlShift, u::AbstractVector, t::Real)
    return control_shift_strategy.c * t
end

function get_control_shift(control_shift_strategy::ZeroControlShift, u::AbstractVector{T}, t::Real) where {T <: AbstractFloat}
    return zero(T)
end


"""
    smooth_f(x::Real) -> Real

Helper function for smooth control transitions. Returns exp(-1/x) for x > 0, 0 otherwise.
"""
smooth_f(x::Real) = x > zero(x) ? exp(-1 / x) : zero(x)

"""
    smooth_g(x::Real) -> Real

Helper function for smooth control transitions. Returns normalized smooth_f.
"""
smooth_g(x::Real) = smooth_f(x) / (smooth_f(x) + smooth_f(1 - x))

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
function smooth_control!(target, t, control_t, current_value, previous_value, τ_smooth::T) where {T <: AbstractFloat}
    progress = smooth_g((t - control_t) / τ_smooth)
    return @turbo @. target = previous_value + (current_value - previous_value) * progress
end

function normalize_width_points(width_points::Int)
    if width_points <= 0
        return 0
    end
    if iseven(width_points)
        return width_points + 1
    end
    return width_points
end

function build_spatial_kernel(width_points::Int, ::Type{T}) where {T <: AbstractFloat}
    width_points = normalize_width_points(width_points)
    if width_points == 0
        return T[]
    end
    if width_points == 1
        return T[one(T)]
    end
    half = (width_points - 1) ÷ 2
    weights = Vector{T}(undef, width_points)
    for k in -half:half
        r = abs(k) / half
        weights[k + half + 1] = smooth_g(1 - r)
    end
    total = sum(weights)
    @assert total > zero(T)
    weights ./= total
    return weights
end

function smooth_spatial!(
        target::Vector{T},
        scratch::Vector{T},
        kernel::Vector{T},
    ) where {T <: AbstractFloat}
    if isempty(kernel)
        return nothing
    end
    N = length(target)
    half = (length(kernel) - 1) ÷ 2
    padded_length = N + 2 * half
    @assert length(scratch) >= padded_length

    padded = @view scratch[1:padded_length]
    @inbounds begin
        if half > 0
            copyto!(view(padded, 1:half), view(target, (N - half + 1):N))
            copyto!(view(padded, (half + 1):(half + N)), target)
            copyto!(view(padded, (half + N + 1):padded_length), view(target, 1:half))
        else
            copyto!(view(padded, 1:N), target)
        end
    end

    @turbo for i in 1:N
        acc = zero(T)
        for k in 1:(2 * half + 1)
            acc += kernel[k] * padded[i + k - 1]
        end
        target[i] = acc
    end
    return nothing
end
