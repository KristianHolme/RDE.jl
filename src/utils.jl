"""
    split_sol(uλ::Vector{T}) where T <: Real -> Tuple{Vector{T}, Vector{T}}

Split a combined state vector into its velocity (u) and reaction progress (λ) components.

# Arguments
- `uλ::Vector{T}`: Combined state vector [u; λ] of length 2N

# Returns
- `Tuple{Vector{T}, Vector{T}}`: Tuple containing (u, λ), each of length N

# Example
```julia
u, λ = split_sol(uλ)
```
"""
function split_sol(uλ::Vector{T}) where T <: Real
    N = Int(length(uλ)/2)
    u = uλ[1:N]
    λ = uλ[N+1:end]
    return u, λ
end

"""
    split_sol_view(uλ::Vector{T}) where T <: Real -> Tuple{SubArray, SubArray}

Split a combined state vector into views of its velocity (u) and reaction progress (λ) components.
This version avoids allocations by returning views instead of copying the data.

# Arguments
- `uλ::Vector{T}`: Combined state vector [u; λ] of length 2N

# Returns
- `Tuple{SubArray, SubArray}`: Tuple containing views into (u, λ), each of length N

# Example
```julia
u, λ = split_sol_view(uλ)
```
"""
function split_sol_view(uλ::Vector{T}) where T <: Real
    N = Int(length(uλ)/2)
    u = @view uλ[1:N]
    λ = @view uλ[N+1:end]
    return u, λ
end

"""
    split_sol(uλs::Vector{Vector{T}}) where T <: Real -> Tuple{Vector{Vector{T}}, Vector{Vector{T}}}

Split a vector of combined state vectors into separate vectors for velocity (u) and reaction progress (λ).

# Arguments
- `uλs::Vector{Vector{T}}`: Vector of combined state vectors

# Returns
- `Tuple{Vector{Vector{T}}, Vector{Vector{T}}}`: Tuple containing vectors of u and λ components

# Example
```julia
us, λs = split_sol(uλs)  # Split multiple states at once
```
"""
function split_sol(uλs::Vector{Vector{T}}) where T <: Real
    tuples = split_sol.(uλs)
    us = getindex.(tuples, 1)
    λs = getindex.(tuples, 2)
    return us, λs
end

"""
    energy_balance(u::Vector{T}, λ::Vector{T}, params::RDEParam) where T <: Real -> T

Calculate the domain energy balance:
```math
Ė_{domain} = ∫₀ᴸ (q(1-λ)ω(u) - ϵξ(u))dx
```

# Arguments
- `u::Vector{T}`: Velocity field
- `λ::Vector{T}`: Reaction progress
- `params::RDEParam`: Model parameters

# Returns
- `T`: Domain energy balance value

See also: [`energy_balance(uλ::Vector{T}, params::RDEParam)`](@ref)
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
    simpsons_rule_integral = periodic_simpsons_rule(integrand, dx)
    return simpsons_rule_integral
end

"""
    periodic_simpsons_rule(u::Vector{T}, dx::T) where {T<:Real} -> T

Compute the integral of a periodic function using Simpson's rule.

# Arguments
- `u::Vector{T}`: Values of the function at equally spaced points
- `dx::T`: Spacing between points

# Returns
- `T`: Integral value
"""
function periodic_simpsons_rule(u::Vector{T}, dx::T) where {T<:Real}
    dx / 3 * sum((2 * u[1:2:end] + 4 * u[2:2:end]))
end

"""
    energy_balance(uλ::Vector{T}, params::RDEParam) where T <: Real -> T

Calculate energy balance from a combined state vector.

# Arguments
- `uλ::Vector{T}`: Combined state vector [u; λ]
- `params::RDEParam`: Model parameters

# Returns
- `T`: Domain energy balance value
"""
function energy_balance(uλ::Vector{T}, params::RDEParam) where T <: Real
    u, λ = split_sol(uλ)
    energy_balance(u, λ, params)
end

"""
    energy_balance(uλs::Vector{Vector{T}}, params::RDEParam) where {T<:Real} -> Vector{T}

Calculate energy balance for multiple states.

# Arguments
- `uλs::Vector{Vector{T}}`: Vector of state vectors
- `params::RDEParam`: Model parameters

# Returns
- `Vector{T}`: Vector of energy balance values
"""
function energy_balance(uλs::Vector{Vector{T}}, params::RDEParam) where {T<:Real}
    energy_balance.(uλs, Ref(params))
end

"""
    chamber_pressure(uλ::Vector{T}, params::RDEParam) where T <: Real -> T

Calculate the mean chamber pressure from either a velocity field or combined state vector.

# Arguments
- `uλ::Vector{T}`: Either velocity field u or combined state vector [u; λ]
- `params::RDEParam`: Model parameters

# Returns
- `T`: Mean chamber pressure
"""
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

"""
    chamber_pressure(uλs::Vector{Vector{T}}, params::RDEParam) where T <: Real -> Vector{T}

Calculate chamber pressure for multiple states.

# Arguments
- `uλs::Vector{Vector{T}}`: Vector of state vectors
- `params::RDEParam`: Model parameters

# Returns
- `Vector{T}`: Vector of chamber pressures
"""
function chamber_pressure(uλs::Vector{Vector{T}}, params::RDEParam) where T <: Real
    [chamber_pressure(uλ, params) for uλ in uλs]
end

"""
    periodic_ddx(u::AbstractArray, dx::Real) -> Vector

Calculate the first derivative of a periodic function using a 3-point stencil.

# Arguments
- `u::AbstractArray`: Function values at equally spaced points
- `dx::Real`: Spacing between points

# Returns
- Vector of derivative values
"""
function periodic_ddx(u::AbstractArray, dx::Real)
    d = similar(u)
    for i in eachindex(u)
        d[i] = (-3*u[i] + 4*u[mod1(i+1, length(u))] - u[mod1(i+2, length(u))]) / (2*dx)
    end
    return d
end

"""
    periodic_diff(u::AbstractArray) -> Vector

Calculate differences with periodic boundary conditions.

# Arguments
- `u::AbstractArray`: Function values at equally spaced points

# Returns
- Vector of differences
"""
function periodic_diff(u::AbstractArray)
    d = similar(u)
    d[2:end] = diff(u)
    d[1] = u[1] - u[end]
    return d
end

"""
    shock_locations(u::AbstractArray, dx::Real) -> CircularArray{Bool}

Find the locations of shocks in a periodic function.

# Arguments
- `u::AbstractArray`: Function values at equally spaced points
- `dx::Real`: Spacing between points

# Returns
- `CircularArray{Bool}`: Boolean array indicating shock locations
"""
function shock_locations(u::AbstractArray, dx::Real)
    N = length(u)
    L = N*dx
    minu, maxu = extrema(u)
    span = maxu - minu
    if span < 1e-1
        return CircularArray(fill(false, N))
    end
    threshold = span/dx*0.06
    u_diff = periodic_ddx(u, dx)
    shocks = CircularArray(-u_diff .> threshold)
    potential_shocks = findall(shocks)

    backwards_block_distance = L*0.06
    backwards_block = ceil(Int, backwards_block_distance/dx)
    for i in potential_shocks
        if any(shocks[i+1:i+backwards_block])
            shocks[i] = false
        end
    end
    return shocks
end

"""
    shock_indices(u::AbstractArray, dx::Real) -> CircularArray{Int}

Find the indices of shocks in a periodic function.

# Arguments
- `u::AbstractArray`: Function values at equally spaced points
- `dx::Real`: Spacing between points

# Returns
- `CircularArray{Int}`: Indices of shocks
"""
function shock_indices(u::AbstractArray, dx::Real)
    return findall(shock_locations(u, dx))
end

"""
    count_shocks(u::AbstractArray, dx::Real) -> Int

Count the number of shocks in a periodic function.

# Arguments
- `u::AbstractArray`: Function values at equally spaced points
- `dx::Real`: Spacing between points

# Returns
- `Int`: Number of shocks detected
"""
function count_shocks(u::AbstractArray, dx::Real)
    return sum(shock_locations(u, dx))
end

"""
    shift_inds(us::AbstractArray, x::AbstractArray, ts::AbstractArray, c::Union{Real, AbstractArray}) -> Vector

Shift solution arrays in a moving frame with velocity c.

# Arguments
- `us::AbstractArray`: Array of solution vectors
- `x::AbstractArray`: Spatial grid points
- `ts::AbstractArray`: Time points
- `c::Union{Real, AbstractArray}`: Frame velocity (scalar or array)

# Returns
- Vector of shifted solutions
"""
function shift_inds(us::AbstractArray, x::AbstractArray, ts::AbstractArray, c::Union{Real, AbstractArray})
    us = CircularArray.(us)
    if c isa Real
        c = fill(c, length(ts)-1)
    end
    pos = [0.0; cumsum(c.*diff(ts))]

    shifted_us = similar(us)
    dx = x[2] - x[1]
    shifted_us[1] = us[1]
    for j in 2:length(ts)
        shift = Int(round(pos[j]/dx))
        shifted_us[j] = us[j][1+shift:end+shift]
    end
    return shifted_us
end

const SHOCK_DATA = let
    data_file = joinpath(@__DIR__, "..", "data", "shocks.jld2")
    if !isfile(data_file)
        throw(ErrorException("Shock data file not found: $data_file"))
    end
    Dict(n => load(data_file, "u$n") for n in 1:4)
end

function get_n_shocks_init_func(n::Int)
    if !(1 <= n <= 4)
        throw(ArgumentError("n must be between 1 and 4"))
    end
    @assert n in keys(SHOCK_DATA) "Shock data for n=$n not found"
    wave = SHOCK_DATA[n]
    x = range(0, 2π, length=513)[1:end-1]
    itp = linear_interpolation(x, wave, extrapolation_bc=Periodic())
    function u_init(x)
        return itp(x)
    end
    return u_init
end

"""
    apply_periodic_shift!(target::AbstractVector, source::AbstractVector, shift::Integer) -> AbstractVector

Apply a periodic shift to `source` and store the result in `target`.
Positive shift moves elements to the left (forward in space).
The operation is performed in-place without allocations.

# Arguments
- `target::AbstractVector`: Vector to store the shifted result
- `source::AbstractVector`: Vector to be shifted
- `shift::Integer`: Number of positions to shift (can be positive or negative)

# Returns
- The modified target vector

# Throws
- `AssertionError`: If target and source have different lengths
"""
function apply_periodic_shift!(target::AbstractVector, source::AbstractVector, shift::Integer)
    N = length(source)
    @assert length(target) == N "target and source must have the same length"
    
    shift = mod(shift, N)
    if shift == 0
        target .= source
        return target
    end
    
    target[1:N-shift] .= @view source[shift+1:N]
    target[N-shift+1:N] .= @view source[1:shift]
    
    return target
end

# Add smooth control function
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