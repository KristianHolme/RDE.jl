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
function split_sol(uλ::Vector{T}) where {T <: Real}
    N = Int(length(uλ) / 2)
    u = uλ[1:N]
    λ = uλ[(N + 1):end]
    return u, λ
end

"""
    cfl_dtmax(params::RDEParam, u::AbstractVector; safety=0.9)

Compute a conservative maximum time step from CFL constraints for the
advection–diffusion system using the current velocity `u`.

Returns a scalar dtmax; can be used with adaptive integrators to cap step size.
"""
function cfl_dtmax(params::RDEParam, u::AbstractVector; safety = 0.43)
    Δx = params.L / params.N
    umax = RDE.turbo_maximum_abs(u)

    # Advection stability (only for u-equation). If umax == 0, no advective restriction.
    adv_dt = umax > 0 ? (Δx / umax) : Inf

    # Diffusion stability for both u and λ equations; if coefficient is zero, no diffusive restriction.
    diff_dt_u = iszero(params.ν_1) ? Inf : (Δx^2 / (2 * params.ν_1))
    diff_dt_λ = iszero(params.ν_2) ? Inf : (Δx^2 / (2 * params.ν_2))

    return safety * min(adv_dt, diff_dt_u, diff_dt_λ)
end

"""
    cfl_dtmax(params::RDEParam{T}, u::AbstractVector{T}, cache::AbstractRDECache{T}; safety=T(0.43)) where {T<:AbstractFloat}

Compute a conservative maximum time step including advection, u/λ diffusion, and a
reaction-timescale limit using current control arrays in `cache`. Emits a debug log
showing which limiter is active.
"""
function cfl_dtmax(params::RDEParam{T}, u::AbstractVector{T}, cache::AbstractRDECache{T}; safety::T = T(0.6)) where {T <: AbstractFloat}
    Δx::T = params.L / params.N

    # Advection (only u advects)
    umax::T = RDE.turbo_maximum_abs(u)
    adv_dt::T = umax > zero(T) ? (Δx / umax) : T(Inf)

    # Diffusion limits
    diff_dt_u::T = iszero(params.ν_1) ? T(Inf) : (Δx^2 / (2 * params.ν_1))
    diff_dt_λ::T = iszero(params.ν_2) ? T(Inf) : (Δx^2 / (2 * params.ν_2))

    # Reaction limit for λ using cached ω and β
    react_sum_max::T = RDE.turbo_max_sum(cache.ωu, cache.βu)
    react_dt::T = react_sum_max > zero(T) ? inv(react_sum_max) : T(Inf)

    # Collect limits and pick the smallest
    minval = min(adv_dt, diff_dt_u, diff_dt_λ, react_dt)
    dtmax::T = safety * minval
    return dtmax
end

function split_sol_view(uλ::Vector{T}) where {T <: Real}
    N = Int(length(uλ) / 2)
    u = @view uλ[1:N]
    λ = @view uλ[(N + 1):end]
    return u, λ
end

function split_sol(uλs::Vector{Vector{T}}) where {T <: Real}
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
function energy_balance(u::Vector{T}, λ::Vector{T}, params::RDEParam) where {T <: Real}
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
function periodic_simpsons_rule(u::Vector{T}, dx::T) where {T <: Real}
    return dx / 3 * sum((2 * u[1:2:end] + 4 * u[2:2:end]))
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
function energy_balance(uλ::Vector{T}, params::RDEParam) where {T <: Real}
    u, λ = split_sol(uλ)
    return energy_balance(u, λ, params)
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
function energy_balance(uλs::Vector{Vector{T}}, params::RDEParam) where {T <: Real}
    return energy_balance.(uλs, Ref(params))
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
function chamber_pressure(uλ::Vector{T}, params::RDEParam) where {T <: Real}
    if length(uλ) != params.N
        @assert length(uλ) == 2 * params.N
        u, = split_sol(uλ)
    else
        u = uλ
    end
    L = params.L
    dx = L / params.N
    mean_pressure = periodic_simpsons_rule(u, dx) / L
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
function chamber_pressure(uλs::Vector{Vector{T}}, params::RDEParam) where {T <: Real}
    return [chamber_pressure(uλ, params) for uλ in uλs]
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
    N = length(u)
    @turbo for i in 1:(N - 3)
        d[i] = (-3 * u[i] + 4 * u[i + 1] - u[i + 2]) / (2 * dx)
    end
    @turbo for i in (N - 2):N
        d[i] = (-3 * u[i] + 4 * u[mod1(i + 1, N)] - u[mod1(i + 2, N)]) / (2 * dx)
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
function shock_locations(u::AbstractArray{T}, dx::T) where {T <: AbstractFloat}
    N = length(u)
    L = N * dx
    minu, maxu = RDE.turbo_extrema(u)
    span = maxu - minu
    if span < T(1.0e-1)
        return CircularArray(fill(false, N))
    end
    # threshold = span / dx * T(0.06)
    threshold = span * T(0.06 * 512 / L)
    u_diff = periodic_ddx(u, dx)
    shocks = CircularArray(-u_diff .> threshold)
    backwards_block_distance = L * T(0.06)
    backwards_block = ceil(Int, backwards_block_distance / dx)
    @inbounds @views begin
        i = findfirst(shocks)
        while i !== nothing
            if any(shocks[(i + 1):(i + backwards_block)])
                shocks[i] = false
            end
            i = findnext(shocks, i + 1)
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
function shock_indices(u::AbstractArray{T}, dx::T) where {T <: AbstractFloat}
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
function count_shocks(u::AbstractArray{T}, dx::T) where {T <: AbstractFloat}
    shocks::Int = sum(shock_locations(u, dx))
    return shocks
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
function shift_inds_old(us::AbstractArray{T}, x::AbstractArray{T}, ts::AbstractArray{T}, c::T) where {T <: AbstractFloat}
    c = fill(c, length(ts) - one(T))
    pos = [zero(T); cumsum(c .* diff(ts))]
    return shift_by_interdistances_old(us, x, pos)
end

function shift_inds_old(us::AbstractArray{T}, x::AbstractArray{T}, ts::AbstractArray{T}, c::AbstractArray{T}) where {T <: AbstractFloat}
    pos = [zero(T); cumsum(c .* diff(ts))]
    return shift_by_interdistances_old(us, x, pos)
end

function shift_inds(us::Vector{Vector{T}}, x::AbstractVector{T}, ts::AbstractVector{T}, c::T) where {T <: AbstractFloat}
    c = fill(c, length(ts) - one(T))
    pos = [zero(T); cumsum(c .* diff(ts))]
    return shift_by_interdistances(us, x, pos)
end

function shift_inds(us::Vector{Vector{T}}, x::AbstractVector{T}, ts::AbstractVector{T}, c::AbstractVector{T}) where {T <: AbstractFloat}
    pos = [zero(T); cumsum(c .* diff(ts))]
    return shift_by_interdistances(us, x, pos)
end

function shift_by_interdistances_old(us::AbstractArray{T}, x::AbstractArray{T}, pos::AbstractArray{T}) where {T <: AbstractFloat}
    us = CircularArray.(us)
    shifted_us = similar(us)
    dx = x[2] - x[1]
    for j in 1:length(us)
        shift = Int(round(pos[j] / dx))
        shifted_us[j] = us[j][(1 + shift):(end + shift)]
    end
    return shifted_us
end

function shift_by_interdistances(us::AbstractArray, x::AbstractArray, pos::AbstractArray)
    shifted_us = similar(us)
    dx = x[2] - x[1]
    shifts = Int.(round.(pos ./ dx))
    shifted_us = circshift.(us, -shifts)
    return shifted_us
end

const SHOCK_DATA = let
    data_file = joinpath(@__DIR__, "..", "data", "shocks.jld2")
    if !isfile(data_file)
        # throw(ErrorException("Shock data file not found: $data_file"))
        @warn "Shock data file not found: $data_file"
        return nothing
    end
    try
        Dict(n => Dict(:u => (load(data_file, "u$n")), :λ => (load(data_file, "λ$n"))) for n in 1:4)
    catch e
        @warn "Failed to load shock data: $e"
        return nothing
    end
end

const SHOCK_MATRICES = let
    if !@isdefined(SHOCK_DATA) || (@isdefined(SHOCK_DATA) && SHOCK_DATA === nothing)
        @warn "SHOCK_DATA not available, returning empty shock matrices"
        (shocks = zeros(Float64, 0, 0), fuels = zeros(Float64, 0, 0))
    else
        shocks = hcat(SHOCK_DATA[1][:u], SHOCK_DATA[2][:u], SHOCK_DATA[3][:u], SHOCK_DATA[4][:u])
        fuels = hcat(SHOCK_DATA[1][:λ], SHOCK_DATA[2][:λ], SHOCK_DATA[3][:λ], SHOCK_DATA[4][:λ])
        (shocks = shocks, fuels = fuels)
    end
end

const SHOCK_PRESSURES = [0.5f0, 0.64f0, 0.84f0, 0.96f0]

const SHOCK_SPEED_MODEL = let
    data_file = joinpath(@__DIR__, "..", "data", "speed_model.jld2")
    if !isfile(data_file)
        @warn "Shock speed model file not found: $data_file"
        return nothing
    end
    try
        load_object(data_file)
    catch e
        @warn "Failed to load shock speed model: $e"
        return nothing
    end
end

function predict_speed(u_p::T, n_shocks::Integer) where {T <: AbstractFloat}
    new_data = DataFrame(u_p = [u_p], shocks = [n_shocks])
    ŷ = predict(SHOCK_SPEED_MODEL, new_data)[1]
    return T(ŷ)
end

function predict_speed(u_p::AbstractArray{T}, n_shocks::Integer) where {T <: AbstractFloat}
    new_data = DataFrame(
        u_p = collect(u_p),
        shocks = fill(n_shocks, length(u_p))
    )
    ŷ = predict(SHOCK_SPEED_MODEL, new_data)
    return T.(ŷ)
end

function predict_speed(u_p::T, n_shocks::AbstractArray{Int}) where {T <: AbstractFloat}
    new_data = DataFrame(
        u_p = fill(u_p, length(n_shocks)),
        shocks = collect(n_shocks)
    )
    ŷ = predict(SHOCK_SPEED_MODEL, new_data)
    return T.(ŷ)
end

function predict_speed(u_p::AbstractArray{T}, n_shocks::AbstractArray{Int}) where {T <: AbstractFloat}
    if length(u_p) != length(n_shocks)
        throw(DimensionMismatch("Length of u_p ($(length(u_p))) must match length of n_shocks ($(length(n_shocks)))"))
    end
    new_data = DataFrame(
        u_p = collect(u_p),
        shocks = collect(n_shocks)
    )
    ŷ = predict(SHOCK_SPEED_MODEL, new_data)
    return T.(ŷ)
end

"""
    softmax(x::AbstractVector, temp::Real=1.0) -> Vector

Compute the softmax of a vector with temperature scaling.

# Arguments
- `x::AbstractVector`: Input vector
- `temp::Real`: Temperature parameter (default=1.0). Higher values make distribution more uniform.

# Returns
- Vector of same length as input containing softmax probabilities
"""
function softmax(x::AbstractVector, temp::Real = 1.0)
    x_scaled = x ./ temp
    exp_x = exp.(x_scaled .- maximum(x_scaled))
    return exp_x ./ sum(exp_x)
end


"""
    outofdomain(uλ, prob, t)

Check if the solution has left the physical domain.

# Arguments
- `uλ`: Current state [u; λ]
- `prob`: RDE problem
- `t`: Current time

# Returns
- `true` if solution is unphysical (u < 0 or λ ∉ [0,1])
- `false` otherwise
"""
function outofdomain(uλ::Vector{T}, prob, t) where {T <: Real}
    # T = eltype(uλ)
    N = prob.params.N

    # Check both u and λ values in a single loop with early return
    @inbounds for i in 1:N
        u_val = uλ[i]
        λ_val = uλ[N + i]

        # Short-circuit as soon as any value is out of domain
        if u_val < zero(T) || u_val > T(25) || λ_val < zero(T) || λ_val > one(T)
            return true
        end
    end

    return false
end

"""
    get_dx(prob::RDEProblem{T}) where T -> T

Get the spatial grid spacing (dx) for a RDE problem.

The grid spacing is calculated as the difference between the first two grid points.

# Arguments
- `prob::RDEProblem{T}`: The RDE problem containing the spatial grid

# Returns
- `T`: The grid spacing dx

# Example
```julia
dx = get_dx(prob)
```
"""
function get_dx(prob::RDEProblem{T})::T where {T}
    return get_dx(prob.params)
end

function get_dx(params::RDEParam{T})::T where {T}
    return params.L / params.N
end

"""
    turbo_maximum(arr) -> Real

Compute the maximum value of an array using @turbo for performance.

# Arguments
- `arr`: Input array

# Returns
- Maximum value in the array
"""
function turbo_maximum(arr)
    maxval = arr[1]
    @turbo for i in eachindex(arr)
        maxval = max(maxval, arr[i])
    end
    return maxval
end

"""
    turbo_minimum(arr) -> Real

Compute the minimum value of an array using @turbo for performance.

# Arguments
- `arr`: Input array

# Returns
- Minimum value in the array
"""
function turbo_minimum(arr)
    minval = arr[1]
    @turbo for i in eachindex(arr)
        minval = min(minval, arr[i])
    end
    return minval
end

"""
    turbo_maximum_abs(arr) -> Real

Compute the maximum absolute value of an array using @turbo for performance.

# Arguments
- `arr`: Input array

# Returns
- Maximum absolute value in the array
"""
function turbo_maximum_abs(arr)
    maxval = abs(arr[1])
    @turbo for i in eachindex(arr)
        maxval = max(maxval, abs(arr[i]))
    end
    return maxval
end

"""
    turbo_max_sum(u::Vector{T}, v::Vector{T}) where {T<:Real} -> T

Compute the maximum of element-wise sums u[i] + v[i] using @turbo for performance.

# Arguments
- `u::Vector{T}`: First input vector
- `v::Vector{T}`: Second input vector

# Returns
- Maximum value of u[i] + v[i] across all indices
"""
function turbo_max_sum(u::Vector{T}, v::Vector{T}) where {T <: Real}
    maxval = zero(T)
    @turbo for i in eachindex(u)
        maxval = max(maxval, u[i] + v[i])
    end
    return maxval
end

"""
    turbo_extrema(arr) -> (minval, maxval)

Compute both the minimum and maximum of an array in one `@turbo` pass.

# Arguments
- `arr`: Input array

# Returns
- Tuple `(minval, maxval)` with the array extrema
"""
function turbo_extrema(arr)
    minval = arr[1]
    maxval = arr[1]
    @turbo for i in eachindex(arr)
        minval = min(minval, arr[i])
        maxval = max(maxval, arr[i])
    end
    return minval, maxval
end

function turbo_diff_norm(u::AbstractVector{T}, v::AbstractVector{T}) where {T <: Real}
    @assert length(u) == length(v)
    nrm = zero(T)
    @turbo for i in eachindex(u)
        nrm += (u[i] - v[i])^2
    end
    return sqrt(nrm)
end

"""
    turbo_column_maximum(A::AbstractMatrix{T}) where {T} -> Vector{T}

Compute the maximum value in each column of a matrix using @turbo for performance.

# Arguments
- `A::AbstractMatrix{T}`: Input matrix

# Returns
- `Vector{T}`: Vector containing the maximum value of each column
"""
function turbo_column_maximum(A::AbstractMatrix{T}) where {T}
    nrows, ncols = size(A)
    result = Vector{T}(undef, ncols)
    @turbo for j in 1:ncols
        maxval = A[1, j]
        for i in 2:nrows
            maxval = max(maxval, A[i, j])
        end
        result[j] = maxval
    end
    return result
end
