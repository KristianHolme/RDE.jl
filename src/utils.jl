function split_sol(uλ::Vector{T}) where T <: Real
    N = Int(length(uλ)/2)
    u = uλ[1:N]
    λ = uλ[N+1:end]
    return u, λ
end
function split_sol(uλs::Vector{Vector{T}}) where T <: Real
    tuples = split_sol.(uλs)
    us = getindex.(tuples, 1)
    λs = getindex.(tuples, 2)
    return us, λs
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

function chamber_pressure(uλs::Vector{Vector{T}}, params::RDEParam) where T <: Real
    [chamber_pressure(uλ, params) for uλ in uλs]
end

"""
Calculate the first derivative of a periodic function using a 3-point stencil.
"""
function periodic_ddx(u::AbstractArray, dx::Real)
    d = similar(u)
    for i in eachindex(u)
        d[i] = (-3*u[i] + 4*u[mod1(i+1, length(u))] - u[mod1(i+2, length(u))]) / (2*dx)
    end
    return d
end

"""
like regular diff, but with periodic boundary conditions.
"""
function periodic_diff(u::AbstractArray, dx::Real)
    d = similar(u)
    d[2:end] = diff(u)
    d[1] = u[1] - u[end]
    return d
end

"""
Find the indices of shocks in a periodic function.
"""
function shock_indices(u::AbstractArray, dx::Real)
    N = length(u)
    L = N*dx
    minu, maxu = extrema(u)
    span = maxu - minu
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

function count_shocks(u::AbstractArray, dx::Real)
    return sum(shock_indices(u, dx))
end
