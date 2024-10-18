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