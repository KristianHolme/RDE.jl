function split_sol(uλ::Vector{Float64})
    N = Int(length(uλ)/2)
    u = uλ[1:N]
    λ = uλ[N+1:end]
    return u, λ
end
function split_sol(uλs::Vector{Vector{Float64}})
    tuples = split_sol.(uλs)
    us = getindex.(tuples, 1)
    λs = getindex.(tuples, 2)
    return us, λs
end