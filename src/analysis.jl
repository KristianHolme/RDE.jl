function l2_error(u1::Vector, u2::Vector, dx)
    periodic_simpsons_rule((u1-u2).^2, dx) |> sqrt
end
function l2_errors(uλ1::Vector{Vector}, uλ2::Vector{Vector}, dx)
    
    u1, λ1 = split_sol(uλ1)
    u2, λ2 = split_sol(uλ2)

    u_error = zeros(size(u1))
    λ_error = zeros(size(λ1))

    for i ∈ eachindex(u1)
        u_error[i] = l2_error(u1[i], u2[i], dx)
        λ_error[i] = l2_error(λ1[i], λ2[i], dx)
    end
    return u_error, λ_error
end

function l2_errors(sol1::ODESolution, sol2::ODESolution, dx)
    l2_errors(sol1.u, sol2.u, dx)
end
#TODO rethink procedure: take into account different time points for different solutinos
#use sol(t) to interpolate
function l2_errors(prob1::RDEProblem, prob2::RDEProblem)
    if prob1.params.N != prob2.params.N
       
        if prob1.params.N < prob2.params.N
            return l2_errors(prob2, prob1)
        else
            t1 = prob1.sol.t
            t2 = prob2.sol.t
            #TODO not finished!!
            x1 = prob1.x
            x2 = prob2.x
            u1 = prob1.sol.u
            u2 = interpolate(prob2.sol.u, prob1.x)
        end
    else
        @assert prob1.dx == prob2.dx
        dx = prob1.dx
        l2_errors(prob1.sol, prob2.sol, dx)
    end
end