function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::Default)
    x = prob.x
    prob.u0 = (3f0 / 2f0) .* sech.(x .- 1f0).^20f0
    prob.λ0 = 0.5f0 .* ones(length(x))
    prob.params.u_p = 0.5f0
    nothing
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::NShock)
    n = reset_strategy.n
    if !(1 ≤ n ≤ 4)
        throw(ArgumentError("n must be between 1 and 4"))
    end
    @assert n in keys(SHOCK_DATA) "Shock data for n=$n not found"
    wave = SHOCK_DATA[n][:u]
    fuel = SHOCK_DATA[n][:λ]
    pressure = SHOCK_PRESSURES[n]
    x = range(0, 2π, length=513)[1:end-1]
    itp_u = linear_interpolation(x, wave, extrapolation_bc=Periodic())
    itp_λ = linear_interpolation(x, fuel, extrapolation_bc=Periodic())
    prob.u0 = itp_u.(prob.x)
    prob.λ0 = itp_λ.(prob.x)
    prob.params.u_p = pressure
    @debug "set u_p to $(prob.params.u_p)"
    nothing
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomCombination)
    shocks = hcat(SHOCK_DATA[1][:u], SHOCK_DATA[2][:u], SHOCK_DATA[3][:u], SHOCK_DATA[4][:u])
    fuels = hcat(SHOCK_DATA[1][:λ], SHOCK_DATA[2][:λ], SHOCK_DATA[3][:λ], SHOCK_DATA[4][:λ])
    T = eltype(shocks)
    weights = softmax(rand(T, 4), reset_strategy.temp)
    prob.u0 = shocks * weights
    prob.λ0 = fuels * weights
    prob.params.u_p = SHOCK_PRESSURES' * weights
    @debug "set u_p to $(prob.params.u_p)"
    nothing
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomShockOrCombination)
    if rand() < reset_strategy.shock_prob
        reset_state_and_pressure!(prob, NShock(rand(1:4)))
    else
        reset_state_and_pressure!(prob, RandomCombination(temp=reset_strategy.temp))
    end
    nothing
end