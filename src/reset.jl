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
    ref_x = range(0, 2π, length=513)[1:end-1]
    itp_u = linear_interpolation(ref_x, wave, extrapolation_bc=Periodic())
    itp_λ = linear_interpolation(ref_x, fuel, extrapolation_bc=Periodic())
    prob.u0 = itp_u.(prob.x)
    prob.λ0 = itp_λ.(prob.x)
    prob.params.u_p = pressure
    nothing
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomShock)
    reset_state_and_pressure!(prob, NShock(rand(1:4)))
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomCombination)
    shocks = hcat(SHOCK_DATA[1][:u], SHOCK_DATA[2][:u], SHOCK_DATA[3][:u], SHOCK_DATA[4][:u])
    fuels = hcat(SHOCK_DATA[1][:λ], SHOCK_DATA[2][:λ], SHOCK_DATA[3][:λ], SHOCK_DATA[4][:λ])
    T = eltype(shocks)
    weights = softmax(rand(T, 4), reset_strategy.temp)
    wave = shocks * weights
    fuel = fuels * weights
    ref_x = range(0, 2π, length=513)[1:end-1]
    itp_u = linear_interpolation(ref_x, wave, extrapolation_bc=Periodic())
    itp_λ = linear_interpolation(ref_x, fuel, extrapolation_bc=Periodic())
    prob.u0 = itp_u.(prob.x)
    prob.λ0 = itp_λ.(prob.x)
    prob.params.u_p = SHOCK_PRESSURES' * weights
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

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::ShiftReset)
    reset_state_and_pressure!(prob, reset_strategy.reset_strategy)
    shift = rand(1:length(prob.u0))
    circshift!(prob.u0, shift)
    circshift!(prob.λ0, shift)
    nothing
end