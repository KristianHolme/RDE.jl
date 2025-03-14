function default_u(x)
    return (3f0 / 2f0) .* sech.(x .- 1f0).^20f0
end

function default_λ(x)
    return 0.5f0 .* ones(length(x))
end

function resample_data(data::Vector{T}, N::Int) where T
    L = length(data)
    if N == L
        return copy(data)
    elseif N < L
        # For downsampling, use regular sampling
        indices = round.(Int, range(1, L, length=N))
        return data[indices]
    else
        # For upsampling, use FFT interpolation
        # This preserves periodicity and is fast for power-of-2 sizes
        fft_data = fft(data)
        n_freq = length(fft_data) ÷ 2
        
        # Pad or truncate in frequency domain
        new_fft = zeros(Complex{T}, N)
        if N ≤ 2n_freq
            new_fft[1:N÷2] = fft_data[1:N÷2]
            new_fft[end-N÷2+1:end] = fft_data[end-N÷2+1:end]
        else
            new_fft[1:n_freq] = fft_data[1:n_freq]
            new_fft[end-n_freq+1:end] = fft_data[end-n_freq+1:end]
        end
        
        # Scale to preserve amplitude
        new_fft .*= N/L
        return real.(ifft(new_fft))
    end
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::Default)
    x = prob.x
    prob.u0 = default_u(x)
    prob.λ0 = default_λ(x)
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
    # ref_x = range(0, 2π, length=513)[1:end-1]
    # itp_u = linear_interpolation(ref_x, wave, extrapolation_bc=Periodic())
    # itp_λ = linear_interpolation(ref_x, fuel, extrapolation_bc=Periodic())
    # prob.u0 = itp_u.(prob.x)
    # prob.λ0 = itp_λ.(prob.x)
    prob.u0 = resample_data(wave, prob.params.N)
    prob.λ0 = resample_data(fuel, prob.params.N)
    prob.params.u_p = pressure
    nothing
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomShock)
    reset_state_and_pressure!(prob, NShock(rand(1:4)))
end

# function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomCombination)
#     shocks = hcat(SHOCK_DATA[1][:u], SHOCK_DATA[2][:u], SHOCK_DATA[3][:u], SHOCK_DATA[4][:u])
#     fuels = hcat(SHOCK_DATA[1][:λ], SHOCK_DATA[2][:λ], SHOCK_DATA[3][:λ], SHOCK_DATA[4][:λ])
#     T = eltype(shocks)
#     weights = softmax(rand(T, 4), reset_strategy.temp)
#     wave = shocks * weights
#     fuel = fuels * weights
#     ref_x = range(0, 2π, length=513)[1:end-1]
#     itp_u = linear_interpolation(ref_x, wave, extrapolation_bc=Periodic())
#     itp_λ = linear_interpolation(ref_x, fuel, extrapolation_bc=Periodic())
#     prob.u0 = itp_u.(prob.x)
#     prob.λ0 = itp_λ.(prob.x)
#     prob.params.u_p = SHOCK_PRESSURES' * weights
#     nothing
# end

# At module level, pre-compute the matrices


function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomCombination)
    T = eltype(SHOCK_MATRICES.shocks)
    weights = softmax(rand(T, 4), reset_strategy.temp)
    
    # Use pre-computed matrices
    wave = SHOCK_MATRICES.shocks * weights
    fuel = SHOCK_MATRICES.fuels * weights
    
    # Resample to match problem size
    prob.u0 = resample_data(wave, prob.params.N)
    prob.λ0 = resample_data(fuel, prob.params.N)
    
    # Set pressure
    prob.params.u_p = SHOCK_PRESSURES' * weights
    return nothing
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