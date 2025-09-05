function default_u(x)
    return (3.0f0 / 2.0f0) .* sech.(x .- 1.0f0) .^ 20.0f0
end

function default_λ(x)
    return 0.5f0 .* ones(length(x))
end

function resample_data(data::Vector{T}, N::Int) where {T}
    L = length(data)
    if N == L
        return copy(data)
    elseif N < L
        # For downsampling, use regular sampling
        indices = round.(Int, range(1, L, length = N))
        return data[indices]
    else
        # For upsampling, use FFT interpolation
        # This preserves periodicity and is fast for power-of-2 sizes
        fft_data = fft(data)
        n_freq = length(fft_data) ÷ 2

        # Pad or truncate in frequency domain
        new_fft = zeros(Complex{T}, N)
        if N ≤ 2n_freq
            new_fft[1:(N ÷ 2)] = fft_data[1:(N ÷ 2)]
            new_fft[(end - N ÷ 2 + 1):end] = fft_data[(end - N ÷ 2 + 1):end]
        else
            new_fft[1:n_freq] = fft_data[1:n_freq]
            new_fft[(end - n_freq + 1):end] = fft_data[(end - n_freq + 1):end]
        end

        # Scale to preserve amplitude
        new_fft .*= N / L
        return real.(ifft(new_fft))
    end
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::Default)
    x = prob.x
    prob.u0 .= default_u(x)
    prob.λ0 .= default_λ(x)
    prob.params.u_p = 0.5f0
    return nothing
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
    prob.u0 .= resample_data(wave, prob.params.N)
    prob.λ0 .= resample_data(fuel, prob.params.N)
    prob.params.u_p = pressure
    return nothing
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomShock)
    return reset_state_and_pressure!(prob, NShock(rand(1:4)))
end


function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomCombination)
    T = eltype(SHOCK_MATRICES.shocks)
    weights = softmax(rand(T, 4), reset_strategy.temp)

    # Use pre-computed matrices
    wave = SHOCK_MATRICES.shocks * weights
    fuel = SHOCK_MATRICES.fuels * weights

    # Resample to match problem size
    prob.u0 .= resample_data(wave, prob.params.N)
    prob.λ0 .= resample_data(fuel, prob.params.N)

    # Set pressure
    prob.params.u_p = SHOCK_PRESSURES' * weights
    return nothing
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomShockOrCombination)
    if rand() < reset_strategy.shock_prob
        reset_state_and_pressure!(prob, NShock(rand(1:4)))
    else
        reset_state_and_pressure!(prob, RandomCombination(temp = reset_strategy.temp))
    end
    return nothing
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::ShiftReset)
    reset_state_and_pressure!(prob, reset_strategy.reset_strategy)
    shift = rand(1:length(prob.u0))
    circshift!(prob.u0, shift)
    circshift!(prob.λ0, shift)
    return nothing
end

struct SineCombination <: AbstractReset
    modes::Vector{Int}
end

function SineCombination(; modes = 2:9)
    return SineCombination(collect(modes))
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::SineCombination)
    x = prob.x
    modes = reset_strategy.modes
    shifts = rand(Float32, length(modes)) .* 2.0f0π

    # Create combination of sine waves
    M = stack([sin.(Float32(i) .* x .+ shifts[ix]) ./ (3.0f0 * i) for (ix, i) in enumerate(modes)])
    prob.u0 .= vec(1.0f0 .+ max.(0.0f0, sum(M, dims = 2)))

    # Set default lambda
    prob.λ0 = default_λ(x)

    # Set pressure to default
    prob.params.u_p = 0.5f0
    return nothing
end

struct WeightedCombination <: AbstractReset
    weights::Vector{Float32}
    function WeightedCombination(weights::Vector{Float32})
        length(weights) == 4 || throw(ArgumentError("weights must have length 4"))
        sum(weights) ≈ 1 || throw(ArgumentError("weights must sum to 1"))
        all(w -> w >= 0, weights) || throw(ArgumentError("weights must be non-negative"))
        return new(weights)
    end
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::WeightedCombination)
    wave = SHOCK_MATRICES.shocks * reset_strategy.weights
    fuel = SHOCK_MATRICES.fuels * reset_strategy.weights

    # Resample to match problem size
    prob.u0 .= resample_data(wave, prob.params.N)
    prob.λ0 .= resample_data(fuel, prob.params.N)

    # Set pressure
    prob.params.u_p = SHOCK_PRESSURES' * reset_strategy.weights
    return nothing
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::CustomPressureReset)
    x = prob.x
    prob.u0 .= reset_strategy.f(x)
    prob.λ0 .= default_λ(x)
    prob.params.u_p = 0.5f0
    return nothing
end

@kwdef mutable struct CycleShockReset <: AbstractReset
    n::Int = 1 #current shock index
    shocks::Vector{Int} = [1, 2, 3, 4]
end

#TODO: use a cycle(iterator), but only for julia 1.11+
function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::CycleShockReset)
    reset_state_and_pressure!(prob, NShock(reset_strategy.shocks[mod1(reset_strategy.n, length(reset_strategy.shocks))]))
    return reset_strategy.n += 1
end
