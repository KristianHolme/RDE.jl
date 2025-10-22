function default_u(x::T) where {T <: AbstractFloat}
    return T(1.5) * sech(x - T(1.0))^20
end

function default_λ(x::T) where {T <: AbstractFloat}
    return T(0.5)
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

function reset_state_and_pressure!(prob::RDEProblem{T, M, R, C}, reset_strategy::Default) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    x = prob.x
    prob.u0 .= default_u.(x)
    prob.λ0 .= default_λ.(x)
    prob.params.u_p = T(0.5)
    return nothing
end

function reset_state_and_pressure!(prob::RDEProblem{T, M, R, C}, reset_strategy::NShock) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    n = reset_strategy.n
    if !(1 ≤ n ≤ 4)
        throw(ArgumentError("n must be between 1 and 4"))
    end
    @assert n in keys(SHOCK_DATA) "Shock data for n=$n not found"
    wave = SHOCK_DATA[n][:u]
    fuel = SHOCK_DATA[n][:λ]
    pressure = SHOCK_PRESSURES[n]
    prob.u0 .= T.(resample_data(wave, prob.params.N))
    prob.λ0 .= T.(resample_data(fuel, prob.params.N))
    prob.params.u_p = T(pressure)
    return nothing
end

function reset_state_and_pressure!(prob::RDEProblem, reset_strategy::RandomShock)
    return reset_state_and_pressure!(prob, NShock(rand(1:4)))
end


function reset_state_and_pressure!(prob::RDEProblem{T, M, R, C}, reset_strategy::RandomCombination) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    weights = softmax(rand(T, 4), T(reset_strategy.temp))

    # Use pre-computed matrices
    wave = SHOCK_MATRICES.shocks * weights
    fuel = SHOCK_MATRICES.fuels * weights

    # Resample to match problem size
    prob.u0 .= T.(resample_data(wave, prob.params.N))
    prob.λ0 .= T.(resample_data(fuel, prob.params.N))

    # Set pressure
    prob.params.u_p = T(SHOCK_PRESSURES' * weights)
    return nothing
end

function reset_state_and_pressure!(prob::RDEProblem{T, M, R, C}, reset_strategy::RandomShockOrCombination) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    if rand(T) < T(reset_strategy.shock_prob)
        reset_state_and_pressure!(prob, NShock(rand(1:4)))
    else
        reset_state_and_pressure!(prob, RandomCombination(temp = T(reset_strategy.temp)))
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

function reset_state_and_pressure!(prob::RDEProblem{T, M, R, C}, reset_strategy::SineCombination) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    x = prob.x
    modes = reset_strategy.modes
    shifts = rand(T, length(modes)) .* (T(2) * T(π))

    # Create combination of sine waves
    Mt = stack([sin.(T(i) .* x .+ shifts[ix]) ./ (T(3) * T(i)) for (ix, i) in enumerate(modes)])
    prob.u0 .= vec(T(1) .+ max.(T(0), sum(Mt, dims = 2)))

    # Set default lambda
    prob.λ0 .= default_λ.(x)

    # Set pressure to default
    prob.params.u_p = T(0.5)
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

function reset_state_and_pressure!(prob::RDEProblem{T, M, R, C}, reset_strategy::WeightedCombination) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    wT = T.(reset_strategy.weights)
    wave = SHOCK_MATRICES.shocks * wT
    fuel = SHOCK_MATRICES.fuels * wT

    # Resample to match problem size
    prob.u0 .= T.(resample_data(wave, prob.params.N))
    prob.λ0 .= T.(resample_data(fuel, prob.params.N))

    # Set pressure
    prob.params.u_p = T(SHOCK_PRESSURES' * wT)
    return nothing
end

function reset_state_and_pressure!(prob::RDEProblem{T, M, R, C}, reset_strategy::CustomPressureReset) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    x = prob.x
    prob.u0 .= T.(reset_strategy.f(x))
    prob.λ0 .= default_λ.(x)
    prob.params.u_p = T(0.5)
    return nothing
end

@kwdef mutable struct CycleShockReset <: AbstractReset
    n::Int = 1 #current shock index
    shocks::Vector{Int} = [1, 2, 3, 4]
end

#TODO: use a cycle(iterator), but only for julia 1.11+
function reset_state_and_pressure!(prob::RDEProblem{T, M, R, C}, reset_strategy::CycleShockReset) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    reset_state_and_pressure!(prob, NShock(reset_strategy.shocks[mod1(reset_strategy.n, length(reset_strategy.shocks))]))
    reset_strategy.n += 1
    return nothing
end

struct RandomReset <: AbstractReset end

function reset_state_and_pressure!(prob::RDEProblem{T, M, R, C}, reset_strategy::RandomReset) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    x = prob.x
    L = prob.params.L

    # Generate random periodic trigonometric series with decaying amplitudes
    # y(x) = baseline + Σ_{k=1..K} (a_k / k^p) * cos(2π k x / L + φ_k)
    function random_trig_series(
            x::AbstractVector{T}, L::T; K::Int = 12,
            amplitude::T = T(1), decay::T = T(1.5), baseline::T = T(0)
        )
        y = fill(baseline, length(x))
        twoπ = T(2) * T(π)
        @inbounds for k in 1:K
            amp_k = amplitude / (T(k)^decay)
            ϕ = twoπ * rand(T)
            y .+= amp_k .* cos.(twoπ * T(k) .* x ./ L .+ ϕ)
        end
        return y
    end

    # Velocity: nonnegative, around O(1)
    u_rand = random_trig_series(x, L; K = 12, amplitude = T(0.8), decay = T(1.4), baseline = T(1))
    @. u_rand = max(u_rand, T(0))

    # Reaction progress: clamp to [0, 1], centered around 0.5
    λ_rand = random_trig_series(x, L; K = 12, amplitude = T(0.3), decay = T(1.6), baseline = T(0.5))
    @. λ_rand = clamp(λ_rand, T(0), T(1))

    prob.u0 .= u_rand
    prob.λ0 .= λ_rand
    prob.params.u_p = T(0.5)
    return nothing
end

# ----------------------------------------------------------------------------
# EvalCycleShockReset
# ----------------------------------------------------------------------------
mutable struct EvalCycleShockReset <: AbstractReset
    init_shocks::Vector{Int}
    current_config::Int
    repetitions_per_config::Int
    function EvalCycleShockReset(repetitions_per_config::Int)
        init_shocks = vcat([repeat(setdiff(1:4, i), inner = repetitions_per_config) for i in 1:4]...)
        return new(init_shocks, 1, repetitions_per_config)
    end
end

function reset_state_and_pressure!(prob::RDEProblem{T, M, R, C}, reset_strategy::EvalCycleShockReset) where {T <: AbstractFloat, M <: AbstractMethod, R <: AbstractReset, C <: AbstractControlShift}
    reset_strategy.current_config += 1
    if reset_strategy.current_config > length(reset_strategy.init_shocks)
        reset_strategy.current_config = 1
    end
    current_config = reset_strategy.current_config
    temp_reset_strategy = NShock(reset_strategy.init_shocks[current_config])
    reset_state_and_pressure!(prob, temp_reset_strategy)
    return nothing
end
