mutable struct RDEParam{T<:AbstractFloat}
    N::Int               # Number of spatial points
    L::T              # Domain length
    ν_1::T           # Viscosity coefficient
    ν_2::T
    u_c::T            # Parameter in ω(u)
    α::T            # Parameter in ω(u)
    q_0::T            # Source term coefficient
    u_0::T            # Parameter in ξ(u, u_0)
    n::Int                 # Exponent in ξ(u, u_0)
    k_param::T        # Parameter in β(u, s)
    u_p::T            # Parameter in β(u, s)
    s::T              # Parameter in β(u, s)
    ϵ::T             # Small parameter in ξ(u)
    tmax::T          # Maximum simulation time
    x0::T             # Initial position
    saveframes::Int       # Number of time steps to save
end
Base.length(params::RDEParam) = 1

function RDEParam{T}(;
    N=256,
    L=2π,
    ν_1=0.1,
    ν_2=0.1,
    u_c=1.1,
    α=0.3,
    q_0=1.0,
    u_0=0.0,
    n=1,
    k_param=5.0,
    u_p=0.5,
    s=3.5,
    ϵ=0.15,
    tmax=26.0,
    x0=1.0,
    saveframes=75) where {T<:AbstractFloat}
    RDEParam{T}(N, L, ν_1, ν_2, u_c, α, q_0, u_0, n, k_param, u_p, s, ϵ, tmax, x0, saveframes)
end

RDEParam(;kwargs...) = RDEParam{Float64}(;kwargs...)

"""Cache for computed values in the solver. To avoid allocations during simulation."""
mutable struct RDECache{T<:AbstractFloat}
    u_hat::Vector{Complex{T}}      # Complex array of size N÷2+1
    u_x_hat::Vector{Complex{T}}    # Complex array of size N÷2+1
    u_x::Vector{T}           # Real array of size N
    u_xx_hat::Vector{Complex{T}}   # Complex array of size N÷2+1
    u_xx::Vector{T}          # Real array of size N

    λ_hat::Vector{Complex{T}}      # Complex array of size N÷2+1
    λ_xx_hat::Vector{Complex{T}}   # Complex array of size N÷2+1
    λ_xx::Vector{T}          # Real array of size N

    ωu::Vector{T}            # Real array of size N
    ξu::Vector{T}            # Real array of size N
    βu::Vector{T}            # Real array of size N

    function RDECache{T}(N::Int) where {T<:AbstractFloat}
        N_complex = div(N, 2) + 1          # Size for complex arrays in rfft
        u_hat = Vector{Complex{T}}(undef, N_complex)
        u_x_hat = Vector{Complex{T}}(undef, N_complex)
        u_xx_hat = Vector{Complex{T}}(undef, N_complex)
        u_x = Vector{T}(undef, N)
        u_xx = Vector{T}(undef, N)

        λ_hat = Vector{Complex{T}}(undef, N_complex)
        λ_xx_hat = Vector{Complex{T}}(undef, N_complex)
        λ_xx = Vector{T}(undef, N)

        ωu = Vector{T}(undef, N)
        ξu = Vector{T}(undef, N)
        βu = Vector{T}(undef, N)
        return new{T}(u_hat, u_x_hat, u_x, u_xx_hat, u_xx, λ_hat, λ_xx_hat, λ_xx, ωu, ξu, βu)
    end
end

RDECache(N::Int) = RDECache{Float64}(N)

mutable struct RDEProblem{T<:AbstractFloat}
    # Parameters with defaults and explanations
    params::RDEParam{T}
    # Precomputed variables (initialized in init!)
    dx::T              # Spatial resolution
    x::Vector{T}       # Spatial grid
    k::Vector{T}       # Wavenumbers
    ik::Vector{Complex{T}}   # Spectral derivative operator (i*k)
    k2::Vector{T}      # Square of wavenumbers for Laplacian
    u0::Vector{T}      # Initial condition for u(x, 0)
    u_init::Function
    λ0::Vector{T}      # Initial condition for λ(x, 0)
    λ_init::Function
    sol::Union{Nothing,ODESolution}  # Solution (initially nothing)
    dealiasing::Vector{T}
    cache::RDECache{T}
    fft_plan::FFTW.rFFTWPlan{T}
    ifft_plan::FFTW.ScaledPlan
    calc_derivatives::Function

    # Constructor accepting keyword arguments to override defaults
    function RDEProblem{T}(params::RDEParam{T};
        u_init= (x, x0) -> (3 / 2) * (sech(x - x0))^(20),
        λ_init=x -> 0.5,
        dealias=true,
        method=:pseudospectral) where {T<:AbstractFloat}

        prob = new{T}()
        prob.params = params
        prob.dx = prob.params.L / prob.params.N
        prob.x = prob.dx * (0:prob.params.N-1)
        prob.ik, prob.k2 = create_spectral_derivative_arrays(params)
        prob.dealiasing = create_dealiasing_vector(params)
        if !dealias
            prob.dealiasing = ones(T, length(prob.dealiasing))
        end

        prob.u_init = u_init
        prob.λ_init = λ_init
        set_init_state!(prob)

        prob.λ0 = λ_init.(prob.x)
        prob.sol = nothing
        prob.cache = RDECache{T}(params.N)
        prob.fft_plan = plan_rfft(prob.u0; flags=FFTW.MEASURE)
        prob.ifft_plan = plan_irfft(prob.cache.u_hat, length(prob.u0); flags=FFTW.MEASURE)
        set_init_state!(prob) #as u0 may have been wiped while creating fft plans
        
        if method == :pseudospectral
            prob.calc_derivatives = pseudospectral_derivatives!
        elseif method == :fd
            prob.calc_derivatives = fd_derivatives!
        else
            error("method must be :pseudospectral or :fd")
        end
        return prob
    end
end

RDEProblem(params::RDEParam{T}; kwargs...) where {T<:AbstractFloat} = RDEProblem{T}(params; kwargs...)

function create_dealiasing_vector(params::RDEParam{T}) where {T<:AbstractFloat}
    N = params.N
    N_complex = div(N, 2) + 1
    k = collect(0:N_complex-1)
    k_cutoff = div(N, 3)

    # Construct the dealiasing vector using broadcasting
    dealiasing = @. ifelse(k <= k_cutoff, one(T), zero(T))

    return dealiasing
end

function create_spectral_derivative_arrays(params::RDEParam{T}) where {T<:AbstractFloat}
    N = params.N
    L = params.L
    N_complex = div(N, 2) + 1
    k = collect(T, 0:N_complex-1) .* T(2π / L)
    ik = 1im .* k
    k2 = k .^ 2
    return ik, k2
end

function set_init_state!(prob::RDEProblem)
    prob.u0 = prob.u_init.(prob.x, prob.params.x0)
    prob.λ0 = prob.λ_init.(prob.x)
end