# Display methods for RDEParam
function Base.show(io::IO, params::RDEParam)
    println(io, "RDEParam:")
    println(io, "  N: $(params.N)")
    println(io, "  L: $(params.L)")
    println(io, "  ν_1: $(params.ν_1)")
    println(io, "  ν_2: $(params.ν_2)")
    println(io, "  u_c: $(params.u_c)")
    println(io, "  α: $(params.α)")
    println(io, "  q_0: $(params.q_0)")
    println(io, "  u_0: $(params.u_0)")
    println(io, "  n: $(params.n)")
    println(io, "  k_param: $(params.k_param)")
    println(io, "  u_p: $(params.u_p)")
    println(io, "  s: $(params.s)")
    println(io, "  ϵ: $(params.ϵ)")
    println(io, "  tmax: $(params.tmax)")
    println(io, "  x0: $(params.x0)")
end

Base.length(params::RDEParam) = 1

# Display methods for RDEProblem
function Base.show(io::IO, prob::RDEProblem)
    println(io, "RDEProblem:")
    println(io, "  params: $(prob.params)")
    println(io, "  u0: $(typeof(prob.u0))")
    println(io, "  λ0: $(typeof(prob.λ0))")
    println(io, "  x: $(typeof(prob.x))")
    println(io, "  reset_strategy: $(prob.reset_strategy)")
    println(io, "  sol: $(typeof(prob.sol))")
    println(io, "  method: $(prob.method)")
    println(io, "  control_shift_strategy: $(prob.control_shift_strategy)")
end

"""
    RDEProblem(params::RDEParam{T}; kwargs...) where {T<:AbstractFloat}

Construct an RDE problem with given parameters.

# Arguments
- `params::RDEParam{T}`: Model parameters

# Keywords
- `reset_strategy::AbstractReset = DefaultReset()`: Reset strategy
- `method::AbstractMethod = FiniteDifferenceMethod{T}()`: Numerical method
- `control_shift_strategy::AbstractControlShift = ZeroControlShift()`: Control shift strategy

# Returns
- `RDEProblem{T}`: Initialized problem

# Examples
```julia
# Using finite difference method (default)
prob = RDEProblem(params)

# Using pseudospectral method with dealiasing
prob = RDEProblem(params, method=PseudospectralMethod{Float64}(dealias=true))

# Using pseudospectral method without dealiasing
prob = RDEProblem(params, method=PseudospectralMethod{Float64}(dealias=false))
```
"""
function RDEProblem(params::RDEParam{T};
    reset_strategy::AbstractReset = Default(),
    method::AbstractMethod = FiniteDifferenceMethod{T}(),
    control_shift_strategy::AbstractControlShift = ZeroControlShift()) where {T<:AbstractFloat}

    x = range(0, params.L, length=params.N+1)[1:end-1]
    dx = x[2] - x[1]

    # Initialize the method's cache
    init_cache!(method, params, dx)

    prob = RDEProblem{T}(params, Vector{T}(undef, params.N), Vector{T}(undef, params.N), 
                         x, reset_strategy, nothing, method, control_shift_strategy)
    set_init_state!(prob) #state may have been erased when creating fft plans in pseudospectral cache
    return prob
end

"""
    set_init_state!(prob::RDEProblem)

Initialize the state vectors of an RDE problem using the initialization functions.
"""
function set_init_state!(prob::RDEProblem)
    reset_state_and_pressure!(prob, prob.reset_strategy)
    @assert all(isfinite.(prob.u0)) "NaN or Inf values detected in u0"
    @assert all(isfinite.(prob.λ0)) "NaN or Inf values detected in λ0"
end