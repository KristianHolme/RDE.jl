# __precompile__(false)
module RDE
    using CircularArrays
    using FFTW
    using FileIO
    using Interpolations
    using JLD2
    using LinearAlgebra
    using Logging
    using LoopVectorization
    using Makie
    using Observables
    using OrdinaryDiffEq
    using ProgressMeter
    using Random
    using Statistics
    
    using PrecompileTools
    
    # Core simulator exports
    export RDEParam, RDEProblem, solve_pde!
    export plot_solution, plot_shifted_history
    export animate_RDE
    export get_n_shocks_init_func, random_shock_init_func, random_shock_combination_init_func
    export energy_balance, chamber_pressure
    export RDE_RHS!

    # Core simulator includes
    include("structs.jl")
    include("utils.jl")
    include("solver.jl")
    include("plotting.jl")
    include("animations.jl")
    include("analysis.jl")

    @compile_workload begin
        try
            # Simulate tiny case for a short time
            prob = RDEProblem(RDEParam(;N=64, tmax = 0.01))
            solve_pde!(prob)
        catch e
            @warn "Precompilation failure: $e"
        end
    end
end
