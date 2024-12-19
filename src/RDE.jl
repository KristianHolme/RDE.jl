# __precompile__(false)
module RDE

    using CommonRLInterface
    using CircularArrays
    using DomainSets
    using FFTW
    using Interpolations
    using JLD2
    using LinearAlgebra
    using LoopVectorization
    using Makie
    using Observables
    using OrdinaryDiffEq
    using POMDPs
    using POMDPTools
    using ProgressMeter
    using Random
    using Statistics
    
    using PrecompileTools
    
    export RDEParam, RDEProblem, RDEEnv, solve_pde!
    export ConstantRDEPolicy, run_policy, PolicyRunData
    export SinusoidalRDEPolicy, StepwiseRDEPolicy, RandomRDEPolicy
    export plot_solution, plot_policy, plot_policy_data, plot_shifted_history
    export animate_policy, animate_policy_data, animate_RDE
    export interactive_control
    export get_n_shocks_init_func
    export get_standard_normalized_actions, AbstractActionType, ScalarPressureAction, 
           VectorPressureAction, ScalarAreaScalarPressureAction
    export FourierObservation, StateObservation, SampledStateObservation

    include("action_types.jl")
    include("structs.jl")
    include("utils.jl")
    include("solver.jl")
    include("RLenv.jl")
    include("plotting.jl")
    include("animations.jl")
    include("interactive_control.jl")


    @compile_workload begin
        try
            #simulate tiny case for a short time
            prob = RDEProblem(RDEParam(;N=64, tmax = 0.01));
            solve_pde!(prob);
        catch e
            @warn "Precompilation failure: $e"
        end
    end
end
