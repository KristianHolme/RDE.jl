# __precompile__(false)
module RDE

    using CommonRLInterface
    using CircularArrays
    using DomainSets
    using FFTW
    using FileIO
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
    export AbstractRDEReward, ShockSpanReward, ShockPreservingReward, ShockPreservingSymmetryReward
    export set_reward!

    include("action_types.jl")
    include("structs.jl")
    include("utils.jl")
    include("solver.jl")
    include("RLenv.jl")
    include("rewards.jl")
    include("plotting.jl")
    include("animations.jl")
    include("interactive_control.jl")

    @compile_workload begin
        try
            # Simulate tiny case for a short time
            prob = RDEProblem(RDEParam(;N=64, tmax = 0.01))
            solve_pde!(prob)

            # Run small environment with random policy
            env = RDEEnv(;
                dt=0.01,
                smax=4.0,
                u_pmax=1.2,
                params=RDEParam(;N=32, tmax=0.05),
                τ_smooth=0.01,
                momentum=0.8,
                observation_strategy=FourierObservation(8),
                action_type=ScalarPressureAction()
            )
            policy = RandomRDEPolicy(env)
            data = run_policy(policy, env, saves_per_action=2)
        catch e
            @warn "Precompilation failure: $e"
        end
    end
end
