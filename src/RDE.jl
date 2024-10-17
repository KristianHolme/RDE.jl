module RDE

# Write your package code here.
    using CommonRLInterface
    using DifferentialEquations
    using FFTW
    using Interpolations
    using LinearAlgebra
    using LoopVectorization
    using Makie
    using Observables
    using POMDPs
    using POMDPTools
    using ProgressMeter
    
    using PrecompileTools
    
    export RDEParam, RDEProblem, RDEEnv, solve_pde!
    export ConstantRDEPolicy, run_policy, PolicyRunData
    export plot_solution, plot_policy, plot_policy_data, animate_policy, animate_RDE
    export interactive_RDE_control
    


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
            prob = RDEProblem(RDEParam(;N=8, tmax = 0.01));
            solve_pde!(prob);
        catch e
            @warn "Precompilation failure: $e"
        end
    end
end
