module RDE

# Write your package code here.
    using DifferentialEquations
    using FFTW
    using LinearAlgebra
    using LoopVectorization
    using Makie
    using Observables
    using POMDPs
    using POMDPTools
    using CommonRLInterface
    using Interpolations
    using ProgressMeter

    using PrecompileTools

    
    export RDEParam, RDEProblem, RDEEnv, solve_pde!
    export ConstantRDEPolicy, run_policy, PolicyRunData
    export plot_solution, plot_policy, plot_policy_data, animate_policy, animate_RDE
    


    include("solver.jl")
    include("RLenv.jl")
    include("plotting.jl")
    include("animations.jl")
    include("interactive_control.jl")
    include("utils.jl")


    @compile_workload begin
        try
            #simulate tiny case for a short time
            prob = RDEProblem(RDEParam(;N=32, tmax = 0.01));
            solve_pde!(prob, progress=true);
        catch e
            @warn "Precompilation failure: $e"
        end
    end
    

end
