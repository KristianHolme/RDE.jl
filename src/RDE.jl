# __precompile__(false)
module RDE
using CircularArrays
using DataFrames
using FFTW
using FileIO
using GLM
using Interpolations
using JLD2
using LinearAlgebra
using Logging
using LoopVectorization
using Makie
using Observables
using OrdinaryDiffEq
using DiffEqCallbacks
using Pkg.Artifacts
using ProgressMeter
using Random
using Statistics
using StatsModels

using PrecompileTools

# Core simulator exports
export RDEParam, RDEProblem, solve_pde!
export plot_solution
export animate_RDE
export energy_balance, chamber_pressure
export RDE_RHS!
export AbstractMethod, FiniteVolumeMethod, reset_cache!
export AbstractLimiter, MinmodLimiter, MCLimiter
export AbstractReset, Default, NShock, RandomCombination,
    RandomShockOrCombination, RandomShock, ShiftReset,
    SineCombination, WeightedCombination, CustomPressureReset, CycleShockReset,
    RandomReset, EvalCycleShockReset
export AbstractControlShift, ZeroControlShift, LinearControlShift

# Core simulator includes
include("control.jl")      # Control strategies
include("types.jl")        # Base types
include("reset.jl")        # Reset strategies
include("methods.jl")      # Method implementations
include("structs.jl")      # Problem construction
include("utils.jl")        # Utility functions
include("solver.jl")       # Solver implementation
include("plotting.jl")     # Plotting functions
include("animations.jl")   # Animation functions

@compile_workload begin
    try
        # Simulate tiny case for a short time
        prob = RDEProblem(RDEParam(; N = 64, tmax = 0.01))
        solve_pde!(prob)
    catch e
        @warn "Precompilation failure: $e"
    end
end
end
