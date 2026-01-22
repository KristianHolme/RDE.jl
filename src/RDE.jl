# __precompile__(false)
module RDE
using CircularArrays
using DataFrames
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
export AbstractMethod, FiniteVolumeMethod
export AbstractLimiter, MinmodLimiter, MCLimiter
export AbstractReset, Default, NShock, RandomCombination,
    RandomShockOrCombination, RandomShock, ShiftReset,
    SineCombination, WeightedCombination, CustomPressureReset, CycleShockReset,
    RandomReset, EvalCycleShockReset
export AbstractControlShift, ZeroControlShift, LinearControlShift

# Utilities used by downstream packages
export SHOCK_PRESSURES, count_shocks, shock_indices, split_sol, split_sol_view
export periodic_diff, smooth_g, get_control_shift, get_dx
export turbo_column_maximum, turbo_column_maximum!, turbo_diff_norm, turbo_extrema
export turbo_maximum, turbo_minimum, turbo_maximum_abs, turbo_maximum_abs_diff

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

@setup_workload begin
    params = RDEParam(; N = 32, tmax = 0.01)
    prob = RDEProblem(params)
    uλ_0 = vcat(prob.u0, prob.λ0)
    tspan = (zero(params.tmax), params.tmax)
    ode_problem = ODEProblem(RDE_RHS!, uλ_0, tspan, prob)
    @compile_workload begin
        duλ = similar(uλ_0)
        RDE_RHS!(duλ, uλ_0, prob, zero(params.tmax))
        solve_pde_step(
            prob,
            ode_problem;
            adaptive = false,
            dt = params.tmax / 10,
            callback = nothing
        )
    end
end
end
