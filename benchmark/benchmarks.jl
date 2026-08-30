using BenchmarkTools
using RDE
using RDE.OrdinaryDiffEq
using RDE.OrdinaryDiffEqSSPRK

include("bench_utils.jl")
using .BenchUtils

const SUITE = BenchmarkGroup()

solver = BenchmarkGroup()
SUITE["solver"] = solver

solver["solve_pde"] = @benchmarkable begin
    solve_pde!(
        prob;
        saveframes = BenchUtils.DEFAULT_SAVEFRAMES,
        alg = SSPRK33(),
        adaptive = false
    )
end setup = begin
    prob = BenchUtils.setup_solve_problem()
end

solver["solve_pde_no_smoothing_no_shift"] = @benchmarkable begin
    solve_pde!(
        prob;
        saveframes = BenchUtils.DEFAULT_SAVEFRAMES,
        alg = SSPRK33(),
        adaptive = false
    )
end setup = begin
    prob = BenchUtils.setup_solve_problem(;
        spatial_smoothing_width = 0,
        control_shift_strategy = ZeroControlShift()
    )
end

solver["solve_pde_no_smoothing_shift"] = @benchmarkable begin
    solve_pde!(
        prob;
        saveframes = BenchUtils.DEFAULT_SAVEFRAMES,
        alg = SSPRK33(),
        adaptive = false
    )
end setup = begin
    prob = BenchUtils.setup_solve_problem(;
        spatial_smoothing_width = 0,
        control_shift_strategy = LinearControlShift(one(Float32))
    )
end

solver["solve_pde_smoothing_no_shift"] = @benchmarkable begin
    solve_pde!(
        prob;
        saveframes = BenchUtils.DEFAULT_SAVEFRAMES,
        alg = SSPRK33(),
        adaptive = false
    )
end setup = begin
    prob = BenchUtils.setup_solve_problem(;
        spatial_smoothing_width = BenchUtils.DEFAULT_SPATIAL_WIDTH,
        control_shift_strategy = ZeroControlShift()
    )
end

solver["solve_pde_smoothing_shift"] = @benchmarkable begin
    solve_pde!(
        prob;
        saveframes = BenchUtils.DEFAULT_SAVEFRAMES,
        alg = SSPRK33(),
        adaptive = false
    )
end setup = begin
    prob = BenchUtils.setup_solve_problem(;
        spatial_smoothing_width = BenchUtils.DEFAULT_SPATIAL_WIDTH,
        control_shift_strategy = LinearControlShift(one(Float32))
    )
end

rhs = BenchmarkGroup()
SUITE["rhs"] = rhs

rhs["RDE_RHS!"] = @benchmarkable begin
    RDE_RHS!(duλ, uλ, prob, t)
end setup = begin
    prob, uλ, duλ, t = BenchUtils.setup_rhs()
end

control = BenchmarkGroup()
SUITE["control"] = control

control["update_control_shifted!"] = @benchmarkable begin
    RDE.update_control_shifted!(cache, control_shift_strategy, u, t)
end setup = begin
    cache, control_shift_strategy, u, t = BenchUtils.setup_control_shift()
end

control["smooth_control!"] = @benchmarkable begin
    RDE.smooth_control!(target, t, control_t, current, previous, τ_smooth)
end setup = begin
    target, t, control_t, current, previous, τ_smooth = BenchUtils.setup_smooth_control()
end

control["smooth_spatial!"] = @benchmarkable begin
    RDE.smooth_spatial!(target, scratch, kernel)
end setup = begin
    target, scratch, kernel = BenchUtils.setup_smooth_spatial()
end

cfl = BenchmarkGroup()
SUITE["cfl"] = cfl

cfl["cfl_dtFE"] = @benchmarkable begin
    RDE.cfl_dtFE(uλ, prob, t)
end setup = begin
    uλ, prob, t = BenchUtils.setup_cfl_dtFE()
end

utils = BenchmarkGroup()
SUITE["utils"] = utils

utils["shock_locations"] = @benchmarkable begin
    RDE.shock_locations(u, dx)
end setup = begin
    u, dx = BenchUtils.setup_shock_inputs()
end

utils["shock_indices"] = @benchmarkable begin
    RDE.shock_indices(u, dx)
end setup = begin
    u, dx = BenchUtils.setup_shock_inputs()
end
