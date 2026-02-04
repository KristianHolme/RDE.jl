module BenchUtils

using RDE
const DEFAULT_N = 512
const DEFAULT_TMAX = 5.0f0
const DEFAULT_SAMPLES = 200
const DEFAULT_SAVEFRAMES = 10
const DEFAULT_SPATIAL_WIDTH = 9

function make_params(; T = Float32, N = DEFAULT_N, tmax = DEFAULT_TMAX)
    params = RDEParam{T}(N = N, tmax = tmax)
    return params
end

function make_problem(;
        T = Float32,
        N = DEFAULT_N,
        tmax = DEFAULT_TMAX,
        reset_strategy = Default(),
        control_shift_strategy = ZeroControlShift(),
        method = FiniteVolumeMethod{T}()
    )
    params = make_params(; T = T, N = N, tmax = tmax)
    prob = RDEProblem(
        params;
        reset_strategy = reset_strategy,
        method = method,
        control_shift_strategy = control_shift_strategy
    )
    return prob
end

function setup_solve_problem(; T = Float32, N = DEFAULT_N, tmax = DEFAULT_TMAX)
    prob = make_problem(; T = T, N = N, tmax = tmax)
    return prob
end

function setup_rhs(; T = Float32, N = DEFAULT_N, tmax = DEFAULT_TMAX)
    prob = make_problem(; T = T, N = N, tmax = tmax)
    uλ = vcat(prob.u0, prob.λ0)
    duλ = similar(uλ)
    t = zero(T)
    return prob, uλ, duλ, t
end

function setup_shock_inputs(; T = Float32, N = DEFAULT_N)
    prob = make_problem(; T = T, N = N, tmax = DEFAULT_TMAX)
    dx = get_dx(prob)
    u = copy(prob.u0)
    return u, dx
end

function setup_control_shift(;
        T = Float32,
        N = DEFAULT_N,
        tmax = DEFAULT_TMAX,
        width_points = DEFAULT_SPATIAL_WIDTH
    )
    control_shift_strategy = LinearControlShift(T(1))
    prob = make_problem(
        ; T = T,
        N = N,
        tmax = tmax,
        control_shift_strategy = control_shift_strategy
    )
    cache = prob.method.cache
    set_spatial_control_smoothing!(cache, width_points)
    u = prob.u0
    t = T(2) * cache.dx
    return cache, control_shift_strategy, u, t
end

function setup_smooth_control(; T = Float32, N = DEFAULT_N)
    target = zeros(T, N)
    current = fill(T(2), N)
    previous = fill(T(1), N)
    t = T(0.5)
    control_t = zero(T)
    τ_smooth = one(T)
    return target, t, control_t, current, previous, τ_smooth
end

function setup_smooth_spatial(;
        T = Float32,
        N = DEFAULT_N,
        width_points = DEFAULT_SPATIAL_WIDTH
    )
    prob = make_problem(; T = T, N = N, tmax = DEFAULT_TMAX)
    cache = prob.method.cache
    set_spatial_control_smoothing!(cache, width_points)
    target = copy(cache.s_t)
    scratch = cache.spatial_scratch
    kernel = cache.spatial_kernel
    return target, scratch, kernel
end

function setup_cfl_dtFE(; T = Float32, N = DEFAULT_N, tmax = DEFAULT_TMAX)
    prob = make_problem(; T = T, N = N, tmax = tmax)
    uλ = vcat(prob.u0, prob.λ0)
    t = zero(T)
    return uλ, prob, t
end

end
