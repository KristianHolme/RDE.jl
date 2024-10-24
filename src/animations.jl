using ProgressMeter

function animate_policy(π::P, env::RDEEnv; kwargs...) where P <: Policy
    data = run_policy(π, env;)
    animate_policy_data(data, env; kwargs...)
end

function animate_policy_data(data::PolicyRunData, env::RDEEnv;
        dir_path="./videos/", fname="policy", format=".mp4", fps=25)
    time_idx = Observable(1)
    time_steps = length(data.sparse_ts)
    fig = plot_policy_data(env, data; time_idx, player_controls=false, show_mouse_vlines=false)

    if !isdir(dir_path)
        mkdir(dir_path)
    end

    path  = joinpath(dir_path, fname*format)
    p = Progress(time_steps, desc="Recording animation...");
    record(fig, path, 1:time_steps, framerate=fps) do i
        time_idx[] = i
        next!(p)
    end
end


function animate_RDE(RDE::RDEProblem; dir_path="./videos/", fname="RDE", format=".mp4", fps=25)
    if isnothing(RDE.sol)
        solve_pde!(RDE)
    end

    time_idx = Observable(1)
    time_steps = length(RDE.sol.t)
    fig = plot_solution(RDE; time_idx, player_controls=false)

    if !isdir(dir_path)
        mkdir(dir_path)
    end

    path = joinpath(dir_path, fname*format)

    p = Progress(time_steps, desc="Recording animation...");
    record(fig, path, 1:time_steps, framerate=fps) do i
        time_idx[] = i
        next!(p)
    end
end