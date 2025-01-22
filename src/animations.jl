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