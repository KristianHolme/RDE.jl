using RDE
using WGLMakie
using Observables


# Run a stepwise policy that creates different modes with varying shock numbers
env = RDEEnv(dt=0.2f0);
π = StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 350.0f0], 
[[3.5f0, 0.64f0], [3.5f0, 0.86f0], [3.5f0, 0.64f0], [3.5f0, 0.94f0]]);
data = run_policy(π, env; tmax=500.0);

x = env.prob.x
dx = x[2] - x[1]
ts = data.sparse_ts;
us, λs = RDE.split_sol(data.sparse_states);
time_idx = Observable(1)
u = @lift(us[$time_idx])
shock_ends = @lift(shock_locations($u, dx))
shock_locs = Observable(Float32[0.0])
shocks_alpha = Observable(0.0f0)

on(shock_ends) do ends
    if any(ends)
        shocks_alpha[] = 1.0f0
        shock_locs[] = x[ends]
    else
        shocks_alpha[] = 0.0f0
    end
end
begin
    fig = Figure()
    ax = Axis(fig[1, 1], limits=(nothing, (0.0, 3.0)))
    lines!(ax, x, u, label="u")
    vlines!(ax, shock_locs, color=:red, label="Shocks", alpha=shocks_alpha, linestyle=:dash)
    fig
end



# Run through the timeseries
for i in eachindex(ts)
    time_idx[] = i
    sleep(1/25)
end

