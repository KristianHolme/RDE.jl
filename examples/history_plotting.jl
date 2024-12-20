using RDE
using GLMakie

env = RDEEnv(dt=0.1f0);
π = StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 350.0f0], 
[[3.5f0, 0.64f0], [3.5f0, 0.86f0], [3.5f0, 0.64f0], [3.5f0, 0.94f0]]);
data = run_policy(π, env; tmax=500.0);



x = env.prob.x
dx = x[2] - x[1]
t = data.sparse_ts
u = data.sparse_states
us, λs = RDE.split_sol(u)

fig = plot_shifted_history(us, x, t, 1.795, u_ps=data.u_ps)