using RDE
using GLMakie

env = RDEEnv(dt=0.1f0);
π = StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 350.0f0], 
[[3.5f0, 0.64f0], [3.5f0, 0.86f0], [3.5f0, 0.64f0], [3.5f0, 0.94f0]]);
data = run_policy(π, env; tmax=500.0)
fig = plot_policy_data(env, data)

animate_policy_data(data, env; fname="stepwise_control", fps=60)