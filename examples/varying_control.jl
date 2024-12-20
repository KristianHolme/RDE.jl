using RDE
using GLMakie

# Create an instance of RDEEnv with custom parameters if desired
env = RDEEnv(; params=RDEParam(tmax=100.0f0))
env.dt = 0.4
# Create a SinusoidalRDEPolicy
policy = SinusoidalRDEPolicy(env; w_1=1.0f0, w_2=0.0f0)

# Run the policy
data = run_policy(policy, env)

# Plot the results
fig = plot_policy_data(env, data)

# Display the figure
display(fig)
