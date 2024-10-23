using RDE
using GLMakie

# Create an instance of RDEEnv with custom parameters if desired
env = RDEEnv(; N=256, tmax=26.0)

# Create a SinusoidalRDEPolicy
policy = SinusoidalRDEPolicy(env; w_1=1.0, w_2=0.0)

# Run the policy
data = run_policy(policy, env)

# Plot the results
fig = plot_policy_data(env, data)

# Display the figure
display(fig)
