using RDE
using GLMakie
# Create an instance of RDEProblem with custom parameters if desired
params = RDEParam(;N=64, tmax=26.0)
rde_prob = RDEProblem(params);

# Solve the PDE
@time out = solve_pde!(rde_prob; progress=true);

# Plot solution
plot_solution(rde_prob);