using RDE
using CairoMakie
using OrdinaryDiffEq

"""
Example demonstrating the use of the upwind method for RDE simulation.

This example:
1. Creates RDE problems with different numerical methods
2. Solves the PDE systems
3. Visualizes and compares the results using Makie
"""

# Create parameters with a higher advection-to-diffusion ratio to highlight 
# the benefits of upwinding
params = RDEParam(tmax = 10.0, N=2048)

# Create problems with different methods
upwind1_prob = RDEProblem(params, method=UpwindMethod(order=1))
upwind2_prob = RDEProblem(params, method=UpwindMethod(order=2))
fd_prob = RDEProblem(params, method=FiniteDifferenceMethod())
ps_prob = RDEProblem(params, method=PseudospectralMethod())

# Solve all problems
println("Solving with 1st-order upwind method...")
solve_pde!(upwind1_prob, saveat=0.1)

println("Solving with 2nd-order upwind method...")
solve_pde!(upwind2_prob, saveat=0.1)

println("Solving with finite difference method...")
solve_pde!(fd_prob, saveat=0.1)

println("Solving with pseudospectral method...")
solve_pde!(ps_prob, saveat=0.1)

# Plot the results at a specific time point
function plot_comparison(upwind1_prob, upwind2_prob, fd_prob, ps_prob, time_index=50)
    fig = Figure(size=(900, 900))
    
    # Extract solutions at the specified time
    u_upwind1 = upwind1_prob.sol[1:upwind1_prob.params.N, time_index]
    λ_upwind1 = upwind1_prob.sol[upwind1_prob.params.N+1:end, time_index]
    
    u_upwind2 = upwind2_prob.sol[1:upwind2_prob.params.N, time_index]
    λ_upwind2 = upwind2_prob.sol[upwind2_prob.params.N+1:end, time_index]
    
    u_fd = fd_prob.sol[1:fd_prob.params.N, time_index]
    λ_fd = fd_prob.sol[fd_prob.params.N+1:end, time_index]
    
    u_ps = ps_prob.sol[1:ps_prob.params.N, time_index]
    λ_ps = ps_prob.sol[ps_prob.params.N+1:end, time_index]
    
    x = upwind1_prob.x
    
    # Plot u - full view
    ax1 = Axis(fig[1, 1], xlabel="x", ylabel="u", title="Solution Comparison: u")
    lines!(ax1, x, u_upwind1, label="1st-order Upwind", linewidth=2)
    lines!(ax1, x, u_upwind2, label="2nd-order Upwind", linewidth=2, color=:red)
    lines!(ax1, x, u_fd, label="Finite Difference", linestyle=:dash)
    lines!(ax1, x, u_ps, label="Pseudospectral", linestyle=:dot)
    axislegend(ax1, position=:lt)
    
    # Plot u - zoomed to highlight differences
    # Find a region with a steep gradient for zooming
    deriv = diff(u_ps)
    _, max_idx = findmax(abs.(deriv))
    zoom_center = x[max_idx]
    zoom_width = 0.2  # Width of zoom window
    
    zoom_indices = findall(x -> zoom_center-zoom_width <= x <= zoom_center+zoom_width, x)
    
    ax2 = Axis(fig[1, 2], xlabel="x", ylabel="u", title="Zoomed View of Steep Gradient")
    lines!(ax2, x[zoom_indices], u_upwind1[zoom_indices], label="1st-order Upwind", linewidth=2)
    lines!(ax2, x[zoom_indices], u_upwind2[zoom_indices], label="2nd-order Upwind", linewidth=2, color=:red)
    lines!(ax2, x[zoom_indices], u_fd[zoom_indices], label="Finite Difference", linestyle=:dash)
    lines!(ax2, x[zoom_indices], u_ps[zoom_indices], label="Pseudospectral", linestyle=:dot)
    
    # Plot λ - full view
    ax3 = Axis(fig[2, 1], xlabel="x", ylabel="λ", title="Solution Comparison: λ")
    lines!(ax3, x, λ_upwind1, label="1st-order Upwind", linewidth=2)
    lines!(ax3, x, λ_upwind2, label="2nd-order Upwind", linewidth=2, color=:red)
    lines!(ax3, x, λ_fd, label="Finite Difference", linestyle=:dash)
    lines!(ax3, x, λ_ps, label="Pseudospectral", linestyle=:dot)
    axislegend(ax3, position=:lt)
    
    # Plot λ - zoomed to highlight differences
    deriv = diff(λ_ps)
    _, max_idx = findmax(abs.(deriv))
    zoom_center = x[max_idx]
    
    zoom_indices = findall(x -> zoom_center-zoom_width <= x <= zoom_center+zoom_width, x)
    
    ax4 = Axis(fig[2, 2], xlabel="x", ylabel="λ", title="Zoomed View of Steep Gradient")
    lines!(ax4, x[zoom_indices], λ_upwind1[zoom_indices], label="1st-order Upwind", linewidth=2)
    lines!(ax4, x[zoom_indices], λ_upwind2[zoom_indices], label="2nd-order Upwind", linewidth=2, color=:red)
    lines!(ax4, x[zoom_indices], λ_fd[zoom_indices], label="Finite Difference", linestyle=:dash)
    lines!(ax4, x[zoom_indices], λ_ps[zoom_indices], label="Pseudospectral", linestyle=:dot)
    
    return fig
end

# Generate the comparison plots
fig = plot_comparison(upwind1_prob, upwind2_prob, fd_prob, ps_prob)

# Save the figure
save("upwind_comparison.png", fig)

# You can also visualize the full space-time solution
plot_solution(upwind1_prob)
plot_solution(upwind2_prob) 