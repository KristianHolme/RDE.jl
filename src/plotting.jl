"""
    plot_solution(prob::RDEProblem; time_idx=Observable(1), player_controls=true)

Plot the solution of an RDE problem with interactive visualization.

# Arguments
- `prob::RDEProblem`: The RDE problem containing solution data

# Keywords
- `time_idx::Observable{Int}=Observable(1)`: Observable for current time index
- `player_controls::Bool=true`: Whether to include playback controls

# Returns
- `Figure`: Makie figure containing the visualization

The plot includes:
- Velocity field (u) and reaction progress (λ) in both linear and circular representations
- Energy balance over time
- Chamber pressure over time
- Interactive time controls (if `player_controls=true`)
"""
function plot_solution(prob::RDEProblem; 
        time_idx = Observable(1),
        player_controls = true)
    params = prob.params
    sol = prob.sol
    x = prob.x
    N = params.N
    num_times = length(sol.t)
    t_values = sol.t
    u_max = Observable(get_u_max(sol.u))    

    # Create Observables for dynamic data
    u_data = @lift(sol.u[$time_idx][1:N])
    λ_data = @lift(sol.u[$time_idx][N+1:end])

    fig = Figure(size=(1000, 600))
    upper_area = fig[1,1] = GridLayout()
    plotting_area = fig[2,1] = GridLayout()
    metrics_area = fig[3,1][1,1] = GridLayout()
    

    main_plotting(plotting_area, x, u_data, λ_data, params;u_max = u_max)

    # Add energy_balance_plot
    energy_bal = energy_balance(sol.u, params)
    ax_eb = Axis(metrics_area[1,1], title="Energy balance", xlabel="t", ylabel="Ė")
    lines!(ax_eb, sol.t, energy_bal)
    vlines!(ax_eb, @lift(sol.t[$time_idx]), color=:green, alpha=0.5)

    # Add chamber pressure
    chamber_p = chamber_pressure(sol.u, params)
    ax_cp = Axis(metrics_area[1,2], title="Chamber pressure", xlabel="t", ylabel="̄u²")
    lines!(ax_cp, sol.t, chamber_p)
    vlines!(ax_cp, @lift(sol.t[$time_idx]), color=:green, alpha=0.5)

    # Time label
    label = Label(upper_area[1,1], text=@lift("Time: $(round(t_values[$time_idx], digits=2))"), tellwidth=false)

    if player_controls
        play_ctrl_area = fig[3,1][1,2] = GridLayout()
        colsize!(play_ctrl_area, 1, Auto(2.0))
        plot_controls(play_ctrl_area, time_idx, num_times)
    end
    Makie.trim!(fig.layout)
    
    display(fig)
    fig
end

"""
    get_u_max(us)

Calculate the maximum velocity across all time steps.
"""
function get_u_max(us)
    full_mat = stack(us)
    N = Int(length(us[1]) / 2)
    u_max = maximum(full_mat[1:N, :])
    return u_max
end

"""
    plot_controls(play_ctrl_area::GridLayout, time_idx::Observable, num_times::Int)

Add interactive playback controls to a figure.

# Arguments
- `play_ctrl_area::GridLayout`: Layout area for controls
- `time_idx::Observable{Int}`: Observable for current time index
- `num_times::Int`: Total number of time steps

Features:
- Play/Pause button
- Step forward/backward buttons
- Jump to start/end buttons
- Time slider
- Animation speed control
- Loop control
"""
function plot_controls(play_ctrl_area::GridLayout, time_idx::Observable, num_times::Int)
    # Slider
    sld = Slider(play_ctrl_area[1,1], range=1:num_times, startvalue=1)
    on(sld.value) do val
        time_idx[] = Int(round(val))
    end

    # Button area
    button_area = play_ctrl_area[2,1] = GridLayout()

    # step to start button
    start_button = Button(button_area[1, 1], label="<<", tellwidth=false)

    on(start_button.clicks) do _
        time_idx[] = 1
        set_close_to!(sld, time_idx[])
    end
    # step back button
    prev_button = Button(button_area[1, 2], label="<", tellwidth=false)

    on(prev_button.clicks) do _
        if time_idx[] > 1
            time_idx[] -= 1
        end
        set_close_to!(sld, time_idx[])
    end
    

     # step forward button button
     next_button = Button(button_area[1, 3], label=">", tellwidth=false)

     on(next_button.clicks) do _
         if time_idx[] < num_times
             time_idx[] += 1
         end
         set_close_to!(sld, time_idx[])
     end
     # step to end button
    end_button = Button(button_area[1, 4], label=">>", tellwidth=false)

    on(end_button.clicks) do _
        time_idx[] = num_times
        set_close_to!(sld, time_idx[])
    end

    # Play/Pause button
    play = Button(button_area[1, 5], label="Play", tellwidth=false)
    playing = Observable(false)

    on(play.clicks) do _
        playing[] = !playing[]
        if playing[]
            play.label[] = "Pause"
        else
            play.label[] = "Play"
        end
    end

    #Loop button
    loop = Button(button_area[1, 6], label="Loop", tellwidth=false)
    looping = Observable(true)

    on(loop.clicks) do _
        looping[] = !looping[]
        if looping[]
            loop.label[] = "Loop"
        else
            loop.label[] = "Once"
        end
    end
   

    # Animation speed Slider
    anim_speed = Observable(1.0)
    anim_sld = Slider(play_ctrl_area[2,2], range=0.1:0.1:5.0, startvalue=1.0)
    on(anim_sld.value) do val
        anim_speed[] = val
    end
    
    #Animation speed label
    anim_speed_label = Label(play_ctrl_area[1,2], text=@lift("Speed: $(round($anim_speed, digits=2))"), tellwidth=false)


    # Animation loop
    @async begin
        while true
            if playing[]
                time_idx[] = min(time_idx[] + 1, num_times)
                if time_idx[] == num_times
                    if !looping[]
                        playing[] = false
                        play.label[] = "Play"
                    else
                        time_idx[] = 1
                    end
                end
                set_close_to!(sld, time_idx[])
            end
            sleep(0.05/anim_speed[])
        end
    end
end

"""
    plot_subfunctions(layout::GridLayout, x, u_data::Observable, params; kwargs...)

Plot auxiliary functions ω(u), ξ(u), and β(u) in a separate layout.

# Arguments
- `layout::GridLayout`: Layout area for plots
- `x`: Spatial grid points
- `u_data::Observable`: Observable containing velocity field
- `params`: RDE parameters

# Keywords
- `u_max=Observable(10.0)`: Maximum velocity for scaling
- `s=Observable(params.s)`: Parameter s in β function
- `u_p=Observable(params.u_p)`: Parameter u_p in β function
"""
function plot_subfunctions(layout::GridLayout, x, u_data::Observable, 
                           params;
                           u_max=Observable(10.0),
                           s = Observable(params.s),
                           u_p = Observable(params.u_p))

    hist_β_max = Observable(maximum(β.(u_data[], s[], u_p[], params.k_param)))
    hist_ω_max = Observable(maximum(ω.(u_data[], params.u_c, params.α)))

    #Axes for ω, ξ, and β
    ω_max = @lift(maximum(ω.($u_data, params.u_c, params.α)))
    ξ_max = @lift(max(1e-3, ξ($u_max, params.u_0, params.n)).*1.05)
    β_max = @lift(maximum(β.($u_data, $s, $u_p, params.k_param)))

    on(β_max) do val
        hist_β_max[] = max(hist_β_max[], val)
    end

    on(ω_max) do val
        hist_ω_max[] = max(hist_ω_max[], val)
    end

    ax_ω = Axis(layout[1,1], xticksvisible=false, xlabelvisible = false,
                xticklabelsvisible = false, ylabel="ω(u)",
                limits=@lift((nothing, (0, max($hist_ω_max*1.05,1e-3)))))
    ax_ξ = Axis(layout[2,1], xticksvisible=false, xlabelvisible = false,
                xticklabelsvisible = false, ylabel="ξ(u)",
                limits=@lift((nothing,(nothing,$ξ_max.*1.05))))
    ax_β = Axis(layout[3,1], xlabel="x", ylabel="β(u)", 
                limits=@lift((nothing, (0.0,max($hist_β_max*1.05, 1e-3)))))

    ω_data = @lift(ω.($u_data, params.u_c, params.α))
    ξ_data = @lift(ξ.($u_data, params.u_0, params.n))
    β_data = @lift(β.($u_data, $s, $u_p, params.k_param))

    lines!(ax_ω, x, ω_data)
    lines!(ax_ξ, x, ξ_data)
    lines!(ax_β, x, β_data)

    return [ax_ω, ax_ξ, ax_β]
end

"""
    main_plotting(layout::GridLayout, x, u_data::Observable, λ_data::Observable, params; kwargs...)

Create main plots for velocity and reaction progress fields.

# Arguments
- `layout::GridLayout`: Layout area for plots
- `x`: Spatial grid points
- `u_data::Observable`: Observable containing velocity field
- `λ_data::Observable`: Observable containing reaction progress
- `params`: RDE parameters

# Keywords
- `u_max=Observable(10.0)`: Maximum velocity for scaling
- `s=Observable(params.s)`: Parameter s in β function
- `u_p=Observable(params.u_p)`: Parameter u_p in β function
- `show_mouse_vlines::Bool=true`: Whether to show vertical lines at mouse position
- `include_subfunctions::Bool=false`: Whether to include auxiliary function plots
"""
function main_plotting(layout::GridLayout, x, u_data::Observable, 
        λ_data::Observable,
        params; 
        u_max=Observable(10.0),
        s = Observable(params.s),
        u_p = Observable(params.u_p),
        show_mouse_vlines = true,
        include_subfunctions = false)
    
    if include_subfunctions
        plotting_area_subfuncs = layout[1, 1] = GridLayout()
        plotting_area_main = layout[1, 2] = GridLayout()
        colsize!(layout, 1, Auto(0.3))
        subfuncs_axes = plot_subfunctions(plotting_area_subfuncs, x, u_data, params; u_max, s, u_p)
    else
        plotting_area_main = layout[1, 1] = GridLayout()  
        subfuncs_axes = []
    end
    
    # hist_β_max = Observable(maximum(β.(u_data[], s[], u_p[], params.k_param)))
    # hist_ω_max = Observable(maximum(ω.(u_data[], params.u_c, params.α)))

    # #Axes for ω, ξ, and β
    # ω_max = @lift( maximum(ω.($u_data, params.u_c, params.α)) )
    # ξ_max = @lift(max(1e-3, ξ($u_max, params.u_0, params.n)).*1.05)
    # β_max = @lift(maximum(β.($u_data, $s, $u_p, params.k_param)))

    # on(β_max) do val
    #     hist_β_max[] = max(hist_β_max[], val)
    # end

    # on(ω_max) do val
    #     hist_ω_max[] = max(hist_ω_max[], val)
    # end
    # ax_ω = Axis(plotting_area_subfuncs[1,1], xticksvisible=false, xlabelvisible = false,
    #             xticklabelsvisible = false, ylabel="ω(u)",
    #             limits=@lift((nothing, (0, max($hist_ω_max*1.05,1e-3)))))
    # ax_ξ = Axis(plotting_area_subfuncs[2,1], xticksvisible=false, xlabelvisible = false,
    #             xticklabelsvisible = false, ylabel="ξ(u)",
    #             limits=@lift((nothing,(nothing,$ξ_max.*1.05))))
    # ax_β = Axis(plotting_area_subfuncs[3,1], xlabel="x", ylabel="β(u)", limits=@lift((nothing, (0.0,max($hist_β_max*1.05, 1e-3)))))

    

    #Plotting u and λ
    ax_u = Axis(plotting_area_main[1,1], xlabel="x", ylabel="u(x, t)", title="u(x, t)", limits=@lift((nothing, (0.0,max($u_max*1.05, 1e-3)))))
    ax_λ = Axis(plotting_area_main[2,1], xlabel="x", ylabel="λ(x, t)", title="λ(x, t)", limits=(nothing, (-0.05,1.05)))
    ax_u_circ = Axis3(plotting_area_main[1,2], limits=@lift((nothing, nothing, (0,max($u_max*1.05, 1e-3)))), protrusions=0)
    ax_λ_circ = Axis3(plotting_area_main[2,2], limits=(nothing, nothing, (-0.05,1.05)), protrusions=0)
    
    circle_indices = [1:params.N;1]
    
    u_data_circle = @lift($u_data[circle_indices])
    λ_data_circle = @lift($λ_data[circle_indices])

    # ω_data = @lift(ω.($u_data, params.u_c, params.α))
    # ξ_data = @lift(ξ.($u_data, params.u_0, params.n))
    # β_data = @lift(β.($u_data, $s, $u_p, params.k_param))

    # Plot lines
    # lines!(ax_ω, x, ω_data)
    # lines!(ax_ξ, x, ξ_data)
    # lines!(ax_β, x, β_data)
    lines!(ax_u, x, u_data, color=:blue)
    lines!(ax_λ, x, λ_data, color=:red)

    #plot circles
    L = params.L
    r = L/(2π)
    xs_circle = r .* cos.(x[circle_indices] .* 2π/L)
    ys_circle = r .* sin.(x[circle_indices] .* 2π/L)
    lines!(ax_u_circ, xs_circle, ys_circle, u_data_circle, color=:blue)
    lines!(ax_λ_circ, xs_circle, ys_circle, λ_data_circle, color=:red)
    hidedecorations!(ax_u_circ, grid = false)
    hidedecorations!(ax_λ_circ, grid = false)

    # Hovering location Observable
    if show_mouse_vlines
        x_cursor = Observable(0.0)
        # Add vertical lines to each axis, linked to the x_cursor observable
        linear_axes = [subfuncs_axes; ax_u; ax_λ]
        for ax in linear_axes
            vlines!(ax, x_cursor, color = :black, linestyle = :dash)
        end
        # @show layout
        # Function to update the x_cursor when the mouse is over any axis
        on(events(layout.parent.parent.scene).mouseposition) do position
            for ax in linear_axes
                if is_mouseinside(ax.scene)
                    position = mouseposition(ax.scene)
                    # Transform screen coordinates to data coordinates
                    x_cursor[] = position[1]
                end
            end
        end
    end
end

"""
    plot_policy(π::P, env::RDEEnv) where P <: Policy

Plot the results of running a policy in an RDE environment.

# Arguments
- `π::P`: Policy to evaluate
- `env::RDEEnv`: RDE environment

# Returns
- `Figure`: Makie figure containing the visualization
"""
function plot_policy(π::P, env::RDEEnv) where P <: Policy
    data = run_policy(π, env)
    plot_policy_data(env, data)
end

"""
    plot_policy_data(env::RDEEnv, data::PolicyRunData; kwargs...)

Create an interactive visualization of policy execution data.

# Arguments
- `env::RDEEnv`: RDE environment
- `data::PolicyRunData`: Data from policy execution

# Keywords
- `time_idx::Observable{Int}=Observable(1)`: Observable for current time index
- `player_controls::Bool=true`: Whether to include playback controls

# Returns
- `Figure`: Makie figure containing the visualization

The plot includes:
- Velocity and reaction progress fields
- Energy balance
- Chamber pressure
- Rewards
- Number of shocks
- Control parameters (s and u_p)
- Interactive time controls (if enabled)
"""
function plot_policy_data(env::RDEEnv, data::PolicyRunData; 
        time_idx = Observable(1),
        player_controls=true,
        kwargs...)
    action_ts = data.action_ts
    ss = data.ss
    u_ps = data.u_ps
    energy_bal = data.energy_bal
    chamber_p = data.chamber_p
    rewards = data.rewards

    state_ts = data.state_ts
    states = data.states

    N = env.prob.params.N


    function sparse_to_dense_ind(dense_time::Vector, sparse_time::Vector, dense_ind::Int)
        dense_val = dense_time[dense_ind]
        sparse_ind = argmin(abs.(sparse_time .- dense_val))
        # @info sparse_ind dense_ind sp_val "$(dense_time[dense_ind])"
        return sparse_ind
    end


    
    u_data = @lift(states[$time_idx][1:N])
    λ_data = @lift(states[$time_idx][N+1:end])
    # @info sparse_to_dense_ind(ts, sparse_ts, 3)
    s = @lift(ss[sparse_to_dense_ind(state_ts, action_ts, $time_idx)])
    u_p = @lift(u_ps[sparse_to_dense_ind(state_ts, action_ts, $time_idx)])
    
    fig = Figure(size=(1000,900))
    upper_area = fig[1,1] = GridLayout()
    main_layout = fig[2,1] = GridLayout()
    metrics_action_area = fig[3,1] = GridLayout()
    

    rowsize!(fig.layout, 3, Auto(0.5))


    label = Label(upper_area[1,1], text=@lift("Time: $(round(state_ts[$time_idx], digits=2))"), tellwidth=false)

    main_plotting(main_layout, env.prob.x, u_data, λ_data, 
                env.prob.params;
                # u_max = Observable(maximum([maximum(spst) for spst in sparse_states])),
                u_max = Observable(3),
                s = s,
                u_p = u_p,
                include_subfunctions = false,
                kwargs...)


    fine_time = @lift(state_ts[$time_idx])
    
    ax_eb = Axis(metrics_action_area[1,1], title="Energy balance", ylabel="Ė")
    hidexdecorations!(ax_eb, grid = false)
    lines!(ax_eb, state_ts, energy_bal)
    vlines!(ax_eb, fine_time, color=:green, alpha=0.5)

    # Add chamber pressure
    ax_cp = Axis(metrics_action_area[2,1], title="Chamber pressure", xlabel="t", ylabel="̄u²")
    lines!(ax_cp, state_ts, chamber_p)
    vlines!(ax_cp, fine_time, color=:green, alpha=0.5)

    # Add rewards and shocks
    ax_rewards = Axis(metrics_action_area[1,2], title="Rewards", ylabel="r")
    hidexdecorations!(ax_rewards, grid = false)
    lines!(ax_rewards, action_ts, rewards)
    vlines!(ax_rewards, fine_time, color=:green, alpha=0.5)

    ax_shocks = Axis(metrics_action_area[2,2], title="Shocks", xlabel="t")
    dx = env.prob.x[2] - env.prob.x[1]
    us, = split_sol(states)
    lines!(ax_shocks, state_ts, count_shocks.(us, dx))
    vlines!(ax_shocks, fine_time, color=:green, alpha=0.5)

    ax_s = Axis(metrics_action_area[1:2,3], xlabel="t", ylabel="s", yticklabelcolor=:forestgreen)
    ax_u_p = Axis(metrics_action_area[1:2,3], ylabel="u_p", yaxisposition = :right, yticklabelcolor=:royalblue)
    hidespines!(ax_u_p)
    hidexdecorations!(ax_u_p)
    hideydecorations!(ax_u_p, ticklabels=false, ticks=false, label=false)
    lines!(ax_s, action_ts, ss, color=:forestgreen)
    lines!(ax_u_p, action_ts, u_ps, color=:royalblue)
    #Time indicator
    vlines!(ax_s, fine_time, color=:green, alpha=0.5)

    # @show length(sparse_ts)
    if player_controls
        play_ctrl_area = fig[4,1] = GridLayout()
        plot_controls(play_ctrl_area, time_idx, length(state_ts))
    end

    fig
end

"""
    plot_shifted_history(us::AbstractArray, x::AbstractArray, ts::AbstractArray, c::Union{Real, AbstractArray}; u_ps=nothing)

Create a space-time plot of the solution in a moving reference frame.

# Arguments
- `us::AbstractArray`: Array of velocity fields
- `x::AbstractArray`: Spatial grid points
- `ts::AbstractArray`: Time points
- `c::Union{Real, AbstractArray}`: Frame velocity (scalar or array)

# Keywords
- `u_ps=nothing`: Optional array of pressure values to plot

# Returns
- `Figure`: Makie figure containing:
  - Heatmap of velocity field in moving frame
  - Number of shocks over time
  - Pressure values over time (if provided)
"""
function plot_shifted_history(us::AbstractArray, x::AbstractArray,
         ts::AbstractArray, c::Union{Real, AbstractArray}; u_ps=nothing, rewards=nothing)
    shifted_us = shift_inds(us, x, ts, c)

    fig = Figure(size=(1800, 600))
    ax = Axis(fig[1,1], title="u(x+ct, t)", xlabel="t",
            ylabel="x", yzoomlock=true, ypanlock=true,
            limits=(extrema(ts), extrema(x)), xautolimitmargin=(0.0, 0.0))
    hm = heatmap!(ax, ts, x, stack(shifted_us)', colorscale=identity)
    Colorbar(fig[1,2], hm)
    
    counts = count_shocks.(us, x[2] - x[1])
    ax2 = Axis(fig[2,1], xlabel="t", ylabel="Number of shocks", 
                limits=(nothing, (-0.05, maximum(counts)*1.05)),
                xautolimitmargin=(0.0, 0.0))
    lines!(ax2, ts, counts)
    linkxaxes!(ax, ax2)

    if u_ps !== nothing
        ax3 = Axis(fig[end+1,1], xlabel="t", ylabel="u_p", 
                    limits=(nothing, (minimum(u_ps)-0.05, maximum(u_ps)*1.05)),
                    xautolimitmargin=(0.0, 0.0))
        lines!(ax3, ts, u_ps)
        linkxaxes!(ax, ax3)
    end
    if rewards !== nothing
        ax4 = Axis(fig[end+1,1], xlabel="t", ylabel="Reward", 
                    limits=(nothing, (minimum(rewards)-0.05, maximum(rewards)*1.05)),
                    xautolimitmargin=(0.0, 0.0))
        lines!(ax4, ts, rewards)
        linkxaxes!(ax, ax4)
    end
    autolimits!(ax) 
    fig
end