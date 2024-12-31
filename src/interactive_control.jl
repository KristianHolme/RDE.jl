"""
    get_timestep_scale(val)

Calculate an appropriate scale for timestep adjustments based on the current value.

# Arguments
- `val`: Current value to scale

# Returns
- Scaled step size (0.01 * val)
"""
function get_timestep_scale(val)
    return 0.01*val
end

"""
    interactive_control(env::RDEEnv; callback=nothing)

Create an interactive visualization and control interface for an RDE simulation.

# Keywords
- `callback`: Optional function called after each action with the environment as argument

# Returns
- `Tuple{RDEEnv, Figure}`: The RDE environment and Makie figure

# Controls
## Keyboard Controls
- `→`: Step simulation forward
- `↑`: Increase timestep
- `↓`: Decrease timestep
- `w`: Increase s parameter
- `s`: Decrease s parameter
- `e`: Increase u_p parameter
- `d`: Decrease u_p parameter
- `r`: Reset simulation

## Interactive Elements
- Sliders for s, u_p, and timestep control
- Real-time plots:
  - Velocity field (u)
  - Reaction progress (λ)
  - Energy balance
  - Chamber pressure
- Time and parameter value displays

# Example
```julia
env, fig = interactive_control()
# Add custom callback
env, fig = interactive_control(callback=(env)->println("t = \$(env.t)"))
```

# Notes
- The visualization updates in real-time as the simulation progresses
- Parameter changes are applied smoothly using the current timestep as the smoothing time
- The energy balance and chamber pressure plots auto-scale as the simulation runs
"""

function interactive_control(env::RDEEnv;callback=nothing)
    params = env.prob.params
    N = params.N
    fig = Figure(size=(1000, 600))
    upper_area = fig[1,1] = GridLayout()
    plotting_area = fig[2,1] = GridLayout()
    energy_area = fig[3,1][1,1] = GridLayout()
    label_area  = fig[3,1][1,2] = GridLayout()
    obs_area = fig[4,1] = GridLayout()

    # Observables for real-time plotting
    u_data = Observable(env.state[1:N])
    λ_data = Observable(env.state[N+1:end])
    u_max = @lift(maximum($u_data))
    obs_data = Observable(observe(env))

    energy_bal_pts = Observable(Point2f[(env.t, energy_balance(env.state, params))])
    chamber_p_pts = Observable(Point2f[(env.t, chamber_pressure(env.state, params))])
    reward_pts = Observable(Point2f[(env.t, env.reward)])
    
    # Control parameters with smooth transitions
    control_s = Observable(params.s)
    s_start = params.s
    on(control_s) do Val
        env.prob.cache.s_current .= Val
    end
    control_u_p = Observable(params.u_p)
    u_p_start = params.u_p
    on(control_u_p) do val
        env.prob.cache.u_p_current .= val
    end
    time_step = Observable(env.dt)
    on(time_step) do val
        env.dt = val
        env.prob.cache.τ_smooth = val
    end

    # Interactive sliders
    slider_s = Slider(label_area[2,1], range = 0:0.001:env.smax, startvalue = control_s[])
    on(slider_s.value) do val
        control_s[] = val
    end
    slider_u_p = Slider(label_area[2,2], range = 0:0.001:env.u_pmax, startvalue = control_u_p[])
    on(slider_u_p.value) do val
        control_u_p[] = val
    end
    slider_dt = Slider(label_area[2,3], range = 0:0.001:1, startvalue = time_step[])
    on(slider_dt.value) do val
        time_step[] = val
        env.prob.cache.τ_smooth = val
    end

    time = Observable(env.t)
    
    """
    Update all visualization observables with current environment state.
    """
    function update_observables!()
        u_data[] = env.state[1:N]
        λ_data[] = env.state[N+1:end]
        time[] = env.t
        obs_data[] = observe(env)
    end

    # Create main visualization
    main_plotting(plotting_area, env.prob.x, u_data, λ_data, env.prob.params;
                u_max=u_max,s=control_s, u_p=control_u_p)

    # Create observation plot
    ax_obs = Axis(obs_area[1,1], title="Observation", xlabel="index", ylabel="value")
    n_obs = length(observe(env))
    barplot!(ax_obs, 1:n_obs, obs_data)
    on(obs_data) do obs
        ylims!(ax_obs, (min(-2.0, minimum(obs)), max(2.0, maximum(obs))))
    end

    # Energy balance plot with auto-scaling
    eb_start_xmax = 0.5
    ax_eb = Axis(energy_area[1,1], title="Energy balance", xlabel="t", ylabel=L"Ė", limits=(0,eb_start_xmax, nothing, nothing))
    lines!(ax_eb, energy_bal_pts)
    on(energy_bal_pts) do _
        y_vals = getindex.(energy_bal_pts[], 2)
        ylims!(ax_eb, (min(0, minimum(y_vals)), maximum(y_vals)))
        xlims!(ax_eb, (0, max(eb_start_xmax, energy_bal_pts[][end][1])))
    end

    # Chamber pressure plot with auto-scaling
    cp_start_xmax = 0.5
    ax_cp = Axis(energy_area[1,2], title="Chamber Pressure", xlabel="t", ylabel=L"P_c", limits=(0,cp_start_xmax, nothing, nothing))
    lines!(ax_cp, chamber_p_pts)
    on(chamber_p_pts) do _
        y_vals = getindex.(chamber_p_pts[], 2)
        ylims!(ax_cp, (min(0, minimum(y_vals)), maximum(y_vals)))
        xlims!(ax_cp, (0, max(cp_start_xmax, chamber_p_pts[][end][1])))
    end

    # Reward plot with auto-scaling
    ax_reward = Axis(energy_area[1,3], title="Reward", xlabel="t", ylabel="r", limits=(0,cp_start_xmax, nothing, nothing))
    lines!(ax_reward, reward_pts)
    on(reward_pts) do _
        y_vals = getindex.(reward_pts[], 2)
        ylims!(ax_reward, (min(-2.0, minimum(y_vals)), max(1.15, maximum(y_vals))))
        @debug "reward_pts[][end][1] = $(reward_pts[][end][1])"
        xlims!(ax_reward, (0, max(0.1, reward_pts[][end][1]*1.15)))
    end

    rowsize!(fig.layout, 3, Auto(0.3))

    # Keyboard control setup
    pressed_keys = Set{Keyboard.Button}()
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press
            push!(pressed_keys, event.key)
        elseif event.action == Keyboard.release
            delete!(pressed_keys, event.key)
        end
    end

    """
    Background task that continuously checks for pressed keys and executes corresponding actions.
    """
    function key_action_loop()
        while events(fig.scene).window_open[]
            if events(fig.scene).hasfocus[]
                for key in pressed_keys
                    if key == Keyboard.up
                        # Increase time step on arrow up
                        time_step[] += get_timestep_scale(time_step.val)
                        set_close_to!(slider_dt, time_step[])
                    elseif key == Keyboard.down
                        # Decrease time step on arrow down
                        time_step[] -= get_timestep_scale(time_step.val)
                        set_close_to!(slider_dt, time_step[])
                    elseif key == Keyboard.right
                        try
                            act!(env, [0.0]) #cached values are already set
                            energy_bal_pts[] =  push!(energy_bal_pts[], Point2f(env.t, energy_balance(env.state, params)))
                            chamber_p_pts[] = push!(chamber_p_pts[], Point2f(env.t, chamber_pressure(env.state, params)))
                            reward_pts[] = push!(reward_pts[], Point2f(env.t, env.reward))
                            update_observables!()
                            if callback !== nothing
                                @debug "calling callback"
                                callback(env)
                            end
                        catch e
                            @error "error taking action e=$e"
                            rethrow(e)
                        end
                    elseif key == Keyboard.w
                        change = get_timestep_scale(control_s.val)
                        control_s[] += change
                        set_close_to!(slider_s, control_s[])
                    elseif key == Keyboard.s
                        control_s[] -= get_timestep_scale(control_s.val)
                        set_close_to!(slider_s, control_s[])
                    elseif key == Keyboard.e
                        control_u_p[] += get_timestep_scale(control_u_p.val)
                        set_close_to!(slider_u_p, control_u_p[])
                    elseif key == Keyboard.d
                        control_u_p[] -= get_timestep_scale(control_u_p.val)
                        set_close_to!(slider_u_p, control_u_p[])
                    elseif key == Keyboard.r
                        reset!(env)
                        update_observables!()
                        control_s[] = s_start
                        control_u_p[] = u_p_start
                        set_close_to!(slider_s, control_s[])
                        set_close_to!(slider_u_p, control_u_p[])
                        energy_bal_pts[] = Point2f[(env.t, energy_balance(env.state, params))]
                        chamber_p_pts[] = Point2f[(env.t, chamber_pressure(env.state, params))]
                        reward_pts[] = Point2f[(env.t, env.reward)]
                    end
                end
            end
            sleep(0.1)  # Control loop rate
        end
    end

    # Labels for time and parameters
    label = Label(upper_area[1,1], text=@lift("Time: $(round($time, digits=2))"), tellwidth=false)
    s_label = Label(label_area[1,1], text=@lift("s = $(round($control_s, digits=3))"), tellwidth=false, tellheight = false)
    u_p_label = Label(label_area[1,2], text=@lift("u_p = $(round($control_u_p, digits=3))"), tellwidth=false, tellheight = false)
    dt_label = Label(label_area[1,3], text=@lift("Δt = $($time_step)"), tellwidth=false, tellheight = false)

    display(fig)
    @async key_action_loop()
    return env, fig
end
