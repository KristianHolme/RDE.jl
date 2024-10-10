
function get_timestep_scale(val)
    step = 10^(floor(log10(val/10)))
    return step
end
function interactive_RDE_control(;kwargs...)
    env = RDEEnv(;kwargs...)
    params = env.prob.params
    N = params.N
    fig = Figure(size=(1000, 600))
    upper_area = fig[1,1] = GridLayout()
    plotting_area = fig[2,1] = GridLayout()
    energy_area = fig[3,1][1,1] = GridLayout()
    label_area  = fig[3,1][1,2] = GridLayout()

    u_data = Observable(env.state[1:N])
    λ_data = Observable(env.state[N+1:end])
    u_max = @lift(maximum($u_data))

    energy_bal_pts = Observable(Point2f[(env.t, energy_balance(env.state, params))])
    
    
    # Variables to be controlled by keys
    control_s = Observable(params.s/env.smax)
    s_start = params.s
    on(control_s) do Val
        params.s = Val
    end
    control_u_p = Observable(params.u_p/env.u_pmax)
    u_p_start = params.u_p
    on(control_u_p) do val
        params.u_p = val
    end
    time_step = Observable(env.dt)
    on(time_step) do val
        env.dt = val
    end
    time = Observable(env.t)
    function update_observables!()
        u_data[] = env.state[1:N]
        λ_data[] = env.state[N+1:end]
        time[] = env.t
    end

    main_plotting(plotting_area, env.prob.x, u_data, λ_data, env.prob.params;
                u_max=u_max,s=control_s, u_p=control_u_p)

    #plot energy_balance
    eb_start_xmax = 0.5
    ax_eb = Axis(energy_area[1,1], title="Energy balance", xlabel="t", ylabel=L"Ė", limits=(0,eb_start_xmax, nothing, nothing))
    lines!(ax_eb, energy_bal_pts)
    on(energy_bal_pts) do _
        y_vals = getindex.(energy_bal_pts[], 2)
        ylims!(ax_eb, (min(0, minimum(y_vals)), maximum(y_vals)))
        xlims!(ax_eb, (0, max(eb_start_xmax, energy_bal_pts[][end][1])))
    end
    rowsize!(fig.layout, 3, Auto(0.3))

    # Create a set to keep track of currently pressed keys
    pressed_keys = Set{Keyboard.Button}()

    # Function to handle key press and release events
    on(events(fig).keyboardbutton) do event
        if event.action == Keyboard.press
            push!(pressed_keys, event.key)
        elseif event.action == Keyboard.release
            delete!(pressed_keys, event.key)
        end
    end

    # Function to perform actions based on pressed keys
    function key_action_loop()
        # @info "entered loop"
        @show(events(fig.scene).window_open[])
        while events(fig.scene).window_open[]
            # @show events(fig.scene).hasfocus[]
            if events(fig.scene).hasfocus[]
                # @info "fig is focused"
                for key in pressed_keys
                    if key == Keyboard.up
                        # Increase time step on arrow up
                        time_step[] += get_timestep_scale(time_step.val)

                    elseif key == Keyboard.down
                        # Decrease time step on arrow down
                        time_step[] -= get_timestep_scale(time_step.val)

                    elseif key == Keyboard.left
                        println("Left arrow held down")
                        # Define your action for the left arrow here

                    elseif key == Keyboard.right
                        act!(env, [control_s[], control_u_p[]])
                        energy_bal_pts[] =  push!(energy_bal_pts[], Point2f(env.t, energy_balance(env.state, params)))
                        update_observables!()

                    elseif key == Keyboard.w
                        control_s[] += get_timestep_scale(control_s.val)

                    elseif key == Keyboard.s
                        control_s[] -= get_timestep_scale(control_s.val)

                    elseif key == Keyboard.e
                        control_u_p[] += get_timestep_scale(control_u_p.val)

                    elseif key == Keyboard.d
                        control_u_p[] -= get_timestep_scale(control_u_p.val)

                    elseif key == Keyboard.r
                        reset!(env)
                        update_observables!()
                        control_s[] = s_start/env.smax
                        control_u_p[] = u_p_start/env.u_pmax
                        energy_bal_pts[] = Point2f[(env.t, energy_balance(env.state, params))]

                    end
                end
            end
            # @info "going to sleep"
            sleep(0.1)  # Adjust the repeat rate as needed
        end
    end

    


    # Time label
    label = Label(upper_area[1,1], text=@lift("Time: $(round($time, digits=2))"), tellwidth=false)
    #s label
    s_label = Label(label_area[1,1], text=@lift("s = $(round($control_s*env.smax, digits=2))"), tellwidth=false, tellheight = false)
    u_p_label = Label(label_area[1,2], text=@lift("u_p = $(round($control_u_p*env.u_pmax, digits=2))"), tellwidth=false, tellheight = false)
    dt_label = Label(label_area[1,3], text=@lift("Δt = $($time_step)"), tellwidth=false, tellheight = false)

    display(fig)
    # Start the key action loop as a background task
    @async key_action_loop()
end