using POMDPs
using POMDPTools
using Crux
using Flux
1
mdp = convert(MDP, RDEEnv())

as = POMDPs.actions(mdp)
S = ContinuousSpace((mdp.env.prob.params.N*2,))
layer_size = 32
A() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., layer_size, relu),
                            Dense(layer_size, layer_size, relu),
                            Dense(layer_size, length(as))), as)
V() = ContinuousNetwork(Chain(Dense(Crux.dim(S)..., layer_size, relu),
                              Dense(layer_size, layer_size, relu),
                              Dense(layer_size, 1)))



𝒮_ppo = PPO(π=ActorCritic(A(), V()), S=S, N=5000, ΔN=200, max_steps=10000)
@time π_ppo = POMDPs.solve(𝒮_ppo, mdp)



p = plot_learning(𝒮_ppo)

PolRunData = run_policy(π_ppo, mdp.env;tmax = 26.0)
animate_policy_data(PolRunData, mdp.env; fname="ppo")