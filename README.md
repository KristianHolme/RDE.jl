# RDE.jl

[![Build Status](https://github.com/KristianHolme/RDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/RDE.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Overview

RDE.jl provides a solver for the rotating detonation engine (RDE) model equations presented in [Koch et al. (2020)](#references):

```math
u_{t}+ uu_{x} = (1-\lambda)\omega(u)q_0 + \nu_1 u_{xx} + \epsilon \xi (u, u_0)
```
```math
\lambda_t = (1-\lambda)\omega(u) - \beta (u, u_p, s)\lambda + \nu_{2}\lambda_{xx}
```

## Features

- **Multiple Discretization Methods**: Supports both finite difference and pseudospectral methods for spatial discretization
- **Reinforcement Learning Interface**: Integration with [CommonRLInterface.jl](https://github.com/JuliaReinforcementLearning/CommonRLInterface.jl)
  - Various observation strategies (direct state, Fourier-based)
  - Flexible action spaces (scalar pressure, stepwise control)
  - Customizable reward functions
  - Interactive control capabilities

## Installation

You can install RDE.jl using Julia's built-in package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add https://github.com/KristianHolme/RDE.jl
```

Or, you can use the Pkg API from the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/KristianHolme/RDE.jl")
```

## Examples

### Basic Usage

### PDE Solving
```julia
using RDE
using GLMakie

# Create and solve a basic RDE problem
params = RDEParam()
rde_prob = RDEProblem(params)
solve_pde!(rde_prob)
plot_solution(rde_prob)
```

### Stepwise Control
```julia
# Initialize environment with parameters
env = RDEEnv(RDEParam(tmax=500.0), dt=20.0f0)

# Create stepwise policy
π = StepwiseRDEPolicy(env, 
    [20.0f0, 100.0f0, 200.0f0, 350.0f0],  # Time points
    [[3.5f0, 0.64f0],                      # Control values
     [3.5f0, 0.86f0], 
     [3.5f0, 0.64f0], 
     [3.5f0, 0.94f0]])

# Run simulation
data = run_policy(π, env)
fig = plot_policy_data(env, data)

# Create animation
animate_policy_data(data, env; fname="stepwise_control", fps=60)
```
https://github.com/user-attachments/assets/154fcc8c-82f1-4158-95ff-5928c8c30e51



### Interactive Control
```julia
using RDE, GLMakie

# Launch interactive control interface
env, fig = interactive_control(params=RDEParam())
```


## Deep Reinforcement Learning

The package provides extensive support for Deep Reinforcement Learning (DRL) through integration with multiple frameworks:

### Stable-Baselines3 Integration
```julia
using RDE
using RLBridge
using PyCall

# Create environment with specific parameters
env = RDEEnv(;
    dt=0.1,
    τ_smooth=0.01,
    params=RDEParam(tmax=100.0),
    observation_strategy=FourierObservation(16),  # Fourier-based observations
    action_type=ScalarPressureAction(),          # Control chamber pressure
    reward_type=ShockPreservingReward(target_shock_count=3)  # Maintain 3 shocks
)

# Convert to Gym environment for SB3
gym_env = convert_to_gym(env)

# Train with PPO
sb = pyimport("sbx") # stable_baselines jax
model = sb.PPO("MlpPolicy", gym_env, device="cpu")
model.learn(total_timesteps=1_000_000)

# Evaluate trained policy
policy = SBPolicy(env, model.policy)
data = run_policy(policy, env)
plot_policy_data(env, data)
```

### Vectorized Environments
For faster training, the package supports parallel environment execution:

```julia
# Create multiple environments
envs = [RDEEnv(dt=0.1, τ_smooth=0.01) for _ in 1:8]
vec_env = RDEVecEnv(envs)

# Convert to SB3 VecEnv
sb_vec_env = convert_to_vec_env[](vec_env)

# Train with vectorized environments
model = sb.PPO("MlpPolicy", sb_vec_env)
model.learn(total_timesteps=1_000_000)
```

### Customizable Components

#### Observation Strategies
- `StateObservation`: Direct state observations
- `FourierObservation`: Fourier coefficients of the state
- `ExperimentalObservation`: Custom observation space

#### Action Types
- `ScalarPressureAction`: Control chamber pressure
- `ScalarAreaScalarPressureAction`: Control both pressure and injectionarea

#### Reward Functions
- `ShockSpanReward`: Maximize shock wave spacing
- `ShockPreservingReward`: Maintain specific number of shocks
- `ExperimentalReward`: Customizable reward components

## References

```bibtex
@article{PhysRevE.101.013106,
  title = {Mode-locked rotating detonation waves: Experiments and a model equation},
  author = {Koch, James and Kurosaka, Mitsuru and Knowlen, Carl and Kutz, J. Nathan},
  journal = {Phys. Rev. E},
  volume = {101},
  issue = {1},
  pages = {013106},
  numpages = {11},
  year = {2020},
  month = {Jan},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.101.013106},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.101.013106}
}

@article{Koch_2021,
   title={Multiscale physics of rotating detonation waves: Autosolitons and modulational instabilities},
   volume={104},
   ISSN={2470-0053},
   url={http://dx.doi.org/10.1103/PhysRevE.104.024210},
   DOI={10.1103/physreve.104.024210},
   number={2},
   journal={Physical Review E},
   publisher={American Physical Society (APS)},
   author={Koch, James and Kurosaka, Mitsuru and Knowlen, Carl and Kutz, J. Nathan},
   year={2021},
   month=aug }
```
