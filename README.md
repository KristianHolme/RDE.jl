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
- **Flexible Parameter Control**: Easy modification of system parameters and initial conditions
- **Analysis Tools**: Built-in functions for energy balance and chamber pressure calculations

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

```julia
using RDE
using GLMakie

# Create and solve a basic RDE problem
params = RDEParam()
rde_prob = RDEProblem(params)
solve_pde!(rde_prob)
plot_solution(rde_prob)
```

### Custom Initial Conditions
```julia
# Initialize with specific number of shocks
u_init = get_n_shocks_init_func(3)  # 3 shocks
params = RDEParam(u_init=u_init)
prob = RDEProblem(params)
solve_pde!(prob)

# Or use random shock initialization
u_init = random_shock_init_func()
params = RDEParam(u_init=u_init)
```

### Analysis
```julia
# Calculate energy balance
energy = energy_balance(prob.sol.u, prob.params)

# Calculate chamber pressure
pressure = chamber_pressure(prob.sol.u, prob.params)

# Visualize results
using GLMakie

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel="t", ylabel="Energy Balance")
ax2 = Axis(fig[2, 1], xlabel="t", ylabel="Chamber Pressure")

lines!(ax1, prob.sol.t, energy)
lines!(ax2, prob.sol.t, pressure)

fig
```

For reinforcement learning applications with RDE systems or functionality for interactive control, please see the companion package [RDE_Env.jl](https://github.com/KristianHolme/RDE_Env.jl).

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
