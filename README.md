# RDE.jl

[![Build Status](https://github.com/KristianHolme/RDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/RDE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/JET.jl-enabled-blue)](https://github.com/aviatesk/JET.jl)

## Overview

RDE.jl provides a solver for the rotating detonation engine (RDE) model equations presented in [Koch et al. (2020)](#references):

```math
u_{t}+ uu_{x} = (1-\lambda)\omega(u)q_0 + \nu_1 u_{xx} + \epsilon \xi (u, u_0)
```
```math
\lambda_t = (1-\lambda)\omega(u) - \beta (u, u_p, s)\lambda + \nu_{2}\lambda_{xx}
```

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

### Solver options

The solver defaults to a conservative finite-volume discretization with an explicit SSPRK integrator and CFL-based steps. You can override the integrator and time stepping options:

```julia
solve_pde!(rde_prob; alg = OrdinaryDiffEq.SSPRK33(), adaptive = false)
```

### Custom Initial Conditions
```julia
reset_strategy = NShock(2) # Initialize with 2 shocks
prob = RDEProblem(params;reset_strategy)
solve_pde!(prob)

# Or use random shock initialization (1-4 shocks)
reset_strategy = RandomShock() 
params = RDEParam()
prob = RDEProblem(params;reset_strategy)

# or use a custom function for the u-variable
reset_strategy = CustomPressureReset() do x
  T = eltype(x)
  return abs(x)/T(π)
end
prob = RDEProblem(params;reset_strategy)
solve_pde!(prob)
plot_solution(prob)
```

## Artifacts

Some reset strategies and predictors rely on artifact data (shock profiles and speed model). These are initialized at package load time; if the artifacts are missing, the package will warn and the related features will throw a clear error when used.



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
