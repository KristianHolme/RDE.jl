# RDE

[![Build Status](https://github.com/KristianHolme/RDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/RDE.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package provides a solver for the rotating detonation engine (RDE) model equations presented in [References](#references):

$u_{t}+ uu_{x} = (1-\lambda)\omega(u)q_0 + \nu_1 u_{xx} + \epsilon \xi (u, u_0)$

$\lambda_t = (1-\lambda)\omega(u) - \beta (u, u_p, s)\lambda + \nu_{2}\lambda_{xx}$.

The solver can use different methods for the spatial discretization. The default is a finite difference method, but a pseudospectral method is also available.

This package also provides an interface to the PDE solver using [CommonRLInterface.jl](https://github.com/JuliaReinforcementLearning/CommonRLInterface.jl).

# Examples

A [Stepwise control](examples/stepwise_control.jl) can be used to control the RDE, and induce bifurcations between different mode-locked states. Visualization of the simulation are mad using [Makie.jl](https://github.com/JuliaPlots/Makie.jl):
```julia
using RDE
using GLMakie

env = RDEEnv(dt=0.1f0);
π = StepwiseRDEPolicy(env, [20.0f0, 100.0f0, 200.0f0, 350.0f0], 
[[3.5f0, 0.64f0], [3.5f0, 0.86f0], [3.5f0, 0.64f0], [3.5f0, 0.94f0]]);
data = run_policy(π, env; tmax=500.0)
fig = plot_policy_data(env, data)

animate_policy_data(data, env; fname="stepwise_control", fps=60)
```


https://github.com/user-attachments/assets/154fcc8c-82f1-4158-95ff-5928c8c30e51


# References
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
