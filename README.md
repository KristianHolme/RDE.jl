# RDE

[![Build Status](https://github.com/KristianHolme/RDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/KristianHolme/RDE.jl/actions/workflows/CI.yml?query=branch%3Amain)

This package provides a solver for the rotating detonation engine (RDE) model equations presented in SOURCES:

$u_{t}+ uu_{x} = (1-\lambda)\omega(u)q_0 + \nu_1 u_{xx} + \epsilon \xi (u, u_0)$

$\lambda_t = (1-\lambda)\omega(u) - \beta (u, u_p, s)\lambda + \nu_{2}\lambda_{xx}$.

The solver uses a pseudospectral approach.

This package also provides an interface to the PDE solver using [CommonRLInterface.jl](https://github.com/JuliaReinforcementLearning/CommonRLInterface.jl).


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