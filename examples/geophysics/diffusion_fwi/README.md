<!-- markdownlint-disable -->
# Diffusion model for full-waveform inversion (FWI)

## Problem overview 

Full Waveform Inversion (FWI) is a seismic imaging technique that reconstructs
subsurface properties, also called velocity model, by fitting the recorded
seismic waveform. It underpins a range of applications, including:

- Hydro-carbon exploration and production, where an accurate velocity model
  guides drilling decisions.
- CO₂ storage, ensuring the integrity of underground reservoirs used
  for carbon capture and sequestration.
- Global and regional seismology, helping characterise tectonic processes and
  earthquakes.
- Analogous elastic/acoustic imaging modalities such as medical ultrasound and
  non-destructive testing.

The present example is tailored to the elastic wave equation in the context of
hydro-carbon exploration, but the same framework can be applied to other wave
equations and applications. The propagation of elastic waves in the subsurface
is governed by the following PDE:

Let $\mathbf{x}(r)=\bigl[ V_\mathrm{P}(r),\,V_\mathrm{S}(r),\,\rho(r) \bigr]$ denote the spatially varying P-wave velocity, S-wave velocity and density at location $r=(z,x,y)$. We introduce the elastic wave operator $\mathcal{A}(\mathbf{x})$. For a seismic source $s$ with force term $f_s(t,r)$, the forward problem reads:

$$
\mathcal{A}(\mathbf{x})\,u_s 
:= \rho(r)\,\frac{\partial^{2} u_s}{\partial t^{2}}
- \rho(r) V_\mathrm{S}^{2}(r)\,\nabla \times (\nabla \times u_s)
- \rho(r) V_\mathrm{P}^{2}(r)\,\nabla (\nabla \cdot u_s)
$$

The forward elastic wave equation for a point source located at $r_s$ with time history $\dot{S}(t)$ is

$$
\mathcal{A}(\mathbf{x})\,u_s(t,r) = \dot{S}(t)\,\delta(r-r_s), \qquad u_s|_{t=0}=0,\;\partial_t u_s|_{t=0}=0.
$$

FWI seeks to an inverse problem of finding the velocity model that best fits
the observed seismic data. Given observed data, standard FWI uses classical
optimization techniques to solve the following minimization problem:

$$
\min_{\mathbf{x}} \Phi(\mathbf{x}) = \sum_{s=1}^{N_s} \bigl\| P\,\mathcal{A}(\mathbf{x})^{-1} f_s - y_s \bigr\|_2^{2}
$$

where $N_s$ is the number of independent source experiments.

In realistic conditions (limited number of observations, limited resolution,
noise, etc.), the inverse problem defined by this equation is ill-posed
(that is, it has multiple solutions). This one-to-many mapping is the main
difficulty of FWI and makes it particularly suitable to be solved with
generative models. This example uses a diffusion model to solve the FWI inverse
problem.

![FWI acquisition geometry and inversion workflow](../assets/fwi_acquisition.png)

Here we remind a few key concepts essential to FWI in the context of hydro-carbon
exploration:

- *Velocity model* $\mathbf{x}(r)=\bigl[ V_\mathrm{P},\,V_\mathrm{S},\,\rho \bigr]$ – a 3-D image over coordinates $(z,x,y)$ spanning several kilometres and discretised at metre-scale resolution.

- *Seismic observations* $y=[u_z, u_x, u_y]$ – vertical and horizontal particle-velocity components recorded at the surface after emitting a seismic source; they contain reflections, refractions and surface waves.

- *Sources / shots* – positions where independent excitations are fired; typically tens to hundreds of shots distributed along the surface, each producing its own dataset $y_s$.

- *Receivers* – sensors that record $u_z, u_x, u_y$ at the surface; arranged in 2-D grids of $\mathcal{O}(10^3)$ sensors with a few-metre spacing.


## Getting started

# TODO: add prerequisite in diffusion models + add links to other examples

## Dataset preprocessing

> **⚠️  Warning:** Note that the E-FWI dataset is distributed under a non-commercial
license.

## Training

## Sampling and model evaluation

### Zero-shot sampling

### Physics-informed sampling

## References

> **⚠️  Warning:** Note that the E-FWI dataset is distributed under a non-commercial
license.
- [E-FWI: Multi-parameter Benchmark Datasets for Elastic Full Waveform Inversion of Geophysical Properties](https://arxiv.org/abs/2306.12386)
- [E-FWI datasets](https://smileunc.github.io/projects/efwi/datasets)
- [Deepwave (Richardson A.)](https://zenodo.org/records/8381177)
- [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687)

