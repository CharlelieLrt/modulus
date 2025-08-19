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
equations and applications.

Let us first introduce a few key concepts essential to FWI in the context of
hydro-carbon exploration:
- *Velocity model* $\mathbf{x}(r)=\bigl[
  V_\mathrm{P},\,V_\mathrm{S},\,\rho \bigr]$ – a 3-D image over
  coordinates $r = (z,x,y)$, where $z$ is the depth, and $x$ and $y$ are the
  surface coordinates. The P-wave velocity is denoted by $V_\mathrm{P}$, the
  S-wave velocity by $V_\mathrm{S}$, and $\rho$ is the density. The velocity
  model spans several kilometres and is discretised at metre-scale resolution.

- *Sources / shots* – positions $r_s = (0, x_s, y_s)_{1 \leq s \leq S}$ where independent
  excitations are fired; typically thousands of shots distributed along the
  surface. The cost associated with acquiring these shots and processing the
  data is a major component of the total cost of the FWI.

 <!-- TODO: make sure those are really the particle-velocity components. And make sure the name and math notation is consistent throughout -->
 <!-- TODO: make sure the time-resolution is correct -->
- *Receivers* – sensors at locations $r_i = (0, x_i, y_i)_{1 \leq i \leq R}$ that record
  horizontal particle-velocity components $u_x$ and $u_y$ and vertical
  particle-velocity component $u_z$ at the surface. Receivers are typically
  arranged in a 2-D grid of $\mathcal{O}(10^3)$ sensors with a few-meter
  spacing and record data with a time-resolution of a few tenth of seconds.

- *Seismic observations* - the particle-velocity components recorded at all
  receivers for a given shot $s$ can be combined into a 3D image $y_s = [u_z, u_x, u_y]$ over
  coordinates $(t, x_i, y_i)$ that contains reflections, refractions and
  surface waves. The $S$ independent sources can be further combined to form a
  large set $Y$ of 3D observations.

The goal of FWI is to reconstruct the velocity model $\mathbf{x}(r)$ from the
entire set of seismic observations $Y$. To do so, standard FWI uses classical
optimization techniques combined with the elastic wave equation, defined below.

<p align="center">
<img src="../../../docs/img/FWI_schematic.png" width="800"/>
</p>

$$
\mathcal{A}_{\mathbf{x}} \{\mathbf{u}\} (r, t) = \dot{S}(t)\,\delta(r-r_s)  \tag{1}
$$
 where $\dot{S}(t)\,\delta(r-r_s)$ is the source at location $r_s$ with time
 signal $\dot{S}(t)$, and $\mathcal{A}_{\mathbf{x}}$ is the elastic wave
 operator defined by:

 $$
\mathcal{A}_{\mathbf{x}} \{\mathbf{u}\} = \rho\ \frac{\partial^{2}
\mathbf{u}}{\partial t^{2}} - \nabla \bigl[  \rho \bigl( V^2_P - 2
V^2_S (\nabla \cdot \mathbf{u})  \bigr) \bigr] - \nabla \cdot \bigl[ \rho
V^2_S \bigl( \nabla \mathbf{u} + \nabla \mathbf{u}^T \bigr) \bigr] 
$$

We denote $\mathcal{R} (\mathbf{x}, s)$ the solution operator associated to the
wave equation $(1)$. This operator maps a velocity model $\mathbf{x}$ and a
source $\dot{S}(t)\,\delta(r-r_s)$ to the solution of the PDE at the receiver
locations: it therefore provide a simulated seismic observation $\hat{y}_s$.

FWI seeks to solve an inverse problem of finding the velocity model $\mathbf{x}$
that best fits the observed seismic data $Y$. Given observed data, standard FWI
uses classical optimization techniques to solve the following minimization
problem:

$$
\mathbf{x}^* = \arg \min_{\mathbf{x}} \Phi(\mathbf{x}) = \sum_{s=1}^{S} \bigl\|
\mathcal{R} (\mathbf{x}, s) - y_s \bigr\|_2^{2}
$$

In realistic conditions (limited number of observations, limited resolution,
noise, etc.), the inverse problem defined by this equation is ill-posed
(that is, it has multiple solutions). This one-to-many mapping is the main
difficulty of FWI and makes it particularly suitable to be solved with
generative models. This example uses a diffusion model to solve the FWI inverse
problem.


## Getting started

This example requires basic knowledge of [denoising diffusion
models](../../generative/README.md); it is also recommended to be familiar with
other examples using diffusion models, such as
[StormCast](../../weather/stormcast/README.md) or
[CorrDiff](../../weather/corrdiff/README.md).

Start by installing PhysicsNeMo (if not already installed) and copying this
folder (`examples/geophysics/diffusion_fwi`) to a system with a GPU available.
This example comprises a succession of three steps:

1. [Dataset preprocessing](#dataset-preprocessing)
2. [Training](#training)
3. [Sampling and model evaluation](#sampling-and-model-evaluation)

## Dataset preprocessing

This examples builds on the [E-FWI
dataset](https://smileunc.github.io/projects/efwi/datasets), initially
published as [E-FWI: Multi-parameter Benchmark Datasets for Elastic Full
Waveform Inversion of Geophysical
Properties](https://arxiv.org/abs/2306.12386). We complement the original
dataset by providing a data generation pipeline to: (1) expand the dataset to
the case of variable density $\rho(r)$, and (2) generate particle-velocity
observations from veloicty models in a consistent manner.

> **⚠️  Warning:** The E-FWI dataset is distributed under a non-commercial license [CC
> BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### Step 1: Download and reorganize the E-FWI dataset

The first step is to download the E-FWI dataset. In addition, since the E-FWI
dataset is composed of multiple sub-datasets (CFB, CFA, FVB, etc.), we provide
utility functions to merge them into a single dataset and reorganize the data
into a more convenient format. To pre-process the entire dataset, navigate to
the `./data` directory and run:

```bash
python download_data.py --download --reorganize --clean --shuffle --name all
```

This will download the dataset as a collection of `.npz` files in the directory
`./data/all/samples`. For more information about the possible options, run:

```bash
python download_data.py --help
```

### Step 2: Generate seismic observations

The second step consists in re-generating seismic observations from the
velocity models with variable density. The original wave speeds $V_\mathrm{P}$ and
$V_\mathrm{S}$ from the E-FWI datasets are retained and the density is generated
by the `generate_data.py` script. This scripts then solve an elastic wave
equationb using [Deepwave](https://zenodo.org/records/8381177) to generate the
seismic observations. Since this step can be time-consuming, it is advised to
do it on a machine with GPUs. To do so, still in the `./data` directory run:

```bash
python generate_data.py --in_dir ./all --out_dir <path_to_output_directory>
```

This script will generate a new set of `.npz` files in the directory
`<path_to_output_directory>/samples`.

### Step 3: Compute dataset statistics

Finally, the last step of the dataset preprocessing is to compute statistics of
the train and test sets. To do so, run:

```bash
python compute_stats.py --dir <path_to_output_directory> --batch_size 512 --num_workers 4
```

This script will compute dataset statistics and save them in the file
`<path_to_output_directory>/stats.json`. It supports distributed processing
based on `torch.distributed`, so it is advised to run it on a machine with
multiple GPUs. If doing so, replace the `python` command with:

```bash
torchrun --standalone --nnodes=<NUM_NODES> --nproc_per_node=<NUM_GPUS_PER_NODE> compute_stats.py --dir <path_to_output_directory> --batch_size 512 --num_workers 4
```

Once these steps are completed, you should have a dataset ready to be used for
training.

## Training

>**Configuration basics**
>
>Training is handled by `train.py`, configured using
>[hydra](https://hydra.cc/docs/intro/) based on the contents of the `config`
>directory. Hydra allows for YAML-based modular and hierarchical configuration
>management and supports command-line overrides for quick testing and
>experimentation. The `config` directory includes the following subdirectories:
> - `dataset`: specifies the dataset used for training as well as the resolution, number of variables, and other parameters of the dataset
> - `model`: specifies the model type and model-specific hyperparameters
> - `sampler`: specifies hyperparameters used in the sampling process for diffusion models
> - `training`: specifies training-specific hyperparameters and settings like checkpoint/log frequency and where to save training outputs
> - `inference` specifies inference-specific settings like which initial condition to run, which model checkpoints to use, etc.
> - `hydra`: specifies basic hydra settings, like where to store outputs (based on the training or inference outputs directories)


More extensive configuration modifications can be made by creating a new top-level configuration file similar to `regression` or `diffusion`. See `diffusion.yaml` for an example of how to specify a top-level config that uses default configuration settings with additional custom modifications added on top.

At runtime, hydra will parse the config subdirectory and command line over-rides into a runtime configuration object `cfg`, which will have all settings accessible via both attribute or dictionary-like interfaces. For example, the total training batch size can be accessed either as `cfg.training.batch_size` or `cfg['training']['batch_size']`.

The training script `train.py` will initialize the training experiment and launch the main training loop, which is defined in `utils/trainer.py`. Outputs (training logs, checkpoints, etc.) will be saved to a directory specified by the following `training` config items:
```yaml
training.outdir: 'rundir' # Root path under which to save training outputs
training.experiment_name: 'stormcast' # Name for the training experiment
training.run_id: '0' # Unique ID to use for this training run 
training.rundir: ./${training.outdir}/${training.experiment_name}/${training.run_id} # Path where experiement outputs will be saved
```
As you can see, the `training.run_id` setting can be used for distinguishing between different runs of the same configuration. The final training output directory is constructed by composing together the `training.outdir` root path (defaults to `rundir`), the `training.experiment_name`, and the `training.run_id`. For inference runs, equivalent options are available in the `stormcast_inference.yaml` config file used with the `inference.py` script.



## Sampling and model evaluation

### Zero-shot sampling

### Physics-informed sampling

## References

> **⚠️  Warning:** The E-FWI dataset is distributed under a non-commercial license [CC
> BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- [E-FWI: Multi-parameter Benchmark Datasets for Elastic Full Waveform Inversion of Geophysical Properties](https://arxiv.org/abs/2306.12386)
- [E-FWI datasets](https://smileunc.github.io/projects/efwi/datasets)
- [Deepwave (Richardson A.)](https://zenodo.org/records/8381177)
- [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687)

