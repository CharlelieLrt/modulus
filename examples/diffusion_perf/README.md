<!-- markdownlint-disable -->
# Diffusion Performance Benchmark

A self-contained benchmark that measures the cost of training and running
inference for a 2D EDM-style diffusion model under four implementations:

1. **Pure-PyTorch baseline**: no framework code, FP32, hand-rolled Heun
     sampler and EDM training loop. Reference point for both wall-clock and
     for the amount of user code required.
2. **PhysicsNeMo (no opts)**: the same model running through
     `physicsnemo.diffusion` (preconditioner, scheduler, sampler, loss),
     FP32, no optimizations enabled.
3. **PhysicsNeMo + full optimizations**: setting (2) with
     `{amp_bf16, torch.compile, apex.contrib.group_norm}` turned on.
4. **PhysicsNeMo + multi-diffusion + full optimizations**: setting (3)
     wrapped in `MultiDiffusionModel2D`, with the patch shape automatically
     bounded so that one patch always fits on the GPU.

Each implementation is exercised on three benchmarks:

* **Training**, multi-GPU DDP (4 ranks by default).
  * **Inference, no guidance**, single GPU.
  * **Inference + DPS data-consistency guidance**, single GPU.

The benchmark reports three quantities of interest (QoIs) per configuration:

* Throughput in **global-resolution samples per second per GPU**. For
    multi-diffusion configurations this still counts whole samples, not
    patches, so all four implementations are directly comparable.
* Peak GPU memory in GB per rank (max across ranks for DDP runs).
  * Model FLOP utilization (MFU) as a fraction of the GPU's BF16 peak.

It also records the number of **user-facing lines of code** required to
implement each variant, delimited by `# LOC-START` / `# LOC-END` markers in
`train.py`, `generate.py`, and `generate_dps_guidance.py`. Framework glue
(YAML serialization, FLOP probes, DDP setup, OOM guards, etc.) sits outside
those markers and is therefore not counted.

The benchmark is portable across GPU types. Calibration determines the
largest global domain that the non-multi-diffusion implementation can fit
inside a configurable fraction of GPU memory; everything else flows from
there.

---

## Layout

```
examples/diffusion_perf/
├── train.py                  # train_baseline, train_physicsnemo, train_physicsnemo_multidiffusion
├── generate.py               # generate_baseline, generate_physicsnemo, generate_physicsnemo_multidiffusion
├── generate_dps_guidance.py  # generate_dps_baseline, generate_dps_physicsnemo, generate_dps_physicsnemo_multidiffusion
├── calibrate.py              # Finds MAX_DOMAIN; writes results/_max_domain.yaml. Prerequisite of run_sweep.
├── run_sweep.py              # Sweep orchestrator (4 settings x DOMAIN_SWEEP x 3 benchmarks)
├── plot.py                   # Renders per-(benchmark, QoI) PNGs and the LoC comparison
├── bench/                    # Instrumentation (NOT counted in the LoC table)
│   ├── config.py             # Backbone kwargs, sweep ranges, MAX_GLOBAL_DOMAIN, GPU peak TFLOPS / VRAM
│   ├── adapter.py            # SongUNetAdapter (legacy SongUNet -> DiffusionModel protocol)
│   ├── calibration.py        # patch_shape_for(domain, max_domain), power_of_2_sweep
│   ├── timing.py             # CUDA-event timing, median + IQR
│   ├── memory.py             # Peak allocated memory, OOM guard
│   ├── flops.py              # FlopCounterMode wrapper used to compute MFU
│   ├── loc.py                # LOC marker counter
│   └── results.py            # YAML writer / ResultBuilder
├── results/                  # Per-run YAML outputs + plots/<device>/*.png + _max_domain.yaml
└── tests/test_smoke.py       # Tiny end-to-end smoke test

```

The benchmark backbone is fixed across all variants so that comparisons are
apples-to-apples. The defaults in `bench/config.py` produce an ~80M
parameter SongUNet:

| key | value |
|---|---|
| `model_channels` | 128 |
| `channel_mult` | `[1, 2, 2, 2, 2]` (5 levels, bottleneck at H/16) |
| `num_blocks` | 4 |
| `attn_resolutions` | computed per-call as the **two deepest** UNet levels (`[H/8, H/16]`) so self-attention always lands at the bottleneck and the level above it, regardless of global resolution |
| `dropout` | 0.13 |
| `embedding_type` | positional |

Sweep and loop defaults (all in `bench/config.py`):

| key | value | notes |
|---|---|---|
| `MAX_GLOBAL_DOMAIN` | 8192 | upper bound of the sweep, overridable via CLI |
| `DOMAIN_SWEEP_FULL` | `(64, 128, 256, 512, 1024, 2048, 4096, 8192)` | global domain edges in pixels |
| `CHANNELS` | 16 | data channels |
| `BATCH_SIZE_TRAIN` | 4 | per-rank training batch size |
| `BATCH_SIZE_INFER` | 1 | inference batch size |
| `SOLVER_STEPS` | 18 | Heun steps for inference |
| `WARMUP_STEPS` / `MEASURE_STEPS` | 6 / 15 | training timing loop |
| `WARMUP_STEPS_INFER` / `MEASURE_STEPS_INFER` | 3 / 5 | inference timing loop |
| `FULL_OPTS_TRAIN`, `FULL_OPTS_INFER` | `{amp_bf16, compile, apex_gn}` | "full opts" set |
| `PATCH_ALIGN` | 16 | every MD patch edge is a multiple of 16 (5 UNet levels => 4 downsamples) |
| `OBSERVATION_FRAC`, `OBSERVATION_STD`, `OBSERVATION_CHANNEL_FRAC` | 0.005 / 0.05 / 0.5 | sparse observation mask used by the DPS benchmark |

---

## Running the full benchmark

```bash
# 0. (one-time, per machine) prerequisites: NVIDIA driver + CUDA + a PhysicsNeMo
#    development environment with apex.contrib.group_norm available.

# 1. Calibrate MAX_DOMAIN for this GPU (writes results/_max_domain.yaml).
python -m examples.diffusion_perf.calibrate

# 2. Run all three benchmark suites end-to-end (takes a few hours on L40s).
python -m examples.diffusion_perf.run_sweep --suite all

# 3. Plot results into results/plots/<device>/ .
python -m examples.diffusion_perf.plot --device L40s

```

The three steps are designed to be re-runnable. Calibration is cached: it
exits early if `results/_max_domain.yaml` already exists (`--force` re-runs
it). The sweep orchestrator writes one YAML per (function, domain, opts)
tuple; existing files are overwritten on re-run.

---

## Calibration (`calibrate.py`)

`run_sweep.py` is a hard dependency on the output of `calibrate.py`. The
calibration finds the largest global domain `D*` such that
`train_physicsnemo` with `FULL_OPTS_TRAIN` fits in `<= MEM_FRAC_CAP`
(default 90%) of GPU memory at the configured training batch size, with all
4 ranks running in DDP. That `D*` is then used by all benchmarks as the
multi-diffusion patch cap:

```
effective_patch_size = min(MAX_GLOBAL_DOMAIN, D*)

```

so that multi-diffusion never OOMs. Internally:

* Phase 1 doubles the domain (64, 128, 256, ...) until the run either
    OOMs or exceeds `MEM_FRAC_CAP`.
* Phase 2 bisects between the last good domain and the first
    over-cap / OOM domain in multiples of `PATCH_ALIGN = 16`.

Output (`results/_max_domain.yaml`):

```yaml
max_domain: 608
max_domain_util: 0.882928...        # observed peak memory utilization at D*
first_oom_domain: 624               # first domain that exceeded the cap
mem_frac_cap: 0.9
batch_size: 4
patch_align: 16
opts: [amp_bf16, apex_gn, compile]
device: L40s
timestamp: ...
probe_log:                          # full sequence of probes
- {domain: 64,  phase: doubling, status: ok,  util: 0.044}
  - {domain: 128, phase: doubling, status: ok,  util: 0.073}
  - ...

```

CLI:

```bash
python -m examples.diffusion_perf.calibrate \
    [--batch-size 4]        # override training batch size used for the probe
    [--cap 8192]            # upper bound of phase 1
    [--warmup 6] [--measure 15]
    [--force]               # ignore cached _max_domain.yaml and re-run

```

To raise / lower the memory safety margin, edit `MEM_FRAC_CAP` at the top
of `calibrate.py` (default `0.90`).

---

## Sweep orchestrator (`run_sweep.py`)

Spawns one subprocess per (function, domain, opts) tuple to isolate
`torch.compile` cache leakage and DDP state. For each benchmark suite, runs
four settings at every domain in the sweep:

| index | setting label in `plot.py` | function | opts |
|---|---|---|---|
| 1 | `baseline` | `*_baseline` | none |
| 2 | `physicsnemo` | `*_physicsnemo` | none |
| 3 | `physicsnemo+opt` | `*_physicsnemo` | `{amp_bf16, compile, apex_gn}` |
| 4 | `MD+opt` | `*_physicsnemo_multidiffusion` | `{amp_bf16, compile, apex_gn}` |

Non-MD settings stop probing larger domains as soon as one OOMs. The MD
setting is expected never to OOM because the patch shape is bounded by the
calibration result.

CLI:

```bash
python -m examples.diffusion_perf.run_sweep \
    --suite {training,inference,inference_dps,all} \
    [--max-global-domain 8192]      # truncate the sweep from the top
    [--domains 512 1024 2048]       # explicit domain list (overrides --max-global-domain)
    [--settings baseline md]        # subset of the 4 settings (default: all 4)
    [--skip-existing]               # skip cases whose result YAML already exists
    [--warmup 6] [--measure 15]              # training timing loop
    [--warmup-infer 3] [--measure-infer 5]   # inference timing loop

```

Setting names accepted by `--settings`: `baseline`, `framework`,
`framework_opts`, `md` (the four rows in the table above).

Examples:

```bash
# Short sweep up to 2048 only.
python -m examples.diffusion_perf.run_sweep --suite all --max-global-domain 2048

# Re-run only the multi-diffusion column at d=4096 and d=8192 (e.g. after a
# framework change that affects patched sampling).
python -m examples.diffusion_perf.run_sweep \
    --suite inference --domains 4096 8192 --settings md

# Resume a partial sweep: skip every case whose result YAML already exists.
python -m examples.diffusion_perf.run_sweep --suite all --skip-existing

```

Training runs use `torchrun --nproc-per-node=4` automatically; inference
runs are single-GPU.

---

## Single-config invocations

You can run any single configuration directly without going through
`run_sweep.py`. This is what the orchestrator does under the hood.

```bash
# Training, framework, full opts, 256x256 domain, B=4/rank x 4 ranks DDP
torchrun --nproc-per-node=4 -m examples.diffusion_perf.train \
    --function train_physicsnemo --domain 256 --opts amp_bf16,compile,apex_gn \
    --batch-size 4 --warmup 6 --measure 15

# Training, multi-diffusion, requires patch-shape
torchrun --nproc-per-node=4 -m examples.diffusion_perf.train \
    --function train_physicsnemo_multidiffusion --domain 1024 \
    --opts amp_bf16,compile,apex_gn --batch-size 4 \
    --patch-shape 608 608

# Inference, framework, full opts, single GPU
python -m examples.diffusion_perf.generate \
    --function generate_physicsnemo --domain 256 \
    --opts amp_bf16,compile,apex_gn --warmup 3 --measure 5

# Inference, multi-diffusion + DPS guidance, single GPU
python -m examples.diffusion_perf.generate_dps_guidance \
    --function generate_dps_physicsnemo_multidiffusion --domain 1024 \
    --opts amp_bf16,compile,apex_gn --patch-shape 608 608 --chunk-size 1

```

Common flags:

| flag | applies to | description |
|---|---|---|
| `--function NAME` | all | which of the 3 functions inside the script to call |
| `--domain D` | all | global domain edge in pixels |
| `--opts a,b,c` | all | comma-separated list, subset of `{amp_bf16, compile, apex_gn}` |
| `--batch-size B` | train | per-rank batch size (default 4) |
| `--patch-shape Hp Wp` | MD only | required for `*_multidiffusion` |
| `--chunk-size C` | MD inference only | patches denoised per backbone call; default 1 |
| `--warmup N`, `--measure N` | all | timing loop sizes |

---

## Output schema

Each subprocess writes one YAML to `results/`. The path encodes the run
identity:

```
results/<function>_<device>_d<domain>_b<batch>_opt-<sorted-opts-or-"none">.yaml

Examples:
  train_physicsnemo_L40s_d256_b4_opt-amp_bf16-apex_gn-compile.yaml
  train_baseline_L40s_d256_b4_opt-none.yaml
  generate_physicsnemo_multidiffusion_L40s_d4096_b1_opt-amp_bf16-apex_gn-compile.yaml
  generate_dps_baseline_L40s_d128_b1_opt-none.yaml

```

Re-running the same configuration overwrites the previous YAML, so the
results directory always reflects the most recent measurement per
configuration.

Schema (one example):

```yaml
function: train_physicsnemo
timestamp: 2026-05-13T22:23:09.399687+00:00Z
device:
  name: L40s                # short label used in filenames
  raw_name: NVIDIA L40
  bf16_peak_tflops: 362.0   # used for MFU
  fp16_peak_tflops: 362.0
  total_memory_gb: 48.0
  capability: [8, 9]
  apex_gn_available: true
world_size: 4
config:
  domain: [256, 256]
  batch_size_per_rank: 4
  channels: 10
  optimizations: [amp_bf16, apex_gn, compile]
  num_steps_measured: 15
  num_steps_warmup: 6
  solver: null               # set for inference runs
  solver_steps: null
  patch_shape: null          # set for MD runs
  num_patches: null
backbone:
  class: SongUNet
  params: 79998090
  flops_per_step: 16074029400064  # measured by FlopCounterMode
results:
  status: ok                 # or "oom" / "error"
  step_time_ms_median: 205.97
  step_time_ms_p25: 205.63
  step_time_ms_p75: 206.71
  samples_per_sec_per_gpu_median: 19.42   # GLOBAL-resolution samples / s / GPU
  samples_per_sec_per_gpu_p25: 19.35
  samples_per_sec_per_gpu_p75: 19.45
  num_measured: 15
  peak_memory_allocated_gb_max_rank: 8.98
  peak_memory_utilization: 0.187
  mfu: 0.216                 # achieved_tflops / device.bf16_peak_tflops
  achieved_tflops_per_gpu: 78.04
loc:
  marked_lines: 16           # LOC-START / LOC-END count for this function
git:
  commit: f4bc2336...
  branch: diffusion-performance-profile

```

OOM runs have `status: oom` and no timing block; error runs have
`status: error` and an `error: <repr>` field. The orchestrator stops
probing larger domains for non-MD settings as soon as one of these is seen.

---

## Plotting (`plot.py`)

Reads every `results/*.yaml` for a given `--device`, deduplicates by
timestamp (keeping the most recent per configuration), and emits one PNG
per (benchmark, QoI) tuple plus one LoC comparison:

```
results/plots/<device>/
├── training__throughput.png
├── training__peak_memory.png
├── training__mfu.png
├── inference__throughput.png
├── inference__peak_memory.png
├── inference__mfu.png
├── inference_dps__throughput.png
├── inference_dps__peak_memory.png
├── inference_dps__mfu.png
└── loc_comparison.png

```

Visual style:

| implementation | bar style |
|---|---|
| pure PyTorch (baseline) | solid gray |
| PhysicsNeMo | NVIDIA-green outline, white fill |
| PhysicsNeMo + opt | NVIDIA-green outline, green diagonal hatching |
| PhysicsNeMo + multi-diffusion + opt | NVIDIA-green outline, solid green fill |
| OOM run | bold black "X" marker at the bar position |

X-axis is the global domain in pixels (categorical groups). Y-axis is the
QoI; throughput is log-scaled, peak memory and MFU are linear.

The LoC plot compares only the non-multi-diffusion versions (baseline vs
PhysicsNeMo) for each of the three benchmarks. Baseline is normalized to
100%; the absolute line counts are annotated on top of each bar.

CLI:

```bash
python -m examples.diffusion_perf.plot \
    [--device L40s]                # device.name to filter on
    [--batch-size-train 4]         # per-rank training BS to filter on
    [--batch-size-infer 1]         # inference BS to filter on
    [--results-dir <path>]         # source YAML directory (default: results/)
    [--out-dir <path>]             # override output directory
                                    # (default: <results-dir>/plots/<device>)

```

To plot a different GPU's results, point `--device` at the short label that
the result YAMLs use under `device.name` (e.g. `H100-SXM-80GB`,
`A100-SXM-80GB`, `B100`). New GPU types can be added to
`bench.config.GPU_PEAK_TFLOPS_BF16`, `GPU_TOTAL_MEMORY_GB`, and the
`_DEVICE_NAME_PATTERNS` table.

---

## Porting to a different GPU

The benchmark is designed to produce comparable numbers across GPU types.
The minimum steps are:

1. **Register the device** if it is not already in `bench/config.py`. Add
     entries to `GPU_PEAK_TFLOPS_BF16`, `GPU_TOTAL_MEMORY_GB`, and
     `_DEVICE_NAME_PATTERNS`. The short label you choose appears in every
     result YAML filename and in `plot.py`'s `--device` argument.
2. **Pick `MAX_GLOBAL_DOMAIN`.** Default is 8192. On GPUs with more
     memory, set it higher and extend `DOMAIN_SWEEP_FULL` to the next
     power of 2; on lower-memory GPUs, leave it at 8192 (calibration will
     just cap multi-diffusion patches earlier). Override at the CLI:
     ```bash
     python -m examples.diffusion_perf.run_sweep --suite all --max-global-domain 16384
     ```
3. **Re-run calibration** with `--force` (deletes the cached
     `_max_domain.yaml`):
     ```bash
     python -m examples.diffusion_perf.calibrate --force
     ```
4. **Run the sweep** as usual:
     ```bash
     python -m examples.diffusion_perf.run_sweep --suite all
     ```
5. **Plot**:
     ```bash
     python -m examples.diffusion_perf.plot --device <your-device-label>
     ```

Tunables you may want to adjust per GPU:

* `BATCH_SIZE_TRAIN`, `BATCH_SIZE_INFER` in `bench/config.py`.
  * `MEM_FRAC_CAP` in `calibrate.py` (default 0.90; lower for more headroom).
  * `WARMUP_STEPS{,_INFER}` and `MEASURE_STEPS{,_INFER}` for shorter /
    longer timing windows.

---

## Smoke test

```bash
python -m pytest examples/diffusion_perf/tests/test_smoke.py -v

```

Requires a CUDA device. Verifies that each entry point runs end-to-end at a
tiny domain (64²) with `warmup=2, measure=2`.
