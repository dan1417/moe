# Hardware-aware NVIDIA ML environment installer

> **In this MoE repo (CPT / MiMo-V2 on B300):** prefer  
> `python scripts/setup/optimize_ml_env.py --stage diagnose` then  
> `python scripts/setup/optimize_ml_env.py --root ml-env-cpt --stage all --fast`.  
> Legacy wrapper: `bash scripts/setup/setup_env.sh`.  
> Generic bootstrap: `python scripts/setup/bootstrap_ml.py --dry-run`.  
> Verify: `bash scripts/setup/check_env.sh`.

Research and selection policy: **6 September 2026** (CPT B300 path verified same day).

## Workload-aware CPT optimizer (recommended for this repo)

Interview-driven defaults (continued pretraining, mini→110B, seq 2048, prefer MXFP8 if stable, 1×B300 now / 8× later):

```bash
python scripts/setup/optimize_ml_env.py --stage diagnose
python scripts/setup/optimize_ml_env.py --stage plan
python scripts/setup/optimize_ml_env.py --root ml-env-cpt --stage all --fast
```

Creates a **new** `ml-env-cpt/` (does not touch `.venv`). Stages: diagnose → plan → install → validate → build-mini → bench → report.

### Candidates measured on the same install

| ID | Runtime | Role |
|---|---|---|
| A_fa4_bf16 | FA4 + BF16 | Correctness / throughput baseline |
| B_fa4_mxfp8 | FA4 + MXFP8 | Speed candidate (needs working TE) |
| C_eager_bf16 | eager + BF16 | Fallback reference |

### Critical TE finding (2026-09-06)

- `pip install transformer-engine` alone installs an **empty meta package** → MXFP8 probe fails.
- Official GitHub TE torch wheels are tagged for **NGC container torch** (`cu13torch26.xx`) and **ABI-mismatch** pip `torch==2.11.0+cu130`.
- Verified working recipe: install `nvidia-cuda-nvcc==13.0.88` + `nvidia-cuda-cccl==13.0.85`, point includes at venv `nvidia/{cu13,cudnn,nccl}`, then  
  `pip install --no-build-isolation "transformer-engine[pytorch,core_cu13]==2.18.0"`.

### CUDA layers (do not force them to be identical)

| Layer | What it means here |
|---|---|
| Driver | 580.x reports CUDA API 13.0 — ceiling for user-mode CUDA |
| PyTorch wheel | `cu130` ships its own CUDA **13.0 runtime** inside the venv |
| Local nvcc | Needed to **build** `transformer-engine-torch` sdist; not required to run Torch/FA4 |

## Generic bootstrap_ml.py

Place `bootstrap_ml.py` in your working directory. Use an idle Linux x86_64 GPU
allocation and Python 3.12 with `venv`/`ensurepip` available.

```bash
python3 bootstrap_ml.py
```

This creates a NEW `ml-env` directory with a training environment and a separate
vLLM inference environment. It does not alter the system Python, NVIDIA driver,
CUDA Toolkit, firmware, GPU power limits, existing venvs, or shell startup files.
It does not need sudo. Package installation executes trusted package installation
code; review the script and use a machine/allocation you control.

First inspect the plan without installing anything:

```bash
python3 bootstrap_ml.py --dry-run
```

For your 8×4090 machine, an easier first run that still supports Transformers
training AND inference is:

```bash
python3 bootstrap_ml.py --profile training
source ml-env/training/bin/activate
```

Do not run both install commands with the same output directory. Existing output
directories are refused, including failed installations. Use a new `--root`:

```bash
python3 bootstrap_ml.py --root ml-env-second
```

There is no command that can make an environment universally optimal for every
model. This installer provides a compatibility-checked baseline and measured
diagnostics. It does NOT choose your model, TP/PP/FSDP/ZeRO recipe, token batch,
precision recipe, or learning rate without model-specific testing.

## What it detects and decides

| Input | Decision/action |
|---|---|
| CUDA-visible GPU names, count, capability, memory | Select a reviewed CUDA wheel; test every visible GPU |
| Host NVIDIA driver and CUDA driver API | Reject stacks beyond the conservative driver policy |
| Topology, PCIe widths/generation, P2P/NVLink | Record bottlenecks; test NCCL; avoid forcing transport settings |
| System Python, existing packages, current Torch | Record without uninstalling/replacing them |
| `nvcc`, C++ compiler and toolkit | Needed only for explicitly permitted source builds |
| CPU affinity, NUMA, host/cgroup RAM, `/dev/shm`, memlock | Record constraints and warnings; no global affinity changes |
| Existing GPU processes | Stop by default to avoid interfering with training jobs |
| Compatible extension wheels | Try architecture-relevant candidates with the base stack constrained |
| Required package conflict | Stop or skip an optional extension; never silently replace Torch |
| GPU count 1/2/8 | Same software recipe, different number of device tests and NCCL ranks |

This edition supports Linux x86_64 GPUs at compute capabilities 7.5, 8.0, 8.6,
8.9, 9.0, 10.0, 10.3 and 12.0. This includes T4, A100, RTX 30/40,
H100/H200, B200/B300 and RTX 50. Old GPUs, unknown future architectures,
Jetson/Orin, GB10/DGX Spark, ARM servers, macOS and Windows stop with an explicit
message. That is intentional: a desktop Linux wheel is not a universal Jetson,
ARM, or legacy GPU solution. MIG visibility may differ between driver and Torch;
a count mismatch stops rather than guessing which resources are allocated.

## Selection policy, not a fastest-version claim

| Conditions | Pinned initial Torch / torchvision | CUDA runtime |
|---|---|---|
| Supported GPUs, driver ≥580.65.06, driver API ≥13.0 | 2.13.0 / 0.28.0 | cu130 |
| Older compatible driver ≥570.26, API ≥12.8; not B300 | 2.11.0 / 0.26.0 | cu128 |
| Older compatible driver ≥560.28.03, API ≥12.6; pre-Blackwell | 2.13.0 / 0.28.0 | cu126 |

These combinations are listed in the official [PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/).
The policy deliberately pins specific releases instead of assuming the newest
package is fastest. A package resolver still must confirm a wheel exists for your
Python/platform; if not, installation stops. No nightly or random GitHub wheel
fallback is used. Future recipe updates should be separately reviewed and tested.

Explicit selection is possible with `--cuda cu128`, but incompatible hardware or
driver combinations are still rejected. CUDA 13.3 is NOT intrinsically optimal
for B300. Earlier chat guidance suggesting a universal 13.3 requirement was too
strong. B300 is restricted to cu130 by this script's reviewed policy, not because
every CUDA 12.x binary is inherently incompatible.

Driver policy uses conservative GA toolkit driver versions from NVIDIA's
[CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html),
[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html), and
[CUDA 12.6 release notes](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html).
NVIDIA supports additional minor/forward compatibility cases, but JIT/PTX has
restrictions. The script does not install `cuda-compat` to work around a driver.
See the [CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## Extensions

Default `--extensions auto` tries ONLY binary resolutions that preserve already
installed package versions. It may legitimately skip all external extensions.
PyTorch SDPA and Inductor remain available without a system `nvcc`.

| GPUs | Automatic candidates |
|---|---|
| Ampere | FlashAttention-2 |
| Ada / RTX 4090 | Transformer Engine and FlashAttention-2 |
| Datacenter Hopper / B200 / B300 | Transformer Engine and FlashAttention-4 |
| RTX Blackwell | Transformer Engine; SDPA baseline |
| Turing | SDPA only |

Mixed architectures use a compatible intersection. FP8 support is not limited to
Hopper: Transformer Engine documents Ada support as well. Installing it does not
convert arbitrary PyTorch/Transformers layers into FP8 layers.

Select extensions explicitly:

```bash
python3 bootstrap_ml.py --root ml-env-custom --profile training --extensions fa2,bnb
```

`bnb` is optional bitsandbytes (quantized model/adapter workflows), not installed
by auto policy. FA2 and FA4 are not installed together because they share the
`flash_attn` namespace. Apex, xFormers, DeepSpeed, TensorRT, NeMo, JAX and TensorFlow
are not installed by this recipe: their model-specific integrations and binary
constraints warrant separate reviewed profiles. Native PyTorch DDP/FSDP and
Transformers/Accelerate/PEFT are included. Torchaudio is not included in this
Torch 2.13 baseline; add audio-specific packages through a separately tested recipe.

Source compilation is OFF by default. Enable only when you already have a complete
matching toolkit, supported C++ compiler and build dependencies:

```bash
python3 bootstrap_ml.py --root ml-env-build --profile training --extensions te,fa4 --allow-source
```

This option first tries wheels. If a source build is needed, it checks `CUDA_HOME`
or `nvcc` on PATH against `torch.version.cuda`, caps parallel compilation using
host/cgroup memory, compiles a small CUDA extension, then installs the requested
extension from the official package index. The toolkit's major/minor must match
for this policy; exact patch equality is not required. Build failures are retained
in logs and stop the installation. The extension's own compiler requirements still
apply. `TORCH_CUDA_ARCH_LIST` is only honored by some build systems; actual kernel
execution is the verification, not merely setting an environment variable.

We intentionally do not run `apt install cuda` or replace host drivers. Ask the
administrator for the appropriate CUDA *toolkit-only* development setup or a
matching NVIDIA devel container if compilation is required. Do not confuse a
partial `nvidia-cuda-nvcc` wheel with a complete toolkit.

Sources: [Transformer Engine requirements and installation](https://github.com/NVIDIA/TransformerEngine),
[FlashAttention installation and architecture support](https://github.com/Dao-AILab/flash-attention),
[NVIDIA Linux CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## Two environments are intentional

`training/` includes Torch, torchvision, Transformers, Accelerate, PEFT, Datasets,
tokenizers, safetensors, NumPy/SciPy/pandas/scikit-learn, Pillow, TensorBoard,
IPython kernel support, monitoring libraries, and compatible optional extensions.
It supports ordinary PyTorch and Transformers inference as well as training.

`inference/` (default `--profile all`) installs vLLM and lets its wheel choose its
own PyTorch/CUDA dependencies. It does not reuse the training Torch. The resolved
versions are captured before installation and the resulting CUDA stack is tested.
If no binary resolution works, or the resolved runtime cannot execute, inference
fails explicitly without damaging the training venv. No full vLLM source build
or blind CUDA-index override is attempted. In this case exit code 2 means partial
success, not a fully working all-profile setup.

This separation follows [vLLM's installation guidance](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
about incompatibilities between compiled kernels and different Torch/CUDA builds.

## Tests run on the target GPU machine

1. Matrix multiply and backward on every visible GPU.
2. Causal SDPA forward/backward compared with an FP32 math reference.
3. A tiny randomly initialized Llama model: training optimizer step and generation.
4. `torch.compile` forward/backward against eager output/gradient.
5. Optional FA2/FA4 attention, TE BF16/FP8 Linear, or bitsandbytes 4-bit Linear tests.
6. For multiple GPUs, one process per GPU with bounded NCCL all-reduce, all-gather,
   and reduce-scatter tests, plus 16 MiB all-reduce timing.
7. Synthetic SDPA auto/flash/cuDNN/math forward-and-backward timings per GPU.
8. `pip check`, final freeze, pip inspection and installation provenance.
9. vLLM custom CUDA RMSNorm on every GPU if serving environment is requested.

Attention reference tolerance is explicitly loose enough for BF16 (rtol 0.06,
atol 0.04). These tests detect gross failures, NOT accuracy regressions for all
models. They do not validate packed masks, sliding windows, arbitrary GQA,
long contexts, FP4, convergence, or your full training loop. vLLM RMSNorm is not
a full model-server test. No model weights or datasets are downloaded.

The benchmark records 1×8×512×64 attention, a single synthetic shape. Its winner
is NOT applied globally. Keep model SDPA auto dispatch and benchmark realistic
training batches before overriding kernels. Installing an external backend alone
does not change your model's attention code.

If `--skip-nccl` is set, readiness explicitly says distributed tests were skipped.
`--allow-busy` overrides the idle-allocation guard; only use it when you control
the running jobs and accept interference. Passing smoke tests is not performance
certification. Consult [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
and [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

## Read the result

```bash
cat ml-env/manifest.json
cat ml-env/reports/attention-benchmark.json
```

Other outputs:

- `inventory.json`: observed hardware, host topology, existing software; may contain
  machine UUIDs and process IDs. Review before sharing.
- `plan.json`: selected policy and reasons.
- `training.lock.txt`, `inference.lock.txt`: exact installed package snapshots.
- `reports/*-resolve.json`, `*-installed.json`: package URLs/hashes and metadata
  from pip. Treat these as private if using a configured authenticated mirror.
- `reports/*-resolved-constraints.txt`: pins for each pre-install resolution.
- `reports/*.log`: full subprocess results.
- `run-training.sh`, `run-inference.sh`: launch commands with per-environment JIT
  caches. Only created after that profile's tests pass.

```bash
bash ml-env/run-training.sh python your_training_script.py
bash ml-env/run-training.sh torchrun --standalone --nproc_per_node=8 your_ddp_script.py
```

The second command requires your script to implement distributed training.
Eight installed GPUs do not automatically parallelize ordinary Python code.

Or activate the interpreter directly:

```bash
source ml-env/training/bin/activate
```

The launcher deliberately does not force TF32, FP8, NCCL transports, CPU binding,
allocator tuning or model backend changes. A portable training starting point is
FP32 parameters/optimizer state with BF16 autocast when native BF16 is available,
otherwise FP16 autocast with GradScaler; many large-model frameworks instead
manage BF16 model states and FP32 masters themselves. Follow the chosen framework's
recipe. Use packed data, appropriate batching, checkpointing and fused optimizers
only after measuring the model's speed and memory requirements.

## Reproducibility and upgrades

The Torch recipe is pinned. Other package versions are resolved from stable
indexes on first install, then pinned for that transaction and recorded. Two first
installs on different days may resolve different extras/vLLM versions. This is
NOT a hermetic offline lock or a promise that unbounded future dependencies work.

Archive the resulting installer copy, inventory, manifest, lock files and pip
reports. For a repeat run, use a clean Python venv and the recorded CUDA index;
preinstall the exact recorded Torch build from that index, then install the lock
against PyPI. Preserve/download original wheels and their hashes for strict offline
reproduction. Do not replace one core library independently and assume compiled
extensions remain compatible. Re-run model validation after any upgrade.

Pip configuration/administrator constraints are honored, not bypassed. A managed
NGC image may impose package constraints that cause this independent recipe to
stop. In that case prefer the image's validated environment; do not remove its
constraints just to make this installer pass. NVIDIA recommends its integrated
containers for Transformer Engine, but a container's Torch can be a development
build, not the same as a public stable wheel. See [NVIDIA PyTorch release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html).

## Corrections to earlier guidance

- `sm_89` or `sm_103` absent from `get_arch_list()` is not by itself an error.
  Generic compatible cubins/PTX can run. Architecture-specific `a`/family feature
  targets have narrower rules. See [NVIDIA Blackwell compatibility](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html).
- Driver/toolkit/PyTorch versions do not obey a universal numeric equality.
  A supported driver/runtime relationship plus actual execution matters.
- `nvcc: command not found` means it is not on PATH; a toolkit may exist elsewhere.
- PCIe topology describes routes, not measured bandwidth. Physical sockets cannot
  always be inferred just from NUMA labels. A maximum lane field does not prove the
  slot is wired at that width.
- For DDP, accumulation reduces synchronization only when implemented correctly
  (e.g. `no_sync()` on nonfinal accumulation microbatches).
- No generic ordering such as pipeline > FSDP > TP holds for every model.
- Synthetic GPU checks and `nvidia-smi` do not certify long-term hardware health.

## Verification performed during creation

The deliverable includes offline unit tests covering GPU/driver selection,
1/2/8 B300 detection, mixed GPUs, extension selection, existing-directory safety,
synthetic-inventory safety, subprocess handling and launcher syntax.

```bash
python3 -m unittest -v test_bootstrap_ml.py
```

The creation environment has no accessible NVIDIA GPU. Actual package installation,
GPU kernels, NCCL and per-device performance were NOT executed there. Those are
implemented as target-machine tests. No claim of H100/H200/B200/B300/4090 hardware
validation is made before you run them on your allocation.

Exit codes: 0 = selected profile passed (inspect skipped-extension/test statuses);
1 = failed/stopped; 2 = training passed but separate inference failed.

