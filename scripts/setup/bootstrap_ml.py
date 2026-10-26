#!/usr/bin/env python3
"""Hardware-aware NVIDIA ML environment bootstrap, policy dated 2026-09-05.

Run: python3 bootstrap_ml.py --dry-run
     python3 bootstrap_ml.py

Creates fresh environments only. No sudo, driver changes, system toolkit installs,
GPU resets, shell-profile edits, model downloads, or automatic remote-code trust.
See ML_ENV_GUIDE.md for scope, research sources, and reproducibility limitations.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
try:
    import resource
except ImportError:
    resource = None  # Non-Linux systems receive an explicit preflight message.
import shlex
import shutil
import signal
import subprocess
import sys
import time

POLICY_DATE = "2026-09-05"
# Published stable wheel combinations, NOT a claim of fastest versions.
# Full toolkit-release driver floors deliberately avoid relying on minor-version
# compatibility for JIT/PTX. No driver upgrade is performed.
STACKS = {
    "cu130": {"torch": "2.13.0", "torchvision": "0.28.0", "cuda": "13.0",
              "driver_floor": "580.65.06", "min_cc": 75},
    "cu128": {"torch": "2.11.0", "torchvision": "0.26.0", "cuda": "12.8",
              "driver_floor": "570.26", "min_cc": 75},
    "cu126": {"torch": "2.13.0", "torchvision": "0.28.0", "cuda": "12.6",
              "driver_floor": "560.28.03", "min_cc": 75},
}
KNOWN_CC = {75, 80, 86, 89, 90, 100, 103, 120}
BASE_PACKAGES = [
    "transformers", "accelerate", "datasets", "peft", "safetensors",
    "huggingface-hub", "tokenizers", "sentencepiece", "protobuf",
    "numpy", "scipy", "pandas", "scikit-learn", "pillow", "tensorboard",
    "psutil", "nvidia-ml-py", "packaging", "einops", "tqdm", "ipykernel",
]
# TE: bare transformer-engine is an empty meta package (fails at import).
# For CUDA 13 / Blackwell use core_cu13 + pytorch extras. transformer-engine-torch
# is typically an sdist that must build against the installed Torch (see optimize_ml_env.py).
EXT_PACKAGES = {"fa2": "flash-attn", "fa4": "flash-attn-4",
                "te": "transformer-engine[pytorch,core_cu13]", "bnb": "bitsandbytes"}
SOURCE_URLS = {
    "pytorch_wheels": "https://pytorch.org/get-started/previous-versions/",
    "driver_compatibility": "https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html",
    "blackwell_binary_compatibility": "https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html",
    "cuda_installation": "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html",
    "transformer_engine": "https://github.com/NVIDIA/TransformerEngine",
    "flash_attention": "https://github.com/Dao-AILab/flash-attention",
    "vllm": "https://docs.vllm.ai/en/stable/getting_started/installation/gpu/",
    "nccl": "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html",
    "tuning": "https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
}


class SetupError(RuntimeError):
    pass


def version_tuple(value):
    parts = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not parts:
        raise SetupError(f"Cannot parse version: {value!r}")
    return tuple(int(v or 0) for v in parts.groups())


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "unavailable"


def run(argv, *, timeout=30, env=None, log=None, check=False, announce=False):
    """No shell interpolation; terminate only our process group on timeout."""
    argv = [str(v) for v in argv]
    if announce:
        print("  $", shlex.join(argv), flush=True)
    try:
        with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=env, start_new_session=True) as proc:
            try:
                output, _ = proc.communicate(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    output, _ = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    output, _ = proc.communicate()
                if log:
                    Path(log).write_text(output + "\nTIMEOUT / INTERRUPTED\n")
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise SetupError(f"Timed out after {timeout}s: {shlex.join(argv[:5])}")
            code = proc.returncode
    except FileNotFoundError:
        output, code = f"Command not found: {argv[0]}", 127
    if log:
        Path(log).write_text(output)
    if check and code:
        tail = output[-3000:]
        raise SetupError(f"Command failed ({code}). Log: {log}\n{tail}")
    return {"returncode": code, "output": output}


def probe_driver():
    """Query CUDA without requiring a PyTorch install; no tensor allocations."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        raise SetupError("CUDA driver library is inaccessible. Request GPU access from the host/container administrator.") from exc
    def call(name, *args):
        code = getattr(lib, name)(*args)
        if code:
            raise SetupError(f"CUDA driver {name} failed with code {code}; check GPU allocation/fabric/driver.")
    call("cuInit", 0)
    count, driver_api = ctypes.c_int(), ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    call("cuDriverGetVersion", ctypes.byref(driver_api))
    devices = []
    for i in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        name, pci = ctypes.create_string_buffer(256), ctypes.create_string_buffer(64)
        memory = ctypes.c_size_t()
        call("cuDeviceGet", ctypes.byref(device), i)
        call("cuDeviceGetName", name, len(name), device)
        call("cuDeviceGetAttribute", ctypes.byref(major), 75, device)
        call("cuDeviceGetAttribute", ctypes.byref(minor), 76, device)
        call("cuDeviceTotalMem_v2", ctypes.byref(memory), device)
        call("cuDeviceGetPCIBusId", pci, len(pci), device)
        devices.append({"index": i, "name": name.value.decode(),
                        "cc": major.value * 10 + minor.value,
                        "memory_bytes": memory.value, "pci_bus_id": pci.value.decode()})
    return {"devices": devices, "driver_api": driver_api.value}


def inventory():
    result = {"platform": platform.system(), "machine": platform.machine(),
              "python": platform.python_version(), "python_executable": sys.executable,
              "os_release": read_text("/etc/os-release"), "commands": {}}
    commands = {
        "smi": ["nvidia-smi"], "list": ["nvidia-smi", "-L"],
        "driver": ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        "topology": ["nvidia-smi", "topo", "-m"],
        "p2p_pcie": ["nvidia-smi", "topo", "-p2p", "p"],
        "p2p_nvlink": ["nvidia-smi", "topo", "-p2p", "n"],
        "pcie": ["nvidia-smi", "--query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max", "--format=csv"],
        "health": ["nvidia-smi", "-q"], "nvcc": ["nvcc", "--version"],
        "cpu": ["lscpu"], "numa": ["numactl", "--hardware"],
        "fabric_manager": ["systemctl", "is-active", "nvidia-fabricmanager"],
        "processes": ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader"],
    }
    for key, argv in commands.items():
        try:
            result["commands"][key] = run(argv)
        except SetupError as exc:
            result["commands"][key] = {"returncode": 124, "output": str(exc)}
    driver_lines = result["commands"]["driver"]["output"].splitlines()
    if result["commands"]["driver"]["returncode"] or not driver_lines:
        raise SetupError("nvidia-smi cannot query the driver. No packages were installed.")
    result["driver"] = driver_lines[0].strip()
    result.update(probe_driver())
    result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    result["mem_available_kib"] = next((int(line.split()[1]) for line in
        read_text("/proc/meminfo").splitlines() if line.startswith("MemAvailable:")), 0)
    result["cgroup_memory_max"] = read_text("/sys/fs/cgroup/memory.max")
    result["cgroup_memory_current"] = read_text("/sys/fs/cgroup/memory.current")
    result["memlock_limits"] = resource.getrlimit(resource.RLIMIT_MEMLOCK) if resource else None
    result["shm_free_bytes"] = shutil.disk_usage("/dev/shm").free if Path("/dev/shm").exists() else None
    result["visible_devices_env"] = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    result["ngc_container"] = os.environ.get("NVIDIA_PYTORCH_VERSION", "unset")
    # Names only: never dump token-bearing environment variables or pip credentials.
    result["environment_overrides"] = sorted(k for k in os.environ if
        k.startswith(("NCCL_", "TORCH_", "PYTORCH_", "PIP_")) or
        k in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "CUDA_HOME"))
    code = "import json,importlib.metadata as m; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"
    result["existing_packages"] = run([sys.executable, "-c", code])
    result["existing_torch"] = run([sys.executable, "-c",
        "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_arch_list())"])
    # Retain enough detail to select toolkit without mistaking an off-PATH
    # installation for a missing toolkit. Listing a few standard paths is not an
    # attempt to change the administrator's selection.
    result["toolkit_candidates"] = [str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/nvcc"))]
    if os.environ.get("CUDA_HOME"):
        result["selected_cuda_home"] = os.environ["CUDA_HOME"]
    return result


def choose_stack(info, cuda="auto"):
    if info["platform"] != "Linux" or info["machine"] not in ("x86_64", "AMD64"):
        raise SetupError("This recipe supports Linux x86_64 only. ARM/Jetson/Windows need a platform-specific recipe.")
    if not info["devices"]:
        raise SetupError("No CUDA-visible GPU. Check the scheduler/container allocation and CUDA_VISIBLE_DEVICES.")
    capabilities = {g["cc"] for g in info["devices"]}
    unknown = capabilities - KNOWN_CC
    if unknown:
        raise SetupError(f"No reviewed recipe for compute capabilities {sorted(unknown)}. Refusing to guess a wheel; use a vendor-supported legacy/platform stack.")
    candidates = [cuda] if cuda != "auto" else ["cu130", "cu128", "cu126"]
    for name in candidates:
        recipe = STACKS[name]
        if version_tuple(info["driver"]) < version_tuple(recipe["driver_floor"]):
            continue
        # Blackwell Ultra is deliberately CUDA 13 only in this policy.
        if 103 in capabilities and name != "cu130":
            continue
        if max(capabilities) >= 100 and name == "cu126":
            continue
        if min(capabilities) < recipe["min_cc"]:
            continue
        api = info.get("driver_api", 0)
        major, minor, _ = version_tuple(recipe["cuda"])
        if api < major * 1000 + minor * 10:
            continue
        return {"index": name, **recipe}
    raise SetupError("No reviewed CUDA wheel fits this driver/GPU combination. Ask the administrator for a compatible driver or choose a vendor-validated container; this script will not change drivers.")


def extension_candidates(info, requested):
    caps = {g["cc"] for g in info["devices"]}
    if requested == "none":
        return []
    if requested != "auto":
        values = requested.split(",")
        if len(values) != len(set(values)) or set(values) - EXT_PACKAGES.keys():
            raise SetupError("--extensions must be auto, none, or comma-separated fa2,fa4,te,bnb without duplicates.")
        if "fa2" in values and "fa4" in values:
            raise SetupError("FA2 and FA4 share a Python namespace. Use separate environments, not both here.")
        if "fa2" in values and not caps <= {80, 86, 89, 90}:
            raise SetupError("This FA2 recipe is limited to Ampere/Ada/Hopper; do not force it on Blackwell.")
        if "fa4" in values and not caps <= {90, 100, 103}:
            raise SetupError("This FA4 recipe is limited to datacenter Hopper/Blackwell, not RTX Blackwell.")
        if "te" in values and min(caps) < 80:
            raise SetupError("This Transformer Engine recipe requires Ampere or later.")
        return values
    selected = []
    if min(caps) >= 89:
        selected.append("te")
    if caps <= {90, 100, 103}:
        selected.append("fa4")
    elif caps <= {80, 86, 89, 90}:
        selected.append("fa2")
    return selected


def notes(info):
    out = ["GPU count changes launch size and memory planning, not the PyTorch version.",
           "Hardware inventory does not establish a universal optimal TP/PP/FSDP layout."]
    topo = info["commands"].get("topology", {}).get("output", "")
    if len(info["devices"]) > 1:
        out.append("NCCL must pass an all-reduce test before this environment is marked ready.")
        if re.search(r"\bNV\d+\b", topo):
            out.append("NVLink paths are reported; benchmark real collectives before tuning parallelism.")
        else:
            out.append("No NVLink path detected. Communication-heavy sharding may be expensive; absence of topology data is not proof of no NVLink.")
        if "SYS" in topo:
            out.append("Cross-NUMA paths reported. Do not hard-code CPU affinity without checking scheduler CPU allocation.")
    p2p = info["commands"].get("p2p_pcie", {}).get("output", "")
    matrix = "\n".join(line for line in p2p.splitlines() if re.match(r"\s*GPU\d+\s", line))
    if "NS" in matrix and not re.search(r"\bOK\b", matrix):
        out.append("PCIe P2P reports unsupported pairs. NCCL may stage via host memory; do not automatically disable transports globally.")
    pcie = info["commands"].get("pcie", {}).get("output", "")
    try:
        for row in list(csv.reader(io.StringIO(pcie)))[1:]:
            if len(row) >= 6 and row[4].strip().isdigit() and int(row[4]) <= 4:
                out.append(f"Physical GPU {row[0].strip()} currently has PCIe x{row[4].strip()}; benchmark under load. 'max' is not proof of wired slot width.")
    except (ValueError, csv.Error):
        pass
    if info.get("shm_free_bytes") is not None and info["shm_free_bytes"] < 1024**3:
        out.append("Less than 1 GiB /dev/shm free. Container multi-GPU/DataLoader use may need a larger --shm-size; script does not change it.")
    if len({g["cc"] for g in info["devices"]}) > 1:
        out.append("Mixed GPU architectures: test each GPU; separate homogeneous jobs are often easier to tune.")
    if info.get("ngc_container", "unset") != "unset":
        out.append("NVIDIA container detected. Its existing optimized stack is recorded and untouched; prefer it for vendor-specific TE/NeMo training recipes.")
    return out


def installation_plan(info, args):
    stack = choose_stack(info, args.cuda)
    return {"policy_date": POLICY_DATE, "gpu_count": len(info["devices"]),
            "devices": info["devices"], "stack": stack,
            "profile": args.profile, "extension_candidates": extension_candidates(info, args.extensions),
            "source_builds_allowed": args.allow_source, "notes": notes(info),
            "baseline": "BF16 when natively supported, otherwise FP16 AMP; PyTorch SDPA auto dispatch",
            "inference": "Transformers in training venv; separate vLLM venv for --profile all",
            "not_automatic": ["driver/firmware changes", "toolkit installation", "FP8/FP4 conversion",
                "model parallelism", "training hyperparameters", "NCCL transport overrides", "model downloads"],
            "sources": SOURCE_URLS}


def pip_env(root):
    env = os.environ.copy()
    # Do not bypass PIP_CONSTRAINT or managed package restrictions. Honor custom
    # constraints (may cause a safe resolution failure). No credential dumps.
    for key in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        if env.get(key):
            raise SetupError(f"{key} is set and could escape/isolate the venv incorrectly. Review it before rerunning.")
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_CACHE_DIR"] = str(root / "cache" / "pip")
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["TORCH_EXTENSIONS_DIR"] = str(root / "cache" / "torch_extensions")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(root / "cache" / "inductor")
    env["TRITON_CACHE_DIR"] = str(root / "cache" / "triton")
    env["HF_HOME"] = str(root / "cache" / "huggingface")
    return env


class Installer:
    def __init__(self, root, info, plan, args):
        self.root, self.info, self.plan, self.args = root, info, plan, args
        self.env = pip_env(root)
        self.counter = 0
        self.manifest = {"status": "INSTALLING", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                         "plan": plan, "stages": {}, "extension_status": {}, "commands": []}
        self.save()

    def save(self):
        json_write(self.root / "manifest.json", self.manifest)

    def command(self, name, argv, timeout=1800, check=True, env=None):
        self.counter += 1
        log = self.root / "reports" / f"{self.counter:03d}-{name}.log"
        self.manifest["commands"].append({"name": name, "log": str(log)})
        self.save()
        result = run(argv, timeout=timeout, env=env or self.env, log=log, check=check, announce=True)
        return result

    def venv(self, name):
        target = self.root / name
        self.command(f"{name}-venv", [sys.executable, "-m", "venv", target], timeout=120)
        py = target / "bin" / "python"
        self.command(f"{name}-pip", [py, "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
            "--only-binary=:all:", "--upgrade", "pip", "setuptools", "wheel", "packaging"])
        return py

    def install(self, py, name, packages, *, constraints=None, index="https://pypi.org/simple", source=False, env=None, optional=False):
        base = [py, "-m", "pip", "install", "--index-url", index]
        base += ["--no-build-isolation"] if source else ["--only-binary=:all:"]
        if constraints:
            base += ["--constraint", constraints]
        report = self.root / "reports" / f"{name}-resolve.json"
        resolved = self.command(f"{name}-resolve", base + ["--dry-run", "--report", report] + packages,
                                timeout=7200 if source else 1800, check=not optional, env=env)
        if resolved["returncode"]:
            return False  # No package install occurred on binary-only resolution failure.
        # Lock exactly this resolution before installing, including transitive deps.
        data = json.loads(report.read_text())
        resolved_constraints = self.root / "reports" / f"{name}-resolved-constraints.txt"
        pins = [f"{x['metadata']['name']}=={x['metadata']['version']}" for x in data.get("install", [])]
        resolved_constraints.write_text("\n".join(pins) + "\n")
        self.command(f"{name}-install", base + ["--constraint", resolved_constraints, "--report",
            self.root / "reports" / f"{name}-installed.json"] + packages,
            timeout=7200 if source else 3600, env=env)
        return True

    def lock(self, py, name):
        result = self.command(f"{name}-freeze", [py, "-m", "pip", "freeze", "--all"], timeout=60)
        path = self.root / f"{name}.lock.txt"
        path.write_text(result["output"])
        self.command(f"{name}-check", [py, "-m", "pip", "check"], timeout=120)
        self.command(f"{name}-inspect", [py, "-m", "pip", "inspect"], timeout=120)
        return path

    def worker(self, py, name, kind, extra=None, timeout=600):
        dest = self.root / "reports" / f"{name}.json"
        argv = [py, self.root / "bootstrap_ml.py", "--worker", kind, "--output", dest]
        if extra:
            argv.extend(extra)
        self.command(name, argv, timeout=timeout)
        return json.loads(dest.read_text())

    def source_environment(self, py):
        result = self.command("toolkit-version", [py, "-c", "import torch; print(torch.version.cuda)"], timeout=60)
        torch_cuda = result["output"].strip()
        home = os.environ.get("CUDA_HOME")
        nvcc = Path(home) / "bin" / "nvcc" if home else Path(shutil.which("nvcc") or "/nonexistent/nvcc")
        if not nvcc.is_file():
            # Prefer pip nvidia-cuda-nvcc inside the target venv (no apt/sudo).
            probe = self.command(
                "pip-nvcc-probe",
                [py, "-c",
                 "import site; from pathlib import Path\n"
                 "cands=[]\n"
                 "for sp in site.getsitepackages():\n"
                 " p=Path(sp)/'nvidia'/'cu13'/'bin'/'nvcc'\n"
                 " print(p if p.is_file() else '')\n"],
                timeout=60, check=False,
            )
            for line in probe["output"].splitlines():
                line = line.strip()
                if line and Path(line).is_file():
                    nvcc = Path(line)
                    break
        if not nvcc.is_file():
            raise SetupError(
                f"Source build requires CUDA Toolkit {torch_cuda} nvcc. "
                "Install pip package nvidia-cuda-nvcc into the venv or use a devel container. "
                "Prefer scripts/setup/optimize_ml_env.py for the verified CPT TE recipe."
            )
        compiler = run([str(nvcc), "--version"])
        match = re.search(r"release (\d+\.\d+)", compiler["output"])
        if not match or match.group(1) != torch_cuda:
            raise SetupError(f"For source builds this recipe requires nvcc major.minor == {torch_cuda}. Current output: {compiler['output']}")
        if not shutil.which("c++"):
            raise SetupError("No C++ compiler. Use your administrator's matching CUDA development environment.")
        self.install(py, "build-prereqs", ["ninja", "cmake", "pybind11", "psutil", "nvidia-cudnn-frontend"],
                     constraints=self.root / "training.lock.txt")
        env = self.env.copy()
        env["CUDA_HOME"] = str(nvcc.parent.parent)
        env["PATH"] = str(nvcc.parent) + os.pathsep + env.get("PATH", "")
        # cuDNN / NCCL headers from torch's nvidia-* wheels
        site_probe = self.command(
            "nvidia-roots",
            [py, "-c",
             "import site,json; from pathlib import Path\n"
             "out={}\n"
             "for sp in site.getsitepackages():\n"
             " n=Path(sp)/'nvidia'\n"
             " if (n/'cudnn'/'include'/'cudnn.h').is_file(): out['cudnn']=str(n/'cudnn')\n"
             " if (n/'nccl'/'include'/'nccl.h').is_file(): out['nccl']=str(n/'nccl')\n"
             "print(json.dumps(out))\n"],
            timeout=60, check=False,
        )
        try:
            roots = json.loads(site_probe["output"].strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            roots = {}
        includes = [str(nvcc.parent.parent / "include")]
        libs = [str(nvcc.parent.parent / "lib")]
        for key in ("cudnn", "nccl"):
            if key in roots:
                includes.append(str(Path(roots[key]) / "include"))
                libs.append(str(Path(roots[key]) / "lib"))
                if key == "cudnn":
                    env["CUDNN_PATH"] = roots[key]
                    env["CUDNN_HOME"] = roots[key]
        env["CPLUS_INCLUDE_PATH"] = ":".join(includes)
        env["C_INCLUDE_PATH"] = env["CPLUS_INCLUDE_PATH"]
        env["LIBRARY_PATH"] = ":".join(libs)
        env["LD_LIBRARY_PATH"] = ":".join(libs) + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else "")
        env["NVTE_FRAMEWORK"] = "pytorch"
        env["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{cc//10}.{cc%10}" for cc in sorted({d['cc'] for d in self.info['devices']}))
        available = self.info.get("mem_available_kib", 0) * 1024
        limit = self.info.get("cgroup_memory_max", "max")
        current = self.info.get("cgroup_memory_current", "0")
        if str(limit).isdigit() and str(current).isdigit():
            available = min(available, max(0, int(limit) - int(current)))
        jobs = max(1, min(4, len(self.info.get("cpu_affinity", [0])), available // (8 * 1024**3) or 1))
        env["MAX_JOBS"] = str(jobs)
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
        env["NVTE_BUILD_THREADS_PER_JOB"] = "1"
        # Packages may ignore TORCH_CUDA_ARCH_LIST: real kernel tests remain required.
        self.command("toolkit-compile", [py, self.root / "bootstrap_ml.py", "--worker", "compile-extension",
            "--output", self.root / "reports" / "toolkit-compile.json"], timeout=600, env=env)
        return env

    def training(self):
        py = self.venv("training")
        s = self.plan["stack"]
        self.install(py, "torch", [f"torch=={s['torch']}", f"torchvision=={s['torchvision']}"],
                     index=f"https://download.pytorch.org/whl/{s['index']}")
        core_lock = self.lock(py, "torch-core")
        core = self.worker(py, "core-smoke", "core")
        if core["cuda"] != s["cuda"]:
            raise SetupError("Installed Torch CUDA runtime differs from the selected recipe.")
        if len(core["devices"]) != self.plan["gpu_count"]:
            raise SetupError("PyTorch device count differs from the initial CUDA inventory; check visibility/MIG allocation.")
        self.install(py, "ml-packages", BASE_PACKAGES, constraints=core_lock)
        self.worker(py, "hf-training-inference", "hf", timeout=600)
        locked = self.lock(py, "training")
        build_env = None
        for ext in self.plan["extension_candidates"]:
            package = EXT_PACKAGES[ext]
            if ext == "fa4" and s["cuda"].startswith("13."):
                package += "[cu13]"
            explicit = self.args.extensions != "auto"
            # TE pytorch extension is an sdist: binary-only resolve usually fails.
            # Prefer optional binary attempt, then --no-build-isolation source with toolkit.
            if ext == "te":
                available = self.install(py, ext, [package], constraints=locked, optional=True, source=False)
                if not available:
                    # Allow TE source when auto/requested: needs nvcc (host or pip nvidia-cuda-nvcc).
                    try:
                        self.install(py, "te-build-prereqs",
                                     ["ninja", "nvidia-cuda-nvcc==13.0.88", "nvidia-cuda-cccl==13.0.85"],
                                     constraints=locked, optional=True)
                        build_env = build_env or self.source_environment(py)
                    except SetupError as exc:
                        self.manifest["extension_status"][ext] = (
                            f"SKIPPED: TE needs toolkit to build torch extension ({exc}). "
                            "Use scripts/setup/optimize_ml_env.py for the verified CPT recipe."
                        )
                        print(self.manifest["extension_status"][ext], flush=True)
                        self.save()
                        if explicit:
                            raise
                        continue
                    available = self.install(py, ext + "-source", [package], constraints=locked,
                        source=True, env=build_env, optional=not explicit)
            else:
                available = self.install(py, ext, [package], constraints=locked, optional=True)
                if not available and self.args.allow_source:
                    build_env = build_env or self.source_environment(py)
                    available = self.install(py, ext + "-source", [package], constraints=locked,
                        source=True, env=build_env, optional=False)
            if not available:
                self.manifest["extension_status"][ext] = "SKIPPED: no compatible binary resolution; see resolver log. SDPA baseline remains available."
                print(f"Optional extension {ext}: SKIPPED (no compatible binary resolution). SDPA remains available.", flush=True)
                self.save()
                if explicit:
                    raise SetupError(f"Requested extension {ext} has no compatible wheel. See the resolver log; --allow-source requires a matching full toolkit.")
                continue
            self.worker(py, f"{ext}-gpu-test", ext, timeout=900)
            self.manifest["extension_status"][ext] = "PASSED synthetic GPU tests; not automatically used by arbitrary models"
            locked = self.lock(py, "training")
            self.save()
        self.worker(py, "final-core-smoke", "core")
        self.worker(py, "compile-smoke", "compile", timeout=600)
        if self.plan["gpu_count"] > 1 and not self.args.skip_nccl:
            self.nccl(py, "training")
        elif self.plan["gpu_count"] > 1:
            self.manifest["stages"]["training_nccl"] = "SKIPPED by user; distributed readiness unverified"
        self.worker(py, "attention-benchmark", "attention-bench", timeout=900)
        self.lock(py, "training")
        self.manifest["stages"]["training"] = "PASSED"
        self.save()
        return py

    def nccl(self, py, name):
        output = self.root / "reports" / f"{name}-nccl.json"
        env = self.env.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        self.command(name + "-nccl", [py, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={self.plan['gpu_count']}", "--max_restarts=0",
            self.root / "bootstrap_ml.py", "--worker", "nccl", "--output", output], timeout=300, env=env)
        self.manifest["stages"][name + "_nccl"] = "PASSED small/16MiB all-reduce, all-gather, reduce-scatter"
        self.save()

    def inference(self):
        py = self.venv("inference")
        # vLLM chooses its own compatible Torch/runtime dependencies. Never force
        # the training environment's Torch into this separate serving environment.
        self.install(py, "vllm", ["vllm"])
        self.lock(py, "inference")
        self.worker(py, "inference-core", "core")
        self.worker(py, "vllm-kernel", "vllm")
        if self.plan["gpu_count"] > 1 and not self.args.skip_nccl:
            self.nccl(py, "inference")
        self.manifest["stages"]["inference"] = "PASSED CUDA + custom RMSNorm; full model serving not tested"
        self.save()


def worker(kind, output):
    """Executed ONLY in the target venv, never import Torch in bootstrap parent."""
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel
    torch.manual_seed(1234)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SetupError("Installed Torch cannot access CUDA.")
    result = {"kind": kind, "torch": torch.__version__, "cuda": torch.version.cuda}
    def native_dtype():
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    def finite(x):
        if x is None or not torch.isfinite(x).all().item():
            raise SetupError("Nonfinite/missing output or gradient.")
    def attention_case(fn, dtype):
        # Same square causal attention for all APIs; API layouts are normalized.
        xs = [torch.randn(2, 4, 128, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
        refs = [x.detach().float().requires_grad_() for x in xs]
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(*refs, is_causal=True, dropout_p=0.0)
        y = fn(*xs)
        grad = torch.randn_like(y)
        y.backward(grad)
        ref.backward(grad.float())
        finite(y)
        for a, b in zip([y] + [x.grad for x in xs], [ref] + [x.grad for x in refs]):
            finite(a)
            torch.testing.assert_close(a.float(), b, rtol=0.06, atol=0.04)
    if kind == "core":
        result.update({"architectures": torch.cuda.get_arch_list(),
                       "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
                       "cudnn": torch.backends.cudnn.version(), "devices": []})
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                p = torch.cuda.get_device_properties(i)
                dtype = native_dtype()
                x = torch.randn(128, 128, device="cuda", dtype=dtype, requires_grad=True)
                w = torch.randn_like(x, requires_grad=True)
                y = x @ w
                torch.testing.assert_close(y.float(), x.float() @ w.float(), rtol=0.06, atol=0.2)
                y.float().square().mean().backward()
                finite(x.grad)
                attention_case(lambda q,k,v: F.scaled_dot_product_attention(q,k,v,is_causal=True), dtype)
                result["devices"].append({"index": i, "name": p.name, "cc": [p.major, p.minor],
                    "memory_bytes": p.total_memory, "sm_count": p.multi_processor_count,
                    "bf16_native": dtype == torch.bfloat16, "dtype": str(dtype)})
                torch.cuda.synchronize()
        result["peer_access"] = [[None if i == j else torch.cuda.can_device_access_peer(i,j)
            for j in range(torch.cuda.device_count())] for i in range(torch.cuda.device_count())]
    elif kind == "hf":
        from transformers import LlamaConfig, LlamaForCausalLM
        import accelerate, datasets, peft, safetensors, sklearn, pandas, torchvision
        # Tiny random model, no Hub connection and no trust_remote_code.
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                cfg = LlamaConfig(vocab_size=128, hidden_size=128, intermediate_size=256,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=256, use_cache=False, bos_token_id=1, eos_token_id=2, pad_token_id=0)
                cfg._attn_implementation = "sdpa"
                model = LlamaForCausalLM(cfg).to(device="cuda", dtype=torch.float32).train()
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
                dtype = native_dtype()
                scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
                ids = torch.randint(3, 128, (2, 64), device="cuda")
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(input_ids=ids, labels=ids).loss
                finite(loss)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                    generated = model.generate(ids[:1,:8], max_new_tokens=2, do_sample=False, use_cache=True)
                assert generated.shape[1] > 8
                del model, opt
                torch.cuda.empty_cache()
        result["training_and_generation"] = "passed on all visible GPUs"
    elif kind == "compile":
        def fn(x):
            return torch.sin(x) * x + 1
        compiled = torch.compile(fn, fullgraph=True)
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                x = torch.randn(4096, device="cuda", requires_grad=True)
                y = compiled(x)
                torch.testing.assert_close(y, fn(x))
                y.sum().backward()
                torch.testing.assert_close(x.grad, torch.sin(x) + x * torch.cos(x), atol=1e-5, rtol=1e-4)
        result["inductor_forward_backward"] = "passed"
    elif kind in ("fa2", "fa4"):
        if kind == "fa2":
            from flash_attn import flash_attn_func
        else:
            from flash_attn.cute import flash_attn_func
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                def apply(q,k,v):
                    xs = [x.transpose(1,2).contiguous() for x in (q,k,v)]
                    y = flash_attn_func(*xs, causal=True)
                    if isinstance(y, tuple):
                        y = y[0]
                    return y.transpose(1,2)
                attention_case(apply, native_dtype())
        result["forward_backward_vs_fp32_reference"] = "passed; square causal head_dim=64 only"
    elif kind == "te":
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
        result["fp8"] = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                dtype = native_dtype()
                layer = te.Linear(128, 128, params_dtype=dtype).cuda()
                x = torch.randn(32,128,device="cuda",dtype=dtype,requires_grad=True)
                y = layer(x)
                y.float().square().mean().backward()
                finite(x.grad)
                cc = torch.cuda.get_device_capability(i)
                if cc >= (8,9):
                    layer.zero_grad(set_to_none=True)
                    x.grad = None
                    context = getattr(te, "autocast", None) or te.fp8_autocast
                    r = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
                    kw = {"enabled": True, "recipe": r} if hasattr(te,"autocast") else {"enabled": True,"fp8_recipe": r}
                    with context(**kw):
                        y = layer(x)
                        loss = y.float().square().mean()
                    loss.backward()
                    finite(x.grad)
                    result["fp8"].append({"device": i,"linear_forward_backward": "passed"})
        result["note"] = "Linear smoke only; no convergence/TE attention/FP4 certification"
    elif kind == "bnb":
        import bitsandbytes as bnb
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                layer = bnb.nn.Linear4bit(128,128,bias=False,compute_dtype=native_dtype(),quant_type="nf4").to("cuda")
                x = torch.randn(4,128,device="cuda",dtype=native_dtype(),requires_grad=True)
                y = layer(x)
                y.float().square().mean().backward()
                finite(x.grad)
        result["4bit_linear_backward"] = "passed"
    elif kind == "vllm":
        import vllm
        from vllm import _custom_ops as ops
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                x = torch.randn(8,128,device="cuda",dtype=native_dtype())
                w = torch.ones(128,device="cuda",dtype=x.dtype)
                y = torch.empty_like(x)
                ops.rms_norm(y,x,w,1e-6)
                ref = x.float() * torch.rsqrt(x.float().square().mean(-1,keepdim=True) + 1e-6)
                torch.testing.assert_close(y.float(),ref,rtol=0.03,atol=0.03)
        result["vllm"] = vllm.__version__
        result["note"] = "Custom CUDA RMSNorm passed; model-specific serving still requires validation"
    elif kind == "compile-extension":
        from torch.utils.cpp_extension import load_inline
        cuda = r'''#include <torch/extension.h>
        __global__ void add_one(float* x) { x[0] += 1.0f; }
        torch::Tensor probe(torch::Tensor x) { add_one<<<1,1>>>(x.data_ptr<float>()); return x; }
        '''
        ext = load_inline(name="ml_bootstrap_probe", cpp_sources="torch::Tensor probe(torch::Tensor x);",
            cuda_sources=cuda, functions=["probe"], verbose=True)
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                x = torch.zeros(1,device="cuda")
                torch.cuda.synchronize()
                y = ext.probe(x)
                torch.cuda.synchronize()
                assert y.item() == 1
        result["compiled_extension"] = "passed on all GPUs"
    elif kind == "nccl":
        import torch.distributed as dist
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=dt.timedelta(seconds=120))
        rank, size = dist.get_rank(), dist.get_world_size()
        result["world_size"] = size
        x = torch.full((1024,), float(rank+1), device="cuda")
        dist.all_reduce(x)
        assert torch.all(x == size*(size+1)/2).item()
        gathered = torch.empty(size*1024,device="cuda")
        dist.all_gather_into_tensor(gathered, torch.full((1024,),float(rank),device="cuda"))
        assert all(torch.all(gathered[j*1024:(j+1)*1024] == j).item() for j in range(size))
        reduced = torch.empty(1024,device="cuda")
        dist.reduce_scatter_tensor(reduced, torch.ones(size*1024,device="cuda"))
        assert torch.all(reduced == size).item()
        x = torch.ones(4*1024*1024,device="cuda")
        for _ in range(3):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        elapsed = torch.tensor((time.perf_counter()-start)/5,device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        result["allreduce_16MiB_seconds_max_rank"] = elapsed.item()
        result["nccl"] = torch.cuda.nccl.version()
        dist.destroy_process_group()
        if rank != 0:
            return
    elif kind == "attention-bench":
        result["measurements"] = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                for label, backend in [("auto", None), ("flash", SDPBackend.FLASH_ATTENTION),
                                       ("cudnn", SDPBackend.CUDNN_ATTENTION), ("math", SDPBackend.MATH)]:
                    entry = {"device": i,"backend": label,"shape_B_H_S_D": [1,8,512,64]}
                    try:
                        ctx = sdpa_kernel(backend) if backend is not None else nullcontext()
                        with ctx:
                            attention_case(lambda q,k,v:F.scaled_dot_product_attention(q,k,v,is_causal=True), native_dtype())
                            xs = [torch.randn(1,8,512,64,device="cuda",dtype=native_dtype(),requires_grad=True) for _ in range(3)]
                            def step():
                                for x in xs: x.grad = None
                                F.scaled_dot_product_attention(*xs,is_causal=True).float().square().mean().backward()
                            for _ in range(3): step()
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            t = time.perf_counter()
                            for _ in range(10): step()
                            torch.cuda.synchronize()
                            entry.update({"ms_forward_backward":(time.perf_counter()-t)*100,
                                "peak_allocated_bytes":torch.cuda.max_memory_allocated(), "status":"passed"})
                    except Exception as exc:
                        entry.update({"status":"unsupported_or_failed", "reason":str(exc)[:2000]})
                        if label == "auto":
                            raise
                    result["measurements"].append(entry)
        result["policy"] = "Keep SDPA auto dispatch. One synthetic shape is not sufficient to force a backend globally."
    else:
        raise SetupError(f"Unknown worker: {kind}")
    json_write(output, result)
    print(json.dumps(result, indent=2))


RUNTIME_TEMPLATE = '''#!/usr/bin/env bash
set -euo pipefail
ML_ENV_ROOT={root}
source "$ML_ENV_ROOT/{name}/bin/activate"
export PYTHONNOUSERSITE=1
export HF_HOME="$ML_ENV_ROOT/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$ML_ENV_ROOT/cache/{name}-extensions"
export TORCHINDUCTOR_CACHE_DIR="$ML_ENV_ROOT/cache/{name}-inductor"
export TRITON_CACHE_DIR="$ML_ENV_ROOT/cache/{name}-triton"
# No forced NCCL transport, CUDA architecture flag, allocator, CPU binding, or
# numerical precision. Those require the workload and job allocation.
if [ "$#" -eq 0 ]; then
  exec python
fi
exec "$@"
'''


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("ml-env"), help="New output directory; existing paths are never overwritten")
    p.add_argument("--profile", choices=["all","training"], default="all", help="all adds a separate vLLM environment")
    p.add_argument("--cuda", choices=["auto",*STACKS], default="auto")
    p.add_argument("--extensions", default="auto", help="auto (binary-only candidates), none, or fa2,fa4,te,bnb")
    p.add_argument("--allow-source", action="store_true", help="Allow official PyPI sdists for selected extensions with a matching installed toolkit")
    p.add_argument("--skip-nccl", action="store_true", help="Skip multi-GPU tests; results marked unverified")
    p.add_argument("--allow-busy", action="store_true", help="Permit small GPU tests while other compute processes are reported")
    p.add_argument("--dry-run", action="store_true", help="Read-only inventory and plan; no downloads, file writes or tensor tests")
    p.add_argument("--diagnose", action="store_true", help="Read-only full JSON inventory and plan")
    p.add_argument("--inventory-json", type=Path, help="Offline fixture input, allowed ONLY with --dry-run/--diagnose")
    p.add_argument("--worker", choices=["core","hf","compile","fa2","fa4","te","bnb","vllm","nccl","compile-extension","attention-bench"], help=argparse.SUPPRESS)
    p.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        if args.output is None:
            raise SetupError("Internal worker needs --output.")
        worker(args.worker,args.output)
        return 0
    if args.inventory_json and not (args.dry_run or args.diagnose):
        raise SetupError("Synthetic inventory is forbidden for real installation.")
    if platform.system() != "Linux" and not args.inventory_json:
        raise SetupError("Run this installer on Linux x86_64. Native Windows/macOS are outside this recipe.")
    if sys.version_info[:2] not in ((3,10),(3,11),(3,12),(3,13)):
        raise SetupError("Use Python 3.12 preferably, or 3.10–3.13 for this recipe. No system Python changes are made.")
    print("Detecting hardware and existing software (read-only)...", flush=True)
    info = json.loads(args.inventory_json.read_text()) if args.inventory_json else inventory()
    plan = installation_plan(info,args)
    print(json.dumps({"inventory":info,"plan":plan} if args.diagnose else plan, indent=2))
    if args.dry_run or args.diagnose:
        return 0
    # Existing optimized images and global restrictions remain intact.
    pip_env(args.root.absolute())  # preflight before creating output files
    root = args.root.expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise SetupError(f"{root} already exists. Choose a NEW --root. Nothing will be overwritten or deleted.")
    parent = root.parent
    if not parent.is_dir():
        raise SetupError("Output parent must already exist; choose a directory you own.")
    if len(str(root)) > 180:
        raise SetupError("Choose a shorter path (under 180 characters) for reliable venv command shebangs.")
    if importlib.util.find_spec("venv") is None or importlib.util.find_spec("ensurepip") is None:
        raise SetupError("Python venv/ensurepip is missing. On Ubuntu ask the administrator to install python3-venv (or your Python-version-specific venv package), then rerun. This script will not use sudo.")
    required_gib = 70 if args.profile == "all" else 30
    free_gib = shutil.disk_usage(parent).free / 1024**3
    if free_gib < required_gib:
        raise SetupError(f"Need at least {required_gib} GiB free for downloads/cache/envs; found {free_gib:.1f}. Use --profile training or a larger filesystem.")
    processes = info["commands"].get("processes",{})
    busy = processes.get("returncode") == 0 and bool(processes.get("output", "").strip())
    if busy and not args.allow_busy:
        raise SetupError("Existing GPU compute processes detected. Run on an idle allocation or explicitly use --allow-busy.")
    for gpu in info["devices"]:
        if gpu["memory_bytes"] < 2 * 1024**3:
            raise SetupError("Visible GPU/MIG allocation has under 2 GiB. This smoke-test recipe needs a larger allocation.")
    print(f"Creating {root}; downloading packages may take tens of minutes. No host packages/drivers will change.", flush=True)
    root.mkdir(mode=0o700)
    (root / "reports").mkdir()
    (root / "cache").mkdir()
    shutil.copy2(Path(__file__), root / "bootstrap_ml.py")
    json_write(root / "inventory.json",info)
    json_write(root / "plan.json",plan)
    json_write(root / "runtime-defaults.json", {
        "attention": "sdpa auto dispatch (set in model code)",
        "amp": "BF16 if native; FP16 + GradScaler otherwise",
        "tf32": "not forced; enable after precision validation",
        "fp8_fp4": "not forced; requires model/framework integration and accuracy validation",
        "distributed_world_size": plan["gpu_count"],
        "parallelism": "undecided: requires model sizes, optimizer, context, batch and topology benchmarks",
        "nccl": "automatic discovery; existing user/admin settings retained",
        "model_weights_downloaded": False,
    })
    install = Installer(root,info,plan,args)
    try:
        install.training()
        for name in ("training",):
            launcher = root / f"run-{name}.sh"
            launcher.write_text(RUNTIME_TEMPLATE.format(root=shlex.quote(str(root)),name=name))
            launcher.chmod(0o700)
        if args.profile == "all":
            try:
                install.inference()
                launcher = root / "run-inference.sh"
                launcher.write_text(RUNTIME_TEMPLATE.format(root=shlex.quote(str(root)),name="inference"))
                launcher.chmod(0o700)
            except Exception as exc:
                install.manifest["stages"]["inference"] = "FAILED: " + str(exc)
                install.manifest["status"] = "PARTIAL: training passed, inference failed"
                install.save()
                print(f"Training venv is available. Inference failed: {exc}\nSee {root / 'manifest.json'}", file=sys.stderr)
                return 2
        install.manifest["status"] = "PASSED_WITH_SKIPPED_DISTRIBUTED_TESTS" if args.skip_nccl and plan["gpu_count"] > 1 else "PASSED_SMOKE_TESTS"
        install.manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        install.manifest["installer_sha256"] = hashlib.sha256((root / "bootstrap_ml.py").read_bytes()).hexdigest()
        install.save()
        print(f"\n{install.manifest['status']}\nActivate: source {shlex.quote(str(root / 'training/bin/activate'))}\nReport: {root / 'manifest.json'}\nExtensions: {json.dumps(install.manifest['extension_status'])}\nNot a certification of model convergence or maximum throughput.")
        return 0
    except BaseException as exc:
        install.manifest["status"] = "FAILED_OR_INTERRUPTED"
        install.manifest["error"] = str(exc)
        install.save()
        print(f"Partial files preserved at {root}. Inspect logs; use a NEW --root for retry.",file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SetupError, OSError) as exc:
        print(f"\nSTOP: {exc}",file=sys.stderr)
        sys.exit(1)
