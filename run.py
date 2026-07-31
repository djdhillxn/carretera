"""Command-line workflows for training, evaluation, analysis, rendering, and I/O."""

import argparse
import atexit
import collections
import copy
import csv
import datetime
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import warnings
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

from carla_env import (
    ACTION_NAMES,
    CHANGE_LEFT,
    CHANGE_RIGHT,
    HighwayDecisionEnv,
    calculate_reward_components,
)
from policies import (
    KeepLanePolicy,
    RandomPolicy,
    RuleBasedOvertakingPolicy,
    build_policy,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
REQUIRED_TOP_LEVEL_KEYS = [
    "project",
    "paths",
    "carla",
    "environment",
    "controller",
    "lane_change",
    "traffic",
    "reward",
    "ppo",
    "evaluation",
    "video",
    "rule_based",
]
ENVIRONMENT_OVERRIDES = {
    "CARLA_ARCHIVE_URL": ("carla", "package", "download_url", str),
    "CARLA_ARCHIVE_DRIVE": ("carla", "package", "drive_archive", str),
    "CARLA_ARCHIVE_LOCAL": ("carla", "package", "local_archive", str),
    "CARLA_ARCHIVE_METADATA": ("carla", "package", "drive_metadata", str),
    "CARLA_HOST": ("carla", "host", str),
    "CARLA_PORT": ("carla", "port", int),
    "CARLA_TM_PORT": ("carla", "traffic_manager_port", int),
    "CARLA_ROOT": ("carla", "package", "local_root", str),
    "CARLA_CACHE_DIR": ("carla", "package", "local_cache_root", str),
    "CARLA_SERVER_MODE": ("carla", "server", "mode", str),
    "HIGHWAY_RL_ARTIFACT_ROOT": ("paths", "artifact_root", str),
    "HIGHWAY_RL_DRIVE_ROOT": ("paths", "drive_root", str),
}
NETWORK_ACQUISITION_CALLS = 0
PREPARE_STATE = {"stage": "platform_validation"}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resolve_path(value, base=REPOSITORY_ROOT):
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path):
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError("Config file does not exist: %s" % config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a YAML mapping.")
    for environment_name, target in ENVIRONMENT_OVERRIDES.items():
        value = os.environ.get(environment_name)
        if value in (None, ""):
            continue
        converter = target[-1]
        keys = target[:-1]
        destination = config
        for key in keys[:-1]:
            destination = destination[key]
        destination[keys[-1]] = converter(value)
    config["_config_path"] = str(config_path)
    config["paths"]["artifact_root"] = str(
        resolve_path(config["paths"]["artifact_root"])
    )
    config["paths"]["drive_root"] = str(
        resolve_path(config["paths"]["drive_root"])
    )
    package = config["carla"]["package"]
    for key in (
        "local_archive",
        "local_root",
        "extraction_staging_root",
        "local_cache_root",
    ):
        package[key] = str(resolve_path(package[key]))
    if os.environ.get("CARLA_ROOT"):
        config["carla"]["server"]["root"] = package["local_root"]
    if package.get("drive_archive"):
        package["drive_archive"] = str(resolve_path(package["drive_archive"]))
    else:
        package["drive_archive"] = str(
            resolve_path(
                Path(config["paths"]["drive_root"])
                / package["drive_cache_subdirectory"]
                / package["archive_name"]
            )
        )
    if package.get("drive_metadata"):
        package["drive_metadata"] = str(resolve_path(package["drive_metadata"]))
    else:
        package["drive_metadata"] = str(
            Path(package["drive_archive"]).with_name(package["metadata_name"])
        )
    config["carla"]["server"]["root"] = str(
        resolve_path(config["carla"]["server"]["root"])
    )
    validate_config_data(config)
    return config


def validate_config_data(config):
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError("Missing required config sections: %s" % ", ".join(missing))
    if str(config["carla"]["version"]) != "0.9.16":
        raise ValueError("The project scope requires CARLA 0.9.16.")
    if "package" not in config["carla"]:
        raise ValueError("Missing carla.package configuration.")
    if config["carla"]["map"] != "Town04":
        raise ValueError("The project scope requires Town04.")
    if config["environment"]["action_repeat_ticks"] != 20:
        raise ValueError("action_repeat_ticks must remain 20.")
    simulated_interval = (
        config["carla"]["fixed_delta_seconds"]
        * config["environment"]["action_repeat_ticks"]
    )
    if not math.isclose(
        simulated_interval,
        config["environment"]["decision_interval_seconds"],
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "fixed_delta_seconds multiplied by action_repeat_ticks must equal "
            "decision_interval_seconds."
        )
    if config["ppo"]["policy"] != "MlpPolicy":
        raise ValueError("Only MlpPolicy is in project scope.")
    if config["carla"]["server"]["mode"] not in ("external", "managed"):
        raise ValueError("carla.server.mode must be external or managed.")
    if config["ppo"]["net_arch"] != [128, 128]:
        raise ValueError("The required PPO network is [128, 128].")
    if config["evaluation"]["policies"] != [
        "ppo",
        "random",
        "keep_lane",
        "rule_based",
    ]:
        raise ValueError("Evaluation must contain PPO and exactly three baselines.")
    if set(ACTION_NAMES) != set(range(5)):
        raise ValueError("The action mapping must contain integer actions 0 through 4.")
    return True


def config_hash(config):
    payload = copy.deepcopy(config)
    payload.pop("_config_path", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_artifact_layout(config):
    root = Path(config["paths"]["artifact_root"])
    directories = [
        "manifests",
        "models",
        "logs/train",
        "logs/tensorboard",
        "logs/runtime",
        "evaluations",
        "plots",
        "videos",
        "recordings",
        "frames",
    ]
    for directory in directories:
        (root / directory).mkdir(parents=True, exist_ok=True)
    return root


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_resolved_config(config, path):
    payload = copy.deepcopy(config)
    payload.pop("_config_path", None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def package_versions():
    packages = [
        "carla",
        "numpy",
        "gymnasium",
        "stable-baselines3",
        "pandas",
        "matplotlib",
        "PyYAML",
        "tqdm",
        "tensorboard",
        "opencv-python-headless",
        "torch",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def git_metadata():
    metadata = {"commit": "unavailable", "dirty": "unavailable"}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        metadata["commit"] = result.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        metadata["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        metadata["error"] = str(exc)
    return metadata


def machine_metadata():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "hostname": socket.gethostname(),
    }


def linux_boot_id():
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def os_release_information():
    path = Path("/etc/os-release")
    values = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    except OSError as exc:
        values["error"] = str(exc)
    return values


def pip_version():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return "unavailable: %s" % exc


def directory_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
    except OSError:
        return None
    return total


def nearest_existing_parent(path):
    current = Path(path)
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def disk_free_gb(path):
    parent = nearest_existing_parent(path)
    return round(shutil.disk_usage(parent).free / (1024 ** 3), 2)


def gpu_status():
    status = {
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "devices": [],
        "gpu_name": "",
        "vram": "",
        "driver_version": "",
        "pytorch_cuda_available": False,
        "pytorch_visible_devices": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if status["nvidia_smi_available"]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            for line in result.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) >= 3:
                    status["devices"].append(
                        {
                            "name": fields[0],
                            "memory_total_mib": fields[1],
                            "driver_version": fields[2],
                        }
                    )
            if status["devices"]:
                first = status["devices"][0]
                status["gpu_name"] = first["name"]
                status["vram"] = "%s MiB" % first["memory_total_mib"]
                status["driver_version"] = first["driver_version"]
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            status["nvidia_smi_error"] = str(exc)
    try:
        import torch

        status["pytorch_cuda_available"] = bool(torch.cuda.is_available())
        status["pytorch_visible_devices"] = int(torch.cuda.device_count())
    except (ImportError, RuntimeError) as exc:
        status["pytorch_error"] = str(exc)
    status["compatible"] = bool(
        status["devices"] and status["pytorch_cuda_available"]
    )
    return status


def graphics_environment():
    import glob
    env = os.environ.copy()
    try:
        uid = os.getuid()
    except AttributeError:
        uid = 0
    runtime_dir = Path("/tmp/runtime-%s" % uid)
    try:
        runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(runtime_dir, 0o700)
    except OSError:
        pass
    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    icd_dir = Path("/usr/share/vulkan/icd.d")
    nvidia_icds = []
    if icd_dir.is_dir():
        nvidia_icds = glob.glob(str(icd_dir / "*nvidia*.json"))
    diagnostics = {
        "uid": uid,
        "runtime_dir": str(runtime_dir),
        "nvidia_icds": nvidia_icds,
        "vk_icd_filenames_set": False,
        "vk_icd_filenames_value": "",
    }
    if len(nvidia_icds) == 1:
        env["VK_ICD_FILENAMES"] = nvidia_icds[0]
        diagnostics["vk_icd_filenames_set"] = True
        diagnostics["vk_icd_filenames_value"] = nvidia_icds[0]
    return env, diagnostics


def vulkan_status():
    executable = shutil.which("vulkaninfo")
    status = {
        "vulkaninfo_available": executable is not None,
        "exit_code": None,
        "physical_devices": [],
        "device_names": [],
        "nvidia_device_present": False,
        "software_renderer_present": False,
        "stderr": "",
        "compatible": False,
    }
    if not executable:
        return status
    try:
        run_env, diagnostics = graphics_environment()
        result = subprocess.run(
            [executable, "--summary"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=run_env,
        )
        status["exit_code"] = result.returncode
        combined = result.stdout + "\n" + result.stderr
        for line in combined.splitlines():
            stripped = line.strip()
            if re.search(r"deviceName\s*=", stripped, flags=re.IGNORECASE):
                name = stripped.split("=", 1)[1].strip()
                if name and name not in status["device_names"]:
                    status["device_names"].append(name)
            if re.match(r"GPU\d+:", stripped):
                name = stripped.split(":", 1)[1].strip()
                if name and name not in status["device_names"]:
                    status["device_names"].append(name)
        status["physical_devices"] = [
            {"name": name} for name in status["device_names"]
        ]
        lowered = combined.lower()
        software_names = ("llvmpipe", "lavapipe", "software rasterizer", "swiftshader")
        status["software_renderer_present"] = any(
            name in lowered for name in software_names
        )
        status["nvidia_device_present"] = any(
            "nvidia" in name.lower() for name in status["device_names"]
        )
        status["stderr"] = result.stderr[-4000:]
        status["compatible"] = bool(
            result.returncode == 0
            and status["nvidia_device_present"]
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        status["stderr"] = str(exc)
    return status


def gpu_vulkan_error(gpu, vulkan):
    if not gpu["devices"]:
        return (
            "No NVIDIA GPU is assigned. In Colab, select Runtime \u2192 Change "
            "runtime type \u2192 GPU, reconnect, and rerun initialization."
        )
    if not vulkan["vulkaninfo_available"]:
        return (
            "NVIDIA is visible, but vulkaninfo is unavailable. Run the notebook "
            "system-dependencies cell, then rerun runtime status."
        )
    if not vulkan["compatible"]:
        return (
            "NVIDIA is visible but no working NVIDIA Vulkan device was detected. "
            "A software Vulkan renderer is not supported; reconnect to a new GPU "
            "runtime and rerun the runtime status cell."
        )
    return ""


def python_tag(version_info=None):
    info = version_info or sys.version_info
    return "cp%s%s" % (info.major, info.minor)


def discover_wheels(root):
    root = Path(root)
    distribution = root / "PythonAPI/carla/dist"
    if not distribution.is_dir():
        return []
    return sorted(distribution.glob("*.whl"))


def select_carla_wheel(candidates, tag=None, version="0.9.16"):
    tag = tag or python_tag()
    names = [Path(candidate).name for candidate in candidates]
    matches = []
    for candidate in candidates:
        name = Path(candidate).name
        lowered = name.lower()
        version_match = (
            ("carla-%s" % version).lower() in lowered
            or ("carla_%s" % version).lower() in lowered
        )
        interpreter_match = re.search(
            r"-%s(?:-%s)?-" % (re.escape(tag), re.escape(tag)), lowered
        )
        linux_match = (
            "linux" in lowered
            and ("x86_64" in lowered or "amd64" in lowered)
        )
        if version_match and interpreter_match and linux_match:
            matches.append(Path(candidate))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one CARLA %s Linux x86_64 wheel for %s; found %s. "
            "Discovered wheels: %s"
            % (version, tag, len(matches), ", ".join(names) or "(none)")
        )
    return matches[0]


def archive_member_is_safe(member):
    name = member.name.replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    if member.issym() or member.islnk():
        link = PurePosixPath(member.linkname.replace("\\", "/"))
        if link.is_absolute() or ".." in link.parts:
            return False
    return True


def archive_root_from_names(names):
    normalized = [PurePosixPath(name.replace("\\", "/")) for name in names]
    roots = set()
    for path in normalized:
        if path.name != "CarlaUE4.sh":
            continue
        root = path.parent
        prefix = "" if str(root) == "." else str(root).rstrip("/") + "/"
        if any(
            str(item).startswith(prefix + "PythonAPI/carla/dist/")
            for item in normalized
        ):
            roots.add(str(root))
    if len(roots) != 1:
        raise RuntimeError(
            "Archive must contain exactly one packaged CARLA root; found %s."
            % len(roots)
        )
    return next(iter(roots))


def inspect_carla_archive(path, version="0.9.16", allow_small=False):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("CARLA archive is missing or empty: %s" % path)
    with path.open("rb") as handle:
        header = handle.read(512).lower()
    if b"<html" in header or b"<!doctype html" in header:
        raise RuntimeError("Downloaded file is HTML, not a CARLA archive: %s" % path)
    byte_size = path.stat().st_size
    minimum_bytes = 100 * 1024 * 1024
    if not allow_small and byte_size < minimum_bytes:
        raise RuntimeError(
            "CARLA archive is implausibly small (%s bytes): %s" % (byte_size, path)
        )
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError("CARLA archive is not a readable gzip tar: %s" % exc) from exc

    names = [m.name for m in members]
    normalized_names = [str(PurePosixPath(name.replace("\\", "/"))) for name in names]

    roots = set()
    for p in normalized_names:
        ppath = PurePosixPath(p)
        if ppath.name.lower() == "carlaue4.sh":
            root = str(ppath.parent)
            prefix = "" if root == "." else root.rstrip("/") + "/"
            if any(item.lower().startswith(prefix.lower() + "pythonapi/carla/dist/") for item in normalized_names):
                roots.add(root)
    root_candidates = sorted(list(roots))
    archive_root = root_candidates[0] if len(root_candidates) == 1 else (archive_root_from_names(normalized_names) if normalized_names else ".")
    root_prefix = "" if archive_root == "." else archive_root.rstrip("/") + "/"

    carla_sh_target = (root_prefix + "CarlaUE4.sh").lower()
    carla_shipping_target = (root_prefix + "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping").lower()

    carla_sh_present = any(name.lower().rstrip("/") == carla_sh_target for name in normalized_names)
    carla_shipping_present = any(name.lower().rstrip("/") == carla_shipping_target for name in normalized_names)

    wheel_paths = [
        name for name in normalized_names
        if name.lower().endswith(".whl") and "pythonapi/carla/dist/" in name.lower()
    ]

    curr_tag = python_tag()
    matching_wheels = [
        name for name in wheel_paths
        if version in name
        and "linux" in name.lower()
        and ("x86_64" in name.lower() or "amd64" in name.lower())
        and re.search(r"-%s(?:-%s)?-" % (re.escape(curr_tag), re.escape(curr_tag)), name.lower())
    ]

    # Content/Paks evaluation
    paks_target_prefix = (root_prefix + "CarlaUE4/Content/Paks/").lower()
    paks_members = []
    paks_member_names = []
    for m, norm in zip(members, normalized_names):
        norm_low = norm.lower()
        if norm_low.startswith(paks_target_prefix) or norm_low == paks_target_prefix.rstrip("/"):
            paks_members.append(m)
            paks_member_names.append(norm)

    content_paks_exists = len(paks_members) > 0
    content_paks_member_count = len(paks_members)

    extension_counts = collections.defaultdict(int)
    container_file_sizes = {}
    pak_paths = []
    utoc_paths = []
    ucas_paths = []

    for m, norm in zip(paks_members, paks_member_names):
        ext = Path(norm).suffix.lower()
        extension_counts[ext or "(no_ext)"] += 1
        if m.isfile() and m.size > 0:
            if ext == ".pak":
                pak_paths.append(norm)
                container_file_sizes[norm] = m.size
            elif ext == ".utoc":
                utoc_paths.append(norm)
                container_file_sizes[norm] = m.size
            elif ext == ".ucas":
                ucas_paths.append(norm)
                container_file_sizes[norm] = m.size

    utoc_stems = {Path(p).stem.lower(): p for p in utoc_paths}
    ucas_stems = {Path(p).stem.lower(): p for p in ucas_paths}
    matching_utoc_ucas_stems = sorted(list(set(utoc_stems.keys()).intersection(set(ucas_stems.keys()))))
    unmatched_utoc_stems = sorted(list(set(utoc_stems.keys()) - set(ucas_stems.keys())))
    unmatched_ucas_stems = sorted(list(set(ucas_stems.keys()) - set(utoc_stems.keys())))
    sample_content_paks_paths = paks_member_names[:100]

    # CarlaUE4/Content evaluation
    content_target_prefix = (root_prefix + "CarlaUE4/Content/").lower()
    asset_registry_target = (root_prefix + "CarlaUE4/AssetRegistry.bin").lower()

    content_root_members = []
    content_root_member_names = []
    for m, norm in zip(members, normalized_names):
        norm_low = norm.lower()
        if norm_low.startswith(content_target_prefix) or norm_low == content_target_prefix.rstrip("/"):
            content_root_members.append(m)
            content_root_member_names.append(norm)

    content_root_exists = len(content_root_members) > 0
    content_root_member_count = len(content_root_members)

    content_root_extension_counts = collections.defaultdict(int)
    loose_uasset_paths = []
    loose_umap_paths = []
    loose_uexp_paths = []
    loose_ubulk_paths = []
    loose_asset_total_bytes = 0

    for m, norm in zip(content_root_members, content_root_member_names):
        ext = Path(norm).suffix.lower()
        if m.isfile() and m.size > 0:
            content_root_extension_counts[ext or "(no_ext)"] += 1
            if ext == ".uasset":
                loose_uasset_paths.append(norm)
                loose_asset_total_bytes += m.size
            elif ext == ".umap":
                loose_umap_paths.append(norm)
                loose_asset_total_bytes += m.size
            elif ext == ".uexp":
                loose_uexp_paths.append(norm)
                loose_asset_total_bytes += m.size
            elif ext == ".ubulk":
                loose_ubulk_paths.append(norm)
                loose_asset_total_bytes += m.size
        elif m.isdir():
            content_root_extension_counts["(dir)"] += 1

    sample_content_root_paths = content_root_member_names[:100]

    asset_registry_present = False
    asset_registry_path = ""
    asset_registry_size = 0
    for m, norm in zip(members, normalized_names):
        if norm.lower() == asset_registry_target and m.isfile() and m.size > 0:
            asset_registry_present = True
            asset_registry_path = norm
            asset_registry_size = m.size
            break

    loose_primary_asset_count = len(loose_uasset_paths) + len(loose_umap_paths)
    loose_companion_asset_count = len(loose_uexp_paths) + len(loose_ubulk_paths)

    layout_active = []
    if content_paks_exists and len(pak_paths) >= 1:
        layout_active.append("classic_pak")
    if content_paks_exists and len(utoc_paths) >= 1 and len(ucas_paths) >= 1 and len(matching_utoc_ucas_stems) >= 1:
        layout_active.append("iostore")
    if content_root_exists and asset_registry_present and loose_primary_asset_count >= 1:
        layout_active.append("loose_cooked")

    detected_asset_layout = ", ".join(layout_active) if layout_active else "none"

    unsafe_paths = [m.name for m in members if not archive_member_is_safe(m)]
    town04_archive_evidence = any("town04" in norm.lower() for norm in normalized_names)

    return {
        "archive_path": str(path),
        "byte_size": byte_size,
        "member_count": len(names),
        "archive_root": archive_root,
        "root_candidates": root_candidates,
        "carla_sh_present": carla_sh_present,
        "carla_shipping_present": carla_shipping_present,
        "wheel_paths": wheel_paths,
        "matching_wheels": matching_wheels,
        "content_paks_exists": content_paks_exists,
        "content_paks_member_count": content_paks_member_count,
        "content_paks_extension_counts": dict(extension_counts),
        "sample_content_paks_paths": sample_content_paks_paths,
        "pak_paths": pak_paths,
        "utoc_paths": utoc_paths,
        "ucas_paths": ucas_paths,
        "matching_utoc_ucas_stems": matching_utoc_ucas_stems,
        "unmatched_utoc_stems": unmatched_utoc_stems,
        "unmatched_ucas_stems": unmatched_ucas_stems,
        "container_file_sizes": container_file_sizes,
        "content_root_exists": content_root_exists,
        "content_root_member_count": content_root_member_count,
        "content_root_extension_counts": dict(content_root_extension_counts),
        "sample_content_root_paths": sample_content_root_paths,
        "asset_registry_present": asset_registry_present,
        "asset_registry_path": asset_registry_path,
        "asset_registry_size": asset_registry_size,
        "loose_uasset_paths": loose_uasset_paths,
        "loose_umap_paths": loose_umap_paths,
        "loose_uexp_paths": loose_uexp_paths,
        "loose_ubulk_paths": loose_ubulk_paths,
        "loose_primary_asset_count": loose_primary_asset_count,
        "loose_companion_asset_count": loose_companion_asset_count,
        "loose_asset_total_bytes": loose_asset_total_bytes,
        "detected_asset_layout": detected_asset_layout,
        "unsafe_paths": unsafe_paths,
        "town04_archive_evidence": town04_archive_evidence,
        "inspection_timestamp": utc_now(),
    }


def validate_carla_archive(path, version="0.9.16", allow_small=False, expected_sha256=None, artifact_root=None):
    path = Path(path).resolve()
    inventory = inspect_carla_archive(path, version=version, allow_small=allow_small)

    invariants_failed = []
    if inventory["unsafe_paths"]:
        invariants_failed.append("CARLA archive contains unsafe member paths: %s" % ", ".join(inventory["unsafe_paths"][:5]))
    if len(inventory["root_candidates"]) != 1 and inventory["archive_root"] == ".":
        invariants_failed.append("Archive does not contain exactly one coherent CARLA root.")
    if not inventory["carla_sh_present"]:
        invariants_failed.append("Archive is missing CarlaUE4.sh.")
    if not inventory["carla_shipping_present"]:
        invariants_failed.append("Archive is missing CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping.")
    if not inventory["wheel_paths"]:
        invariants_failed.append("Archive is missing PythonAPI/carla/dist directory or wheels.")
    if not inventory["matching_wheels"]:
        invariants_failed.append("Archive contains no CARLA %s Linux x86_64 wheel matching %s." % (version, python_tag()))

    if inventory["detected_asset_layout"] == "none":
        invariants_failed.append(
            "Archive has no recognized CARLA cooked-asset layout.\n"
            "Expected one of:\n"
            "1. nonempty Classic Pak files under CarlaUE4/Content/Paks;\n"
            "2. matching nonempty IoStore .utoc/.ucas files under that directory; or\n"
            "3. a nonempty CarlaUE4/AssetRegistry.bin plus loose nonempty .uasset or .umap files under CarlaUE4/Content."
        )

    calculated_hash = ""
    if not invariants_failed:
        calculated_hash = file_sha256(path)
        if expected_sha256:
            if calculated_hash.lower() != expected_sha256.lower():
                invariants_failed.append(
                    "Archive SHA-256 checksum mismatch: expected %s, got %s."
                    % (expected_sha256, calculated_hash)
                )

    if artifact_root is None:
        try:
            artifact_root = REPOSITORY_ROOT / "artifacts"
        except Exception:
            artifact_root = Path("artifacts")
    inv_path = artifact_root / "logs/runtime/carla_archive_inventory.json"
    val_path = artifact_root / "logs/runtime/carla_archive_validation.json"

    write_json(inv_path, inventory)

    if invariants_failed:
        validation_result = {
            "valid": False,
            "archive": str(path),
            "byte_size": inventory["byte_size"],
            "member_count": inventory["member_count"],
            "archive_root": inventory["archive_root"],
            "asset_registry_present": inventory["asset_registry_present"],
            "asset_registry_size": inventory["asset_registry_size"],
            "content_root_member_count": inventory["content_root_member_count"],
            "content_root_extension_counts": inventory["content_root_extension_counts"],
            "content_paks_member_count": inventory["content_paks_member_count"],
            "content_paks_extension_counts": inventory["content_paks_extension_counts"],
            "sample_content_root_paths": inventory["sample_content_root_paths"],
            "sample_content_paks_paths": inventory["sample_content_paks_paths"],
            "invariants_failed": invariants_failed,
            "inventory_path": str(inv_path),
            "validated_at": utc_now(),
        }
        write_json(val_path, validation_result)
        raise RuntimeError(
            "CARLA archive validation failed:\n- %s\n"
            "Archive size: %s bytes | Root: %s | Total members: %s\n"
            "AssetRegistry present: %s (size: %s bytes)\n"
            "Content members: %s | Content extensions: %s\n"
            "Content/Paks members: %s | Content/Paks extensions: %s\n"
            "Sample Content paths: %s\n"
            "Sample Content/Paks paths: %s\n"
            "Detailed inventory written to: %s"
            % (
                "\n- ".join(invariants_failed),
                inventory["byte_size"],
                inventory["archive_root"],
                inventory["member_count"],
                inventory["asset_registry_present"],
                inventory["asset_registry_size"],
                inventory["content_root_member_count"],
                inventory["content_root_extension_counts"],
                inventory["content_paks_member_count"],
                inventory["content_paks_extension_counts"],
                inventory["sample_content_root_paths"][:3],
                inventory["sample_content_paks_paths"][:3],
                inv_path,
            )
        )

    target_path = path
    if str(path).endswith(".part"):
        target_path = Path(str(path)[:-5])
        path.replace(target_path)

    validation_result = {
        "valid": True,
        "archive": str(target_path),
        "byte_size": inventory["byte_size"],
        "sha256": calculated_hash,
        "member_count": inventory["member_count"],
        "archive_root": inventory["archive_root"],
        "detected_asset_layout": inventory["detected_asset_layout"],
        "wheels": [Path(name).name for name in inventory["matching_wheels"]],
        "town04_archive_evidence": inventory["town04_archive_evidence"],
        "expected_sha256_configured": bool(expected_sha256),
        "validated_at": utc_now(),
    }
    write_json(val_path, validation_result)
    return validation_result


def command_runtime_inspect_archive(args):
    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    archive_path = resolve_path(args.archive)
    inventory = inspect_carla_archive(
        archive_path,
        version=config["carla"]["version"],
        allow_small=args.allow_small,
    )
    out_path = Path(args.json_output) if args.json_output else (artifact_root / "logs/runtime/carla_archive_inventory.json")
    write_json(out_path, inventory)
    print(json.dumps(inventory, indent=2))
    print("Archive inspection complete. Detailed inventory written to:", out_path)


def discover_carla_root(extraction_root):
    candidates = []
    for executable in Path(extraction_root).rglob("CarlaUE4.sh"):
        root = executable.parent
        if (root / "PythonAPI/carla/dist").is_dir():
            candidates.append(root)
    unique = sorted(set(path.resolve() for path in candidates))
    if len(unique) != 1:
        raise RuntimeError(
            "Expected one extracted CARLA package root; found %s: %s"
            % (len(unique), ", ".join(str(path) for path in unique) or "(none)")
        )
    return unique[0]


def safe_extract_archive(archive_path, staging_root):
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        unsafe = [member.name for member in members if not archive_member_is_safe(member)]
        if unsafe:
            raise RuntimeError(
                "Refusing to extract unsafe archive members: %s"
                % ", ".join(unsafe[:5])
            )
        archive.extractall(staging_root)


def managed_local_paths(config):
    package = config["carla"]["package"]
    local_archive = Path(package["local_archive"]).resolve()
    return {
        local_archive,
        local_archive.with_name(local_archive.name + ".part"),
        local_archive.with_name(local_archive.name + ".from-drive.part"),
        Path(package["local_root"]).resolve(),
        Path(package["extraction_staging_root"]).resolve(),
        Path(package["local_cache_root"]).resolve(),
    }


def safe_managed_path(config, path):
    path = Path(path).resolve()
    forbidden = {
        Path("/").resolve(),
        Path("/content").resolve(),
        REPOSITORY_ROOT.resolve(),
        Path(config["paths"]["drive_root"]).resolve(),
    }
    if path in forbidden or path not in managed_local_paths(config):
        return False
    package = config["carla"]["package"]
    version_token = str(config["carla"]["version"])
    return bool(
        version_token in path.name
        or path == Path(package["local_cache_root"]).resolve()
    )


def remove_managed_path(config, path, dry_run=False):
    path = Path(path).resolve()
    if not safe_managed_path(config, path):
        raise ValueError("Refusing to remove unrecognized managed path: %s" % path)
    if dry_run or not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def read_json_if_valid(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def package_manifest_path(config):
    return Path(config["carla"]["package"]["local_root"]) / (
        ".carretera_runtime_manifest.json"
    )


def runtime_manifest_matches(config, archive_sha256=None):
    package = config["carla"]["package"]
    root = Path(package["local_root"])
    manifest = read_json_if_valid(package_manifest_path(config))
    if not manifest:
        return False
    matches = bool(
        manifest.get("carla_version") == str(config["carla"]["version"])
        and Path(manifest.get("package_root", "")).resolve() == root.resolve()
        and (root / "CarlaUE4.sh").is_file()
        and (root / "PythonAPI/carla/dist").is_dir()
    )
    if archive_sha256 is not None:
        matches = matches and manifest.get("archive_sha256") == archive_sha256
    return matches


def server_record_state(config, record=None):
    record = record if record is not None else read_server_record(config)
    if not record:
        return {"state": "absent", "active": False, "stale": False}
    hostname_match = record.get("hostname") == socket.gethostname()
    boot_match = record.get("boot_id") == linux_boot_id()
    if not hostname_match or not boot_match:
        return {
            "state": "stale_runtime_identity",
            "active": False,
            "stale": True,
            "hostname_match": hostname_match,
            "boot_id_match": boot_match,
        }
    active = server_record_matches_process(record)
    return {
        "state": "active" if active else "inactive_or_identity_mismatch",
        "active": active,
        "stale": False,
        "hostname_match": True,
        "boot_id_match": True,
    }


def python_client_status():
    status = {"distribution_version": "not installed", "import_success": False}
    try:
        status["distribution_version"] = importlib.metadata.version("carla")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        import carla

        status["import_success"] = True
        status["module"] = str(getattr(carla, "__file__", "unknown"))
    except Exception as exc:
        status["import_error"] = str(exc)
    return status


def build_runtime_status(config, verify_archive_hash=False, check_server=False):
    package = config["carla"]["package"]
    local_archive = Path(package["local_archive"])
    drive_archive = Path(package["drive_archive"])
    metadata_path = Path(package["drive_metadata"])
    local_root = Path(package["local_root"])
    executable = local_root / "CarlaUE4.sh"
    local_metadata_path = (
        Path(config["paths"]["artifact_root"])
        / "logs/runtime/CARLA_0.9.16.local.metadata.json"
    )
    drive_metadata = read_json_if_valid(metadata_path)
    local_metadata = read_json_if_valid(local_metadata_path)
    metadata = drive_metadata or local_metadata
    drive_cache_valid, drive_cache_reason = validate_drive_metadata(
        drive_metadata, drive_archive, config["carla"]["version"]
    )
    package_manifest = read_json_if_valid(package_manifest_path(config))
    wheels = discover_wheels(local_root)
    matching_wheel = ""
    wheel_error = ""
    try:
        matching_wheel = str(select_carla_wheel(wheels))
    except RuntimeError as exc:
        wheel_error = str(exc)
    local_hash_matches = None
    local_hash = ""
    if local_archive.is_file() and metadata:
        if verify_archive_hash:
            local_hash = file_sha256(local_archive)
            local_hash_matches = local_hash == metadata.get("sha256")
        elif (
            local_metadata
            and local_metadata.get("local_sha256")
            and local_archive.stat().st_size
            == int(local_metadata.get("byte_size", -1))
            and local_archive.stat().st_mtime_ns
            == int(local_metadata.get("local_mtime_ns", -1))
        ):
            local_hash = local_metadata["local_sha256"]
            local_hash_matches = local_hash == metadata.get("sha256")
    gpu = gpu_status()
    vulkan = vulkan_status()
    drive_mounted = Path("/content/drive/MyDrive").is_dir()
    record = read_server_record(config)
    status = {
        "runtime": {
            "timestamp_utc": utc_now(),
            "hostname": socket.gethostname(),
            "boot_id": linux_boot_id(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "os_release": os_release_information(),
            "python": sys.version,
            "pip": pip_version(),
            "appears_to_be_colab": bool(
                os.environ.get("COLAB_RELEASE_TAG")
                or os.environ.get("COLAB_GPU")
                or Path("/content").exists()
                and Path("/usr/local/lib/python3.").parent.exists()
            ),
            "content_exists": Path("/content").exists(),
            "google_drive_mounted": drive_mounted,
        },
        "gpu": gpu,
        "vulkan": vulkan,
        "storage": {
            "local_disk_probe": str(
                Path("/content") if Path("/content").exists() else nearest_existing_parent(local_archive)
            ),
            "local_free_gb": disk_free_gb(
                Path("/content") if Path("/content").exists() else local_archive
            ),
            "minimum_free_disk_gb": package["minimum_free_disk_gb"],
            "drive_free_gb": (
                disk_free_gb(config["paths"]["drive_root"])
                if Path(config["paths"]["drive_root"]).exists()
                else None
            ),
            "local_archive": str(local_archive),
            "local_archive_exists": local_archive.is_file(),
            "local_archive_size": (
                local_archive.stat().st_size if local_archive.is_file() else 0
            ),
            "drive_archive": str(drive_archive),
            "drive_archive_exists": drive_archive.is_file(),
            "drive_archive_size": (
                drive_archive.stat().st_size if drive_archive.is_file() else 0
            ),
            "drive_cache_valid": drive_cache_valid,
            "drive_cache_error": drive_cache_reason,
            "local_root": str(local_root),
            "local_root_exists": local_root.is_dir(),
            "local_root_size": directory_size(local_root),
            "executable_exists": executable.is_file(),
        },
        "archive": {
            "metadata_path": str(metadata_path),
            "metadata_exists": metadata is not None,
            "drive_metadata_exists": drive_metadata is not None,
            "local_metadata_path": str(local_metadata_path),
            "local_metadata_exists": local_metadata is not None,
            "metadata_sha256": metadata.get("sha256", "") if metadata else "",
            "local_sha256": local_hash,
            "local_hash_matches_metadata": local_hash_matches,
        },
        "package_manifest": {
            "path": str(package_manifest_path(config)),
            "exists": package_manifest is not None,
            "matches_runtime": runtime_manifest_matches(config),
            "contents": package_manifest,
        },
        "python_client": {
            **python_client_status(),
            "python_tag": python_tag(),
            "discovered_wheels": [path.name for path in wheels],
            "matching_wheel": matching_wheel,
            "wheel_error": wheel_error,
        },
        "managed_server": {
            "mode": config["carla"]["server"]["mode"],
            "configured_root": config["carla"]["server"]["root"],
            "executable_exists": executable.is_file(),
            "executable_is_executable": bool(
                executable.is_file() and os.access(executable, os.X_OK)
            ),
            "record": record,
            "record_state": server_record_state(config, record),
            "reachable": None,
        },
    }
    if check_server:
        try:
            with socket.create_connection(
                (config["carla"]["host"], config["carla"]["port"]), timeout=2.0
            ):
                status["managed_server"]["reachable"] = True
        except OSError as exc:
            status["managed_server"]["reachable"] = False
            status["managed_server"]["reachability_error"] = str(exc)
    status["compatibility"] = {
        "linux": platform.system() == "Linux",
        "x86_64": platform.machine().lower() in ("x86_64", "amd64"),
        "supported_python": (
            sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12
        ),
        "local_disk_sufficient": (
            status["storage"]["local_free_gb"]
            >= float(package["minimum_free_disk_gb"])
        ),
        "gpu_vulkan_error": gpu_vulkan_error(gpu, vulkan),
    }
    return status


def save_runtime_status(config, status):
    root = ensure_artifact_layout(config)
    path = root / "logs/runtime/runtime_status.json"
    write_json(path, status)
    return path


def command_runtime_status(args):
    config = load_config(args.config)
    status = build_runtime_status(
        config,
        verify_archive_hash=args.verify_archive_hash,
        check_server=args.check_server,
    )
    path = save_runtime_status(config, status)
    print(json.dumps(status, indent=2))
    print("Saved runtime status:", path)
    if args.strict:
        compatibility = status["compatibility"]
        failures = [
            key
            for key in ("linux", "x86_64", "supported_python", "local_disk_sufficient")
            if not compatibility[key]
        ]
        if (
            config["carla"]["server"]["mode"] == "managed"
            and compatibility["gpu_vulkan_error"]
        ):
            failures.append("gpu_vulkan")
        if failures:
            raise RuntimeError(
                "Runtime compatibility checks failed: %s. %s"
                % (
                    ", ".join(failures),
                    compatibility["gpu_vulkan_error"],
                )
            )


def validate_provisioning_platform(config):
    package = config["carla"]["package"]
    failures = []
    if platform.system() != "Linux":
        failures.append("Linux is required (found %s)." % platform.system())
    if platform.machine().lower() not in ("x86_64", "amd64"):
        failures.append(
            "x86_64/amd64 is required (found %s)." % platform.machine()
        )
    if not (
        sys.version_info.major == 3 and 10 <= sys.version_info.minor <= 12
    ):
        failures.append(
            "Python 3.10 through 3.12 is required (found %s.%s)."
            % (sys.version_info.major, sys.version_info.minor)
        )
    probe = Path("/content") if Path("/content").exists() else Path(
        package["local_archive"]
    )
    free = disk_free_gb(probe)
    required = float(package["minimum_free_disk_gb"])
    if free < required:
        failures.append(
            "Local VM disk has %.2f GiB free; %.2f GiB is required. The local "
            "VM must temporarily hold both the compressed archive and extracted "
            "CARLA package. Local archive: %s. Extraction staging path: %s."
            % (
                free,
                required,
                package["local_archive"],
                package["extraction_staging_root"],
            )
        )
    if failures:
        raise RuntimeError("Provisioning compatibility failed:\n- " + "\n- ".join(failures))
    return {"local_free_gb": free, "minimum_free_disk_gb": required}


def copy_file_with_progress(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = source.stat().st_size
    copied = 0
    next_report = 10
    with source.open("rb") as reader, destination.open("wb") as writer:
        while True:
            chunk = reader.read(16 * 1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            copied += len(chunk)
            percent = int(100 * copied / max(total, 1))
            if percent >= next_report:
                print("Copy progress: %s%%" % percent)
                next_report += 10
    shutil.copystat(source, destination)


def download_archive(url, part_path):
    global NETWORK_ACQUISITION_CALLS

    NETWORK_ACQUISITION_CALLS += 1
    part_path = Path(part_path)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    aria2 = shutil.which("aria2c")
    resolved_url = ""
    if aria2:
        command = [
            aria2,
            "--continue=true",
            "--max-tries=8",
            "--retry-wait=5",
            "--timeout=30",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--dir=%s" % part_path.parent,
            "--out=%s" % part_path.name,
            url,
        ]
        print("Downloading with aria2c:", url)
        subprocess.run(command, check=True)
    else:
        curl = shutil.which("curl")
        if not curl:
            raise RuntimeError("Install aria2 or curl before provisioning CARLA.")
        command = [
            curl,
            "-L",
            "--fail",
            "--retry",
            "8",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(part_path),
            "--write-out",
            "%{url_effective}",
            url,
        ]
        print("Downloading with curl:", url)
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        resolved_url = result.stdout.strip()
        if result.stderr:
            print(result.stderr[-2000:])
    return resolved_url


def validate_drive_metadata(metadata, drive_archive, version):
    if not metadata:
        return False, "metadata is absent or invalid JSON"
    required = ("carla_version", "byte_size", "sha256")
    missing = [key for key in required if key not in metadata]
    if missing:
        return False, "metadata is missing %s" % ", ".join(missing)
    if str(metadata["carla_version"]) != str(version):
        return False, "metadata CARLA version does not match"
    if not Path(drive_archive).is_file():
        return False, "Drive archive is absent"
    if Path(drive_archive).stat().st_size != int(metadata["byte_size"]):
        return False, "Drive archive size does not match metadata"
    return True, ""


def cache_archive_to_drive(config, local_archive, validation, metadata):
    package = config["carla"]["package"]
    drive_archive = Path(package["drive_archive"])
    drive_metadata = Path(package["drive_metadata"])
    drive_archive.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = drive_archive.with_name(drive_archive.name + ".part")
    temporary_metadata = drive_metadata.with_name(drive_metadata.name + ".part")
    if temporary_archive.exists():
        temporary_archive.unlink()
    copy_file_with_progress(local_archive, temporary_archive)
    if temporary_archive.stat().st_size != validation["byte_size"]:
        temporary_archive.unlink(missing_ok=True)
        raise RuntimeError("Drive archive copy has the wrong byte size.")
    temporary_archive.replace(drive_archive)
    metadata = dict(metadata)
    metadata["drive_cached_at"] = utc_now()
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(drive_metadata)
    return True


def acquire_carla_archive(config, args):
    package = config["carla"]["package"]
    version = str(config["carla"]["version"])
    local_archive = Path(package["local_archive"])
    drive_archive = Path(package["drive_archive"])
    drive_metadata = Path(package["drive_metadata"])
    validation_path = (
        ensure_artifact_layout(config)
        / "logs/runtime/carla_archive_validation.json"
    )

    def validate_candidate(path):
        PREPARE_STATE["stage"] = "archive_validation"
        try:
            result = validate_carla_archive(path, version)
            write_json(validation_path, result)
            return result
        except RuntimeError as exc:
            write_json(
                validation_path,
                {
                    "valid": False,
                    "archive": str(path),
                    "validation_timestamp": utc_now(),
                    "error": str(exc),
                },
            )
            raise

    use_drive = not args.no_drive_cache
    drive_validated = False
    source = ""
    resolved_url = ""

    if local_archive.is_file() and not args.force_download:
        try:
            validation = validate_candidate(local_archive)
            source = "existing_local_archive"
            print("Using validated local archive:", local_archive)
        except RuntimeError as exc:
            print("Existing local archive is invalid and will be replaced:", exc)
            local_archive.unlink()
            validation = None
    else:
        validation = None

    if validation is None and use_drive and not args.force_download:
        metadata = read_json_if_valid(drive_metadata)
        valid, reason = validate_drive_metadata(metadata, drive_archive, version)
        if valid:
            temporary = local_archive.with_name(local_archive.name + ".from-drive.part")
            temporary.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                temporary.unlink()
            print("Restoring CARLA archive from Drive:", drive_archive)
            PREPARE_STATE["stage"] = "archive_acquisition"
            copy_file_with_progress(drive_archive, temporary)
            if file_sha256(temporary) != metadata["sha256"]:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    "Drive CARLA archive checksum does not match its metadata."
                )
            if temporary.stat().st_size != int(metadata["byte_size"]):
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    "Drive CARLA archive size changed during restoration."
                )
            validation = validate_candidate(temporary)
            temporary.replace(local_archive)
            validation["archive"] = str(local_archive)
            source = "drive_cache"
            drive_validated = True
        elif package.get("require_drive_cache"):
            raise RuntimeError(
                "No valid Drive cache was found (%s). Run runtime prepare to "
                "download the official CARLA 0.9.16 archive." % reason
            )
        elif drive_archive.exists() or drive_metadata.exists():
            print("Drive cache is incomplete or invalid; downloading instead:", reason)

    if validation is None:
        part = local_archive.with_name(local_archive.name + ".part")
        if args.force_download:
            if part.exists():
                part.unlink()
                print("Removed an old partial file for a clean forced download:", part)
            aria2_sidecar = part.with_name(part.name + ".aria2")
            if aria2_sidecar.exists():
                aria2_sidecar.unlink()
        elif part.is_file() and part.stat().st_size > 0:
            print("Found existing partial/completed download file; validating before downloading:", part)
            try:
                validation = validate_candidate(part)
                source = "existing_part_file"
                print("Existing .part file is valid; using it as local archive.")
            except RuntimeError as exc:
                print("Existing .part file is not a complete valid archive. Will re-download:", exc)
                validation = None

        if validation is None:
            PREPARE_STATE["stage"] = "archive_acquisition"
            download_meta = download_archive(package["download_url"], part)
            validation = validate_candidate(part)
            source = "official_download"
            if isinstance(download_meta, dict):
                resolved_url = download_meta.get("resolved_url", "")

    metadata = {
        "carla_version": version,
        "requested_url": package["download_url"],
        "resolved_url": resolved_url,
        "filename": local_archive.name,
        "byte_size": validation["byte_size"],
        "sha256": validation["sha256"],
        "download_timestamp": utc_now() if source == "official_download" else "",
        "validation_timestamp": validation["validated_at"],
        "platform": platform.platform(),
        "source": source,
        "local_sha256": validation["sha256"],
        "local_mtime_ns": local_archive.stat().st_mtime_ns,
    }
    write_json(validation_path, {**validation, "source": source})
    if use_drive and source != "drive_cache":
        drive_parent = nearest_existing_parent(drive_archive)
        drive_looks_mounted = Path("/content/drive/MyDrive").is_dir()
        if drive_looks_mounted or drive_archive.parent.exists():
            print("Caching validated archive in Drive:", drive_archive)
            drive_validated = cache_archive_to_drive(
                config, local_archive, validation, metadata
            )
        elif package.get("require_drive_cache"):
            raise RuntimeError(
                "Google Drive is not mounted, but require_drive_cache is enabled."
            )
        else:
            print(
                "Google Drive is not mounted; continuing with the validated "
                "local archive only."
            )
    local_metadata = validation_path.with_name("CARLA_0.9.16.local.metadata.json")
    write_json(local_metadata, metadata)
    return validation, source, drive_validated


def install_packaged_carla_wheel(config, wheel, dry_run=False):
    wheel = Path(wheel)
    output_path = (
        ensure_artifact_layout(config)
        / "logs/runtime/carla_client_install.json"
    )
    report = {
        "timestamp": utc_now(),
        "python": sys.version,
        "python_tag": python_tag(),
        "candidate_wheels": [
            item.name
            for item in discover_wheels(config["carla"]["package"]["local_root"])
        ],
        "selected_wheel": str(wheel),
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ],
            check=True,
        )
        verification = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib.metadata, json; import carla; "
                    "print(json.dumps({'distribution_version': "
                    "importlib.metadata.version('carla'), 'module': carla.__file__}))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        verified = json.loads(verification.stdout.strip().splitlines()[-1])
        report.update(verified)
        if verified["distribution_version"] != str(config["carla"]["version"]):
            raise RuntimeError(
                "Installed CARLA distribution is %s, expected %s."
                % (
                    verified["distribution_version"],
                    config["carla"]["version"],
                )
            )
    write_json(output_path, report)
    return report


def inspect_extracted_asset_layout(extracted_root):
    extracted_root = Path(extracted_root).resolve()
    carla_content = extracted_root / "CarlaUE4/Content"
    paks_dir = carla_content / "Paks"
    asset_registry = extracted_root / "CarlaUE4/AssetRegistry.bin"

    asset_registry_present = asset_registry.is_file() and asset_registry.stat().st_size > 0
    asset_registry_size = asset_registry.stat().st_size if asset_registry_present else 0

    pak_files = [f for f in paks_dir.glob("*.pak") if f.is_file() and f.stat().st_size > 0] if paks_dir.is_dir() else []
    utoc_files = [f for f in paks_dir.glob("*.utoc") if f.is_file() and f.stat().st_size > 0] if paks_dir.is_dir() else []
    ucas_files = [f for f in paks_dir.glob("*.ucas") if f.is_file() and f.stat().st_size > 0] if paks_dir.is_dir() else []

    utoc_stems = {f.stem.lower() for f in utoc_files}
    ucas_stems = {f.stem.lower() for f in ucas_files}
    matching_utoc_ucas_count = len(utoc_stems.intersection(ucas_stems))

    uasset_files = []
    umap_files = []
    uexp_files = []
    ubulk_files = []

    if carla_content.is_dir():
        for path in carla_content.rglob("*"):
            if path.is_file() and path.stat().st_size > 0:
                ext = path.suffix.lower()
                if ext == ".uasset":
                    uasset_files.append(path)
                elif ext == ".umap":
                    umap_files.append(path)
                elif ext == ".uexp":
                    uexp_files.append(path)
                elif ext == ".ubulk":
                    ubulk_files.append(path)

    layout_active = []
    if paks_dir.is_dir() and len(pak_files) >= 1:
        layout_active.append("classic_pak")
    if paks_dir.is_dir() and len(utoc_files) >= 1 and len(ucas_files) >= 1 and matching_utoc_ucas_count >= 1:
        layout_active.append("iostore")
    if carla_content.is_dir() and asset_registry_present and (len(uasset_files) + len(umap_files)) >= 1:
        layout_active.append("loose_cooked")

    detected_asset_layout = ", ".join(layout_active) if layout_active else "none"

    all_found = pak_files + utoc_files + ucas_files + uasset_files + umap_files
    sample_files = [str(f.relative_to(extracted_root)) for f in all_found[:50]]

    return {
        "detected_asset_layout": detected_asset_layout,
        "asset_registry_present": asset_registry_present,
        "asset_registry_path": str(asset_registry) if asset_registry_present else "",
        "asset_registry_size": asset_registry_size,
        "content_dir": str(carla_content),
        "content_dir_exists": carla_content.is_dir(),
        "pak_file_count": len(pak_files),
        "utoc_file_count": len(utoc_files),
        "ucas_file_count": len(ucas_files),
        "matching_utoc_ucas_count": matching_utoc_ucas_count,
        "uasset_file_count": len(uasset_files),
        "umap_file_count": len(umap_files),
        "uexp_file_count": len(uexp_files),
        "ubulk_file_count": len(ubulk_files),
        "sample_files": sample_files,
    }


def extract_carla_package(config, archive, validation, force=False):
    package = config["carla"]["package"]
    final_root = Path(package["local_root"])
    staging_root = Path(package["extraction_staging_root"])
    if runtime_manifest_matches(config, validation["sha256"]) and not force:
        PREPARE_STATE["stage"] = "wheel_selection"
        wheel = select_carla_wheel(discover_wheels(final_root))
        print("Existing CARLA extraction matches archive; skipping extraction.")
        return wheel, read_json_if_valid(package_manifest_path(config))
    if final_root.exists():
        remove_managed_path(config, final_root)
    if staging_root.exists():
        remove_managed_path(config, staging_root)
    staging_root.mkdir(parents=True)
    try:
        PREPARE_STATE["stage"] = "archive_extraction"
        safe_extract_archive(archive, staging_root)
        extracted_root = discover_carla_root(staging_root)
        PREPARE_STATE["stage"] = "wheel_selection"
        wheel = select_carla_wheel(discover_wheels(extracted_root))
        executable = extracted_root / "CarlaUE4.sh"
        support_binary = (
            extracted_root
            / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
        )
        if not support_binary.is_file():
            raise RuntimeError(
                "Extracted package is missing CarlaUE4-Linux-Shipping."
            )
        if not executable.is_file():
            raise RuntimeError("Extracted package is missing CarlaUE4.sh.")

        layout_info = inspect_extracted_asset_layout(extracted_root)
        if layout_info["detected_asset_layout"] == "none":
            raise RuntimeError(
                "Extracted package has no recognized CARLA cooked-asset layout.\n"
                "Expected one of:\n"
                "1. nonempty Classic Pak files under CarlaUE4/Content/Paks;\n"
                "2. matching nonempty IoStore .utoc/.ucas files under that directory; or\n"
                "3. a nonempty CarlaUE4/AssetRegistry.bin plus loose nonempty .uasset or .umap files under CarlaUE4/Content."
            )

        try:
            executable.chmod(executable.stat().st_mode | 0o755)
            support_binary.chmod(support_binary.stat().st_mode | 0o755)
        except Exception as exc:
            warnings.warn("Could not set executable permissions: %s" % exc)

        extracted_inventory = {
            "package_root": str(extracted_root),
            "executable_sh": str(executable),
            "executable_shipping": str(support_binary),
            "sh_executable_mode": oct(executable.stat().st_mode),
            "shipping_executable_mode": oct(support_binary.stat().st_mode),
            "wheel_selected": str(wheel),
            "detected_asset_layout": layout_info["detected_asset_layout"],
            "asset_layout_info": layout_info,
            "archive_sha256": validation["sha256"],
            "extracted_at": utc_now(),
        }
        artifact_root = ensure_artifact_layout(config)
        write_json(artifact_root / "logs/runtime/extracted_package_inventory.json", extracted_inventory)

        if extracted_root == staging_root.resolve():
            staging_root.replace(final_root)
        else:
            extracted_root.replace(final_root)
            remove_managed_path(config, staging_root)
        wheel = select_carla_wheel(discover_wheels(final_root))
        manifest = {
            "carla_version": str(config["carla"]["version"]),
            "archive_sha256": validation["sha256"],
            "archive_size": validation["byte_size"],
            "package_root": str(final_root),
            "python_wheel": str(wheel),
            "extracted_at": utc_now(),
            "python": sys.version,
            "hostname": socket.gethostname(),
            "boot_id": linux_boot_id(),
        }
        write_json(package_manifest_path(config), manifest)
        write_json(
            artifact_root / "logs/runtime/carla_package_manifest.json",
            manifest,
        )
        return wheel, manifest
    except Exception:
        invalid_marker = staging_root / ".carretera_extraction_invalid"
        if staging_root.exists():
            invalid_marker.write_text(utc_now() + "\n", encoding="utf-8")
        raise


def provisioning_plan(config, args):
    package = config["carla"]["package"]
    return {
        "dry_run": bool(args.dry_run),
        "platform_required": "Linux x86_64/amd64",
        "python_required": "3.10-3.12",
        "download_url": package["download_url"],
        "local_archive": package["local_archive"],
        "drive_archive": package["drive_archive"],
        "drive_metadata": package["drive_metadata"],
        "staging_root": package["extraction_staging_root"],
        "final_root": package["local_root"],
        "cache_root": package["local_cache_root"],
        "force_download": bool(args.force_download),
        "force_extract": bool(args.force_extract),
        "use_drive_cache": not args.no_drive_cache,
        "keep_local_archive": bool(args.keep_local_archive),
        "will_start_server": False,
        "next_command": "python run.py server start --config %s" % args.config,
    }


def command_runtime_prepare(args):
    global PREPARE_STATE
    PREPARE_STATE = {"stage": "platform_validation"}
    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    
    gpu = None
    vulkan = None
    
    try:
        PREPARE_STATE["stage"] = "platform_validation"
        compatibility = validate_provisioning_platform(config)
        
        PREPARE_STATE["stage"] = "graphics_diagnostics"
        gpu = gpu_status()
        vulkan = vulkan_status()
        error = gpu_vulkan_error(gpu, vulkan)
        if error:
            print("================================================================================")
            print("WARNING: GPU/Vulkan status check failed/unavailable: %s" % error)
            print("Package provisioning will continue, but managed server startup remains protected")
            print("by a strict GPU/Vulkan check.")
            print("================================================================================")
            
        PREPARE_STATE["stage"] = "archive_acquisition"
        plan = provisioning_plan(config, args)
        write_json(artifact_root / "logs/runtime/runtime_prepare_plan.json", plan)
        if args.dry_run:
            status = build_runtime_status(config)
            save_runtime_status(config, status)
            print(json.dumps({"plan": plan, "status": status}, indent=2))
            print("Dry run only: no network, extraction, installation, or deletion occurred.")
            return
            
        print(
            json.dumps(
                {
                    "local_archive": plan["local_archive"],
                    "extraction_staging_root": plan["staging_root"],
                    "final_root": plan["final_root"],
                    "minimum_free_disk_gb": config["carla"]["package"][
                        "minimum_free_disk_gb"
                    ],
                },
                indent=2,
            )
        )
        
        package = config["carla"]["package"]
        local_root = Path(package["local_root"])
        existing_valid = runtime_manifest_matches(config)
        source = "existing_extraction"
        drive_validated = False
        validation = None
        
        if existing_valid and not args.force_download and not args.force_extract:
            manifest = read_json_if_valid(package_manifest_path(config))
            validation = {
                "sha256": manifest["archive_sha256"],
                "byte_size": manifest["archive_size"],
            }
            PREPARE_STATE["stage"] = "wheel_selection"
            wheel = select_carla_wheel(discover_wheels(local_root))
            print("Using existing validated CARLA extraction:", local_root)
        else:
            validation, source, drive_validated = acquire_carla_archive(config, args)
            wheel, manifest = extract_carla_package(
                config,
                package["local_archive"],
                validation,
                force=args.force_extract or args.force_download,
            )
            
        PREPARE_STATE["stage"] = "wheel_installation"
        install_report = install_packaged_carla_wheel(config, wheel)
        
        PREPARE_STATE["stage"] = "final_verification"
        prepared = {
            "timestamp": utc_now(),
            "success": True,
            "compatibility": compatibility,
            "gpu": gpu,
            "vulkan": vulkan,
            "archive_source": source,
            "archive_sha256": validation["sha256"],
            "archive_size": validation["byte_size"],
            "drive_cache_validated": drive_validated,
            "package_root": str(local_root),
            "wheel": str(wheel),
            "client_install": install_report,
            "next_command": plan["next_command"],
        }
        write_json(
            artifact_root / "logs/runtime/runtime_prepare_manifest.json", prepared
        )
        local_archive = Path(package["local_archive"])
        metadata = read_json_if_valid(package["drive_metadata"])
        drive_cache_now_valid, _ = validate_drive_metadata(
            metadata, package["drive_archive"], config["carla"]["version"]
        )
        if (
            local_archive.exists()
            and drive_cache_now_valid
            and package["delete_local_archive_after_extract"]
            and not args.keep_local_archive
        ):
            remove_managed_path(config, local_archive)
            prepared["local_archive_deleted_after_extract"] = True
            write_json(
                artifact_root / "logs/runtime/runtime_prepare_manifest.json", prepared
            )
        print(json.dumps(prepared, indent=2))
        print("CARLA is prepared but not started.")
        print("Next command:", plan["next_command"])
        
    except Exception as exc:
        import traceback
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        tb_str = traceback.format_exc()
        
        gpu_rep = gpu_status()
        vulk_rep = vulkan_status()
        
        try:
            package = config["carla"]["package"]
            resolved_paths = {
                "local_root": str(Path(package["local_root"]).resolve()) if package.get("local_root") else "",
                "local_archive": str(Path(package["local_archive"]).resolve()) if package.get("local_archive") else "",
                "extraction_staging_root": str(Path(package["extraction_staging_root"]).resolve()) if package.get("extraction_staging_root") else "",
                "local_cache_root": str(Path(package["local_cache_root"]).resolve()) if package.get("local_cache_root") else "",
            }
        except Exception:
            resolved_paths = {}
            
        failure_record = {
            "stage": PREPARE_STATE["stage"],
            "timestamp": utc_now(),
            "exception_type": exc_type,
            "exception_message": exc_msg,
            "traceback": tb_str,
            "resolved_paths": resolved_paths,
            "GPU report": gpu_rep,
            "Vulkan report": vulk_rep,
        }
        
        failure_dir = artifact_root / "logs/runtime"
        failure_dir.mkdir(parents=True, exist_ok=True)
        write_json(failure_dir / "runtime_prepare_failure.json", failure_record)
        print("Provisioning failed at stage '%s'. Failure record written." % PREPARE_STATE["stage"])
        raise exc


def command_runtime_clean_local(args):
    config = load_config(args.config)
    package = config["carla"]["package"]
    targets = [
        package["local_archive"],
        package["local_archive"] + ".part",
        package["local_archive"] + ".from-drive.part",
        package["extraction_staging_root"],
        package["local_root"],
        package["local_cache_root"],
    ]
    removed = []
    for target in targets:
        path = Path(target)
        if path.exists():
            remove_managed_path(config, path, dry_run=args.dry_run)
            removed.append(str(path))
    report = {
        "timestamp": utc_now(),
        "dry_run": bool(args.dry_run),
        "removed_or_selected": removed,
        "drive_archive_untouched": package["drive_archive"],
    }
    print(json.dumps(report, indent=2))


def command_validate_config(args):
    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    print("Valid config:", config["_config_path"])
    print("Resolved artifact root:", artifact_root)
    print("Resolved Drive root:", config["paths"]["drive_root"])
    print("Config SHA-256:", config_hash(config))


def runtime_pid_file(config):
    return Path(config["paths"]["artifact_root"]) / "logs/runtime/carla_server.json"


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def server_record_matches_process(record):
    try:
        identity_matches = bool(
            record.get("hostname") == socket.gethostname()
            and record.get("boot_id") == linux_boot_id()
            and Path(record.get("repository_root", "")).resolve()
            == REPOSITORY_ROOT.resolve()
            and int(record.get("process_group", -1))
            == int(record.get("pid", -2))
        )
    except (TypeError, ValueError):
        identity_matches = False
    if not identity_matches:
        return False
    if not process_alive(record["pid"]):
        return False
    try:
        result = subprocess.run(
            ["ps", "-p", str(record["pid"]), "-o", "command="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    expected = str(record.get("executable", ""))
    expected_binary = str(
        Path(record.get("carla_root", ""))
        / "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
    )
    return bool(
        "CarlaUE4" in result.stdout
        and expected
        and (expected in result.stdout or expected_binary in result.stdout)
    )


def read_server_record(config):
    path = runtime_pid_file(config)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def carla_server_executable(config):
    root = config["carla"]["server"].get("root", "")
    if not root:
        raise ValueError(
            "Managed mode requires CARLA_ROOT or carla.server.root."
        )
    executable = resolve_path(Path(root) / "CarlaUE4.sh")
    if not executable.is_file():
        raise FileNotFoundError("Packaged server executable not found: %s" % executable)
    return executable


def tail_file(path, lines=150):
    try:
        values = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(values[-lines:])
    except OSError as exc:
        return "Unable to read server log: %s" % exc


def terminate_owned_process_group(process, grace_seconds=15.0):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)


def wait_for_carla(config, timeout, process=None, log_path=None):
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError("Install carla==0.9.16 before starting managed mode.") from exc
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            recent = tail_file(log_path) if log_path else ""
            raise RuntimeError(
                "Managed CARLA exited during startup with code %s.\n"
                "Recent server log:\n%s" % (process.returncode, recent)
            )
        try:
            client = carla.Client(config["carla"]["host"], config["carla"]["port"])
            client.set_timeout(2.0)
            client_version = client.get_client_version()
            server_version = client.get_server_version()
            expected = str(config["carla"]["version"])
            if client_version != expected or server_version != expected:
                raise ValueError(
                    "CARLA version mismatch: expected client/server %s, got "
                    "client %s and server %s."
                    % (expected, client_version, server_version)
                )
            return {
                "client_version": client_version,
                "server_version": server_version,
            }
        except RuntimeError as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(
        "Managed CARLA did not become reachable within %s seconds: %s"
        % (timeout, last_error)
    )


def command_server_start(args):
    config = load_config(args.config)
    ensure_artifact_layout(config)
    if config["carla"]["server"]["mode"] != "managed":
        raise ValueError(
            "server start is permitted only when carla.server.mode is managed."
        )
    gpu = gpu_status()
    vulkan = vulkan_status()
    compatibility_error = gpu_vulkan_error(gpu, vulkan)
    if compatibility_error:
        raise RuntimeError(compatibility_error)
    previous = read_server_record(config)
    previous_state = server_record_state(config, previous)
    if previous_state["active"]:
        raise RuntimeError(
            "Managed CARLA is already running with PID %s." % previous["pid"]
        )
    if previous:
        runtime_pid_file(config).unlink(missing_ok=True)
        if previous_state["stale"]:
            print("Removed a stale managed-server record from another runtime.")
    executable = carla_server_executable(config)
    quality_key = "quality_rendering" if args.rendering else "quality_training"
    quality = config["carla"]["server"][quality_key]
    command = [
        str(executable),
        "-carla-rpc-port=%s" % config["carla"]["port"],
        "-quality-level=%s" % quality,
    ]
    offscreen_key = "rendering_offscreen" if args.rendering else "training_offscreen"
    if config["carla"]["server"][offscreen_key]:
        command.append("-RenderOffScreen")
    if not config["carla"]["server"]["sound"]:
        command.append("-nosound")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        Path(config["paths"]["artifact_root"])
        / "logs/runtime"
        / ("carla_server_%s.log" % timestamp)
    )
    log_handle = log_path.open("ab")
    package = config["carla"]["package"]
    cache_root = Path(package["local_cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    process_environment, diagnostics = graphics_environment()
    process_environment["CARLA_CACHE_DIR"] = str(cache_root)
    process = subprocess.Popen(
        command,
        cwd=executable.parent,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=process_environment,
    )
    startup_complete = {"value": False}

    def cleanup_incomplete_start():
        if not startup_complete["value"] and process_alive(process.pid):
            terminate_owned_process_group(process)

    atexit.register(cleanup_incomplete_start)
    record = {
        "pid": process.pid,
        "process_group": process.pid,
        "hostname": socket.gethostname(),
        "boot_id": linux_boot_id(),
        "started_at": utc_now(),
        "command": command,
        "log": str(log_path),
        "rendering": bool(args.rendering),
        "executable": str(executable),
        "repository_root": str(REPOSITORY_ROOT),
        "carla_root": str(executable.parent),
        "cache_root": str(cache_root),
    }
    write_json(runtime_pid_file(config), record)
    try:
        versions = wait_for_carla(
            config,
            config["carla"]["server"]["startup_timeout_seconds"],
            process=process,
            log_path=log_path,
        )
    except Exception as exc:
        terminate_owned_process_group(process)
        runtime_pid_file(config).unlink(missing_ok=True)
        details = (
            "%s\nServer command: %s\nLog: %s\nGPU: %s\nVulkan: %s\n"
            "Recent server log:\n%s"
            % (
                exc,
                " ".join(command),
                log_path,
                json.dumps(gpu),
                json.dumps(vulkan),
                tail_file(log_path),
            )
        )
        raise RuntimeError(details) from exc
    finally:
        log_handle.close()
    startup_complete["value"] = True
    atexit.unregister(cleanup_incomplete_start)
    print(
        "Managed CARLA started with PID %s, client/server version %s/%s."
        % (
            process.pid,
            versions["client_version"],
            versions["server_version"],
        )
    )
    print("Log:", log_path)


def command_server_status(args):
    config = load_config(args.config)
    record = read_server_record(config)
    if not record:
        print("No repository-owned managed CARLA process is recorded.")
        return
    state = server_record_state(config, record)
    print("Managed CARLA PID %s state: %s." % (record.get("pid"), state["state"]))
    print(json.dumps(record, indent=2))
    if state["stale"]:
        runtime_pid_file(config).unlink(missing_ok=True)
        print("Removed the stale local record; no process was signaled.")


def command_server_stop(args):
    config = load_config(args.config)
    if config["carla"]["server"]["mode"] != "managed":
        raise ValueError("server stop never acts on an external CARLA server.")
    record = read_server_record(config)
    if not record:
        print("No repository-owned managed CARLA process is recorded; nothing stopped.")
        return
    state = server_record_state(config, record)
    if state["stale"]:
        runtime_pid_file(config).unlink(missing_ok=True)
        print(
            "Removed stale managed-server record from another runtime; no "
            "local process was signaled."
        )
        return
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        runtime_pid_file(config).unlink(missing_ok=True)
        print("Removed an invalid managed-server record; no process was signaled.")
        return
    if process_alive(pid) and not state["active"]:
        raise RuntimeError(
            "Recorded PID %s no longer identifies a CarlaUE4 process; refusing "
            "to signal an unrelated process." % pid
        )
    if process_alive(pid):
        os.killpg(int(record["process_group"]), signal.SIGTERM)
        deadline = time.monotonic() + 15.0
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.25)
        if process_alive(pid):
            os.killpg(int(record["process_group"]), signal.SIGKILL)
        print("Stopped repository-owned managed CARLA PID %s." % pid)
    else:
        print("Recorded managed CARLA PID %s was already stopped." % pid)
    runtime_pid_file(config).unlink(missing_ok=True)


def gpu_information():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "No NVIDIA GPU reported"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "nvidia-smi unavailable"


def command_doctor(args):
    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    runtime = build_runtime_status(config, check_server=True)
    save_runtime_status(config, runtime)
    disk = shutil.disk_usage(artifact_root)
    report = {
        "repository_root": str(REPOSITORY_ROOT),
        "python": sys.version,
        "packages": package_versions(),
        "gpu": gpu_information(),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "artifact_root": str(artifact_root),
        "drive_root": config["paths"]["drive_root"],
        "server_mode": config["carla"]["server"]["mode"],
        "carla_root": config["carla"]["server"].get("root", ""),
        "carla_root_valid": False,
        "server_reachable": False,
        "runtime_compatibility": runtime["compatibility"],
        "runtime": runtime,
    }
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        report["torch_version"] = "not installed"
        report["cuda_available"] = False
    root = config["carla"]["server"].get("root", "")
    if root:
        report["carla_root_valid"] = (resolve_path(root) / "CarlaUE4.sh").is_file()
    doctor_path = artifact_root / "logs/runtime/doctor.json"
    print(json.dumps(report, indent=2))
    if (
        config["carla"]["server"]["mode"] == "managed"
        and runtime["compatibility"]["gpu_vulkan_error"]
    ):
        write_json(doctor_path, report)
        raise RuntimeError(runtime["compatibility"]["gpu_vulkan_error"])
    if (
        config["carla"]["server"]["mode"] == "managed"
        and not report["carla_root_valid"]
    ):
        write_json(doctor_path, report)
        raise RuntimeError(
            "Managed CARLA is not prepared. Run: python run.py runtime prepare "
            "--config %s" % args.config
        )
    if not runtime["python_client"]["import_success"]:
        write_json(doctor_path, report)
        raise RuntimeError(
            "The CARLA Python client is unavailable. Run: python run.py runtime "
            "prepare --config %s" % args.config
        )
    try:
        with socket.create_connection(
            (config["carla"]["host"], config["carla"]["port"]), timeout=3.0
        ):
            report["tcp_connection"] = "ok"
    except OSError as exc:
        report["tcp_connection"] = "failed: %s" % exc
        write_json(doctor_path, report)
        raise ConnectionError(
            "CARLA TCP connection failed at %s:%s. Run: python run.py server "
            "start --config %s. For external mode, set CARLA_HOST/CARLA_PORT. "
            "Details: %s"
            % (
                config["carla"]["host"],
                config["carla"]["port"],
                args.config,
                exc,
            )
        ) from exc

    env = None
    try:
        env = HighwayDecisionEnv(config, mode="train", artifact_root=artifact_root)
        report.update(
            {
                "server_reachable": True,
                "client_version": env.client.get_client_version(),
                "server_version": env.client.get_server_version(),
                "map": env.map.name,
                "map_available": any(
                    item.lower().endswith("/" + config["carla"]["map"].lower())
                    or item.lower() == config["carla"]["map"].lower()
                    for item in env.client.get_available_maps()
                ),
                "synchronous_mode": env.world.get_settings().synchronous_mode,
                "fixed_delta_seconds": env.world.get_settings().fixed_delta_seconds,
                "traffic_manager_port": config["carla"]["traffic_manager_port"],
                "traffic_manager_connected": env.traffic_manager is not None,
                "highway_candidate_count": len(env.highway_candidates),
            }
        )
        if not report["map_available"]:
            raise RuntimeError("Required map %s is not available on CARLA server." % config["carla"]["map"])
        candidate_path = env.write_highway_candidates(
            artifact_root / "logs/runtime/highway_candidates.json"
        )
        report["highway_candidates"] = str(candidate_path)
        print("Selected candidate preview:", json.dumps(env.highway_candidates[0], indent=2))
    finally:
        if env is not None:
            env.close()
    write_json(doctor_path, report)
    print(json.dumps(report, indent=2))


def wilson_interval(successes, total, confidence_z=1.959963984540054):
    if total <= 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1.0 + confidence_z ** 2 / total
    center = (
        proportion + confidence_z ** 2 / (2.0 * total)
    ) / denominator
    half_width = (
        confidence_z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + confidence_z ** 2 / (4.0 * total ** 2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def bootstrap_interval(values, resamples=2000, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(resamples), values.size))
    estimates = values[indices].mean(axis=1)
    return tuple(np.percentile(estimates, [2.5, 97.5]).tolist())


def synthetic_state():
    radius = 80.0
    lane = {
        "available": True,
        "road_id": 1,
        "lane_id": -1,
        "section_id": 0,
        "front_gap_m": radius,
        "front_relative_speed_mps": 0.0,
        "front_speed_mps": 0.0,
        "rear_gap_m": radius,
        "rear_relative_speed_mps": 0.0,
        "rear_speed_mps": 0.0,
    }
    return {
        "speed_kmh": 60.0,
        "speed_mps": 60.0 / 3.6,
        "target_speed_kmh": 65.0,
        "lateral_offset_normalized": 0.0,
        "heading_error_rad": 0.0,
        "lane_change_active": False,
        "left_of_initial_lane": False,
        "lanes": {
            "left": dict(lane, lane_id=-2),
            "current": dict(lane, lane_id=-1),
            "right": dict(lane, lane_id=-3),
        },
    }


def latex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def sync_path_excluded(relative):
    relative = Path(relative)
    lowered = [part.lower() for part in relative.parts]
    normalized = relative.as_posix().lower()
    if normalized in (
        "logs/runtime/carla_server.json",
        "logs/runtime/runtime_status.json",
        "logs/runtime/runtime_prepare_plan.json",
        "logs/runtime/runtime_prepare_failure.json",
    ):
        return True
    if any(
        part.startswith("carla_0.9.16")
        or part in ("carla_cache", ".carla_cache", "shader_cache")
        for part in lowered
    ):
        return True
    name = relative.name.lower()
    if (
        name.startswith("carla_")
        and (name.endswith(".tar.gz") or ".tar.gz." in name)
    ):
        return True
    return False


def sync_directories(source, destination, dry_run=False):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError(
            "Refusing recursive sync between nested roots: %s and %s"
            % (source, destination)
        )
    if not source.exists():
        return {"copied": 0, "skipped": 0, "errors": 0, "source_missing": True}
    counts = {"copied": 0, "skipped": 0, "errors": 0, "source_missing": False}
    for current, directory_names, file_names in os.walk(source, topdown=True):
        current = Path(current)
        retained = []
        for name in sorted(directory_names):
            source_path = current / name
            relative = source_path.relative_to(source)
            if source_path.is_symlink() or sync_path_excluded(relative):
                counts["skipped"] += 1
            else:
                retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            source_path = current / name
            relative = source_path.relative_to(source)
            if source_path.is_symlink() or sync_path_excluded(relative):
                counts["skipped"] += 1
                continue
            destination_path = destination / relative
            try:
                needs_copy = (
                    not destination_path.exists()
                    or source_path.stat().st_size != destination_path.stat().st_size
                    or source_path.stat().st_mtime
                    > destination_path.stat().st_mtime + 1e-6
                )
                if needs_copy:
                    if not dry_run:
                        destination_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, destination_path)
                    counts["copied"] += 1
                else:
                    counts["skipped"] += 1
            except OSError as exc:
                counts["errors"] += 1
                warnings.warn(
                    "Sync failed for %s: %s" % (source_path, exc), RuntimeWarning
                )
    return counts


def command_sync(args):
    config = load_config(args.config)
    local_root = resolve_path(args.local_root or config["paths"]["artifact_root"])
    drive_root = resolve_path(args.drive_root or config["paths"]["drive_root"])
    if (
        local_root == REPOSITORY_ROOT
        or local_root in REPOSITORY_ROOT.parents
        or (local_root / ".git").exists()
    ):
        raise ValueError(
            "Refusing to sync the repository itself; choose the artifacts root."
        )
    if bool(args.to_drive) == bool(args.from_drive):
        raise ValueError("Choose exactly one of --to-drive or --from-drive.")
    source, destination = (
        (local_root, drive_root) if args.to_drive else (drive_root, local_root)
    )
    counts = sync_directories(source, destination, args.dry_run)
    manifest = {
        "timestamp": utc_now(),
        "direction": "to_drive" if args.to_drive else "from_drive",
        "source": str(source),
        "destination": str(destination),
        "dry_run": bool(args.dry_run),
        **counts,
    }
    if not args.dry_run:
        ensure_artifact_layout(config)
        if args.to_drive:
            generated_source = REPOSITORY_ROOT / "reports/generated"
            generated_destination = drive_root / "reports_generated"
            manifest["reports_generated"] = sync_directories(
                generated_source, generated_destination, False
            )
        name = "sync_%s.json" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        write_json(
            Path(config["paths"]["artifact_root"]) / "manifests" / name,
            manifest,
        )
    print(json.dumps(manifest, indent=2))
    return manifest


def command_offline_self_test(args):
    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    tests = {}
    network_calls_before = NETWORK_ACQUISITION_CALLS
    tests["yaml_loading"] = config["project"]["name"] == "carla_highway_rl"
    original = os.environ.get("CARLA_PORT")
    try:
        os.environ["CARLA_PORT"] = "2999"
        override_config = load_config(args.config)
        tests["environment_overrides"] = override_config["carla"]["port"] == 2999
    finally:
        if original is None:
            os.environ.pop("CARLA_PORT", None)
        else:
            os.environ["CARLA_PORT"] = original
    tests["action_mapping"] = ACTION_NAMES == {
        0: "MAINTAIN",
        1: "ACCELERATE",
        2: "DECELERATE",
        3: "CHANGE_LEFT",
        4: "CHANGE_RIGHT",
    }
    state = synthetic_state()
    observation = HighwayDecisionEnv.synthetic_observation(state)
    lower = np.array(
        [0.0, 0.0, -2.0, -1.0, -1.0, 0.0, 0.0, 0.0]
        + [0.0, -1.0, 0.0, -1.0] * 3,
        dtype=np.float32,
    )
    upper = np.array(
        [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        + [1.0, 1.0, 1.0, 1.0] * 3,
        dtype=np.float32,
    )
    tests["observation_shape_bounds"] = bool(
        observation.shape == (20,)
        and observation.dtype == np.float32
        and np.all(observation >= lower)
        and np.all(observation <= upper)
    )
    policies = [
        RandomPolicy(7),
        KeepLanePolicy(),
        RuleBasedOvertakingPolicy(config["rule_based"]),
    ]
    tests["policy_output_validity"] = all(
        policy.act(observation, state) in ACTION_NAMES for policy in policies
    )
    blocked = copy.deepcopy(state)
    blocked["lanes"]["current"]["front_gap_m"] = 20.0
    blocked["lanes"]["current"]["front_relative_speed_mps"] = -5.0
    rule = RuleBasedOvertakingPolicy(config["rule_based"])
    tests["rule_based_blocked_logic"] = rule.act(observation, blocked) == CHANGE_LEFT
    components = calculate_reward_components(config, 10.0, 65.0)
    tests["reward_components"] = math.isclose(
        components["total_reward"], 1.1, abs_tol=1e-9
    )
    tests["wilson_interval"] = np.allclose(
        wilson_interval(5, 10), (0.236593090512564, 0.763406909487436)
    )
    first_bootstrap = bootstrap_interval([1, 2, 3, 4], 500, 11)
    second_bootstrap = bootstrap_interval([1, 2, 3, 4], 500, 11)
    tests["bootstrap_determinism"] = first_bootstrap == second_bootstrap
    tests["path_resolution"] = (
        resolve_path("artifacts") == REPOSITORY_ROOT / "artifacts"
    )
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        Path(first, "sample.txt").write_text("sample", encoding="utf-8")
        sync_result = sync_directories(first, second, dry_run=True)
        tests["sync_dry_run"] = (
            sync_result["copied"] == 1
            and not Path(second, "sample.txt").exists()
        )
    tests["safe_path_deletion_rules"] = bool(
        not safe_managed_path(config, "/")
        and not safe_managed_path(config, "/content")
        and not safe_managed_path(config, REPOSITORY_ROOT)
        and not safe_managed_path(config, config["paths"]["drive_root"])
        and safe_managed_path(
            config, config["carla"]["package"]["local_archive"]
        )
        and safe_managed_path(
            config, config["carla"]["package"]["local_cache_root"]
        )
    )
    expected_drive_archive = (
        Path(config["paths"]["drive_root"])
        / config["carla"]["package"]["drive_cache_subdirectory"]
        / config["carla"]["package"]["archive_name"]
    ).resolve()
    tests["drive_archive_path_derivation"] = (
        Path(config["carla"]["package"]["drive_archive"])
        == expected_drive_archive
        and Path(config["carla"]["package"]["drive_metadata"])
        == expected_drive_archive.with_name(
            config["carla"]["package"]["metadata_name"]
        )
    )
    override_names = {
        "CARLA_ARCHIVE_URL": "https://example.invalid/carla.tar.gz",
        "CARLA_ARCHIVE_LOCAL": "/content/CARLA_0.9.16.override.tar.gz",
        "CARLA_ARCHIVE_DRIVE": (
            "/content/drive/MyDrive/test/CARLA_0.9.16.tar.gz"
        ),
        "CARLA_ROOT": "/content/CARLA_0.9.16.override",
        "CARLA_CACHE_DIR": "/content/carla_cache",
    }
    originals = {name: os.environ.get(name) for name in override_names}
    try:
        os.environ.update(override_names)
        override_config = load_config(args.config)
        tests["package_environment_overrides"] = bool(
            override_config["carla"]["package"]["download_url"]
            == override_names["CARLA_ARCHIVE_URL"]
            and override_config["carla"]["package"]["local_archive"]
            == override_names["CARLA_ARCHIVE_LOCAL"]
            and override_config["carla"]["package"]["drive_archive"]
            == override_names["CARLA_ARCHIVE_DRIVE"]
            and override_config["carla"]["package"]["local_root"]
            == override_names["CARLA_ROOT"]
            and override_config["carla"]["server"]["root"]
            == override_names["CARLA_ROOT"]
        )
    finally:
        for name, value in originals.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        metadata_path = temporary / "metadata.json"
        metadata_value = {
            "carla_version": "0.9.16",
            "byte_size": 123,
            "sha256": "abc123",
        }
        write_json(metadata_path, metadata_value)
        tests["archive_metadata_round_trip"] = (
            read_json_if_valid(metadata_path) == metadata_value
        )
        hash_path = temporary / "hash.txt"
        hash_path.write_bytes(b"deterministic")
        tests["sha256_determinism"] = (
            file_sha256(hash_path)
            == file_sha256(hash_path)
            == "0badac3c6df445ad3aea62da1350683923aba37c685978afed96a515d12921a3"
        )

        def make_synthetic_tar(
            tar_path,
            root="CARLA_0.9.16",
            has_sh=True,
            has_shipping=True,
            has_wheel=True,
            has_paks_dir=True,
            pak_files=None,
            unsafe_member=None,
        ):
            if pak_files is None:
                pak_files = {"CarlaUE4/Content/Paks/Town04.pak": b"pak_data"}
            prefix = "" if root == "." else root.rstrip("/") + "/"
            with tarfile.open(tar_path, "w:gz") as archive:
                if has_sh:
                    name = prefix + "CarlaUE4.sh"
                    info = tarfile.TarInfo(name)
                    payload = b"#!/bin/bash\necho Carla"
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                if has_shipping:
                    name = prefix + "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
                    info = tarfile.TarInfo(name)
                    payload = b"elf_binary_data"
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                if has_wheel:
                    tag = python_tag()
                    name = prefix + ("PythonAPI/carla/dist/carla-0.9.16-%s-%s-linux_x86_64.whl" % (tag, tag))
                    info = tarfile.TarInfo(name)
                    payload = b"wheel_zip_data"
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                if pak_files:
                    for rel_path, data in pak_files.items():
                        name = prefix + rel_path
                        info = tarfile.TarInfo(name)
                        payload = data if isinstance(data, bytes) else data.encode("utf-8")
                        info.size = len(payload)
                        archive.addfile(info, io.BytesIO(payload))
                if unsafe_member:
                    info = tarfile.TarInfo(unsafe_member)
                    payload = b"unsafe"
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

        # 1. Classic Pak accepted
        classic_tar = temporary / "classic.tar.gz"
        make_synthetic_tar(classic_tar)
        tests["archive_classic_pak_accepted"] = validate_carla_archive(classic_tar, allow_small=True)["valid"]

        # 2. IoStore accepted
        iostore_tar = temporary / "iostore.tar.gz"
        make_synthetic_tar(iostore_tar, pak_files={"CarlaUE4/Content/Paks/global.utoc": b"utoc", "CarlaUE4/Content/Paks/global.ucas": b"ucas"})
        tests["archive_iostore_accepted"] = validate_carla_archive(iostore_tar, allow_small=True)["valid"]

        # 3. Pak + IoStore accepted
        both_tar = temporary / "both.tar.gz"
        make_synthetic_tar(both_tar, pak_files={"CarlaUE4/Content/Paks/Town04.pak": b"pak", "CarlaUE4/Content/Paks/global.utoc": b"utoc", "CarlaUE4/Content/Paks/global.ucas": b"ucas"})
        tests["archive_pak_and_iostore_accepted"] = validate_carla_archive(both_tar, allow_small=True)["valid"]

        # 4. .utoc without .ucas rejected
        utoc_only_tar = temporary / "utoc_only.tar.gz"
        make_synthetic_tar(utoc_only_tar, pak_files={"CarlaUE4/Content/Paks/global.utoc": b"utoc"})
        try:
            validate_carla_archive(utoc_only_tar, allow_small=True)
            tests["archive_utoc_only_rejected"] = False
        except RuntimeError:
            tests["archive_utoc_only_rejected"] = True

        # 5. .ucas without .utoc rejected
        ucas_only_tar = temporary / "ucas_only.tar.gz"
        make_synthetic_tar(ucas_only_tar, pak_files={"CarlaUE4/Content/Paks/global.ucas": b"ucas"})
        try:
            validate_carla_archive(ucas_only_tar, allow_small=True)
            tests["archive_ucas_only_rejected"] = False
        except RuntimeError:
            tests["archive_ucas_only_rejected"] = True

        # 6. Empty file rejected
        empty_tar = temporary / "empty.tar.gz"
        make_synthetic_tar(empty_tar, pak_files={"CarlaUE4/Content/Paks/Town04.pak": b""})
        try:
            validate_carla_archive(empty_tar, allow_small=True)
            tests["archive_empty_container_rejected"] = False
        except RuntimeError:
            tests["archive_empty_container_rejected"] = True

        # 7. Loose cooked archive accepted
        loose_tar = temporary / "loose_cooked.tar.gz"
        make_synthetic_tar(
            loose_tar,
            has_paks_dir=False,
            pak_files={
                "CarlaUE4/AssetRegistry.bin": b"registry_bytes",
                "CarlaUE4/Content/Carla/Maps/Town04/Town04.umap": b"umap_bytes",
                "CarlaUE4/Content/Carla/Maps/Town04/Town04_BuiltData.uasset": b"uasset_bytes",
                "CarlaUE4/Content/Carla/Maps/Town04/Town04_BuiltData.uexp": b"uexp_bytes",
            },
        )
        val_loose = validate_carla_archive(loose_tar, allow_small=True)
        tests["archive_loose_cooked_accepted"] = (
            val_loose["valid"] and "loose_cooked" in val_loose["detected_asset_layout"]
        )

        # 8. No cooked assets rejected with recognized-layout error
        nocooked_tar = temporary / "no_cooked.tar.gz"
        make_synthetic_tar(nocooked_tar, has_paks_dir=False, pak_files={})
        try:
            validate_carla_archive(nocooked_tar, allow_small=True)
            tests["archive_no_cooked_assets_rejected"] = False
        except RuntimeError as exc:
            tests["archive_no_cooked_assets_rejected"] = (
                "Archive has no recognized CARLA cooked-asset layout" in str(exc)
            )

        # Extracted filesystem check for loose cooked tree
        extracted_loose_dir = temporary / "extracted_loose"
        (extracted_loose_dir / "CarlaUE4/Content/Carla/Maps/Town04").mkdir(parents=True, exist_ok=True)
        (extracted_loose_dir / "CarlaUE4/AssetRegistry.bin").write_bytes(b"registry")
        (extracted_loose_dir / "CarlaUE4/Content/Carla/Maps/Town04/Town04.umap").write_bytes(b"umap")
        (extracted_loose_dir / "CarlaUE4/Content/Carla/Maps/Town04/Town04.uasset").write_bytes(b"uasset")
        extracted_layout = inspect_extracted_asset_layout(extracted_loose_dir)
        tests["extracted_loose_cooked_accepted"] = (
            "loose_cooked" in extracted_layout["detected_asset_layout"]
        )

        # 8. Missing CarlaUE4.sh rejected
        nosh_tar = temporary / "nosh.tar.gz"
        make_synthetic_tar(nosh_tar, has_sh=False)
        try:
            validate_carla_archive(nosh_tar, allow_small=True)
            tests["archive_missing_sh_rejected"] = False
        except RuntimeError:
            tests["archive_missing_sh_rejected"] = True

        # 9. Missing CarlaUE4-Linux-Shipping rejected
        noshipping_tar = temporary / "noshipping.tar.gz"
        make_synthetic_tar(noshipping_tar, has_shipping=False)
        try:
            validate_carla_archive(noshipping_tar, allow_small=True)
            tests["archive_missing_shipping_rejected"] = False
        except RuntimeError:
            tests["archive_missing_shipping_rejected"] = True

        # 10. Missing wheel rejected
        nowheel_tar = temporary / "nowheel.tar.gz"
        make_synthetic_tar(nowheel_tar, has_wheel=False)
        try:
            validate_carla_archive(nowheel_tar, allow_small=True)
            tests["archive_missing_wheel_rejected"] = False
        except RuntimeError:
            tests["archive_missing_wheel_rejected"] = True

        # 11. Unsafe traversal rejected
        traversal_tar = temporary / "traversal.tar.gz"
        make_synthetic_tar(traversal_tar, unsafe_member="../escape.txt")
        try:
            validate_carla_archive(traversal_tar, allow_small=True)
            tests["archive_unsafe_traversal_rejected"] = False
        except RuntimeError:
            tests["archive_unsafe_traversal_rejected"] = True

        # 12. Inventory details & zero network calls
        inv = inspect_carla_archive(both_tar, allow_small=True)
        tests["archive_inventory_extension_counts_and_samples"] = bool(
            "content_paks_extension_counts" in inv
            and "sample_content_paks_paths" in inv
            and inv["content_paks_extension_counts"].get(".pak") == 1
        )

        flat_names = [
            "CarlaUE4.sh",
            "PythonAPI/carla/dist/carla-0.9.16-cp311-cp311-linux_x86_64.whl",
            "CarlaUE4/Content/Carla/Maps/Town04.umap",
        ]
        nested_names = ["CARLA_0.9.16/" + name for name in flat_names]
        tests["flat_archive_root_discovery"] = (
            archive_root_from_names(flat_names) == "."
        )
        tests["nested_archive_root_discovery"] = (
            archive_root_from_names(nested_names) == "CARLA_0.9.16"
        )
        wheel_selection_results = []
        for tag in ("cp310", "cp311", "cp312"):
            candidate = temporary / (
                "carla-0.9.16-%s-%s-manylinux_2_31_x86_64.whl" % (tag, tag)
            )
            wheel_selection_results.append(
                select_carla_wheel([candidate], tag=tag) == candidate
            )
        tests["wheel_selection_cp310_cp311_cp312"] = all(
            wheel_selection_results
        )
        ambiguous = [
            temporary
            / "carla-0.9.16-cp311-cp311-linux_x86_64.whl",
            temporary
            / "carla-0.9.16-cp311-cp311-manylinux_x86_64.whl",
        ]
        try:
            select_carla_wheel(ambiguous, tag="cp311")
            ambiguous_rejected = False
        except RuntimeError:
            ambiguous_rejected = True
        tests["ambiguous_wheel_rejected"] = ambiguous_rejected
        stale_record = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "boot_id": "definitely-another-boot",
            "repository_root": str(REPOSITORY_ROOT),
            "executable": "/content/CARLA_0.9.16/CarlaUE4.sh",
        }
        tests["stale_server_record_detection"] = server_record_state(
            config, stale_record
        )["stale"]
        manifest_config = copy.deepcopy(config)
        manifest_root = temporary / "CARLA_0.9.16"
        (manifest_root / "PythonAPI/carla/dist").mkdir(parents=True)
        (manifest_root / "CarlaUE4.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        manifest_config["carla"]["package"]["local_root"] = str(manifest_root)
        manifest_config["carla"]["server"]["root"] = str(manifest_root)
        write_json(
            package_manifest_path(manifest_config),
            {
                "carla_version": "0.9.16",
                "archive_sha256": "synthetic",
                "archive_size": 123,
                "package_root": str(manifest_root),
            },
        )
        tests["runtime_manifest_comparison"] = runtime_manifest_matches(
            manifest_config, "synthetic"
        )
        source = temporary / "sync_source"
        destination = temporary / "sync_destination"
        (source / "logs/runtime").mkdir(parents=True)
        (source / "logs/runtime/carla_server.json").write_text(
            "{}", encoding="utf-8"
        )
        (source / "logs/runtime/keep.json").write_text("{}", encoding="utf-8")
        sync_directories(source, destination)
        tests["sync_excludes_active_pid"] = bool(
            not (destination / "logs/runtime/carla_server.json").exists()
            and (destination / "logs/runtime/keep.json").exists()
        )
    dry_args = argparse.Namespace(
        config=args.config,
        dry_run=True,
        force_download=False,
        force_extract=False,
        no_drive_cache=False,
        keep_local_archive=False,
    )
    dry_plan = provisioning_plan(config, dry_args)
    tests["provisioning_dry_run"] = bool(
        dry_plan["dry_run"] and not dry_plan["will_start_server"]
    )
    tests["colab_path_resolution"] = bool(
        Path(config["carla"]["package"]["local_root"]).is_absolute()
        and str(config["carla"]["package"]["local_root"]).startswith("/content/")
        and str(config["carla"]["package"]["drive_archive"]).startswith(
            "/content/drive/MyDrive/"
        )
    )
    tests["no_network_call"] = NETWORK_ACQUISITION_CALLS == network_calls_before

    # Metric & Controller Unit Tests
    class MockEnv:
        lateral_integral = 0.0
        lateral_previous_error = 0.0
        speed_integral = 0.0
        speed_previous_error = 0.0
        def _pid(self, error, integral_name, previous_name, kp, ki, kd, dt):
            raw_integral = getattr(self, integral_name) + error * dt
            clipped_integral = float(np.clip(raw_integral, -50.0, 50.0))
            derivative = (error - getattr(self, previous_name)) / dt
            setattr(self, integral_name, clipped_integral)
            setattr(self, previous_name, error)
            return kp * error + ki * clipped_integral + kd * derivative

    mock_env = MockEnv()
    pid_val = mock_env._pid(10.0, "speed_integral", "speed_previous_error", 1.0, 0.1, 0.0, 1.0)
    for _ in range(20):
        pid_val = mock_env._pid(10.0, "speed_integral", "speed_previous_error", 1.0, 0.1, 0.0, 1.0)
    tests["pid_integral_clipping"] = bool(mock_env.speed_integral == 50.0 and math.isclose(pid_val, 15.0))

    scen_a = {"map": "Town04", "traffic_density": "low", "weather": "ClearNoon", "seed": 42}
    scen_b = {"map": "Town04", "traffic_density": "low", "weather": "ClearNoon", "seed": 42}
    tests["condition_id_stability"] = generate_condition_id(scen_a) == generate_condition_id(scen_b)

    report_paths = generate_report_data(config)
    tests["report_macro_generation"] = all(path.exists() for path in report_paths)
    success = all(tests.values())
    result = {
        "timestamp": utc_now(),
        "success": success,
        "tests": tests,
        "carla_server_required": False,
    }
    write_json(artifact_root / "logs/runtime/offline_self_test.json", result)
    print(json.dumps(result, indent=2))
    if not success:
        failed_keys = [k for k, v in tests.items() if not v]
        raise RuntimeError("One or more offline self-tests failed: %s" % ", ".join(failed_keys))


def manifest_hash(rows):
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compatible_manifest_spawns(env, candidate):
    spawn_points = env.map.get_spawn_points()
    ego_transform = spawn_points[candidate["carla_spawn_index"]]
    ego_waypoint = env.map.get_waypoint(
        ego_transform.location,
        project_to_road=True,
        lane_type=env.target_waypoint.lane_type if env.target_waypoint else None,
    )
    radius = env.config["traffic"]["background_spawn_radius_m"]
    indices = []
    for index, transform in enumerate(spawn_points):
        if index == candidate["carla_spawn_index"]:
            continue
        distance = transform.location.distance(ego_transform.location)
        if distance < 12.0 or distance > radius:
            continue
        waypoint = env.map.get_waypoint(transform.location, project_to_road=True)
        if waypoint is None or waypoint.is_junction:
            continue
        if not env._same_direction(ego_waypoint, waypoint):
            continue
        indices.append(index)
    return sorted(indices)


def generate_condition_id(scenario):
    traffic_density = str(scenario.get("traffic_density", "low"))
    weather = str(scenario.get("weather", "ClearNoon"))
    seed = int(scenario.get("seed", 0))
    canonical = {
        "map": str(scenario.get("map", "Town04")),
        "traffic_density": traffic_density,
        "weather": weather,
        "seed": seed,
        "ego_highway_candidate_index": scenario.get("ego_highway_candidate_index", 0),
        "lead_distance_m": float(scenario.get("lead_distance_m", 35.0)),
        "lead_speed_kmh": float(scenario.get("lead_speed_kmh", 40.0)),
        "npc_spawn_indices": sorted([int(x) for x in scenario.get("npc_spawn_indices", [])]),
        "npc_blueprints": list(scenario.get("npc_blueprints", [])),
        "npc_desired_speeds_kmh": [round(float(x), 4) for x in scenario.get("npc_desired_speeds_kmh", [])],
        "npc_following_distances_m": [round(float(x), 4) for x in scenario.get("npc_following_distances_m", [])],
    }
    raw_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    short_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:8]
    prefix = "%s_%s_seed%s" % (traffic_density, weather.lower(), seed)
    return "%s_%s" % (prefix, short_hash)


def command_make_eval_manifest(args):
    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    filename = (
        "evaluation_manifest_quick.json" if args.quick else "evaluation_manifest.json"
    )
    output = artifact_root / "manifests" / filename
    if output.exists() and not args.force:
        raise FileExistsError(
            "Manifest already exists: %s. Use --force only to deliberately replace it."
            % output
        )
    env = None
    try:
        env = HighwayDecisionEnv(config, mode="train", artifact_root=artifact_root)
        candidate = env.highway_candidates[0]
        env.target_waypoint = env.map.get_waypoint(
            env.map.get_spawn_points()[candidate["carla_spawn_index"]].location,
            project_to_road=True,
        )
        spawn_indices = compatible_manifest_spawns(env, candidate)
        blueprints = [b.id for b in get_safe_vehicle_blueprints(env.world.get_blueprint_library())]
        seeds = (
            config["evaluation"]["quick_seeds"]
            if args.quick
            else config["evaluation"]["seeds"]
        )
        rows = []
        condition_index = 0
        for density in config["evaluation"]["traffic_densities"]:
            for weather in config["evaluation"]["weather_presets"]:
                for seed in seeds:
                    rng = np.random.default_rng(seed + condition_index * 1009)
                    requested = config["traffic"]["density_counts"][density]
                    background_count = max(0, requested - 1)
                    selected_spawns = []
                    if spawn_indices:
                        order = rng.permutation(len(spawn_indices))
                        selected_spawns = [
                            spawn_indices[index]
                            for index in order[:background_count]
                        ]
                    selected_blueprints = [
                        blueprints[int(rng.integers(0, len(blueprints)))]
                        for _ in selected_spawns
                    ]
                    speeds = rng.uniform(
                        *config["traffic"]["npc_speed_range_kmh"],
                        size=len(selected_spawns),
                    ).round(4).tolist()
                    distances = rng.uniform(
                        *config["traffic"]["npc_following_distance_range_m"],
                        size=len(selected_spawns),
                    ).round(4).tolist()
                    scen = {
                        "map": config["carla"]["map"],
                        "seed": int(seed),
                        "weather": weather,
                        "traffic_density": density,
                        "ego_highway_candidate_index": candidate["candidate_index"],
                        "requested_npc_count": requested,
                        "lead_distance_m": round(
                            float(
                                rng.uniform(
                                    *config["traffic"]["lead_distance_range_m"]
                                )
                            ),
                            4,
                        ),
                        "lead_speed_kmh": round(
                            float(
                                rng.uniform(
                                    *config["traffic"]["lead_speed_range_kmh"]
                                )
                            ),
                            4,
                        ),
                        "npc_spawn_indices": selected_spawns,
                        "npc_blueprints": selected_blueprints,
                        "npc_desired_speeds_kmh": speeds,
                        "npc_following_distances_m": distances,
                        "npc_auto_lane_change": config["traffic"][
                            "npc_auto_lane_change"
                        ],
                        "max_episode_seconds": config["environment"][
                            "max_episode_seconds"
                        ],
                        "target_route_distance_m": config["environment"][
                            "target_route_distance_m"
                        ],
                    }
                    scen["condition_id"] = generate_condition_id(scen)
                    rows.append(scen)
                    condition_index += 1
    finally:
        if env is not None:
            env.close()
    payload = {
        "created_at": utc_now(),
        "carla_version": config["carla"]["version"],
        "map": config["carla"]["map"],
        "quick": bool(args.quick),
        "config_hash": config_hash(config),
        "manifest_hash": manifest_hash(rows),
        "scenario_count": len(rows),
        "scenarios": rows,
    }
    write_json(output, payload)
    print("Wrote %s scenarios to %s" % (len(rows), output))
    print("Manifest SHA-256:", payload["manifest_hash"])


def flatten_episode_metrics(metrics):
    row = dict(metrics)
    action_counts = row.pop("action_counts", {})
    for action_name in ACTION_NAMES.values():
        row["action_count_%s" % action_name.lower()] = action_counts.get(action_name, 0)
    reward_totals = row.pop("reward_component_totals", {})
    for name, value in reward_totals.items():
        row["reward_%s" % name] = value
    return row


def append_csv_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def training_metadata(config, run_name, seed, started_at):
    metadata = {
        "project": config["project"]["name"],
        "run_name": run_name,
        "seed": seed,
        "started_at": started_at,
        "carla_version": str(config["carla"]["version"]),
        "carla_package_manifest": read_json_if_valid(
            package_manifest_path(config)
        ),
        "config_hash": config_hash(config),
        "packages": package_versions(),
        "git": git_metadata(),
        "machine": machine_metadata(),
    }
    try:
        import carla

        client = carla.Client(config["carla"]["host"], config["carla"]["port"])
        client.set_timeout(config["carla"]["client_timeout_seconds"])
        metadata["carla_client_version"] = client.get_client_version()
        metadata["carla_server_version"] = client.get_server_version()
    except (ImportError, RuntimeError) as exc:
        metadata["carla_version_error"] = str(exc)
    return metadata


def command_train(args):
    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before training.") from exc

    run_name = args.run_name
    run_directory = artifact_root / "models" / run_name
    checkpoint_directory = run_directory / "checkpoints"
    training_csv = artifact_root / "logs/train" / ("%s_episodes.csv" % run_name)
    tensorboard_directory = artifact_root / "logs/tensorboard" / run_name
    if args.resume:
        if not Path(args.resume).exists():
            raise FileNotFoundError("Resume checkpoint not found: %s" % args.resume)
        run_directory.mkdir(parents=True, exist_ok=True)
    else:
        if run_directory.exists() and any(run_directory.iterdir()):
            raise FileExistsError(
                "Run directory already contains files: %s. Use a new --run-name "
                "or resume an explicit checkpoint." % run_directory
            )
        if training_csv.exists():
            raise FileExistsError(
                "Training episode log already exists: %s." % training_csv
            )
        run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    tensorboard_directory.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_directory / "resolved_config.yaml")
    snapshot_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_resolved_config(
        config, run_directory / ("resolved_config_%s.yaml" % snapshot_stamp)
    )
    metadata = training_metadata(config, run_name, args.seed, utc_now())
    write_json(run_directory / "training_metadata.json", metadata)
    write_json(artifact_root / "manifests/training_metadata.json", metadata)

    class EpisodeCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_checkpoint = 0
            self.episode_count = 0
            self.recent = []

        def _on_training_start(self):
            self.last_checkpoint = self.model.num_timesteps

        def _on_step(self):
            for info in self.locals.get("infos", []):
                metrics = info.get("episode_metrics")
                if metrics is None:
                    continue
                self.episode_count += 1
                row = flatten_episode_metrics(metrics)
                row.update(
                    {
                        "run_name": run_name,
                        "seed": args.seed,
                        "total_timesteps": self.num_timesteps,
                        "timestamp": utc_now(),
                    }
                )
                append_csv_row(training_csv, row)
                self.recent.append(row)
                self.recent = self.recent[-20:]
                rolling = {
                    "episodes": self.episode_count,
                    "total_timesteps": self.num_timesteps,
                    "rolling_return": float(
                        np.mean([item["episode_return"] for item in self.recent])
                    ),
                    "rolling_success": float(
                        np.mean([item["success"] for item in self.recent])
                    ),
                    "rolling_collision": float(
                        np.mean([item["collision"] for item in self.recent])
                    ),
                }
                write_json(run_directory / "rolling_summary.json", rolling)
                print(
                    "episode=%s steps=%s return=%.2f success=%s collision=%s"
                    % (
                        self.episode_count,
                        self.num_timesteps,
                        row["episode_return"],
                        row["success"],
                        row["collision"],
                    )
                )
            frequency = int(config["ppo"]["checkpoint_frequency"])
            if self.num_timesteps - self.last_checkpoint >= frequency:
                path = checkpoint_directory / ("checkpoint_%09d" % self.num_timesteps)
                self.model.save(str(path))
                self.last_checkpoint = self.num_timesteps
                print("Saved checkpoint:", str(path) + ".zip")
            return True

    env = None
    started = time.monotonic()
    try:
        env = HighwayDecisionEnv(config, mode="train", artifact_root=artifact_root)
        monitored = Monitor(env)
        if args.resume:
            model = PPO.load(args.resume, env=monitored, device=config["ppo"]["device"])
            budget = args.additional_timesteps
            if budget is None:
                raise ValueError("--resume requires --additional-timesteps.")
            reset_num_timesteps = False
        else:
            budget = args.total_timesteps or config["ppo"]["total_timesteps"]
            policy_kwargs = {
                "net_arch": config["ppo"]["net_arch"],
                "activation_fn": torch.nn.Tanh,
            }
            model = PPO(
                config["ppo"]["policy"],
                monitored,
                learning_rate=config["ppo"]["learning_rate"],
                n_steps=config["ppo"]["n_steps"],
                batch_size=config["ppo"]["batch_size"],
                n_epochs=config["ppo"]["n_epochs"],
                gamma=config["ppo"]["gamma"],
                gae_lambda=config["ppo"]["gae_lambda"],
                clip_range=config["ppo"]["clip_range"],
                ent_coef=config["ppo"]["ent_coef"],
                vf_coef=config["ppo"]["vf_coef"],
                max_grad_norm=config["ppo"]["max_grad_norm"],
                tensorboard_log=(
                    str(tensorboard_directory)
                    if config["ppo"]["tensorboard"]
                    else None
                ),
                policy_kwargs=policy_kwargs,
                seed=args.seed,
                device=config["ppo"]["device"],
                verbose=1,
            )
            reset_num_timesteps = True
        model.learn(
            total_timesteps=int(budget),
            callback=EpisodeCallback(),
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=run_name,
        )
        model.save(str(run_directory / "final_model"))
        generate_training_plots(config, training_csv)
    finally:
        if env is not None:
            env.close()
    metadata["ended_at"] = utc_now()
    metadata["wall_clock_seconds"] = time.monotonic() - started
    metadata["requested_training_decisions"] = int(budget)
    metadata["training_decisions"] = int(model.num_timesteps)
    metadata["final_model_sha256"] = file_sha256(
        run_directory / "final_model.zip"
    )
    if args.resume:
        metadata["resumed_from"] = str(resolve_path(args.resume))
    write_json(run_directory / "training_metadata.json", metadata)
    write_json(artifact_root / "manifests/training_metadata.json", metadata)
    print("Saved final model:", run_directory / "final_model.zip")


def load_manifest(path):
    path = resolve_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scenarios", [])
    actual_hash = manifest_hash(rows)
    if payload.get("manifest_hash") != actual_hash:
        raise ValueError("Manifest hash does not match its scenario rows: %s" % path)
    return payload, path


def selected_evaluation_policies(args, config):
    if args.policy:
        names = [args.policy]
    elif args.policies:
        names = args.policies
    else:
        names = config["evaluation"]["policies"]
    invalid = [name for name in names if name not in config["evaluation"]["policies"]]
    if invalid:
        raise ValueError("Unknown policies: %s" % ", ".join(invalid))
    return names


def command_evaluate(args):
    import pandas as pd

    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    evaluation_snapshot = (
        artifact_root
        / "evaluations"
        / (
            "resolved_config_%s.yaml"
            % datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
    save_resolved_config(config, evaluation_snapshot)
    manifest, manifest_path = load_manifest(args.manifest)
    scenarios = manifest["scenarios"]
    if args.quick:
        quick_seeds = set(config["evaluation"]["quick_seeds"])
        scenarios = [row for row in scenarios if row["seed"] in quick_seeds]
    start = args.start_index or 0
    end = args.end_index if args.end_index is not None else len(scenarios)
    scenarios = scenarios[start:end]
    policies = selected_evaluation_policies(args, config)
    if "ppo" in policies and not args.model:
        raise ValueError("PPO evaluation requires --model.")
    model_hash_value = file_sha256(args.model) if "ppo" in policies else ""

    default_name = "episode_results_quick.csv" if (manifest.get("quick") or args.quick) else "episode_results.csv"
    output = Path(args.output) if args.output else (artifact_root / "evaluations" / default_name)

    existing = pd.DataFrame()
    completed = set()
    m_hash = manifest["manifest_hash"]
    c_hash = config_hash(config)

    if output.exists() and output.stat().st_size > 0:
        if not args.resume_existing:
            raise FileExistsError(
                "Evaluation output already exists: %s. Use --resume-existing or specify --output."
                % output
            )
        existing = pd.read_csv(output)
        if "manifest_hash" in existing.columns:
            mismatched = existing[existing["manifest_hash"] != m_hash]
            if not mismatched.empty:
                raise RuntimeError(
                    "Existing output CSV %s contains rows from a different manifest_hash (%s vs %s). "
                    "Specify --output <new_file> or remove stale CSV."
                    % (output, mismatched["manifest_hash"].iloc[0], m_hash)
                )
        if "policy" in existing.columns and "condition_id" in existing.columns:
            for _, row in existing.iterrows():
                row_m_hash = str(row.get("manifest_hash", m_hash))
                completed.add((row_m_hash, str(row["policy"]), str(row["condition_id"])))

    env = None
    try:
        first_scenario = scenarios[0] if scenarios else None
        if first_scenario is None:
            raise ValueError("No scenarios selected for evaluation.")
        env = HighwayDecisionEnv(
            config,
            mode="evaluate",
            scenario=first_scenario,
            artifact_root=artifact_root,
        )
        for policy_name in policies:
            policy = build_policy(
                policy_name,
                config,
                env,
                seed=config["project"]["default_seed"],
                model_path=args.model,
            )
            for scenario in scenarios:
                condition_id = scenario["condition_id"]
                key = (m_hash, policy_name, condition_id)
                if key in completed:
                    print("Skipping completed", key)
                    continue
                env.scenario = scenario
                policy.reset(scenario["seed"])
                started = time.monotonic()
                observation, info = env.reset(seed=scenario["seed"])
                terminated = False
                truncated = False
                final_info = info
                while not (terminated or truncated):
                    action = policy.act(observation, env.current_state)
                    observation, reward, terminated, truncated, final_info = env.step(
                        action
                    )
                metrics = flatten_episode_metrics(final_info["episode_metrics"])
                outcome = "success" if metrics.get("success") else ("collision" if metrics.get("collision") else ("offroad" if metrics.get("offroad") else ("stuck" if metrics.get("stuck") else "timeout")))
                row = {
                    "manifest_hash": m_hash,
                    "config_hash": c_hash,
                    "condition_id": condition_id,
                    "policy": policy_name,
                    "seed": scenario["seed"],
                    "traffic_density": scenario["traffic_density"],
                    "weather": scenario["weather"],
                    "outcome": outcome,
                    "duration_seconds": metrics.get("elapsed_seconds", 0.0),
                    "route_progress_m": metrics.get("traveled_distance_m", 0.0),
                    "mean_speed_kmh": metrics.get("mean_speed_kmh", 0.0),
                    "max_speed_kmh": metrics.get("max_speed_kmh", 0.0),
                    "safety_override_activation_events": metrics.get("safety_override_activation_events", 0),
                    "safety_override_active_seconds": metrics.get("safety_override_active_seconds", 0.0),
                    "lane_changes_completed": metrics.get("completed_lane_changes", metrics.get("lane_changes_completed", 0)),
                    "lane_changes_aborted": metrics.get("aborted_lane_changes", metrics.get("lane_change_aborted_count", 0)),
                    "collision_count": 1 if metrics.get("collision") else 0,
                    "offroad_count": 1 if metrics.get("offroad") else 0,
                    "stuck_count": 1 if metrics.get("stuck") else 0,
                    "timestamp": utc_now(),
                    **metrics,
                    "runtime_wall_clock_seconds": time.monotonic() - started,
                    "model_path_or_baseline_version": (
                        str(resolve_path(args.model))
                        if policy_name == "ppo"
                        else policy.version
                    ),
                    "model_sha256": (
                        model_hash_value if policy_name == "ppo" else ""
                    ),
                    "manifest_path": str(manifest_path),
                }
                append_csv_row(output, row)
                completed.add(key)
                print(
                    "%s %s outcome=%s route_progress=%.1fm speed=%.1fkmh"
                    % (
                        policy_name,
                        condition_id,
                        outcome,
                        row["route_progress_m"],
                        row["mean_speed_kmh"],
                    )
                )
    finally:
        if env is not None:
            env.close()
    print("Evaluation rows written to:", output)


SUMMARY_METRICS = {
    "success": "success_rate",
    "collision": "collision_rate",
    "offroad": "offroad_rate",
    "timeout": "timeout_rate",
    "route_completion": "mean_route_completion",
    "mean_speed_kmh": "average_speed_kmh",
    "completion_time_seconds": "successful_completion_time_seconds",
    "completed_lane_changes": "mean_completed_lane_changes",
    "unsafe_lane_change_requests": "mean_unsafe_lane_changes",
    "minimum_ttc_seconds": "mean_minimum_ttc_seconds",
    "episode_return": "mean_episode_return",
}


def group_specifications():
    return [
        ("overall", ["policy"]),
        ("density", ["policy", "traffic_density"]),
        ("weather", ["policy", "weather"]),
        (
            "density_weather",
            ["policy", "traffic_density", "weather"],
        ),
    ]


def summarize_evaluations(data):
    rows = []
    for group_type, columns in group_specifications():
        for keys, group in data.groupby(columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {
                "group_type": group_type,
                "traffic_density": "all",
                "weather": "all",
                "n_episodes": len(group),
            }
            row.update(dict(zip(columns, keys)))
            for source, destination in SUMMARY_METRICS.items():
                values = group[source].dropna()
                row[destination] = float(values.mean()) if len(values) else np.nan
            for binary in ("success", "collision", "offroad", "timeout"):
                lower, upper = wilson_interval(
                    float(group[binary].sum()), len(group)
                )
                row["%s_ci_lower" % binary] = lower
                row["%s_ci_upper" % binary] = upper
            rows.append(row)
    return rows


def bootstrap_summaries(data, config):
    rows = []
    continuous = [
        "route_completion",
        "mean_speed_kmh",
        "completion_time_seconds",
        "completed_lane_changes",
        "unsafe_lane_change_requests",
        "minimum_ttc_seconds",
        "episode_return",
    ]
    resamples = config["evaluation"]["bootstrap_resamples"]
    for group_type, columns in group_specifications():
        for keys, group in data.groupby(columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            identity = dict(zip(columns, keys))
            for metric_index, metric in enumerate(continuous):
                lower, upper = bootstrap_interval(
                    group[metric].to_numpy(),
                    resamples,
                    config["project"]["default_seed"] + metric_index,
                )
                rows.append(
                    {
                        "group_type": group_type,
                        "traffic_density": identity.get("traffic_density", "all"),
                        "weather": identity.get("weather", "all"),
                        **identity,
                        "metric": metric,
                        "mean": float(group[metric].mean()),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "resamples": resamples,
                    }
                )
    return rows


def paired_comparisons(data, config):
    import pandas as pd

    rows = []
    ppo = data[data["policy"] == "ppo"].set_index("condition_id")
    metrics = [
        "success",
        "collision",
        "route_completion",
        "mean_speed_kmh",
        "completion_time_seconds",
    ]
    for baseline_name in ("random", "keep_lane", "rule_based"):
        baseline = data[data["policy"] == baseline_name].set_index("condition_id")
        common = ppo.index.intersection(baseline.index)
        for metric_index, metric in enumerate(metrics):
            paired = pd.concat(
                [ppo.loc[common, metric], baseline.loc[common, metric]], axis=1
            )
            paired.columns = ["ppo", "baseline"]
            paired = paired.dropna()
            differences = paired["ppo"] - paired["baseline"]
            lower, upper = bootstrap_interval(
                differences.to_numpy(),
                config["evaluation"]["bootstrap_resamples"],
                config["project"]["default_seed"] + metric_index,
            )
            rows.append(
                {
                    "baseline": baseline_name,
                    "metric": metric,
                    "n_pairs": len(differences),
                    "ppo_minus_baseline_mean": (
                        float(differences.mean()) if len(differences) else np.nan
                    ),
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return rows


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    print("Wrote plot:", path)


def generate_training_plots(config, training_csv=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plot_root = Path(config["paths"]["artifact_root"]) / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    if training_csv is None:
        training_files = sorted(
            (Path(config["paths"]["artifact_root"]) / "logs/train").glob(
                "*_episodes.csv"
            )
        )
        if not training_files:
            warnings.warn(
                "No training episode CSV exists; skipped the two training plots."
            )
            return False
        training_csv = training_files[-1]
    training = pd.read_csv(training_csv)
    if training.empty:
        warnings.warn("Training episode CSV is empty; skipped training plots.")
        return False
    rolling_return = training["episode_return"].rolling(20, min_periods=1).mean()
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(
        training.index + 1,
        training["episode_return"],
        alpha=0.35,
        label="Raw",
    )
    axis.plot(training.index + 1, rolling_return, label="20-episode mean")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Return")
    axis.set_title("Training episode return")
    axis.legend()
    save_figure(figure, plot_root / "training_episode_return.png")

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(
        training.index + 1,
        training["success"].rolling(20, min_periods=1).mean(),
        label="Success",
    )
    axis.plot(
        training.index + 1,
        training["collision"].rolling(20, min_periods=1).mean(),
        label="Collision",
    )
    axis.set_xlabel("Episode")
    axis.set_ylabel("Rolling rate")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Training outcomes (20-episode rolling rates)")
    axis.legend()
    save_figure(figure, plot_root / "training_success_collision.png")
    return True


def generate_plots(data, summary, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_root = Path(config["paths"]["artifact_root"]) / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    overall = summary[summary["group_type"] == "overall"].copy()
    policy_order = [
        name
        for name in config["evaluation"]["policies"]
        if name in set(overall["policy"])
    ]
    overall = overall.set_index("policy").loc[policy_order].reset_index()

    figure, axis = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(overall))
    width = 0.36
    success_error = np.vstack(
        [
            overall["success_rate"] - overall["success_ci_lower"],
            overall["success_ci_upper"] - overall["success_rate"],
        ]
    )
    collision_error = np.vstack(
        [
            overall["collision_rate"] - overall["collision_ci_lower"],
            overall["collision_ci_upper"] - overall["collision_rate"],
        ]
    )
    axis.bar(
        x - width / 2,
        overall["success_rate"],
        width,
        yerr=success_error,
        capsize=3,
        label="Success",
    )
    axis.bar(
        x + width / 2,
        overall["collision_rate"],
        width,
        yerr=collision_error,
        capsize=3,
        label="Collision",
    )
    axis.set_xticks(x, policy_order)
    axis.set_ylabel("Episode rate")
    axis.set_ylim(0.0, 1.05)
    axis.legend()
    axis.set_title("Held-out success and collision rates (95% Wilson CI)")
    save_figure(figure, plot_root / "policy_success_collision.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(policy_order, overall["mean_route_completion"])
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Mean route completion")
    axis.set_title("Route completion by policy")
    save_figure(figure, plot_root / "policy_route_completion.png")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(policy_order, overall["average_speed_kmh"])
    axes[0].set_ylabel("Mean speed (km/h)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(policy_order, overall["successful_completion_time_seconds"])
    axes[1].set_ylabel("Completion time, successful episodes (s)")
    axes[1].tick_params(axis="x", rotation=20)
    figure.suptitle("Efficiency metrics")
    save_figure(figure, plot_root / "policy_speed_time.png")

    densities = config["evaluation"]["traffic_densities"]
    weather = config["evaluation"]["weather_presets"]
    matrix = np.full((len(policy_order) * len(densities), len(weather)), np.nan)
    labels = []
    for policy_index, policy_name in enumerate(policy_order):
        for density_index, density in enumerate(densities):
            row_index = policy_index * len(densities) + density_index
            labels.append("%s / %s" % (policy_name, density))
            for weather_index, weather_name in enumerate(weather):
                subset = data[
                    (data["policy"] == policy_name)
                    & (data["traffic_density"] == density)
                    & (data["weather"] == weather_name)
                ]
                if len(subset):
                    matrix[row_index, weather_index] = subset["success"].mean()
    figure, axis = plt.subplots(figsize=(8, 8))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(weather)), weather, rotation=20)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title("Success rate by density and weather")
    figure.colorbar(image, ax=axis, label="Success rate")
    save_figure(figure, plot_root / "density_heatmap_success.png")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    behavior = (
        data.groupby("policy")[
            ["completed_lane_changes", "unsafe_lane_change_requests"]
        ]
        .mean()
        .reindex(policy_order)
    )
    behavior.plot.bar(ax=axis)
    axis.set_ylabel("Mean count per episode")
    axis.set_title("Lane-change behavior")
    axis.legend(["Completed", "Rejected unsafe/illegal"])
    save_figure(figure, plot_root / "lane_change_behavior.png")

    ppo = data[data["policy"] == "ppo"]
    if len(ppo):
        action_columns = [
            "action_count_%s" % name.lower() for name in ACTION_NAMES.values()
        ]
        overall_actions = ppo[action_columns].sum()
        density_actions = ppo.groupby("traffic_density")[action_columns].sum()
        normalized = density_actions.div(density_actions.sum(axis=1), axis=0)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        (overall_actions / max(overall_actions.sum(), 1)).plot.bar(ax=axes[0])
        axes[0].set_title("PPO overall")
        axes[0].set_ylabel("Action fraction")
        normalized.plot.bar(ax=axes[1])
        axes[1].set_title("PPO by traffic density")
        axes[1].set_ylabel("Action fraction")
        axes[1].legend(
            [ACTION_NAMES[index] for index in range(5)],
            fontsize=8,
        )
        save_figure(figure, plot_root / "ppo_action_distribution.png")
    else:
        warnings.warn("No PPO rows; skipped ppo_action_distribution.png.")

    generate_training_plots(config)


def command_analyze(args):
    import pandas as pd

    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    episode_path = resolve_path(args.episodes)
    data = pd.read_csv(episode_path)
    required = {
        "policy",
        "condition_id",
        "traffic_density",
        "weather",
        *SUMMARY_METRICS.keys(),
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Episode CSV is missing columns: %s" % ", ".join(missing))
    for binary in ("success", "collision", "offroad", "timeout"):
        data[binary] = data[binary].astype(int)
    summary_rows = summarize_evaluations(data)
    bootstrap_rows = bootstrap_summaries(data, config)
    paired_rows = paired_comparisons(data, config)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(artifact_root / "evaluations/summary_results.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        artifact_root / "evaluations/bootstrap_intervals.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        artifact_root / "evaluations/paired_comparisons.csv", index=False
    )
    generate_plots(data, summary, config)
    print(
        summary[summary["group_type"] == "overall"].to_string(index=False)
    )


def video_selection_candidates(data, category):
    if category == "ppo_safe_overtake":
        return data[
            (data["policy"] == "ppo")
            & (data["success"].astype(bool))
            & (data["completed_lane_changes"] > 0)
            & (data["route_completion"] >= 0.9)
        ].sort_values(["completed_lane_changes", "route_completion"], ascending=False)
    if category == "ppo_rejected_unsafe_lane_change":
        return data[
            (data["policy"] == "ppo")
            & (data["unsafe_lane_change_requests"] > 0)
        ].sort_values("unsafe_lane_change_requests", ascending=False)
    if category == "ppo_dense_traffic_success":
        return data[
            (data["policy"] == "ppo")
            & (data["traffic_density"] == "high")
            & (data["success"].astype(bool))
        ].sort_values("route_completion", ascending=False)
    if category == "keep_lane_blocked":
        return data[
            (data["policy"] == "keep_lane")
            & ((data["timeout"].astype(bool)) | (data["mean_speed_kmh"] < 45.0))
        ].sort_values("mean_speed_kmh")
    if category == "random_failure":
        return data[
            (data["policy"] == "random")
            & (
                data["collision"].astype(bool)
                | data["offroad"].astype(bool)
                | data["stuck"].astype(bool)
                | data["timeout"].astype(bool)
            )
        ].sort_values(["collision", "route_completion"], ascending=[False, True])
    if category == "ppo_failure":
        return data[
            (data["policy"] == "ppo")
            & (
                data["collision"].astype(bool)
                | data["offroad"].astype(bool)
                | data["stuck"].astype(bool)
                | data["timeout"].astype(bool)
            )
        ].sort_values(["collision", "route_completion"], ascending=[False, True])
    raise ValueError("Unknown video category %r" % category)


def command_select_videos(args):
    import pandas as pd

    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    data = pd.read_csv(resolve_path(args.episodes))
    selections = []
    for category in config["video"]["categories"]:
        candidates = video_selection_candidates(data, category)
        fallback_reason = ""
        if candidates.empty:
            print("No matching episode exists for", category)
            if not args.allow_fallback:
                continue
            required_policy = "keep_lane" if category == "keep_lane_blocked" else (
                "random" if category == "random_failure" else "ppo"
            )
            candidates = data[data["policy"] == required_policy].sort_values(
                "route_completion"
            )
            if candidates.empty:
                continue
            fallback_reason = (
                "Explicit fallback requested; no episode met the category criteria."
            )
        selected = candidates.iloc[0]
        selections.append(
            {
                "category": category,
                "policy": selected["policy"],
                "condition_id": selected["condition_id"],
                "reason_selected": fallback_reason or "Matched category criteria.",
                "model": str(resolve_path(args.model)) if args.model else "",
                "mp4_path": "",
                "carla_recording_path": "",
                "frame_count": 0,
                "duration": 0.0,
                "outcome": "",
            }
        )
    output = artifact_root / "manifests/video_manifest.json"
    write_json(
        output,
        {
            "created_at": utc_now(),
            "episodes": str(resolve_path(args.episodes)),
            "selections": selections,
        },
    )
    print("Selected %s videos in %s" % (len(selections), output))


def overlay_frame(frame, env, policy_name, action_name, outcome=None):
    import cv2

    state = env.current_state or {}
    current = state.get("lanes", {}).get("current", {})
    lines = [
        "%s | %s" % (policy_name, action_name),
        "speed %.1f km/h | target %.1f km/h"
        % (state.get("speed_kmh", 0.0), state.get("target_speed_kmh", 0.0)),
        "lane %s | target lane %s"
        % (state.get("current_lane_id", "-"), state.get("target_lane_id", "-")),
        "front gap %.1f m | min TTC %.2f s"
        % (
            current.get("front_gap_m", 0.0),
            env._finite_or_radius(env.minimum_ttc_seconds),
        ),
        "return %.2f | completion %.1f%%"
        % (
            env.episode_return,
            100.0
            * min(
                1.0,
                env.traveled_distance_m
                / env.config["environment"]["target_route_distance_m"],
            ),
        ),
        "lane change accepted=%s rejected=%s"
        % (env.last_lane_change_accepted, env.last_lane_change_rejected),
    ]
    canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (15, 15), (650, 205), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0.0, canvas)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (30, 45 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if outcome:
        cv2.rectangle(canvas, (220, canvas.shape[0] - 100), (1060, canvas.shape[0] - 25), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            outcome,
            (245, canvas.shape[0] - 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def outcome_text(metrics):
    if metrics["success"]:
        return "SUCCESS"
    for field in ("collision", "offroad", "stuck", "timeout"):
        if metrics[field]:
            return "FAILURE: %s" % field.upper()
    return "EPISODE ENDED"


def render_one_video(config, scenario, policy_name, model_path, output_path,
                     recording_path=None):
    import cv2

    env = None
    writer = None
    frame_count = 0
    try:
        env = HighwayDecisionEnv(
            config,
            mode="render",
            scenario=scenario,
            artifact_root=config["paths"]["artifact_root"],
            render_mode="rgb_array",
        )
        if env.world.get_settings().no_rendering_mode:
            raise RuntimeError(
                "Rendering requires world.no_rendering_mode=false. Restart managed "
                "CARLA in rendering mode or correct external server settings."
            )
        policy = build_policy(
            policy_name,
            config,
            env,
            seed=scenario["seed"],
            model_path=model_path,
        )
        policy.reset(scenario["seed"])
        video = config["video"]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*video["codec"]),
            video["fps"],
            (video["width"], video["height"]),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open video writer: %s" % output_path)
        observation, info = env.reset(seed=scenario["seed"])
        if recording_path:
            env.start_recorder(recording_path)
        terminated = False
        truncated = False
        final_info = info
        last_action_name = ACTION_NAMES[0]
        while not (terminated or truncated):
            action = policy.act(observation, env.current_state)
            last_action_name = ACTION_NAMES[action]
            observation, reward, terminated, truncated, final_info = env.step(action)
            frames = env.drain_render_frames()
            if not frames:
                raise RuntimeError("Render environment returned no RGB frame.")
            for frame in frames:
                writer.write(
                    overlay_frame(frame, env, policy_name, last_action_name)
                )
                frame_count += 1
        metrics = final_info["episode_metrics"]
        final_text = outcome_text(metrics)
        final_frame = overlay_frame(
            env.render(), env, policy_name, last_action_name, final_text
        )
        for _ in range(config["video"]["fps"] * 2):
            writer.write(final_frame)
            frame_count += 1
        return {
            "frame_count": frame_count,
            "duration": frame_count / config["video"]["fps"],
            "outcome": final_text,
            "metrics": metrics,
        }
    finally:
        if writer is not None:
            writer.release()
        if env is not None:
            env.close()


def command_render_videos(args):
    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    manifest, manifest_path = load_manifest(args.manifest)
    scenario_lookup = {
        row["condition_id"]: row for row in manifest["scenarios"]
    }
    video_manifest_path = resolve_path(
        args.video_manifest
        or artifact_root / "manifests/video_manifest.json"
    )
    if args.policy and args.condition_id:
        selections = [
            {
                "category": "explicit_%s_%s" % (args.policy, args.condition_id),
                "policy": args.policy,
                "condition_id": args.condition_id,
                "reason_selected": "Explicit CLI selection.",
                "model": str(resolve_path(args.model)) if args.model else "",
                "mp4_path": "",
                "carla_recording_path": "",
                "frame_count": 0,
                "duration": 0.0,
                "outcome": "",
            }
        ]
        payload = {"created_at": utc_now(), "selections": selections}
    else:
        payload = json.loads(video_manifest_path.read_text(encoding="utf-8"))
        selections = payload["selections"]
        if args.category:
            selections = [
                item for item in selections if item["category"] == args.category
            ]
            if not selections:
                print("Category %s is absent from %s (skipped rendering)" % (args.category, video_manifest_path))
                return
    for selection in selections:
        condition_id = selection["condition_id"]
        if condition_id not in scenario_lookup:
            raise KeyError(
                "Video condition %s is absent from %s"
                % (condition_id, manifest_path)
            )
        output = artifact_root / "videos" / ("%s.mp4" % selection["category"])
        if (
            args.resume_existing
            and output.exists()
            and output.stat().st_size > 1024
            and int(selection.get("frame_count", 0)) > 0
            and float(selection.get("duration", 0.0)) > 0.0
        ):
            print("Skipping existing video:", output)
            continue
        recording = None
        if config["video"]["save_carla_recording"]:
            recording = (
                artifact_root
                / "recordings"
                / ("%s.log" % selection["category"])
            )
        result = render_one_video(
            config,
            scenario_lookup[condition_id],
            selection["policy"],
            args.model or selection.get("model"),
            output,
            recording,
        )
        selection.update(
            {
                "model": str(resolve_path(args.model)) if args.model else selection.get("model", ""),
                "mp4_path": str(output),
                "carla_recording_path": str(recording) if recording else "",
                "frame_count": result["frame_count"],
                "duration": result["duration"],
                "outcome": result["outcome"],
            }
        )
        write_json(video_manifest_path, payload)
        print("Rendered:", output)
    write_json(video_manifest_path, payload)


def generated_macro(name, value):
    return "\\newcommand{\\%s}{%s}\n" % (name, latex_escape(value))


def format_percent(value):
    if value is None or not np.isfinite(float(value)):
        return "TBD"
    return "%.1f%%" % (100.0 * float(value))


def generate_report_data(config):
    import pandas as pd

    artifact_root = Path(config["paths"]["artifact_root"])
    generated = REPOSITORY_ROOT / "reports/generated"
    generated.mkdir(parents=True, exist_ok=True)
    values = {
        "TrainingDecisions": "TBD",
        "EvaluationEpisodes": "TBD",
        "PPOSuccessRate": "TBD",
        "PPOSuccessCI": "TBD",
        "PPOCollisionRate": "TBD",
        "PPOCollisionCI": "TBD",
        "PPORouteCompletion": "TBD",
        "PPOAverageSpeed": "TBD",
        "PPOCompletionTime": "TBD",
        "StrongestBaseline": "TBD",
        "PairedSuccessDifference": "TBD",
        "PairedCollisionDifference": "TBD",
        "ModelSeed": "TBD",
        "ManifestHash": "TBD",
        "CARLAVersion": config["carla"]["version"],
    }
    training_metadata_path = artifact_root / "manifests/training_metadata.json"
    if training_metadata_path.exists():
        metadata = json.loads(training_metadata_path.read_text(encoding="utf-8"))
        values["TrainingDecisions"] = metadata.get(
            "training_decisions",
            metadata.get("requested_training_decisions", "TBD"),
        )
        values["ModelSeed"] = metadata.get("seed", "TBD")
    manifest_path = artifact_root / "manifests/evaluation_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        values["ManifestHash"] = manifest.get("manifest_hash", "TBD")
    episodes_path = artifact_root / "evaluations/episode_results.csv"
    summary_path = artifact_root / "evaluations/summary_results.csv"
    paired_path = artifact_root / "evaluations/paired_comparisons.csv"
    summary = None
    paired = None
    if episodes_path.exists():
        episodes = pd.read_csv(episodes_path)
        values["EvaluationEpisodes"] = len(episodes)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        overall = summary[summary["group_type"] == "overall"]
        ppo = overall[overall["policy"] == "ppo"]
        if len(ppo):
            row = ppo.iloc[0]
            values.update(
                {
                    "PPOSuccessRate": format_percent(row["success_rate"]),
                    "PPOSuccessCI": "%s--%s"
                    % (
                        format_percent(row["success_ci_lower"]),
                        format_percent(row["success_ci_upper"]),
                    ),
                    "PPOCollisionRate": format_percent(row["collision_rate"]),
                    "PPOCollisionCI": "%s--%s"
                    % (
                        format_percent(row["collision_ci_lower"]),
                        format_percent(row["collision_ci_upper"]),
                    ),
                    "PPORouteCompletion": format_percent(
                        row["mean_route_completion"]
                    ),
                    "PPOAverageSpeed": "%.1f km/h" % row["average_speed_kmh"],
                    "PPOCompletionTime": (
                        "%.1f s" % row["successful_completion_time_seconds"]
                        if np.isfinite(row["successful_completion_time_seconds"])
                        else "TBD"
                    ),
                }
            )
        baselines = overall[overall["policy"] != "ppo"].sort_values(
            "success_rate", ascending=False
        )
        if len(baselines):
            values["StrongestBaseline"] = baselines.iloc[0]["policy"]
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        baseline = values["StrongestBaseline"]
        if baseline != "TBD":
            subset = paired[paired["baseline"] == baseline]
            success = subset[subset["metric"] == "success"]
            collision = subset[subset["metric"] == "collision"]
            if len(success):
                values["PairedSuccessDifference"] = format_percent(
                    success.iloc[0]["ppo_minus_baseline_mean"]
                )
            if len(collision):
                values["PairedCollisionDifference"] = format_percent(
                    collision.iloc[0]["ppo_minus_baseline_mean"]
                )

    value_path = generated / "report_values.tex"
    value_path.write_text(
        "".join(generated_macro(name, value) for name, value in values.items()),
        encoding="utf-8",
    )
    results_path = generated / "results_table.tex"
    if summary is None:
        results_text = (
            "\\begin{tabular}{lrrrr}\\toprule\n"
            "Policy & Success & Collision & Completion & Speed \\\\\\midrule\n"
            "\\multicolumn{5}{c}{Experiments pending} \\\\\\bottomrule\n"
            "\\end{tabular}\n"
        )
    else:
        overall = summary[summary["group_type"] == "overall"]
        lines = [
            "\\begin{tabular}{lrrrr}\\toprule",
            "Policy & Success & Collision & Completion & Speed (km/h) \\\\\\midrule",
        ]
        for _, row in overall.iterrows():
            lines.append(
                "%s & %s & %s & %s & %.1f \\\\"
                % (
                    latex_escape(row["policy"]),
                    latex_escape(format_percent(row["success_rate"])),
                    latex_escape(format_percent(row["collision_rate"])),
                    latex_escape(format_percent(row["mean_route_completion"])),
                    row["average_speed_kmh"],
                )
            )
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        results_text = "\n".join(lines) + "\n"
    results_path.write_text(results_text, encoding="utf-8")

    paired_table_path = generated / "paired_table.tex"
    if paired is None:
        paired_text = (
            "\\begin{tabular}{llrr}\\toprule\n"
            "Baseline & Metric & Difference & 95\\% CI \\\\\\midrule\n"
            "\\multicolumn{4}{c}{Experiments pending} \\\\\\bottomrule\n"
            "\\end{tabular}\n"
        )
    else:
        lines = [
            "\\begin{tabular}{llrr}\\toprule",
            "Baseline & Metric & Difference & 95\\% CI \\\\\\midrule",
        ]
        for _, row in paired.iterrows():
            lines.append(
                "%s & %s & %.3f & [%.3f, %.3f] \\\\"
                % (
                    latex_escape(row["baseline"]),
                    latex_escape(row["metric"]),
                    row["ppo_minus_baseline_mean"],
                    row["ci_lower"],
                    row["ci_upper"],
                )
            )
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        paired_text = "\n".join(lines) + "\n"
    paired_table_path.write_text(paired_text, encoding="utf-8")
    for plot in (artifact_root / "plots").glob("*.png"):
        shutil.copy2(plot, generated / plot.name)
    return [value_path, results_path, paired_table_path]


def command_report_data(args):
    config = load_config(args.config)
    paths = generate_report_data(config)
    print("Generated report inputs:")
    for path in paths:
        print(path)


def run_short_policy_episode(config, policy_class, seconds):
    test_config = copy.deepcopy(config)
    test_config["environment"]["max_episode_seconds"] = seconds
    test_config["environment"]["target_route_distance_m"] = 100.0
    env = HighwayDecisionEnv(test_config, mode="train")
    try:
        policy = (
            policy_class(test_config["rule_based"])
            if policy_class is RuleBasedOvertakingPolicy
            else policy_class()
        )
        policy.reset(0)
        observation, info = env.reset(seed=0)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy.act(observation, env.current_state)
            observation, reward, terminated, truncated, info = env.step(action)
        return info["episode_metrics"]
    finally:
        env.close()


def command_smoke(args):
    try:
        from gymnasium.utils.env_checker import check_env as gym_check_env
        from stable_baselines3.common.env_checker import check_env as sb3_check_env
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before smoke testing.") from exc
    config = load_config(args.config)
    artifact_root = ensure_artifact_layout(config)
    if args.render_frame_only:
        if not args.manifest:
            raise ValueError("--render-frame-only requires --manifest.")
        manifest, _ = load_manifest(args.manifest)
        if not manifest["scenarios"]:
            raise ValueError("Rendering smoke manifest contains no scenarios.")
        env = HighwayDecisionEnv(
            config,
            mode="render",
            scenario=manifest["scenarios"][0],
            artifact_root=artifact_root,
            render_mode="rgb_array",
        )
        try:
            available_maps = env.client.get_available_maps()
            if not any(item.lower().endswith("/" + config["carla"]["map"].lower()) or item.lower() == config["carla"]["map"].lower() for item in available_maps):
                raise RuntimeError("Required map %s is not available on CARLA server." % config["carla"]["map"])
            observation, info = env.reset(seed=manifest["scenarios"][0]["seed"])
            observation, reward, terminated, truncated, info = env.step(0)
            frame = env.render()
            if frame is None or frame.shape != (
                config["video"]["height"],
                config["video"]["width"],
                3,
            ):
                raise RuntimeError("RGB camera smoke frame has an invalid shape.")
            result = {
                "timestamp": utc_now(),
                "render_frame": "passed",
                "shape": list(frame.shape),
                "frame_id": env.last_render_frame_id,
                "simulator_frames_received": len(env.drain_render_frames()),
                "world_no_rendering_mode": bool(
                    env.world.get_settings().no_rendering_mode
                ),
            }
            write_json(
                artifact_root / "logs/runtime/render_frame_smoke.json", result
            )
            print(json.dumps(result, indent=2))
            return
        finally:
            env.close()
    test_config = copy.deepcopy(config)
    test_config["environment"]["max_episode_seconds"] = 3.0
    test_config["environment"]["target_route_distance_m"] = 50.0
    checker_env = HighwayDecisionEnv(test_config, mode="train")
    try:
        available_maps = checker_env.client.get_available_maps()
        if not any(item.lower().endswith("/" + test_config["carla"]["map"].lower()) or item.lower() == test_config["carla"]["map"].lower() for item in available_maps):
            raise RuntimeError("Required map %s is not available on CARLA server." % test_config["carla"]["map"])
        gym_check_env(checker_env, skip_render_check=True)
        sb3_check_env(checker_env, warn=True)
    finally:
        checker_env.close()
    keep_metrics = run_short_policy_episode(
        test_config, KeepLanePolicy, 5.0
    )
    rule_metrics = run_short_policy_episode(
        test_config, RuleBasedOvertakingPolicy, 5.0
    )
    fresh_env = HighwayDecisionEnv(test_config, mode="train")
    try:
        fresh_env.reset(seed=1)
        actor_count_after_reset = len(fresh_env.actors)
    finally:
        fresh_env.close()
        fresh_env.close()
    result = {
        "timestamp": utc_now(),
        "gymnasium_checker": "passed",
        "stable_baselines3_checker": "passed",
        "keep_lane_episode": keep_metrics,
        "rule_based_episode": rule_metrics,
        "fresh_environment_actor_count": actor_count_after_reset,
        "double_close": "passed",
    }
    write_json(artifact_root / "logs/runtime/smoke_test.json", result)
    print(json.dumps(result, indent=2, default=str))


def build_parser():
    parser = argparse.ArgumentParser(
        description="CARLA highway tactical decision-making workflows"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", default="config.yaml")
    validate.set_defaults(function=command_validate_config)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--config", default="config.yaml")
    doctor.set_defaults(function=command_doctor)

    runtime = subparsers.add_parser(
        "runtime", help="Inspect or provision the hosted CARLA runtime."
    )
    runtime_subparsers = runtime.add_subparsers(
        dest="runtime_command", required=True
    )
    runtime_status = runtime_subparsers.add_parser("status")
    runtime_status.add_argument("--config", default="config.yaml")
    runtime_status.add_argument("--verify-archive-hash", action="store_true")
    runtime_status.add_argument("--check-server", action="store_true")
    runtime_status.add_argument("--strict", action="store_true")
    runtime_status.set_defaults(function=command_runtime_status)
    runtime_prepare = runtime_subparsers.add_parser("prepare")
    runtime_prepare.add_argument("--config", default="config.yaml")
    runtime_prepare.add_argument("--force-download", action="store_true")
    runtime_prepare.add_argument("--force-extract", action="store_true")
    runtime_prepare.add_argument("--dry-run", action="store_true")
    runtime_prepare.add_argument("--no-drive-cache", action="store_true")
    runtime_prepare.add_argument("--keep-local-archive", action="store_true")
    runtime_prepare.add_argument(
        "--skip-gpu-vulkan-check", action="store_true"
    )
    runtime_prepare.set_defaults(function=command_runtime_prepare)
    runtime_clean = runtime_subparsers.add_parser("clean-local")
    runtime_clean.add_argument("--config", default="config.yaml")
    runtime_clean.add_argument("--dry-run", action="store_true")
    runtime_clean.set_defaults(function=command_runtime_clean_local)

    server = subparsers.add_parser("server")
    server_subparsers = server.add_subparsers(dest="server_command", required=True)
    server_start = server_subparsers.add_parser("start")
    server_start.add_argument("--config", default="config.yaml")
    server_start.add_argument("--rendering", action="store_true")
    server_start.add_argument(
        "--skip-gpu-vulkan-check", action="store_true"
    )
    server_start.set_defaults(function=command_server_start)
    server_status = server_subparsers.add_parser("status")
    server_status.add_argument("--config", default="config.yaml")
    server_status.set_defaults(function=command_server_status)
    server_stop = server_subparsers.add_parser("stop")
    server_stop.add_argument("--config", default="config.yaml")
    server_stop.set_defaults(function=command_server_stop)

    offline = subparsers.add_parser("offline-self-test")
    offline.add_argument("--config", default="config.yaml")
    offline.set_defaults(function=command_offline_self_test)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", default="config.yaml")
    smoke.add_argument("--render-frame-only", action="store_true")
    smoke.add_argument("--manifest")
    smoke.set_defaults(function=command_smoke)

    manifest = subparsers.add_parser("make-eval-manifest")
    manifest.add_argument("--config", default="config.yaml")
    manifest.add_argument("--quick", action="store_true")
    manifest.add_argument("--force", action="store_true")
    manifest.set_defaults(function=command_make_eval_manifest)

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="config.yaml")
    train.add_argument("--run-name", required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--total-timesteps", type=int)
    train.add_argument("--resume")
    train.add_argument("--additional-timesteps", type=int)
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="config.yaml")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--model")
    evaluate.add_argument("--policies", nargs="+")
    evaluate.add_argument("--policy")
    evaluate.add_argument("--quick", action="store_true")
    evaluate.add_argument("--start-index", type=int)
    evaluate.add_argument("--end-index", type=int)
    evaluate.add_argument("--resume-existing", action="store_true")
    evaluate.set_defaults(function=command_evaluate)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--config", default="config.yaml")
    analyze.add_argument("--episodes", required=True)
    analyze.set_defaults(function=command_analyze)

    select = subparsers.add_parser("select-videos")
    select.add_argument("--config", default="config.yaml")
    select.add_argument("--episodes", required=True)
    select.add_argument("--model")
    select.add_argument("--allow-fallback", action="store_true")
    select.set_defaults(function=command_select_videos)

    render = subparsers.add_parser("render-videos")
    render.add_argument("--config", default="config.yaml")
    render.add_argument("--manifest", required=True)
    render.add_argument("--video-manifest")
    render.add_argument("--model")
    render.add_argument("--policy")
    render.add_argument("--condition-id")
    render.add_argument("--category")
    render.add_argument("--resume-existing", action="store_true")
    render.set_defaults(function=command_render_videos)

    sync = subparsers.add_parser("sync")
    sync.add_argument("--config", default="config.yaml")
    sync.add_argument("--to-drive", action="store_true")
    sync.add_argument("--from-drive", action="store_true")
    sync.add_argument("--local-root")
    sync.add_argument("--drive-root")
    sync.add_argument("--dry-run", action="store_true")
    sync.set_defaults(function=command_sync)

    report = subparsers.add_parser("report-data")
    report.add_argument("--config", default="config.yaml")
    report.set_defaults(function=command_report_data)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
