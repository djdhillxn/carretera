"""Command-line workflows for training, evaluation, analysis, rendering, and I/O."""

import argparse
import atexit
import copy
import csv
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

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
    "CARLA_HOST": ("carla", "host", str),
    "CARLA_PORT": ("carla", "port", int),
    "CARLA_TM_PORT": ("carla", "traffic_manager_port", int),
    "CARLA_ROOT": ("carla", "server", "root", str),
    "CARLA_SERVER_MODE": ("carla", "server", "mode", str),
    "HIGHWAY_RL_ARTIFACT_ROOT": ("paths", "artifact_root", str),
    "HIGHWAY_RL_DRIVE_ROOT": ("paths", "drive_root", str),
}


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
    validate_config_data(config)
    return config


def validate_config_data(config):
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError("Missing required config sections: %s" % ", ".join(missing))
    if str(config["carla"]["version"]) != "0.9.16":
        raise ValueError("The project scope requires CARLA 0.9.16.")
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
    return "CarlaUE4" in result.stdout


def read_server_record(config):
    path = runtime_pid_file(config)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def wait_for_carla(config, timeout):
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError("Install carla==0.9.16 before starting managed mode.") from exc
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            client = carla.Client(config["carla"]["host"], config["carla"]["port"])
            client.set_timeout(2.0)
            version = client.get_server_version()
            return version
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
    previous = read_server_record(config)
    if previous and server_record_matches_process(previous):
        raise RuntimeError("Managed CARLA is already running with PID %s." % previous["pid"])
    executable = carla_server_executable(config)
    quality_key = "quality_rendering" if args.rendering else "quality_training"
    quality = config["carla"]["server"][quality_key]
    command = [
        str(executable),
        "-carla-port=%s" % config["carla"]["port"],
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
    process = subprocess.Popen(
        command,
        cwd=executable.parent,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    startup_complete = {"value": False}

    def cleanup_incomplete_start():
        if not startup_complete["value"] and process_alive(process.pid):
            os.killpg(process.pid, signal.SIGTERM)

    atexit.register(cleanup_incomplete_start)
    record = {
        "pid": process.pid,
        "process_group": process.pid,
        "started_at": utc_now(),
        "command": command,
        "log": str(log_path),
        "rendering": bool(args.rendering),
        "executable": str(executable),
    }
    write_json(runtime_pid_file(config), record)
    try:
        version = wait_for_carla(
            config, config["carla"]["server"]["startup_timeout_seconds"]
        )
    except Exception:
        if process_alive(process.pid):
            os.killpg(process.pid, signal.SIGTERM)
        raise
    finally:
        log_handle.close()
    startup_complete["value"] = True
    atexit.unregister(cleanup_incomplete_start)
    print("Managed CARLA started with PID %s, server version %s." % (process.pid, version))
    print("Log:", log_path)


def command_server_status(args):
    config = load_config(args.config)
    record = read_server_record(config)
    if not record:
        print("No repository-owned managed CARLA process is recorded.")
        return
    state = (
        "running"
        if server_record_matches_process(record)
        else "not running or identity mismatch"
    )
    print("Managed CARLA PID %s is %s." % (record["pid"], state))
    print(json.dumps(record, indent=2))


def command_server_stop(args):
    config = load_config(args.config)
    record = read_server_record(config)
    if not record:
        print("No repository-owned managed CARLA process is recorded; nothing stopped.")
        return
    pid = int(record["pid"])
    if process_alive(pid) and not server_record_matches_process(record):
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
    print(json.dumps(report, indent=2))
    try:
        with socket.create_connection(
            (config["carla"]["host"], config["carla"]["port"]), timeout=3.0
        ):
            report["tcp_connection"] = "ok"
    except OSError as exc:
        report["tcp_connection"] = "failed: %s" % exc
        write_json(artifact_root / "logs/runtime/doctor.json", report)
        raise ConnectionError(
            "CARLA TCP connection failed at %s:%s. Start CARLA 0.9.16 or set "
            "CARLA_HOST/CARLA_PORT. Details: %s"
            % (config["carla"]["host"], config["carla"]["port"], exc)
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
                    item.endswith("/" + config["carla"]["map"])
                    or item == config["carla"]["map"]
                    for item in env.client.get_available_maps()
                ),
                "synchronous_mode": env.world.get_settings().synchronous_mode,
                "fixed_delta_seconds": env.world.get_settings().fixed_delta_seconds,
                "traffic_manager_port": config["carla"]["traffic_manager_port"],
                "traffic_manager_connected": env.traffic_manager is not None,
                "highway_candidate_count": len(env.highway_candidates),
            }
        )
        candidate_path = env.write_highway_candidates(
            artifact_root / "logs/runtime/highway_candidates.json"
        )
        report["highway_candidates"] = str(candidate_path)
        print("Selected candidate preview:", json.dumps(env.highway_candidates[0], indent=2))
    finally:
        if env is not None:
            env.close()
    write_json(artifact_root / "logs/runtime/doctor.json", report)
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
    for source_path in sorted(source.rglob("*")):
        if source_path.is_symlink():
            counts["skipped"] += 1
            continue
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            if not dry_run:
                destination_path.mkdir(parents=True, exist_ok=True)
            continue
        try:
            needs_copy = (
                not destination_path.exists()
                or source_path.stat().st_size != destination_path.stat().st_size
                or source_path.stat().st_mtime > destination_path.stat().st_mtime + 1e-6
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
    if local_root == REPOSITORY_ROOT or (local_root / ".git").exists():
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
        raise RuntimeError("One or more offline self-tests failed.")


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
        blueprints = sorted(
            [
                item.id
                for item in env.world.get_blueprint_library().filter("vehicle.*")
                if int(item.get_attribute("number_of_wheels")) == 4
            ]
        )
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
                    rows.append(
                        {
                            "condition_id": "condition_%03d" % condition_index,
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
                    )
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
    output = artifact_root / "evaluations/episode_results.csv"
    existing = pd.DataFrame()
    completed = set()
    if output.exists():
        if not args.resume_existing:
            raise FileExistsError(
                "Evaluation output already exists. Use --resume-existing to skip "
                "completed policy/condition pairs."
            )
        existing = pd.read_csv(output)
        if existing.duplicated(["policy", "condition_id"]).any():
            raise ValueError(
                "Existing evaluation CSV contains duplicate policy/condition pairs."
            )
        completed = set(zip(existing["policy"], existing["condition_id"]))

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
                key = (policy_name, scenario["condition_id"])
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
                row = {
                    "policy": policy_name,
                    "condition_id": scenario["condition_id"],
                    "seed": scenario["seed"],
                    "traffic_density": scenario["traffic_density"],
                    "weather": scenario["weather"],
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
                    "resolved_config_hash": config_hash(config),
                    "manifest_hash": manifest["manifest_hash"],
                    "manifest_path": str(manifest_path),
                }
                append_csv_row(output, row)
                completed.add(key)
                print(
                    "%s %s success=%s collision=%s completion=%.3f"
                    % (
                        policy_name,
                        scenario["condition_id"],
                        row["success"],
                        row["collision"],
                        row["route_completion"],
                    )
                )
    finally:
        if env is not None:
            env.close()
    print("Evaluation rows:", output)


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
                raise ValueError(
                    "Category %s is absent from %s"
                    % (args.category, video_manifest_path)
                )
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
                "simulator_frames_received": len(env.drain_render_frames()),
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

    server = subparsers.add_parser("server")
    server_subparsers = server.add_subparsers(dest="server_command", required=True)
    server_start = server_subparsers.add_parser("start")
    server_start.add_argument("--config", default="config.yaml")
    server_start.add_argument("--rendering", action="store_true")
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
