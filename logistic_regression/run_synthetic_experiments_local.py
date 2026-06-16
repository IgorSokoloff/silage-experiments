import argparse
import ast
import importlib
import json
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

from src.synthetic_logreg import (
    SYNTHETIC_DIRICHLET_LOGREG_DATASET,
    synthetic_logreg_extension_suffix,
    synthetic_logreg_regime_params,
    synthetic_logreg_setting_to_nm,
)
from src.utils import myrepr


global_config = {
    "dataset": SYNTHETIC_DIRICHLET_LOGREG_DATASET,
    "loss_func": "log-reg",
    "regularizer_type": "non-cvx",
    "exp_name": "SILAGE",
    "use_ray": 0,
    "print_status": 1,
    "is_grad_comp_init": 1,
    "NUM_LAUNCHES": 1,
    "PRINT_EVERY": 10,
    "COLLECT_EVERY": 2,
    "SAVE_EVERY": 1000,
    # Stop each run once it reaches the epoch budget or the target tolerance.
    "max_epochs": 40.0,
    "max_iters": 5000,
    "tol": 1e-12,
    "factor": 1.0,
    "stop_criteruim_params": "['epochs','sqnorm']",
    "collectable_metrics": "['iters','epochs','sqnorm']",
    "computable_params": "[]",
    "device": "cuda:0",
    "d": 1000,
}

algorithm_overlays = {
    "silage_m_gt_n": {
        "alg_name": "SILAGE_m>n",
        "loadable_datasets": "['X_train','y_train','X_groups_train','y_groups_train']",
        "loadable_params": "['lambda_reg','delta2_emp','L_global_emp']",
        "seed": 1000,
    },
    "silage_n_gt_m": {
        "alg_name": "SILAGE_n>m",
        "loadable_datasets": "['X_train','y_train','X_groups_train','y_groups_train']",
        "loadable_params": "['lambda_reg','delta1_emp','delta2_emp','L_global_emp']",
        "seed": 3000,
    },
    "silage_n_gt_m_grid": {
        "alg_name": "SILAGE_n>m",
        "loadable_datasets": "['X_train','y_train','X_groups_train','y_groups_train']",
        "loadable_params": "['lambda_reg','delta1_emp','delta2_emp','L_global_emp']",
        "seed": 3000,
    },
    "zerosarah": {
        "alg_name": "ZeroSARAH",
        "loadable_datasets": "['X_train','y_train']",
        "loadable_params": "['lambda_reg','L_ij_max_emp']",
        "seed": 4000,
    },
    "zerosarah_grid": {
        "alg_name": "ZeroSARAH",
        "loadable_datasets": "['X_train','y_train']",
        "loadable_params": "['lambda_reg','L_ij_max_emp']",
        "seed": 4000,
    },
    "d_zerosarah": {
        "alg_name": "D-ZeroSARAH",
        "loadable_datasets": "['X_train','y_train','X_groups_train','y_groups_train']",
        "loadable_params": "['lambda_reg','L_ij_max_emp']",
        "seed": 5000,
    },
    "silver": {
        "alg_name": "SILVER",
        "batch_size": 5,
        "loadable_datasets": "['X_train','y_train']",
        "loadable_params": "['lambda_reg','L_ij_max_emp','delta_flat_emp']",
        "seed": 7000,
    },
    "silver_grid": {
        "alg_name": "SILVER",
        "loadable_datasets": "['X_train','y_train']",
        "loadable_params": "['lambda_reg','L_ij_max_emp','delta_flat_emp']",
        "seed": 7000,
    },
    "d_zerosarah_grid": {
        "alg_name": "D-ZeroSARAH",
        "loadable_datasets": "['X_train','y_train','X_groups_train','y_groups_train']",
        "loadable_params": "['lambda_reg','L_ij_max_emp']",
        "seed": 5000,
    },
}

algorithm_scripts = {
    "silage_m_gt_n": ("SILAGE_m_gt_n", "SILAGE_m_gt_n.py", "SILAGE_m_gt_n"),
    "silage_n_gt_m": ("SILAGE_n_gt_m", "SILAGE_n_gt_m.py", "SILAGE_n_gt_m"),
    "silage_n_gt_m_grid": ("SILAGE_n_gt_m", "SILAGE_n_gt_m.py", "SILAGE_n_gt_m"),
    "zerosarah": ("ZeroSARAH", "ZeroSARAH.py", "ZeroSARAH"),
    "zerosarah_grid": ("ZeroSARAH", "ZeroSARAH.py", "ZeroSARAH"),
    "d_zerosarah": ("D_ZeroSARAH", "D_ZeroSARAH.py", "D_ZeroSARAH"),
    "silver": ("SILVER", "SILVER.py", "SILVER"),
    "silver_grid": ("SILVER", "SILVER.py", "SILVER"),
    "d_zerosarah_grid": ("D_ZeroSARAH", "D_ZeroSARAH.py", "D_ZeroSARAH"),
}

algorithm_settings = {
    "silage_m_gt_n": ["m_gt_n"],
    "silage_n_gt_m": ["n_gt_m"],
    "silage_n_gt_m_grid": ["n_gt_m"],
    "zerosarah": ["m_gt_n", "n_gt_m"],
    "zerosarah_grid": ["m_gt_n", "n_gt_m"],
    "d_zerosarah": ["m_gt_n", "n_gt_m"],
    "silver": ["m_gt_n", "n_gt_m"],
    "silver_grid": ["m_gt_n", "n_gt_m"],
    "d_zerosarah_grid": ["m_gt_n", "n_gt_m"],
}

synthetic_regimes = [
    "d1_small_d2_small",
    "d1_small_d2_large",
    "d1_large_d2_small",
    "d1_large_d2_large",
]

# Tuned reproduction runs use factor=1.0. Keep the power grid available for
# manual experiments, but do not opt any public launcher into it by default.
factor_ar = [float(2 ** power) for power in range(10)]
FACTOR_TUNING_ALGORITHMS = set()

# Empirically best constant batch sizes from the already computed tuning logs.
# These are used for factor tuning so the factor sweep compares methods at their
# current best-found batch per synthetic regime.
SILAGE_N_GT_M_BEST_BATCH_BY_REGIME = {
    "d1_small_d2_small": 6,
    "d1_small_d2_large": 1,
    "d1_large_d2_small": 1,
    "d1_large_d2_large": 1,
}
SILVER_BEST_BATCH_BY_SETTING_AND_REGIME = {
    "m_gt_n": {
        "d1_small_d2_small": 46,
        "d1_small_d2_large": 128,
        "d1_large_d2_small": 96,
        "d1_large_d2_large": 96,
    },
    "n_gt_m": {
        "d1_small_d2_small": 46,
        "d1_small_d2_large": 128,
        "d1_large_d2_small": 96,
        "d1_large_d2_large": 50,
    },
}
D_ZEROSARAH_BEST_SAMPLING_BY_SETTING_AND_REGIME = {
    "m_gt_n": {
        "d1_small_d2_small": {"client_subset_size": 1, "batch_size": 6},
        "d1_small_d2_large": {"client_subset_size": 31, "batch_size": 1},
        "d1_large_d2_small": {"client_subset_size": 31, "batch_size": 1},
        "d1_large_d2_large": {"client_subset_size": 31, "batch_size": 1},
    },
    "n_gt_m": {
        "d1_small_d2_small": {"client_subset_size": 41, "batch_size": 1},
        "d1_small_d2_large": {"client_subset_size": 96, "batch_size": 1},
        "d1_large_d2_small": {"client_subset_size": 96, "batch_size": 1},
        "d1_large_d2_large": {"client_subset_size": 96, "batch_size": 1},
    },
}

# Explicit SILAGE_n>m batch-grid sweep for empirical tightness checks.
SILAGE_N_GT_M_BATCH_SIZE_AR = list(range(1, 247, 5)) + [250]

# Keep the ZeroSARAH-family sweeps explicit and editable here. The plain
# launcher stays coarse, while *_grid variants can use wider exploratory sweeps.
ZEROSARAH_BEST_BATCH_BY_SETTING_AND_REGIME = {
    "m_gt_n": {
        "d1_small_d2_small": 192,
        "d1_small_d2_large": 192,
        "d1_large_d2_small": 11,
        "d1_large_d2_large": 11,
    },
    "n_gt_m": {
        "d1_small_d2_small": 192,
        "d1_small_d2_large": 31,
        "d1_large_d2_small": 31,
        "d1_large_d2_large": 31,
    },
}
ZEROSARAH_GRID_BATCH_SIZE_AR = [
    1, 6, 11, 16, 21, 26, 31, 36, 41, 46,
    64, 96, 128, 192, 256, 384, 512, 768,
    1024, 1536, 2048, 3072, 4096, 6144,
    8192, 10240, 12500,
]

# SILVER extends only over the not-yet-computed larger flattened batch sizes.
SILVER_GRID_BATCH_SIZE_AR = [
    50, 64, 96, 128, 192, 256, 384, 512, 768,
    1024, 1536, 2048, 3072, 4096, 6144, 8192,
    10240, 12500,
]

# D-ZeroSARAH uses a setting-dependent grid aligned with the available
# client / local sample sizes.
D_ZEROSARAH_GRID_BATCH_SIZE_AR_BY_SETTING = {
    "m_gt_n": [
        1, 6, 11, 16, 21, 26, 31, 36, 41, 46,
        56, 66, 76, 86, 96, 111, 126, 151,
        176, 201, 226, 250,
    ],
    "n_gt_m": [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 50],
}
D_ZEROSARAH_GRID_CLIENT_SUBSET_SIZE_AR_BY_SETTING = {
    "m_gt_n": [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 50],
    "n_gt_m": [
        1, 6, 11, 16, 21, 26, 31, 36, 41, 46,
        56, 66, 76, 86, 96, 111, 126, 151,
        176, 201, 226, 250,
    ],
}

TMUX_SCHEDULER_SPECS = {
    "silage_m_gt_n": {
        "gpu_slots": [("cuda:1", 8), ("cuda:0", 8)],
        "poll_seconds": 10,
        "mode": "rolling",
    },
    "silage_n_gt_m": {
        "gpu_slots": [("cuda:1", 8), ("cuda:0", 8)],
        "poll_seconds": 10,
        "mode": "rolling",
    },
    "zerosarah": {
        "gpu_slots": [("cuda:1", 8), ("cuda:0", 8)],
        "poll_seconds": 10,
        "mode": "rolling",
    },
    "zerosarah_grid": {
        "gpu_slots": [("cuda:1", 8), ("cuda:0", 8)],
        "poll_seconds": 10,
        "mode": "rolling",
    },
    "d_zerosarah": {
        "gpu_slots": [("cuda:1", 6), ("cuda:0", 6)],
        "poll_seconds": 15,
        "mode": "rolling",
    },
    "silver": {
        "gpu_slots": [("cuda:1", 8), ("cuda:0", 8)],
        "poll_seconds": 10,
        "mode": "rolling",
    },
    "silage_n_gt_m_grid": {
        "gpu_slots": [("cuda:1", 8), ("cuda:0", 8)],
        "poll_seconds": 10,
        "mode": "rolling",
    },
    "silver_grid": {
        "gpu_slots": [("cuda:1", 8), ("cuda:0", 8)],
        "poll_seconds": 10,
        "mode": "rolling",
    },
    "d_zerosarah_grid": {
        "gpu_slots": [("cuda:1", 10), ("cuda:0", 10)],
        "poll_seconds": 15,
        "mode": "rolling",
    },
}


def cfg_to_cli_args(cfg_dict):
    args = []
    for key, value in cfg_dict.items():
        args.extend([f"--{key}", str(value)])
    return args


def resolve_logreg_dir():
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "algorithm.py").exists() and (cwd / "run_synthetic_experiments_local.py").exists():
        return cwd
    experiment_dir = cwd / "experiments" / "logistic_regression"
    if (experiment_dir / "src" / "algorithm.py").exists():
        return experiment_dir
    raise FileNotFoundError("Could not locate the logistic-regression experiment directory from the current working directory.")


def expected_dataset_path(cfg, logreg_dir):
    n_groups, m_per_group = synthetic_logreg_setting_to_nm(cfg["synthetic_setting"])
    regime_params = synthetic_logreg_regime_params(cfg["synthetic_setting"], cfg["synthetic_regime"])
    suffix = synthetic_logreg_extension_suffix(
        setting=cfg["synthetic_setting"],
        regime=cfg["synthetic_regime"],
        n_groups=n_groups,
        m_per_group=m_per_group,
        d=int(cfg["d"]),
        K=int(regime_params["K"]),
        T=int(regime_params["T"]),
    )
    exp_data_ext = f"_{cfg['loss_func']}_{cfg['dataset']}{suffix}"
    return logreg_dir / f"data_{cfg['dataset']}" / f"data{exp_data_ext}"


def expected_comp_params_path(cfg, logreg_dir):
    n_groups, m_per_group = synthetic_logreg_setting_to_nm(cfg["synthetic_setting"])
    regime_params = synthetic_logreg_regime_params(cfg["synthetic_setting"], cfg["synthetic_regime"])
    suffix = synthetic_logreg_extension_suffix(
        setting=cfg["synthetic_setting"],
        regime=cfg["synthetic_regime"],
        n_groups=n_groups,
        m_per_group=m_per_group,
        d=int(cfg["d"]),
        K=int(regime_params["K"]),
        T=int(regime_params["T"]),
    )
    exp_params_ext = f"_{cfg['loss_func']}_{cfg['dataset']}{suffix}"
    return logreg_dir / f"data_{cfg['dataset']}" / f"comp_params{exp_params_ext}"


def sanitize_name(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def n_gt_m_batch_mode_token(cfg):
    if cfg["alg_name"] != "SILAGE_n>m":
        return None
    loadable_params = ast.literal_eval(cfg["loadable_params"])
    if "optimal_b" in loadable_params:
        return "optimal_b"
    return f"b{int(cfg['batch_size'])}"


def zerosarah_batch_token(cfg):
    if cfg["alg_name"] != "ZeroSARAH":
        return None
    return f"b{int(cfg['batch_size'])}"


def flat_batch_token(cfg):
    if cfg["alg_name"] != "SILVER":
        return None
    return f"b{int(cfg['batch_size'])}"


def d_zerosarah_sampling_tokens(cfg):
    if cfg["alg_name"] != "D-ZeroSARAH":
        return []
    return [
        f"s{int(cfg['client_subset_size'])}",
        f"b{int(cfg['batch_size'])}",
    ]


def factor_values_for_algorithm(algorithm_key):
    if algorithm_key in FACTOR_TUNING_ALGORITHMS:
        return list(factor_ar)
    return [1.0]


def job_param_overrides_for_algorithm(algorithm_key, target_setting, regime):
    if algorithm_key == "silage_n_gt_m":
        return [{"batch_size": int(SILAGE_N_GT_M_BEST_BATCH_BY_REGIME[regime])}]

    if algorithm_key == "silver":
        return [{"batch_size": int(SILVER_BEST_BATCH_BY_SETTING_AND_REGIME[target_setting][regime])}]

    if algorithm_key == "silage_n_gt_m_grid":
        return [{"batch_size": int(batch_size)} for batch_size in SILAGE_N_GT_M_BATCH_SIZE_AR]

    if algorithm_key == "zerosarah":
        return [{"batch_size": int(ZEROSARAH_BEST_BATCH_BY_SETTING_AND_REGIME[target_setting][regime])}]

    if algorithm_key == "zerosarah_grid":
        return [{"batch_size": int(batch_size)} for batch_size in ZEROSARAH_GRID_BATCH_SIZE_AR]

    if algorithm_key == "silver_grid":
        return [{"batch_size": int(batch_size)} for batch_size in SILVER_GRID_BATCH_SIZE_AR]

    if algorithm_key == "d_zerosarah":
        best_sampling = D_ZEROSARAH_BEST_SAMPLING_BY_SETTING_AND_REGIME[target_setting][regime]
        return [{
            "client_subset_size": int(best_sampling["client_subset_size"]),
            "batch_size": int(best_sampling["batch_size"]),
        }]

    if algorithm_key == "d_zerosarah_grid":
        batch_sizes = D_ZEROSARAH_GRID_BATCH_SIZE_AR_BY_SETTING[target_setting]
        client_subset_sizes = D_ZEROSARAH_GRID_CLIENT_SUBSET_SIZE_AR_BY_SETTING[target_setting]
        return [
            {"client_subset_size": int(client_subset_size), "batch_size": int(batch_size)}
            for client_subset_size in client_subset_sizes
            for batch_size in batch_sizes
        ]

    return [{}]


def uses_wave_scheduler(algorithm_key):
    return algorithm_key in TMUX_SCHEDULER_SPECS


def wave_scheduler_spec(algorithm_key):
    if algorithm_key not in TMUX_SCHEDULER_SPECS:
        raise ValueError(f"No wave-scheduler spec registered for algorithm={algorithm_key}.")
    return TMUX_SCHEDULER_SPECS[algorithm_key]


def expanded_device_schedule(gpu_slots):
    schedule = []
    for device, count in gpu_slots:
        schedule.extend([str(device)] * int(count))
    return schedule


def assign_devices_to_jobs(jobs, algorithm_key):
    if not uses_wave_scheduler(algorithm_key):
        return jobs

    spec = wave_scheduler_spec(algorithm_key)
    device_schedule = expanded_device_schedule(spec["gpu_slots"])
    wave_size = len(device_schedule)
    if wave_size <= 0:
        raise ValueError(
            f"Wave scheduler for algorithm={algorithm_key} produced an empty device schedule."
        )

    for idx, job in enumerate(jobs):
        job["cfg"]["device"] = device_schedule[idx % wave_size]
    return jobs


def iter_job_waves(jobs, wave_size):
    for start in range(0, len(jobs), wave_size):
        yield jobs[start:start + wave_size]


def tmux_session_alive(session_name, workdir):
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        cwd=str(workdir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def wait_for_tmux_sessions(session_names, workdir, poll_seconds):
    while True:
        active_sessions = [name for name in session_names if tmux_session_alive(name, workdir)]
        if not active_sessions:
            return
        print(
            f"Waiting on {len(active_sessions)} running tmux sessions; "
            f"next poll in {poll_seconds}s."
        )
        time.sleep(poll_seconds)


def start_tmux_session(session_name, shell_cmd, logreg_dir):
    result = subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            f"bash -lc {shlex.quote(shell_cmd)}",
        ],
        cwd=str(logreg_dir),
        check=False,
    )
    return result.returncode == 0


def build_tmux_launch(job, idx, algorithm_key, timestamp_prefix, logs_dir, logreg_dir, conda_sh, conda_env, script_path):
    cfg = job["cfg"]
    run_name = job["run_name"]
    safe_run_name = job["safe_run_name"]
    log_file = logs_dir / f"{timestamp_prefix}_{safe_run_name}.log"
    session_name = sanitize_name(f"{algorithm_key}_{idx}_{run_name}")[:80]

    cli_args = " ".join(shlex.quote(arg) for arg in cfg_to_cli_args(cfg))
    python_cmd = f"python {shlex.quote(str(script_path.name))} {cli_args}"
    setup_parts = []
    if conda_sh is not None:
        setup_parts.append(f"source {shlex.quote(str(conda_sh))}")
    if conda_env:
        setup_parts.append(f"conda activate {shlex.quote(str(conda_env))}")
    setup_parts.extend(
        [
            f"cd {shlex.quote(str(logreg_dir))}",
            f"{python_cmd} > {shlex.quote(str(log_file))} 2>&1",
        ]
    )
    shell_cmd = " && ".join(setup_parts)

    return {
        "cfg": cfg,
        "run_name": run_name,
        "session_name": session_name,
        "log_file": log_file,
        "shell_cmd": shell_cmd,
    }


def print_tmux_launch(launch, idx, total_jobs, scheduler_label=None, slot_idx=None, slot_count=None):
    print("=" * 90)
    prefix = f"[{idx}/{total_jobs}]"
    if scheduler_label is not None:
        prefix = f"[{scheduler_label} | {idx}/{total_jobs}]"
    print(f"{prefix} {launch['run_name']}")
    if slot_idx is not None and slot_count is not None:
        print(f"slot: {slot_idx + 1}/{slot_count}")
    print(f"device: {launch['cfg']['device']}")
    print(f"tmux session: {launch['session_name']}")
    print(f"log file: {launch['log_file']}")
    print(f"command: bash -lc {launch['shell_cmd']}")


def log_run_name(cfg):
    alg_label = cfg["alg_name"].replace(">", "_gt_").replace("<", "_lt_")
    parts = [alg_label]
    if cfg["alg_name"] in {"ZeroSARAH", "D-ZeroSARAH", "SILVER"}:
        # These baselines run on both synthetic settings, so the setting must be
        # part of the public run identity to avoid collisions across m_gt_n / n_gt_m.
        parts.append(cfg["synthetic_setting"])
    parts.append(cfg["synthetic_regime"])
    batch_mode = n_gt_m_batch_mode_token(cfg)
    if batch_mode is None:
        batch_mode = zerosarah_batch_token(cfg)
    if batch_mode is None:
        batch_mode = flat_batch_token(cfg)
    if batch_mode is not None:
        parts.append(batch_mode)
    parts.extend(d_zerosarah_sampling_tokens(cfg))
    parts.append(f"f{myrepr(float(cfg['factor']))}")
    return "__".join(parts)


def build_jobs(logreg_dir, algorithm_key, device_override=None):
    if algorithm_key not in algorithm_overlays:
        raise ValueError(f"Unknown algorithm key: {algorithm_key}")

    jobs = []
    base_cfg = dict(global_config)
    base_cfg.update(algorithm_overlays[algorithm_key])
    if device_override:
        base_cfg["device"] = str(device_override)
    target_settings = algorithm_settings[algorithm_key]

    for target_setting in target_settings:
        n_groups, m_per_group = synthetic_logreg_setting_to_nm(target_setting)
        factor_values = factor_values_for_algorithm(algorithm_key)

        for regime in synthetic_regimes:
            job_overrides = job_param_overrides_for_algorithm(algorithm_key, target_setting, regime)
            for factor in factor_values:
                for job_override in job_overrides:
                    cfg = dict(base_cfg)
                    cfg["synthetic_setting"] = target_setting
                    cfg["synthetic_regime"] = regime
                    cfg["factor"] = float(factor)
                    cfg["n_groups"] = int(n_groups)
                    cfg["m_per_group"] = int(m_per_group)
                    cfg.update(job_override)

                    run_name = log_run_name(cfg)
                    jobs.append({
                        "run_name": run_name,
                        "safe_run_name": sanitize_name(run_name),
                        "cfg": cfg,
                        "expected_dataset_path": str(expected_dataset_path(cfg, logreg_dir)),
                        "expected_comp_params_path": str(expected_comp_params_path(cfg, logreg_dir)),
                    })
    if device_override:
        return jobs
    return assign_devices_to_jobs(jobs, algorithm_key)


def import_runner_class(algorithm_key):
    module_name, _, class_name = algorithm_scripts[algorithm_key]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description="Synthetic algorithm launcher with optional tmux sessions.")
    parser.add_argument(
        "--algorithm",
        action="store",
        dest="algorithm",
        type=str,
        choices=sorted(algorithm_overlays.keys()),
        default="silage_m_gt_n",
        help="Which synthetic algorithm variant to launch.",
    )
    parser.add_argument("--dry_run", action="store", dest="dry_run", type=int, default=0, help="If 1, only print commands without creating tmux sessions.")
    parser.add_argument(
        "--launch_mode",
        action="store",
        dest="launch_mode",
        type=str,
        choices=["tmux", "direct"],
        default="tmux",
        help="Launch via detached tmux sessions or run one selected job directly in-process.",
    )
    parser.add_argument(
        "--conda_sh",
        action="store",
        dest="conda_sh",
        type=str,
        default="",
        help="Optional path to conda.sh for tmux launches. Leave empty if conda is already initialized.",
    )
    parser.add_argument(
        "--conda_env",
        action="store",
        dest="conda_env",
        type=str,
        default="silage",
        help="Conda environment to activate for tmux launches. Use an empty string to skip activation.",
    )
    parser.add_argument(
        "--job_index",
        action="store",
        dest="job_index",
        type=int,
        default=None,
        help="Zero-based job index used only with --launch_mode direct.",
    )
    parser.add_argument(
        "--device",
        action="store",
        dest="device",
        type=str,
        default="",
        help="Optional device override for all jobs, e.g. cuda:0 or cpu.",
    )
    args = parser.parse_args()

    logreg_dir = resolve_logreg_dir()
    _, script_filename, _ = algorithm_scripts[args.algorithm]
    script_path = logreg_dir / script_filename

    configs_dir = logreg_dir / "configs" / "synthetic_experiments"
    logs_dir = logreg_dir / "synthetic_experiments_local_logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    device_override = args.device.strip() or None
    jobs = build_jobs(logreg_dir, args.algorithm, device_override=device_override)
    config_payload = {
        "algorithm": args.algorithm,
        "global_config": global_config,
        "algorithm_overlay": algorithm_overlays[args.algorithm],
    }
    (configs_dir / f"global_local_{args.algorithm}.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    (configs_dir / f"jobs_plan_{args.algorithm}.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    print(f"Prepared {len(jobs)} synthetic jobs for algorithm={args.algorithm}.")
    for idx, job in enumerate(jobs):
        print(f"- [{idx}] {job['run_name']}")
        print(f"    dataset: {job['expected_dataset_path']}")
        print(f"    comp_params: {job['expected_comp_params_path']}")

    if args.launch_mode == "direct":
        if args.job_index is None:
            parser.error("--job_index is required when --launch_mode direct.")
        if args.job_index < 0 or args.job_index >= len(jobs):
            parser.error(f"--job_index must be between 0 and {len(jobs) - 1}.")

        job = jobs[args.job_index]
        cfg = job["cfg"]
        cli_args = cfg_to_cli_args(cfg)
        cli_preview = " ".join(shlex.quote(arg) for arg in cli_args)

        print("=" * 90)
        print(f"Direct debug run for job [{args.job_index}] {job['run_name']}")
        print(f"dataset: {job['expected_dataset_path']}")
        print(f"comp_params: {job['expected_comp_params_path']}")
        print(f"command: python {script_path.name} {cli_preview}")

        if args.dry_run:
            print("Dry run enabled; direct job was not executed.")
            return

        runner_class = import_runner_class(args.algorithm)
        runner = runner_class(cli_args)
        runner.run()
        return

    conda_sh = Path(args.conda_sh).expanduser() if args.conda_sh else None
    conda_env = args.conda_env.strip()
    timestamp_prefix = datetime.now().strftime("%Y-%m-%d;%H:%M:%S")

    if uses_wave_scheduler(args.algorithm):
        wave_spec = (
            {"gpu_slots": [(device_override, 1)], "poll_seconds": 15, "mode": "rolling"}
            if device_override is not None
            else wave_scheduler_spec(args.algorithm)
        )
        device_schedule = expanded_device_schedule(wave_spec["gpu_slots"])
        max_parallel_jobs = len(device_schedule)
        poll_seconds = int(wave_spec["poll_seconds"])
        scheduler_mode = str(wave_spec.get("mode", "waves"))

        print(
            f"\nLaunching {len(jobs)} jobs with scheduler mode={scheduler_mode!r} "
            f"and up to {max_parallel_jobs} concurrent jobs."
        )
        print(f"GPU slot pattern: {wave_spec['gpu_slots']}")

        if scheduler_mode == "rolling":
            active_slots = [None] * max_parallel_jobs
            next_job_idx = 0
            launched_count = 0

            while next_job_idx < len(jobs) or any(active_slots):
                launched_this_pass = False

                for slot_idx, slot_device in enumerate(device_schedule):
                    active_session = active_slots[slot_idx]
                    if active_session is not None and tmux_session_alive(active_session, logreg_dir):
                        continue

                    if active_session is not None:
                        print(f"Completed slot {slot_idx + 1}/{max_parallel_jobs}: {active_session}")
                        active_slots[slot_idx] = None

                    if next_job_idx >= len(jobs):
                        continue

                    job = jobs[next_job_idx]
                    job["cfg"]["device"] = slot_device
                    launch = build_tmux_launch(
                        job=job,
                        idx=next_job_idx + 1,
                        algorithm_key=args.algorithm,
                        timestamp_prefix=timestamp_prefix,
                        logs_dir=logs_dir,
                        logreg_dir=logreg_dir,
                        conda_sh=conda_sh,
                        conda_env=conda_env,
                        script_path=script_path,
                    )
                    print_tmux_launch(
                        launch,
                        idx=next_job_idx + 1,
                        total_jobs=len(jobs),
                        scheduler_label="rolling",
                        slot_idx=slot_idx,
                        slot_count=max_parallel_jobs,
                    )

                    if not args.dry_run:
                        if start_tmux_session(launch["session_name"], launch["shell_cmd"], logreg_dir):
                            active_slots[slot_idx] = launch["session_name"]
                            launched_count += 1
                            launched_this_pass = True
                            print(f"Started tmux session: {launch['session_name']}")
                        else:
                            print(f"Failed to start tmux session: {launch['session_name']}")
                    else:
                        active_slots[slot_idx] = f"dry_run_slot_{slot_idx}_{next_job_idx + 1}"
                        launched_count += 1
                        launched_this_pass = True

                    next_job_idx += 1

                if args.dry_run:
                    if next_job_idx >= len(jobs):
                        break
                    active_slots = [None] * max_parallel_jobs
                    continue

                if next_job_idx >= len(jobs) and not any(active_slots):
                    break

                if not launched_this_pass:
                    active_count = sum(1 for session in active_slots if session is not None)
                    print(
                        f"Rolling scheduler status: {active_count} active sessions, "
                        f"{launched_count}/{len(jobs)} jobs launched; next poll in {poll_seconds}s."
                    )
                    time.sleep(poll_seconds)

            if args.dry_run:
                print("\nDry run enabled; no tmux sessions were created.")
            return

        waves = list(iter_job_waves(jobs, max_parallel_jobs))
        print(
            f"\nLaunching {len(jobs)} jobs in {len(waves)} waves of up to "
            f"{max_parallel_jobs} jobs."
        )

        for wave_idx, wave_jobs in enumerate(waves, start=1):
            print("\n" + "#" * 90)
            print(f"Wave {wave_idx}/{len(waves)}")

            launched_sessions = []
            for offset, job in enumerate(wave_jobs, start=1):
                global_idx = (wave_idx - 1) * max_parallel_jobs + offset
                launch = build_tmux_launch(
                    job=job,
                    idx=global_idx,
                    algorithm_key=args.algorithm,
                    timestamp_prefix=timestamp_prefix,
                    logs_dir=logs_dir,
                    logreg_dir=logreg_dir,
                    conda_sh=conda_sh,
                    conda_env=conda_env,
                    script_path=script_path,
                )

                print_tmux_launch(
                    launch,
                    idx=global_idx,
                    total_jobs=len(jobs),
                    scheduler_label=f"wave {wave_idx}/{len(waves)}",
                    slot_idx=offset - 1,
                    slot_count=len(wave_jobs),
                )

                if not args.dry_run:
                    if start_tmux_session(launch["session_name"], launch["shell_cmd"], logreg_dir):
                        launched_sessions.append(launch["session_name"])
                        print(f"Started tmux session: {launch['session_name']}")
                    else:
                        print(f"Failed to start tmux session: {launch['session_name']}")

            if not args.dry_run and launched_sessions:
                print(
                    f"\nWaiting for wave {wave_idx}/{len(waves)} to finish "
                    f"before launching the next wave."
                )
                wait_for_tmux_sessions(
                    launched_sessions,
                    logreg_dir,
                    poll_seconds,
                )
                print(f"Wave {wave_idx}/{len(waves)} finished.")

        if args.dry_run:
            print("\nDry run enabled; no tmux sessions were created.")
        return

    for idx, job in enumerate(jobs, start=1):
        launch = build_tmux_launch(
            job=job,
            idx=idx,
            algorithm_key=args.algorithm,
            timestamp_prefix=timestamp_prefix,
            logs_dir=logs_dir,
            logreg_dir=logreg_dir,
            conda_sh=conda_sh,
            conda_env=conda_env,
            script_path=script_path,
        )

        print_tmux_launch(launch, idx=idx, total_jobs=len(jobs))

        if not args.dry_run:
            if start_tmux_session(launch["session_name"], launch["shell_cmd"], logreg_dir):
                print(f"Started tmux session: {launch['session_name']}")
            else:
                print(f"Failed to start tmux session: {launch['session_name']}")

    print("\nAttach to one session with:")
    print("tmux attach -t <session_name>")


if __name__ == "__main__":
    main()
