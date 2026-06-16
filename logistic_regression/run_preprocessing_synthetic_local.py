import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from src.synthetic_logreg import (
    synthetic_logreg_extension_suffix,
    synthetic_logreg_regime_params,
    synthetic_logreg_setting_to_nm,
)


# Global defaults for the full synthetic preprocessing pass: generate the
# prepared datasets from scratch and compute every comp-param key used by the
# public synthetic experiment scripts and plotting code.
global_config = {
    "dataset": "synthetic_dirichlet_logreg",
    "loss_func": "log-reg",
    "regularizer_type": "non-cvx",
    "use_ray": 0,
    "is_sparse_dataset": 0,
    
    "is_minimize": 0,
    "load_prepared_dataset": 0,
    "load_raw_dataset": 0,
    "generate_dataset": 1,
    "estimate_diagnostics": 1,
    "save_diagnostics": 1,
    
    "print_args": 1,
    "print_status": 1,
    "hetero": 0,
    "num_workers": 1,
    "computable_params": (
        "['lambda_reg','dim','n_groups','m_per_group',"
        "'delta1_emp','delta2_emp','delta_flat_emp',"
        "'L_ij_max_emp','L_i_max_emp','L_global_emp',"
        "'delta1_wc','delta2_wc',"
        "'L_ij_max_wc','L_i_max_wc','L_global_wc']"
    ),
    "loadable_params": "[]",
    "loadable_datasets": "[]",
    "batchsize": 1,
    "device": "cuda:0",
    "compute_deltas": 1,
    "compute_L_bounds": 1,
    "compute_worstcase": 1,
    "probe_strategy": "init+path",
    "probe_T": 5,
    "probe_path_steps": 20,
    "probe_path_lr_mode": "manual",
    "probe_path_lr": 0.01,
    "delta2_mode": "exact_per_component",
    "delta_power_iters": 20,
    "delta_power_tol": 1e-4,
    "delta2_power_iters": 20,
    "delta_batch_size": 4096,
    "L_power_iters": 20,
    "L_power_tol": 1e-5,
    "L_batch_size": 4096,
    "d": 1000,
}


# We materialize all four requested regimes for each grouped setting.
synthetic_regimes = [
    "d1_small_d2_small",
    "d1_small_d2_large",
    "d1_large_d2_small",
    "d1_large_d2_large",
]

# We keep one deterministic seed per grouped setting, mirroring the existing MNIST launcher style.
synthetic_settings = [
    {"setting": "m_gt_n", "seed": 42},
    {"setting": "n_gt_m", "seed": 142},
]


def cfg_to_cli_args(cfg_dict):
    """Convert a flat config dictionary into the CLI format expected by data_preprocessing.py."""
    args = []
    for key, value in cfg_dict.items():
        args.extend([f"--{key}", str(value)])
    return args


def resolve_logreg_dir():
    """Resolve the logistic-regression experiment directory."""
    cwd = Path.cwd().resolve()
    if (cwd / "data_preprocessing.py").exists():
        return cwd
    experiment_dir = cwd / "experiments" / "logistic_regression"
    if (experiment_dir / "data_preprocessing.py").exists():
        return experiment_dir
    raise FileNotFoundError("Could not locate data_preprocessing.py from current working directory.")


def expected_dataset_path(cfg, logreg_dir):
    """Rebuild the exact synthetic dataset path used by Experiment.init_exp_data_extension()."""
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


def build_jobs(logreg_dir, base_config):
    """Build the requested 4 x 2 synthetic preprocessing grid."""
    jobs = []
    for setting_spec in synthetic_settings:
        for regime in synthetic_regimes:
            cfg = dict(base_config)
            cfg["synthetic_setting"] = setting_spec["setting"]
            cfg["synthetic_regime"] = regime
            cfg["seed"] = int(setting_spec["seed"])

            # Mirror the resolved grouped dimensions in the CLI config for transparency in logs.
            n_groups, m_per_group = synthetic_logreg_setting_to_nm(cfg["synthetic_setting"])
            cfg["n_groups"] = int(n_groups)
            cfg["m_per_group"] = int(m_per_group)

            run_name = f"{cfg['synthetic_setting']}__{cfg['synthetic_regime']}_seed{cfg['seed']}"
            jobs.append({
                "run_name": run_name,
                "cfg": cfg,
                "expected_dataset_path": str(expected_dataset_path(cfg, logreg_dir)),
            })
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the full synthetic preprocessing grid.")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Optional device override, e.g. cuda:0 or cpu. Defaults to the launcher config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logreg_dir = resolve_logreg_dir()
    script_path = logreg_dir / "data_preprocessing.py"
    launcher_config = dict(global_config)
    if args.device:
        launcher_config["device"] = args.device

    # Keep synthetic configs and logs separate from the MNIST launcher outputs.
    configs_dir = logreg_dir / "configs" / "preprocessing_synthetic"
    logs_dir = logreg_dir / "preprocessing_synthetic_local_logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact launcher defaults for reproducibility.
    (configs_dir / "global_local.json").write_text(json.dumps(launcher_config, indent=2), encoding="utf-8")

    jobs = build_jobs(logreg_dir, launcher_config)

    # Save the fully resolved job plan before execution.
    (configs_dir / "jobs_plan.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")

    print(f"Prepared {len(jobs)} synthetic preprocessing jobs.")
    for job in jobs:
        print(f"- {job['run_name']} -> {job['expected_dataset_path']}")

    # Execute the grid one job at a time and stream each job to its own log file.
    results = []
    job_iter = tqdm(jobs, total=len(jobs), desc="Synthetic preprocessing jobs") if tqdm is not None else jobs
    for idx, job in enumerate(job_iter, start=1):
        run_name = job["run_name"]
        cfg = job["cfg"]
        log_file = logs_dir / f"{datetime.now().strftime('%Y-%m-%d;%H:%M:%S')}_{run_name}.log"

        # Each run is a plain subprocess invocation of the shared preprocessing entrypoint.
        cmd = [sys.executable, str(script_path)] + cfg_to_cli_args(cfg)
        print("=" * 90)
        print(f"[{idx}/{len(jobs)}] Running: {run_name}")
        print("Command:", " ".join(cmd))

        # Persist stdout/stderr to a dedicated log file for later inspection.
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(
                cmd,
                cwd=str(logreg_dir),
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
            )

        results.append({
            "run_name": run_name,
            "returncode": proc.returncode,
            "log_file": str(log_file),
            "expected_dataset_path": job["expected_dataset_path"],
        })

        if proc.returncode == 0:
            print(f"[OK] {run_name}")
        else:
            print(f"[FAIL] {run_name} (returncode={proc.returncode})")
            print(f"Inspect log: {log_file}")

    # Print a compact summary once the full synthetic grid finishes.
    print("\n" + "#" * 90)
    print("Run summary")
    for result in results:
        status = "OK" if result["returncode"] == 0 else "FAIL"
        print(f"- {status:4s} | {result['run_name']} | {result['expected_dataset_path']}")
        print(f"        log: {result['log_file']}")


if __name__ == "__main__":
    main()
