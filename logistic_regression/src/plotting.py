from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

from src.synthetic_logreg import (
    SYNTHETIC_DIRICHLET_LOGREG_DATASET,
    synthetic_logreg_extension_suffix,
    synthetic_logreg_regime_params,
    synthetic_logreg_setting_to_nm,
)
from src.utils import load_comp_params_bundle


SYNTHETIC_SETTINGS = ("m_gt_n", "n_gt_m")
SYNTHETIC_REGIMES = (
    "d1_small_d2_small",
    "d1_small_d2_large",
    "d1_large_d2_small",
    "d1_large_d2_large",
)
REGIME_GRID_LAYOUT = (
    ("d1_small_d2_small", "d1_small_d2_large"),
    ("d1_large_d2_small", "d1_large_d2_large"),
)
REGIME_SUBPLOT_TITLES = {
    "d1_small_d2_small": r"$(\delta_1 \text{ small}, \delta_2 \text{ small})$",
    "d1_small_d2_large": r"$(\delta_1 \text{ small}, \delta_2 \text{ large})$",
    "d1_large_d2_small": r"$(\delta_1 \text{ large}, \delta_2 \text{ small})$",
    "d1_large_d2_large": r"$(\delta_1 \text{ large}, \delta_2 \text{ large})$",
}

DEFAULT_TRAJECTORY_METHODS_BY_SETTING = {
    "m_gt_n": ["SILAGE_m>n", "ZeroSARAH", "D-ZeroSARAH", "SILVER"],
    "n_gt_m": ["SILAGE_n>m", "ZeroSARAH", "D-ZeroSARAH", "SILVER"],
}
DEFAULT_TUNING_1D_METHODS_BY_SETTING = {
    "m_gt_n": ["SILVER"],
    "n_gt_m": ["SILAGE_n>m", "SILVER"],
}
DEFAULT_TUNING_2D_METHODS = ("D-ZeroSARAH",)

METHOD_DISPLAY_NAMES = {
    "SILAGE_m>n": "SILAGE (m>n)",
    "SILAGE_n>m": "SILAGE (n>m)",
    "ZeroSARAH": "ZeroSARAH",
    "D-ZeroSARAH": "D-ZeroSARAH",
    "SILVER": "SILVER",
}
METHOD_STYLES = {
    "SILAGE_m>n": {"color": "red", "marker": "*"},
    "SILAGE_n>m": {"color": "red", "marker": "*"},
    "ZeroSARAH": {"color": "#ff7f0e", "marker": "s"},
    "D-ZeroSARAH": {"color": "#2ca02c", "marker": "^"},
    "SILVER": {"color": "#8c564b", "marker": "P"},
}

DEFAULT_SYNTHETIC_PLOT_STYLE = {
    "marker_size": 25,
    "label_fontsize": 35,
    "tick_labelsize": 35,
    "axis_labelsize": 35,
    "legend_fontsize": 31,
    "subplot_title_fontsize": 33,
    "suptitle_fontsize": 35,
}

TITLE_PARAM_KEYS = ("delta1_emp", "delta2_emp", "L_ij_max_emp", "L_global_emp")


@dataclass(frozen=True)
class SyntheticRunRecord:
    method: str
    setting: str
    regime: str
    exp_str: str
    run_dir: Path
    factor_token: str | None = None
    factor: float | None = None
    batch_size: int | None = None
    client_subset_size: int | None = None
    batch_mode: str | None = None


def resolve_logreg_dir() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "algorithm.py").exists() and (cwd / "run_synthetic_experiments_local.py").exists():
        return cwd
    experiment_dir = cwd / "experiments" / "logistic_regression"
    if (experiment_dir / "src" / "algorithm.py").exists():
        return experiment_dir
    raise FileNotFoundError("Could not locate the logistic-regression experiment directory from the current working directory.")


def _resolve_plot_style(config: Mapping[str, Any] | None = None) -> dict[str, float]:
    style = dict(DEFAULT_SYNTHETIC_PLOT_STYLE)
    provided_keys = set()
    if config:
        updates = {key: value for key, value in config.items() if value is not None}
        style.update(updates)
        provided_keys = set(updates.keys())
    label_fontsize = float(style["label_fontsize"])
    if "tick_labelsize" not in provided_keys:
        style["tick_labelsize"] = label_fontsize
    if "axis_labelsize" not in provided_keys:
        style["axis_labelsize"] = label_fontsize
    if "legend_fontsize" not in provided_keys:
        style["legend_fontsize"] = max(1.0, label_fontsize - 4.0)
    if "subplot_title_fontsize" not in provided_keys:
        style["subplot_title_fontsize"] = max(1.0, label_fontsize - 2.0)
    if "suptitle_fontsize" not in provided_keys:
        style["suptitle_fontsize"] = label_fontsize
    return style


def _default_matplotlib_rcparams(style: Mapping[str, float]) -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "FreeSerif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.grid": True,
            "axes.titlesize": style["subplot_title_fontsize"],
            "axes.labelsize": style["axis_labelsize"],
            "legend.fontsize": style["legend_fontsize"],
            "xtick.labelsize": style["tick_labelsize"],
            "ytick.labelsize": style["tick_labelsize"],
            "lines.linewidth": 2,
        }
    )


def _get_nested_setting(mapping: Mapping[str, Any] | None, setting: str, regime: str, default: Any = None) -> Any:
    if not mapping:
        return default
    return mapping.get(setting, {}).get(regime, default)


def _synthetic_comp_params_path(
    logreg_dir: Path,
    setting: str,
    regime: str,
    d: int = 1000,
    dataset: str = SYNTHETIC_DIRICHLET_LOGREG_DATASET,
    loss_func: str = "log-reg",
) -> Path:
    n_groups, m_per_group = synthetic_logreg_setting_to_nm(setting)
    regime_params = synthetic_logreg_regime_params(setting, regime)
    suffix = synthetic_logreg_extension_suffix(
        setting=setting,
        regime=regime,
        n_groups=int(n_groups),
        m_per_group=int(m_per_group),
        d=int(d),
        K=int(regime_params["K"]),
        T=int(regime_params["T"]),
    )
    return logreg_dir / f"data_{dataset}" / f"comp_params_{loss_func}_{dataset}{suffix}"


def _load_title_constants(logreg_dir: Path, setting: str, regime: str, d: int = 1000) -> dict[str, float | None]:
    comp_params_path = _synthetic_comp_params_path(logreg_dir, setting, regime, d=d)
    try:
        bundle = load_comp_params_bundle(str(comp_params_path), requested_keys=list(TITLE_PARAM_KEYS), is_print=0)
    except FileNotFoundError:
        return {key: None for key in TITLE_PARAM_KEYS}
    return {key: _to_scalar(bundle.get(key)) for key in TITLE_PARAM_KEYS}


def _to_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value
    return value


def _format_constant(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return r"\mathrm{N/A}"
    return f"{float(value):.2f}"


def format_synthetic_subplot_title(regime: str) -> str:
    try:
        return REGIME_SUBPLOT_TITLES[regime]
    except KeyError as exc:
        raise ValueError(f"Unsupported synthetic regime: {regime}") from exc


def _parse_factor(name: str) -> tuple[str | None, float | None]:
    match = re.search(r"_f([^_]+)$", name)
    if not match:
        return None, None
    token = match.group(1)
    try:
        return token, float(token.replace(",", "."))
    except ValueError:
        return token, None


def _parse_synthetic_run_dir(run_dir: Path) -> SyntheticRunRecord | None:
    if not run_dir.is_dir() or not run_dir.name.startswith("logs_"):
        return None

    exp_str = run_dir.name[len("logs_") :]
    if "synthetic_dirichlet_logreg" not in exp_str:
        return None

    setting = next((candidate for candidate in SYNTHETIC_SETTINGS if f"ss-{candidate}_sr-" in exp_str), None)
    regime = next((candidate for candidate in SYNTHETIC_REGIMES if f"sr-{candidate}_" in exp_str), None)
    if setting is None or regime is None:
        return None

    factor_token, factor = _parse_factor(exp_str)
    method = None
    batch_size = None
    client_subset_size = None
    batch_mode = None

    if "_SILAGE_m>n_" in exp_str or "_SILAGE_m_n_" in exp_str:
        method = "SILAGE_m>n"
    else:
        match = re.search(r"_SILAGE_n(?:>m|_m)_(optimal_b|b(\d+))_f", exp_str)
        if match:
            method = "SILAGE_n>m"
            batch_mode = match.group(1)
            if match.group(2) is not None:
                batch_size = int(match.group(2))

    if method is None:
        match = re.search(r"_ZeroSARAH_b(\d+)_f", exp_str)
        if match:
            method = "ZeroSARAH"
            batch_size = int(match.group(1))

    if method is None:
        match = re.search(r"_D-ZeroSARAH_s(\d+)_b(\d+)_f", exp_str)
        if match:
            method = "D-ZeroSARAH"
            client_subset_size = int(match.group(1))
            batch_size = int(match.group(2))

    if method is None:
        match = re.search(r"_SILVER_b(\d+)_f", exp_str)
        if match:
            method = "SILVER"
            batch_size = int(match.group(1))

    if method is None:
        return None

    return SyntheticRunRecord(
        method=method,
        setting=setting,
        regime=regime,
        exp_str=exp_str,
        run_dir=run_dir,
        factor_token=factor_token,
        factor=factor,
        batch_size=batch_size,
        client_subset_size=client_subset_size,
        batch_mode=batch_mode,
    )


def discover_synthetic_runs(logreg_dir: Path | None = None) -> list[SyntheticRunRecord]:
    root = resolve_logreg_dir() if logreg_dir is None else Path(logreg_dir)
    logs_dir = root / "logs"
    if not logs_dir.exists():
        return []

    records: list[SyntheticRunRecord] = []
    for candidate in sorted(logs_dir.iterdir()):
        record = _parse_synthetic_run_dir(candidate)
        if record is not None:
            records.append(record)
    return records


def _load_metric_series(run: SyntheticRunRecord, metric_name: str) -> np.ndarray:
    metric_path = run.run_dir / f"{metric_name}_{run.exp_str}.npy"
    if not metric_path.exists():
        raise FileNotFoundError(f"Missing metric file for {run.method}: {metric_path}")
    return np.asarray(np.load(metric_path), dtype=np.float64)


def _load_final_epoch_value(run: SyntheticRunRecord) -> float:
    epochs = _load_metric_series(run, "epochs")
    if epochs.size == 0:
        raise ValueError(f"Empty epochs array for run: {run.run_dir}")
    return float(epochs[-1])


def _trajectory_markevery_indices(sqnorm: np.ndarray, ymin: float | None) -> list[int]:
    """Choose sparse marker positions and suppress markers below ymin.

    The trajectory line itself should remain untouched; only marker placement is
    filtered by the plotting floor so tiny tail values do not create visual
    glitches near the bottom boundary.
    """
    if sqnorm.size == 0:
        return []

    step = max(1, len(sqnorm) // 12)
    marker_indices = np.arange(0, len(sqnorm), step, dtype=int)

    if ymin is not None:
        ymin = float(ymin)
        marker_indices = marker_indices[sqnorm[marker_indices] > ymin]

    return marker_indices.astype(int).tolist()


def _selector_matches(run: SyntheticRunRecord, selector: Mapping[str, Any]) -> bool:
    factor_value = selector.get("factor")
    if factor_value is not None:
        if run.factor is None or not math.isclose(float(run.factor), float(factor_value), rel_tol=0.0, abs_tol=1e-12):
            return False

    selector_wo_factor = {key: value for key, value in selector.items() if key != "factor"}
    if run.method == "SILAGE_m>n":
        return len(selector_wo_factor) == 0
    if run.method == "SILAGE_n>m":
        if "batch_mode" in selector_wo_factor:
            return run.batch_mode == selector_wo_factor["batch_mode"]
        if "batch_size" in selector_wo_factor:
            return run.batch_size == int(selector_wo_factor["batch_size"])
        return False
    if run.method in {"ZeroSARAH", "SILVER"}:
        return "batch_size" in selector_wo_factor and run.batch_size == int(selector_wo_factor["batch_size"])
    if run.method in {"D-ZeroSARAH"}:
        return (
            "batch_size" in selector_wo_factor
            and "client_subset_size" in selector_wo_factor
            and run.batch_size == int(selector_wo_factor["batch_size"])
            and run.client_subset_size == int(selector_wo_factor["client_subset_size"])
        )
    return False


def select_trajectory_run(
    runs: Sequence[SyntheticRunRecord],
    setting: str,
    regime: str,
    method: str,
    selector: Mapping[str, Any],
) -> SyntheticRunRecord:
    matching_runs = [
        run
        for run in runs
        if run.method == method and run.setting == setting and run.regime == regime and _selector_matches(run, selector)
    ]
    if not matching_runs:
        raise LookupError(
            f"No run found for method={method}, setting={setting}, regime={regime}, selector={dict(selector)}."
        )
    if len(matching_runs) > 1:
        raise LookupError(
            f"Selector matched multiple runs for method={method}, setting={setting}, regime={regime}, "
            f"selector={dict(selector)}. Please make the selector more specific or clean duplicate logs."
        )
    return matching_runs[0]


def _group_best_by_batch(records: Sequence[SyntheticRunRecord]) -> list[tuple[int, float]]:
    grouped: dict[int, float] = {}
    for record in records:
        if record.batch_size is None:
            continue
        final_epoch = _load_final_epoch_value(record)
        grouped[record.batch_size] = min(final_epoch, grouped.get(record.batch_size, float("inf")))
    return sorted(grouped.items())


def _group_best_surface(records: Sequence[SyntheticRunRecord]) -> tuple[list[int], list[int], np.ndarray]:
    filtered = [record for record in records if record.batch_size is not None and record.client_subset_size is not None]
    client_values = sorted({int(record.client_subset_size) for record in filtered})
    batch_values = sorted({int(record.batch_size) for record in filtered})
    grid = np.full((len(client_values), len(batch_values)), np.nan, dtype=np.float64)
    client_index = {value: idx for idx, value in enumerate(client_values)}
    batch_index = {value: idx for idx, value in enumerate(batch_values)}

    for record in filtered:
        value = _load_final_epoch_value(record)
        row = client_index[int(record.client_subset_size)]
        col = batch_index[int(record.batch_size)]
        if math.isnan(grid[row, col]):
            grid[row, col] = value
        else:
            grid[row, col] = min(grid[row, col], value)

    return client_values, batch_values, grid


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _panel_missing_text(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def create_synthetic_trajectory_grids(config: Mapping[str, Any] | None = None) -> dict[str, Path]:
    cfg = dict(config or {})
    style = _resolve_plot_style(cfg.get("plot_style"))
    _default_matplotlib_rcparams(style)

    logreg_dir = resolve_logreg_dir()
    runs = discover_synthetic_runs(logreg_dir)
    if not runs:
        raise FileNotFoundError("No synthetic log directories were found under logs/.")

    methods_by_setting = cfg.get("methods_by_setting", DEFAULT_TRAJECTORY_METHODS_BY_SETTING)
    selector_map = cfg.get("trajectory_run_selector", {})
    epoch_tick_step_by_setting = cfg.get("trajectory_epoch_tick_step_by_setting", {})
    epoch_xlim_by_setting = cfg.get("trajectory_epoch_xlim_by_setting", {})
    trajectory_ymin = cfg.get("trajectory_ymin")
    legend_bbox_y = float(cfg.get("trajectory_legend_bbox_y", 0.02))
    tight_layout_rect = cfg.get("trajectory_tight_layout_rect", [0, 0.06, 1, 0.95])
    selected_factor = cfg.get("factor")
    selected_settings = list(cfg.get("settings", SYNTHETIC_SETTINGS))
    invalid_settings = sorted(set(selected_settings) - set(SYNTHETIC_SETTINGS))
    if invalid_settings:
        raise ValueError(f"Unsupported synthetic settings requested: {invalid_settings}")
    d_value = int(cfg.get("d", 1000))
    show_plot = bool(cfg.get("show_plot", 1))
    save_plot = bool(cfg.get("save_plot", 1))
    output_dir = logreg_dir / cfg.get("output_dir", "plots/synthetic_trajectory_grids")
    if save_plot:
        _ensure_output_dir(output_dir)

    outputs: dict[str, Path] = {}
    for setting in selected_settings:
        methods = list(methods_by_setting.get(setting, []))
        fig, axes = plt.subplots(2, 2, figsize=(20, 15), sharey=True)
        legend_handles = []
        legend_labels = []

        n_groups, m_per_group = synthetic_logreg_setting_to_nm(setting)
        fig.suptitle(f"n={n_groups}, m={m_per_group}", fontsize=style["suptitle_fontsize"])

        for row_idx, regime_row in enumerate(REGIME_GRID_LAYOUT):
            for col_idx, regime in enumerate(regime_row):
                ax = axes[row_idx, col_idx]
                ax.set_title(format_synthetic_subplot_title(regime), fontsize=style["subplot_title_fontsize"])

                setting_selectors = selector_map.get(setting, {})
                regime_selectors = setting_selectors.get(regime, {})
                for method in methods:
                    if method not in regime_selectors:
                        raise ValueError(
                            f"Missing trajectory selector for method={method}, setting={setting}, regime={regime}."
                        )
                    selector = dict(regime_selectors[method])
                    if selected_factor is not None:
                        selector.setdefault("factor", float(selected_factor))
                    run = select_trajectory_run(runs, setting, regime, method, selector)
                    epochs = _load_metric_series(run, "epochs")
                    sqnorm = _load_metric_series(run, "sqnorm")
                    num_points = min(len(epochs), len(sqnorm))
                    epochs = epochs[:num_points]
                    sqnorm = np.maximum(sqnorm[:num_points], 1e-24)
                    markevery = _trajectory_markevery_indices(sqnorm, trajectory_ymin)
                    method_style = METHOD_STYLES[method]
                    (line,) = ax.plot(
                        epochs,
                        sqnorm,
                        label=METHOD_DISPLAY_NAMES[method],
                        color=method_style["color"],
                        marker=method_style["marker"],
                        markevery=markevery,
                        markersize=style["marker_size"],
                        markeredgecolor="black",
                    )
                    if METHOD_DISPLAY_NAMES[method] not in legend_labels:
                        legend_handles.append(line)
                        legend_labels.append(METHOD_DISPLAY_NAMES[method])

                ax.set_yscale("log")
                if trajectory_ymin is not None:
                    ax.set_ylim(bottom=float(trajectory_ymin))
                ax.set_xlabel("Epochs" if row_idx == 1 else "")
                ax.set_ylabel(r"$\|\nabla f(x^t)\|^2$" if col_idx == 0 else "")
                ax.grid(True)
                epoch_xlim = _get_nested_setting(epoch_xlim_by_setting, setting, regime)
                if epoch_xlim is not None:
                    ax.set_xlim(left=0.0, right=float(epoch_xlim))
                epoch_tick_step = _get_nested_setting(epoch_tick_step_by_setting, setting, regime)
                if epoch_tick_step is not None:
                    ax.xaxis.set_major_locator(MultipleLocator(float(epoch_tick_step)))
                ax.tick_params(axis="both", which="major", labelsize=style["tick_labelsize"])

        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=min(4, len(legend_labels)),
                bbox_to_anchor=(0.5, legend_bbox_y),
                fontsize=style["legend_fontsize"],
            )
        fig.tight_layout(rect=tight_layout_rect)

        output_path = output_dir / f"trajectory_grid_{setting}.pdf"
        if save_plot:
            fig.savefig(output_path, bbox_inches="tight")
        outputs[setting] = output_path
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return outputs


def _plot_1d_tuning_axis(
    ax: plt.Axes,
    records: Sequence[SyntheticRunRecord],
    methods: Sequence[str],
    style: Mapping[str, float],
) -> list[tuple[Any, str]]:
    plotted_any = False
    missing_methods = []
    legend_entries: list[tuple[Any, str]] = []

    for method in methods:
        method_records = [record for record in records if record.method == method]
        points = _group_best_by_batch(method_records)
        if not points:
            missing_methods.append(method)
            continue
        x_values = [batch_size for batch_size, _ in points]
        y_values = [epoch_value for _, epoch_value in points]
        method_style = METHOD_STYLES[method]
        (line,) = ax.plot(
            x_values,
            y_values,
            label=METHOD_DISPLAY_NAMES[method],
            color=method_style["color"],
            marker=method_style["marker"],
            markersize=style["marker_size"],
            markeredgecolor="black",
        )
        if METHOD_DISPLAY_NAMES[method] not in [label for _, label in legend_entries]:
            legend_entries.append((line, METHOD_DISPLAY_NAMES[method]))
        plotted_any = True

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Final epochs value")
    ax.set_xscale("linear")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.tick_params(axis="both", which="major", labelsize=style["tick_labelsize"])
    if not plotted_any:
        _panel_missing_text(ax, "No 1D sweep logs")

    if missing_methods and plotted_any:
        missing_label = ", ".join(METHOD_DISPLAY_NAMES[method] for method in missing_methods)
        ax.text(
            0.02,
            0.02,
            f"Missing: {missing_label}",
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
        )
    return legend_entries


def _plot_2d_tuning_axis(
    fig: plt.Figure,
    ax: plt.Axes,
    records: Sequence[SyntheticRunRecord],
    method: str,
    style: Mapping[str, float],
    show_method_title: bool = True,
) -> None:
    if not records:
        _panel_missing_text(ax, f"No {METHOD_DISPLAY_NAMES[method]} logs")
        if show_method_title:
            ax.set_title(METHOD_DISPLAY_NAMES[method], fontsize=style["subplot_title_fontsize"])
        return

    client_values, batch_values, grid = _group_best_surface(records)
    if not client_values or not batch_values:
        _panel_missing_text(ax, f"No {METHOD_DISPLAY_NAMES[method]} grid")
        if show_method_title:
            ax.set_title(METHOD_DISPLAY_NAMES[method], fontsize=style["subplot_title_fontsize"])
        return

    if show_method_title:
        ax.set_title(METHOD_DISPLAY_NAMES[method], fontsize=style["subplot_title_fontsize"])
    image = None
    masked_grid = np.ma.masked_invalid(grid)
    is_complete = (
        masked_grid.count() == masked_grid.size
        and len(client_values) > 1
        and len(batch_values) > 1
    )

    if is_complete:
        batch_mesh, client_mesh = np.meshgrid(batch_values, client_values)
        image = ax.contourf(batch_mesh, client_mesh, grid, levels=12, cmap="viridis")
        ax.set_xticks(batch_values)
        ax.set_yticks(client_values)
    else:
        image = ax.imshow(masked_grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(batch_values)))
        ax.set_xticklabels(batch_values, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(client_values)))
        ax.set_yticklabels(client_values)
        ax.text(
            0.98,
            0.02,
            "partial grid",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="bottom",
            color="white",
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Clients")
    ax.tick_params(axis="both", which="major", labelsize=style["tick_labelsize"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def create_synthetic_tuning_summary_grids(config: Mapping[str, Any] | None = None) -> dict[str, Path]:
    cfg = dict(config or {})
    style = _resolve_plot_style(cfg.get("plot_style"))
    _default_matplotlib_rcparams(style)

    logreg_dir = resolve_logreg_dir()
    runs = discover_synthetic_runs(logreg_dir)
    if not runs:
        raise FileNotFoundError("No synthetic log directories were found under logs/.")
    selected_factor = cfg.get("factor")
    if selected_factor is not None:
        runs = [
            run
            for run in runs
            if run.factor is not None
            and math.isclose(float(run.factor), float(selected_factor), rel_tol=0.0, abs_tol=1e-12)
        ]
        if not runs:
            raise FileNotFoundError(
                f"No synthetic log directories were found under logs/ for factor={float(selected_factor)}."
            )

    methods_1d_by_setting = cfg.get("methods_1d_by_setting", DEFAULT_TUNING_1D_METHODS_BY_SETTING)
    methods_2d = list(cfg.get("methods_2d", DEFAULT_TUNING_2D_METHODS))
    selected_settings = list(cfg.get("settings", SYNTHETIC_SETTINGS))
    invalid_settings = sorted(set(selected_settings) - set(SYNTHETIC_SETTINGS))
    if invalid_settings:
        raise ValueError(f"Unsupported synthetic settings requested: {invalid_settings}")
    d_value = int(cfg.get("d", 1000))
    show_plot = bool(cfg.get("show_plot", 1))
    save_plot = bool(cfg.get("save_plot", 1))
    output_dir = logreg_dir / cfg.get("output_dir", "plots/synthetic_tuning_summary_grids")
    if save_plot:
        _ensure_output_dir(output_dir)

    outputs: dict[str, Path] = {}
    for setting in selected_settings:
        n_groups, m_per_group = synthetic_logreg_setting_to_nm(setting)
        methods_1d = list(methods_1d_by_setting.get(setting, []))
        shared_legend_entries: list[tuple[Any, str]] = []

        # If no 2D methods are requested, render a simple 2x2 grid of 1D tuning curves.
        # This is useful when we want one selected 1D method only with the same title style
        # as the synthetic trajectory grids.
        if len(methods_2d) == 0:
            fig, axes = plt.subplots(2, 2, figsize=(20, 15), sharey=True)
            fig.suptitle(f"n={n_groups}, m={m_per_group}", fontsize=style["suptitle_fontsize"])

            for row_idx, regime_row in enumerate(REGIME_GRID_LAYOUT):
                for col_idx, regime in enumerate(regime_row):
                    ax = axes[row_idx, col_idx]
                    ax.set_title(format_synthetic_subplot_title(regime), fontsize=style["subplot_title_fontsize"])

                    regime_records = [record for record in runs if record.setting == setting and record.regime == regime]
                    legend_entries = _plot_1d_tuning_axis(ax, regime_records, methods_1d, style)
                    for handle, label in legend_entries:
                        if label not in [existing_label for _, existing_label in shared_legend_entries]:
                            shared_legend_entries.append((handle, label))

            if shared_legend_entries:
                fig.legend(
                    [handle for handle, _ in shared_legend_entries],
                    [label for _, label in shared_legend_entries],
                    loc="lower center",
                    ncol=min(4, len(shared_legend_entries)),
                    bbox_to_anchor=(0.5, 0.02),
                    fontsize=style["legend_fontsize"],
                )
            fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        elif len(methods_1d) == 0 and len(methods_2d) == 1:
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f"n={n_groups}, m={m_per_group}", fontsize=style["suptitle_fontsize"])
            method_2d = methods_2d[0]

            for row_idx, regime_row in enumerate(REGIME_GRID_LAYOUT):
                for col_idx, regime in enumerate(regime_row):
                    ax = axes[row_idx, col_idx]
                    ax.set_title(format_synthetic_subplot_title(regime), fontsize=style["subplot_title_fontsize"])

                    regime_records = [
                        record
                        for record in runs
                        if record.setting == setting and record.regime == regime and record.method == method_2d
                    ]
                    _plot_2d_tuning_axis(fig, ax, regime_records, method_2d, style, show_method_title=False)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            fig = plt.figure(figsize=(26, 18))
            outer_grid = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.18)
            fig.suptitle(f"n={n_groups}, m={m_per_group}", fontsize=style["suptitle_fontsize"])

            for row_idx, regime_row in enumerate(REGIME_GRID_LAYOUT):
                for col_idx, regime in enumerate(regime_row):
                    panel_grid = outer_grid[row_idx, col_idx].subgridspec(1, 3, width_ratios=[1.6, 1.0, 1.0], wspace=0.35)
                    ax_1d = fig.add_subplot(panel_grid[0, 0])
                    ax_dzs = fig.add_subplot(panel_grid[0, 1])
                    ax_dp = fig.add_subplot(panel_grid[0, 2])

                    ax_1d.set_title(format_synthetic_subplot_title(regime), fontsize=style["subplot_title_fontsize"])

                    regime_records = [record for record in runs if record.setting == setting and record.regime == regime]
                    legend_entries = _plot_1d_tuning_axis(ax_1d, regime_records, methods_1d, style)
                    for handle, label in legend_entries:
                        if label not in [existing_label for _, existing_label in shared_legend_entries]:
                            shared_legend_entries.append((handle, label))

                    dz_records = [record for record in regime_records if record.method == methods_2d[0]]
                    dp_records = [record for record in regime_records if record.method == methods_2d[1]]
                    _plot_2d_tuning_axis(fig, ax_dzs, dz_records, methods_2d[0], style)
                    _plot_2d_tuning_axis(fig, ax_dp, dp_records, methods_2d[1], style)

            if shared_legend_entries:
                fig.legend(
                    [handle for handle, _ in shared_legend_entries],
                    [label for _, label in shared_legend_entries],
                    loc="lower center",
                    ncol=min(4, len(shared_legend_entries)),
                    bbox_to_anchor=(0.5, 0.015),
                    fontsize=style["legend_fontsize"],
                )
            fig.subplots_adjust(top=0.93, bottom=0.08, left=0.04, right=0.98)
        output_path = output_dir / f"tuning_summary_{setting}.pdf"
        if save_plot:
            fig.savefig(output_path, bbox_inches="tight")
        outputs[setting] = output_path
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return outputs


__all__ = [
    "DEFAULT_TRAJECTORY_METHODS_BY_SETTING",
    "DEFAULT_TUNING_1D_METHODS_BY_SETTING",
    "DEFAULT_TUNING_2D_METHODS",
    "METHOD_DISPLAY_NAMES",
    "METHOD_STYLES",
    "SYNTHETIC_REGIMES",
    "SYNTHETIC_SETTINGS",
    "SyntheticRunRecord",
    "create_synthetic_trajectory_grids",
    "create_synthetic_tuning_summary_grids",
    "discover_synthetic_runs",
    "format_synthetic_subplot_title",
    "resolve_logreg_dir",
    "select_trajectory_run",
]
