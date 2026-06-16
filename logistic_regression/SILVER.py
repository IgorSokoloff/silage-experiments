"""
SILVER Option I baseline for flattened synthetic logistic-regression data.

The active implementation uses the repo's regularized-component convention:

    f_r(x) = log_loss_r(x) + lambda * R(x),

so each row returned by torch_logreg_grad_matrix(...) is already the full
regularized component gradient. The perturbation radius is fixed to zero.
"""

import sys

from src.algorithm import *
from src.oracle_functions import (
    torch_logreg_grad,
    torch_logreg_grad_matrix,
)


class SILVER(Algorithm):
    def __init__(self, args=None):
        super().__init__(args)

    def script_directory(self):
        return os.path.dirname(os.path.abspath(__file__))

    def fill_alg_params_dict(self, state, oracle, data, alg_param):
        """Resolve SILVER hyperparameters from the loaded flat synthetic tensors."""
        if self.arg_values["synthetic_setting"] not in SYNTHETIC_SETTING_TO_NM:
            raise ValueError(
                "SILVER is only implemented for the active synthetic settings "
                f"{sorted(SYNTHETIC_SETTING_TO_NM.keys())}."
            )
        if self.arg_values["is_grad_comp_init"] != 1:
            raise ValueError("SILVER requires --is_grad_comp_init 1 in the current implementation.")

        X_train = data["X_train"]
        if not hasattr(X_train, "shape") or len(X_train.shape) != 2:
            raise ValueError("Expected X_train to be a rank-2 flattened feature matrix.")

        dataset_size = int(X_train.shape[0])
        dim = int(X_train.shape[1])
        if dataset_size <= 0 or dim <= 0:
            raise ValueError(f"Invalid flattened dataset shape: {tuple(X_train.shape)}.")

        batch_size = int(alg_param["batch_size"])
        if not (1 <= batch_size <= dataset_size):
            raise ValueError(
                f"SILVER expects batch_size in [1, N], got b={batch_size}, N={dataset_size}."
            )

        alg_param["resolved_device"] = resolve_torch_device(alg_param["device"], alg_param["print_status"])
        alg_param["dataset_size"] = dataset_size
        alg_param["batch_size"] = batch_size
        alg_param["step_size"] = float(
            get_silver_stepsize(
                alg_param["L_ij_max_emp"],
                alg_param["delta_flat_emp"],
                dataset_size=dataset_size,
                batch_size=batch_size,
                factor=alg_param["factor"],
            )
        )
        return alg_param

    def init_oracles_dict(self, state, oracle, data, alg_param):
        """Create flattened Torch helpers used by SILVER."""
        import torch

        regularizer_type = self.arg_values["regularizer_type"]
        lambda_reg = float(alg_param["lambda_reg"])

        def batch_sample_grads(x, sample_indices):
            sample_indices_t = torch.as_tensor(
                sample_indices,
                device=data["X_train"].device,
                dtype=torch.int64,
            ).reshape(-1)
            X_sel = data["X_train"][sample_indices_t]
            y_sel = data["y_train"][sample_indices_t]
            return torch_logreg_grad_matrix(
                x,
                X_sel,
                y_sel,
                lambda_reg,
                regularizer_type,
            )

        def full_grad(x):
            return torch_logreg_grad(
                x,
                data["X_train"],
                data["y_train"],
                lambda_reg,
                regularizer_type,
            )

        return {
            "batch_sample_grads": batch_sample_grads,
            "full_grad": full_grad,
        }

    def init_states_dict(self, state, oracle, data, alg_param):
        """Move flat tensors to device and initialize x^0, y^0, and g^0."""
        import torch

        resolved_device = alg_param["resolved_device"]
        data["X_train"] = data["X_train"].to(device=resolved_device, dtype=torch.float64)
        data["y_train"] = data["y_train"].to(device=resolved_device, dtype=torch.int64)

        x_0 = torch.as_tensor(state["x"], device=resolved_device, dtype=torch.float64)
        if x_0.ndim != 1 or x_0.shape[0] != data["X_train"].shape[1]:
            raise ValueError(f"Loaded w_init has shape {tuple(x_0.shape)} but expected ({data['X_train'].shape[1]},).")

        # SILVER Option I starts from the full table of component gradients.
        # We store y_r as y_memory_base[r] + global_shift to make later
        # all-row shifts exact without performing dense N-by-d additions.
        full_index_t = torch.arange(alg_param["dataset_size"], device=resolved_device, dtype=torch.int64)
        y_memory_base = oracle["batch_sample_grads"](x_0, full_index_t)
        global_shift = torch.zeros_like(x_0)
        base_mean = y_memory_base.mean(dim=0)
        g_0 = base_mean + global_shift

        state["x_prev"] = x_0.detach().clone()
        state["x"] = x_0
        state["g"] = g_0
        state["y_memory_base"] = y_memory_base
        state["global_shift"] = global_shift
        state["base_mean"] = base_mean.detach().clone()

        alg_param["training_epochs"] = 1.0
        alg_param["latest_sqnorm"] = float(torch.dot(g_0, g_0).item())
        alg_param["last_sqnorm_iter"] = 0
        return state, alg_param

    def init_collectable_metrics_dict(self, state, collectable_metric, alg_param, oracle, data):
        if "iters" in collectable_metric.keys():
            collectable_metric["iters"] = [0]
        if "epochs" in collectable_metric.keys():
            collectable_metric["epochs"] = [float(alg_param["training_epochs"])]
        if "sqnorm" in collectable_metric.keys():
            collectable_metric["sqnorm"] = [float(alg_param["latest_sqnorm"])]
        return collectable_metric

    def update_collectable_metrics_dict(self, state, collectable_metric, alg_param, oracle, data, epochs_single_iter):
        if "iters" in collectable_metric.keys():
            collectable_metric["iters"].append(int(alg_param["iter"]))
        if "epochs" in collectable_metric.keys():
            collectable_metric["epochs"].append(float(alg_param["training_epochs"]))
        if "sqnorm" in collectable_metric.keys():
            collectable_metric["sqnorm"].append(float(alg_param["latest_sqnorm"]))
        return collectable_metric

    def update(self, state, data, collectable_metric, alg_param, oracle, update_collectable_metrics_dict):
        """Run one SILVER Option I iteration on the flattened dataset."""
        import torch

        x_t = state["x"]
        g_t = state["g"]
        y_memory_base = state["y_memory_base"]
        global_shift = state["global_shift"]
        base_mean = state["base_mean"]

        dataset_size = int(alg_param["dataset_size"])
        batch_size = int(alg_param["batch_size"])
        rs_sample = alg_param["rs_sample"]

        x_next = x_t - alg_param["step_size"] * g_t

        batch_indices_np = rs_sample.choice(dataset_size, size=batch_size, replace=False).astype(np.int64, copy=False)
        batch_indices_t = torch.as_tensor(batch_indices_np, device=x_t.device, dtype=torch.int64)

        grad_new = oracle["batch_sample_grads"](x_next, batch_indices_t)
        grad_old = oracle["batch_sample_grads"](x_t, batch_indices_t)
        d_t = (grad_new - grad_old).mean(dim=0)

        global_shift_next = global_shift + d_t
        old_base_rows = y_memory_base[batch_indices_t].detach().clone()
        new_base_rows = grad_new - global_shift_next.unsqueeze(0)

        y_memory_base[batch_indices_t] = new_base_rows
        base_mean_next = base_mean + (new_base_rows - old_base_rows).sum(dim=0) / float(dataset_size)
        g_next = base_mean_next + global_shift_next

        state["x_prev"] = x_t
        state["x"] = x_next
        state["g"] = g_next
        state["y_memory_base"] = y_memory_base
        state["global_shift"] = global_shift_next
        state["base_mean"] = base_mean_next

        epochs_single_iter = (2.0 * float(batch_size)) / float(dataset_size)
        alg_param["iter"] += 1
        alg_param["training_epochs"] += epochs_single_iter

        if alg_param["iter"] % alg_param["COLLECT_EVERY"] == 0:
            full_grad = oracle["full_grad"](state["x"])
            alg_param["latest_sqnorm"] = float(torch.dot(full_grad, full_grad).item())
            alg_param["last_sqnorm_iter"] = alg_param["iter"]
            collectable_metric = update_collectable_metrics_dict(
                state,
                collectable_metric,
                alg_param,
                oracle,
                data,
                epochs_single_iter,
            )

        return state, collectable_metric, alg_param


if __name__ == "__main__":
    SILVER(sys.argv[1:]).run()
