"""
ZeroSARAH baseline for the synthetic flattened logistic-regression datasets.

The active implementation follows the repo's gradient-initialized convention:

    * is_grad_comp_init must be 1
    * y_i^0 is initialized with the full table of component gradients at x^0
    * v^0 is initialized as the average of that table, i.e. the exact full gradient

This is intentionally different from the paper's zero-memory warm start and is
treated as the only supported active-path initialization in this codebase.
"""

import sys

from src.algorithm import *
from src.oracle_functions import (
    torch_logreg_grad,
    torch_logreg_grad_matrix,
)


class ZeroSARAH(Algorithm):
    def __init__(self, args=None):
        super().__init__(args)

    def script_directory(self):
        return os.path.dirname(os.path.abspath(__file__))

    def fill_alg_params_dict(self, state, oracle, data, alg_param):
        """Fill ZeroSARAH hyperparameters from the loaded flat synthetic tensors."""
        if self.arg_values["synthetic_setting"] not in SYNTHETIC_SETTING_TO_NM:
            raise ValueError(
                "ZeroSARAH is only implemented for the active synthetic settings "
                f"{sorted(SYNTHETIC_SETTING_TO_NM.keys())}."
            )
        if self.arg_values["is_grad_comp_init"] != 1:
            raise ValueError("ZeroSARAH requires --is_grad_comp_init 1 in the current implementation.")

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
                f"ZeroSARAH expects batch_size in [1, N], got b={batch_size}, N={dataset_size}."
            )

        alg_param["resolved_device"] = resolve_torch_device(alg_param["device"], alg_param["print_status"])
        alg_param["dataset_size"] = dataset_size
        alg_param["batch_size"] = batch_size
        # The full-gradient initialized variant uses nu_0 = 1 only for the first
        # stochastic step, then switches to the theorem-style constant b / (2N).
        alg_param["nu_init"] = 1.0
        alg_param["nu_const"] = float(batch_size) / (2.0 * float(dataset_size))
        alg_param["nu"] = alg_param["nu_const"]
        alg_param["step_size"] = float(
            get_zerosarah_stepsize(
                alg_param["L_ij_max_emp"],
                dataset_size=dataset_size,
                batch_size=batch_size,
                factor=alg_param["factor"],
            )
        )
        return alg_param

    def init_oracles_dict(self, state, oracle, data, alg_param):
        """Create the flattened Torch helpers used by ZeroSARAH."""
        import torch

        regularizer_type = self.arg_values["regularizer_type"]
        lambda_reg = float(alg_param["lambda_reg"])

        def batch_sample_grads(x, sample_indices):
            # Gather arbitrary flattened samples into one dense batch so every
            # minibatch update is computed with a single Torch matrix kernel.
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
            # The sqnorm metric is always the exact full objective gradient.
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
        """Move tensors to device and initialize x^0, y^0, and v^0."""
        import torch

        resolved_device = alg_param["resolved_device"]

        # ZeroSARAH works on the flattened dataset only; grouped tensors are not
        # loaded into the active path for this baseline.
        data["X_train"] = data["X_train"].to(device=resolved_device, dtype=torch.float64)
        data["y_train"] = data["y_train"].to(device=resolved_device, dtype=torch.int64)

        x_0 = torch.as_tensor(state["x"], device=resolved_device, dtype=torch.float64)
        if x_0.ndim != 1 or x_0.shape[0] != data["X_train"].shape[1]:
            raise ValueError(f"Loaded w_init has shape {tuple(x_0.shape)} but expected ({data['X_train'].shape[1]},).")

        # Build the full memory table y^0 explicitly, as requested for the active
        # gradient-initialized mode. This costs one exact pass over the dataset.
        full_index_t = torch.arange(alg_param["dataset_size"], device=resolved_device, dtype=torch.int64)
        y_0 = oracle["batch_sample_grads"](x_0, full_index_t)
        v_0 = y_0.mean(dim=0)

        state["x_prev"] = x_0.detach().clone()
        state["x"] = x_0
        state["v"] = v_0
        state["g"] = v_0
        state["y_memory"] = y_0
        # Keep the running average alongside the full table so later updates do
        # not need to recompute y_memory.mean(0) from scratch.
        state["y_avg"] = v_0.detach().clone()

        alg_param["training_epochs"] = 1.0
        alg_param["latest_sqnorm"] = float(torch.dot(v_0, v_0).item())
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
        """Run one ZeroSARAH iteration on the flattened synthetic dataset."""
        import torch

        x_t = state["x"]
        x_prev = state["x_prev"]
        v_prev = state["v"]
        y_memory_prev = state["y_memory"]
        y_avg_prev = state["y_avg"]

        dataset_size = alg_param["dataset_size"]
        batch_size = alg_param["batch_size"]
        # iter == 0 is the first stochastic update, so we use nu_0 = 1 exactly
        # once and switch to the constant theorem value on all later updates.
        nu_t = alg_param["nu_init"] if int(alg_param["iter"]) == 0 else alg_param["nu_const"]
        rs_sample = alg_param["rs_sample"]

        # Sample one flat minibatch I^t uniformly without replacement.
        batch_indices_np = rs_sample.choice(dataset_size, size=batch_size, replace=False).astype(np.int64, copy=False)
        batch_indices_t = torch.as_tensor(batch_indices_np, device=x_t.device, dtype=torch.int64)

        # Step 2 of the pseudocode: build v^t from x^t, x^{t-1}, v^{t-1}, and
        # the previous memory table y^{t-1}. Each row is already the full
        # regularized component gradient.
        grad_x_t = oracle["batch_sample_grads"](x_t, batch_indices_t)
        grad_x_prev = oracle["batch_sample_grads"](x_prev, batch_indices_t)
        old_y = y_memory_prev[batch_indices_t]

        v_t = (
            (grad_x_t - grad_x_prev).mean(dim=0)
            + (1.0 - nu_t) * v_prev
            + nu_t * ((grad_x_prev - old_y).mean(dim=0) + y_avg_prev)
        )

        # Step 3 of the pseudocode: x^{t+1} = x^t - gamma v^t.
        x_next = x_t - alg_param["step_size"] * v_t

        # Step 4 of the pseudocode: refresh only the sampled memory rows with
        # \nabla f_i(x^t). The running average is updated incrementally so it
        # always equals the mean of y_memory at O(d) cost.
        y_memory = y_memory_prev
        y_memory[batch_indices_t] = grad_x_t
        y_avg = y_avg_prev + (grad_x_t - old_y).sum(dim=0) / float(dataset_size)

        state["x_prev"] = x_t
        state["x"] = x_next
        state["v"] = v_t
        state["g"] = v_t
        state["y_memory"] = y_memory
        state["y_avg"] = y_avg

        alg_param["iter"] += 1
        # Each iteration evaluates b gradients at x^t and b gradients at x^{t+1}.
        alg_param["training_epochs"] += (2.0 * float(batch_size)) / float(dataset_size)

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
                (2.0 * float(batch_size)) / float(dataset_size),
            )

        return state, collectable_metric, alg_param


if __name__ == "__main__":
    ZeroSARAH(sys.argv[1:]).run()
