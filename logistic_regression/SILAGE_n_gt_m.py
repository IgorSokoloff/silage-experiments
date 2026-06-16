"""
SILAGE for the n > m synthetic grouped logistic-regression setting.

The implementation keeps the exact finite-sum structure

    (1/n) sum_i (1/m) sum_j f_{i,j}(x) + lambda * R(x)

by treating each local component f_{i,j} as already regularized. The per-group
control variates therefore store full local estimates, and the global estimator
is just the average of those local estimates.
"""

import sys

from src.algorithm import *
from src.oracle_functions import (
    torch_logreg_grad_matrix,
    torch_logreg_grad,
)


class SILAGE_n_gt_m(Algorithm):
    def __init__(self, args=None):
        super().__init__(args)

    def script_directory(self):
        return os.path.dirname(os.path.abspath(__file__))

    def fill_alg_params_dict(self, state, oracle, data, alg_param):
        """Fill SILAGE_n>m hyperparameters from loaded tensors and diagnostics."""
        if self.arg_values["synthetic_setting"] != "n_gt_m":
            raise ValueError("SILAGE_n>m is only implemented for synthetic_setting='n_gt_m'.")
        if self.arg_values["is_grad_comp_init"] != 1:
            raise ValueError("SILAGE_n>m requires --is_grad_comp_init 1 in the current implementation.")

        X_groups = data["X_groups_train"]
        if not hasattr(X_groups, "shape") or len(X_groups.shape) != 3:
            raise ValueError("Expected X_groups_train to be a rank-3 grouped tensor.")

        n_groups = int(X_groups.shape[0])
        m_per_group = int(X_groups.shape[1])
        if n_groups <= 1 or m_per_group <= 0:
            raise ValueError(f"Invalid grouped shape: n_groups={n_groups}, m_per_group={m_per_group}.")

        loaded_optimal_b = alg_param.get("optimal_b")
        if isinstance(loaded_optimal_b, np.ndarray):
            loaded_optimal_b = loaded_optimal_b.item()

        if loaded_optimal_b is not None:
            batch_size = int(loaded_optimal_b)
            batch_size_mode = "optimal_b"
        else:
            batch_size = int(alg_param["batch_size"])
            batch_size_mode = "constant"

        if not (1 <= batch_size <= n_groups):
            raise ValueError(f"SILAGE_n>m expects batch_size in [1, n], got b={batch_size}, n={n_groups}.")

        alg_param["resolved_device"] = resolve_torch_device(alg_param["device"], alg_param["print_status"])
        alg_param["n_groups"] = n_groups
        alg_param["m_per_group"] = m_per_group
        alg_param["dataset_size"] = int(n_groups * m_per_group)
        if alg_param["dataset_size"] != int(data["X_train"].shape[0]):
            raise ValueError(
                f"Expected dataset_size={alg_param['dataset_size']} to match flattened X_train rows="
                f"{int(data['X_train'].shape[0])}."
            )

        alg_param["batch_size"] = batch_size
        alg_param["selected_batch_size"] = batch_size
        alg_param["batch_size_mode"] = batch_size_mode
        alg_param["step_size"] = float(
            get_silage_n_gt_m_stepsize(
                alg_param["L_global_emp"],
                alg_param["delta1_emp"],
                alg_param["delta2_emp"],
                n_groups=n_groups,
                batch_size=batch_size,
                factor=alg_param["factor"],
            )
        )
        return alg_param

    def init_oracles_dict(self, state, oracle, data, alg_param):
        """Create Torch helpers specialized to grouped synthetic log-reg data."""
        import torch

        regularizer_type = self.arg_values["regularizer_type"]
        lambda_reg = float(alg_param["lambda_reg"])

        def group_sample_grads(x, group_idx):
            # This returns one full regularized gradient row per sample in group i.
            return torch_logreg_grad_matrix(
                x,
                data["X_groups_train"][group_idx],
                data["y_groups_train"][group_idx],
                lambda_reg,
                regularizer_type,
            )

        def group_data_grad(x, group_idx):
            # This is the exact gradient of (1/m) sum_j f_{i,j}(x) for one group i,
            # with the regularizer already included in each component.
            return group_sample_grads(x, group_idx).mean(dim=0)

        def batch_pair_data_grads(x, group_indices, sample_indices):
            # Gather arbitrary (i, j) pairs into one dense batch so SILAGE_n>m can
            # form the line-12 differences without a Python loop over the batch.
            group_indices_t = torch.as_tensor(
                group_indices,
                device=data["X_groups_train"].device,
                dtype=torch.int64,
            ).reshape(-1)
            sample_indices_t = torch.as_tensor(
                sample_indices,
                device=data["X_groups_train"].device,
                dtype=torch.int64,
            ).reshape(-1)
            if group_indices_t.shape != sample_indices_t.shape:
                raise ValueError(
                    f"Expected matching group/sample index shapes, got {tuple(group_indices_t.shape)} and "
                    f"{tuple(sample_indices_t.shape)}."
                )
            X_sel = data["X_groups_train"][group_indices_t, sample_indices_t]
            y_sel = data["y_groups_train"][group_indices_t, sample_indices_t]
            return torch_logreg_grad_matrix(
                x,
                X_sel,
                y_sel,
                lambda_reg,
                regularizer_type,
            )

        def full_grad(x):
            # The full metric sqnorm uses the exact full objective gradient.
            return torch_logreg_grad(
                x,
                data["X_train"],
                data["y_train"],
                lambda_reg,
                regularizer_type,
            )

        return {
            "group_sample_grads": group_sample_grads,
            "group_data_grad": group_data_grad,
            "batch_pair_data_grads": batch_pair_data_grads,
            "full_grad": full_grad,
        }

    def init_states_dict(self, state, oracle, data, alg_param):
        """Move tensors to the selected device and initialize x^0, g_i^0, and g^0."""
        import torch

        resolved_device = alg_param["resolved_device"]

        # Load both flattened and grouped tensors because grouped slices are natural
        # for SILAGE updates, while the flat tensors are cheapest for exact sqnorm.
        data["X_train"] = data["X_train"].to(device=resolved_device, dtype=torch.float64)
        data["y_train"] = data["y_train"].to(device=resolved_device, dtype=torch.int64)
        data["X_groups_train"] = data["X_groups_train"].to(device=resolved_device, dtype=torch.float64)
        data["y_groups_train"] = data["y_groups_train"].to(device=resolved_device, dtype=torch.int64)

        x_0 = torch.as_tensor(state["x"], device=resolved_device, dtype=torch.float64)
        if x_0.ndim != 1 or x_0.shape[0] != data["X_train"].shape[1]:
            raise ValueError(f"Loaded w_init has shape {tuple(x_0.shape)} but expected ({data['X_train'].shape[1]},).")

        n_groups = alg_param["n_groups"]
        # g_i^0 are exact local group gradients of the regularized objective. This
        # costs one full pass over the dataset, so it contributes 1 training epoch.
        g_groups = torch.stack([oracle["group_data_grad"](x_0, i) for i in range(n_groups)], dim=0)
        g_full = g_groups.mean(dim=0)

        state["x"] = x_0
        state["x_prev"] = x_0.detach().clone()
        state["g"] = g_full
        state["g_groups"] = g_groups

        alg_param["training_epochs"] = 1.0
        alg_param["latest_sqnorm"] = float(torch.dot(g_full, g_full).item())
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
        # We append only every COLLECT_EVERY steps; the exact full sqnorm has already
        # been computed before this helper is called.
        if "iters" in collectable_metric.keys():
            collectable_metric["iters"].append(int(alg_param["iter"]))
        if "epochs" in collectable_metric.keys():
            collectable_metric["epochs"].append(float(alg_param["training_epochs"]))
        if "sqnorm" in collectable_metric.keys():
            collectable_metric["sqnorm"].append(float(alg_param["latest_sqnorm"]))
        return collectable_metric

    def update(self, state, data, collectable_metric, alg_param, oracle, update_collectable_metrics_dict):
        """Run one SILAGE_n>m iteration with regularized grouped control variates."""
        import torch

        x_t = state["x"]
        g_t = state["g"]
        g_groups = state["g_groups"]

        # Line 3: x^{t+1} = x^t - gamma g^t.
        x_next = x_t - alg_param["step_size"] * g_t

        n_groups = alg_param["n_groups"]
        m_per_group = alg_param["m_per_group"]
        batch_size = alg_param["selected_batch_size"]
        dataset_size = float(alg_param["dataset_size"])
        rs_sample = alg_param["rs_sample"]

        # Line 4: sample the distinguished group i^t uniformly from [n].
        refreshed_group = int(rs_sample.randint(n_groups))

        # Lines 5 and 11: compute the full refreshed group once at x^{t+1}. We will
        # reuse the sampled row j_{i^t}^t from this matrix when forming d^t so the
        # new sample gradient for the distinguished group is not recomputed.
        refreshed_group_sample_grads = oracle["group_sample_grads"](x_next, refreshed_group)
        g_groups[refreshed_group] = refreshed_group_sample_grads.mean(dim=0)

        other_groups_np = np.concatenate(
            [np.arange(refreshed_group, dtype=np.int64), np.arange(refreshed_group + 1, n_groups, dtype=np.int64)]
        )
        if batch_size > 1:
            omega_groups_np = rs_sample.choice(other_groups_np, size=batch_size - 1, replace=False).astype(np.int64, copy=False)
        else:
            omega_groups_np = np.empty((0,), dtype=np.int64)

        # Build \tilde{\Omega}^t = Omega^t union {i^t} and sample one within-group
        # coordinate j_i^t for every selected group.
        selected_groups_np = np.concatenate(([refreshed_group], omega_groups_np))
        selected_sample_indices_np = rs_sample.randint(m_per_group, size=batch_size).astype(np.int64, copy=False)

        selected_groups_t = torch.as_tensor(selected_groups_np, device=x_t.device, dtype=torch.int64)
        selected_sample_indices_t = torch.as_tensor(selected_sample_indices_np, device=x_t.device, dtype=torch.int64)

        omega_diffs = None
        if batch_size > 1:
            omega_groups_t = selected_groups_t[1:]
            omega_sample_indices_t = selected_sample_indices_t[1:]

            # Lines 7-9: update the explicitly sampled Omega^t groups with their
            # own two-point control-variate corrections in one vectorized batch.
            omega_grad_new = oracle["batch_pair_data_grads"](x_next, omega_groups_t, omega_sample_indices_t)
            omega_grad_old = oracle["batch_pair_data_grads"](x_t, omega_groups_t, omega_sample_indices_t)
            omega_diffs = omega_grad_new - omega_grad_old
            g_groups[omega_groups_t] = g_groups[omega_groups_t] + omega_diffs

        if batch_size < n_groups:
            # Line 12: form d^t from the selected set in one batched tensor. For the
            # refreshed group, reuse the already available x^{t+1} sample gradient.
            refreshed_grad_new = refreshed_group_sample_grads.index_select(0, selected_sample_indices_t[:1]).squeeze(0)
            refreshed_grad_old = oracle["batch_pair_data_grads"](x_t, selected_groups_t[:1], selected_sample_indices_t[:1]).squeeze(0)
            refreshed_diff = (refreshed_grad_new - refreshed_grad_old).unsqueeze(0)

            if omega_diffs is None:
                selected_diffs = refreshed_diff
            else:
                selected_diffs = torch.cat([refreshed_diff, omega_diffs], dim=0)

            d_t = selected_diffs.mean(dim=0)

            remaining_mask = np.ones(n_groups, dtype=bool)
            remaining_mask[selected_groups_np] = False
            remaining_groups_np = np.nonzero(remaining_mask)[0].astype(np.int64, copy=False)
            if remaining_groups_np.size > 0:
                remaining_groups_t = torch.as_tensor(remaining_groups_np, device=x_t.device, dtype=torch.int64)
                g_groups[remaining_groups_t] = g_groups[remaining_groups_t] + d_t.unsqueeze(0)

            # When b < n, one extra old sample gradient is needed for the refreshed
            # group in addition to the full group gradient and the Omega^t sample
            # differences.
            epochs_single_iter = (m_per_group + 1.0 + 2.0 * float(batch_size - 1)) / dataset_size
        else:
            # When b = n, [n] \ \tilde{\Omega}^t is empty, so d^t is not formed and
            # the extra old sample for the refreshed group is unnecessary.
            epochs_single_iter = (m_per_group + 2.0 * float(batch_size - 1)) / dataset_size

        # Line 16: average the full local control variates.
        g_next = g_groups.mean(dim=0)

        state["x_prev"] = x_t
        state["x"] = x_next
        state["g"] = g_next
        state["g_groups"] = g_groups

        alg_param["iter"] += 1
        alg_param["training_epochs"] += epochs_single_iter

        # Full-gradient sqnorm is a metric only; it does not count toward epochs.
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
    SILAGE_n_gt_m(sys.argv[1:]).run()
