"""
D-ZeroSARAH grouped baseline for the synthetic logistic-regression datasets.

The active implementation follows the repo's regularized-component convention:

    f_{i,j}(x) = log_loss_{i,j}(x) + lambda * R(x),

so every stored local row gradient is already fully regularized.

The runtime state also follows the approved full-gradient-initialized variant:

    * y_{i,j}^{-1} = \nabla f_{i,j}(x^0)
    * y_i^{-1}     = (1/m) sum_j y_{i,j}^{-1}
    * y^{-1}       = (1/n) sum_i y_i^{-1}
    * v^{-1}       = 0
    * nu_0         = 1, then nu_t = s b / (2 n m) for later updates
"""

import sys

from src.algorithm import *
from src.oracle_functions import (
    torch_grouped_logreg_grad_tensor,
    torch_logreg_grad,
    torch_logreg_grad_matrix,
)


class D_ZeroSARAH(Algorithm):
    def __init__(self, args=None):
        super().__init__(args)

    def script_directory(self):
        return os.path.dirname(os.path.abspath(__file__))

    def fill_alg_params_dict(self, state, oracle, data, alg_param):
        """Resolve grouped D-ZeroSARAH hyperparameters from loaded tensors."""
        if self.arg_values["synthetic_setting"] not in SYNTHETIC_SETTING_TO_NM:
            raise ValueError(
                "D-ZeroSARAH is only implemented for the active synthetic settings "
                f"{sorted(SYNTHETIC_SETTING_TO_NM.keys())}."
            )
        if self.arg_values["is_grad_comp_init"] != 1:
            raise ValueError("D-ZeroSARAH requires --is_grad_comp_init 1 in the current implementation.")

        X_groups = data["X_groups_train"]
        if not hasattr(X_groups, "shape") or len(X_groups.shape) != 3:
            raise ValueError("Expected X_groups_train to be a rank-3 grouped tensor.")

        n_groups = int(X_groups.shape[0])
        m_per_group = int(X_groups.shape[1])
        dim = int(X_groups.shape[2])
        if n_groups <= 0 or m_per_group <= 0 or dim <= 0:
            raise ValueError(
                f"Invalid grouped training shape: {(n_groups, m_per_group, dim)}."
            )
        X_train = data["X_train"]
        if not hasattr(X_train, "shape") or len(X_train.shape) != 2:
            raise ValueError("Expected X_train to be a rank-2 flat training tensor.")
        dataset_size = int(X_train.shape[0])
        if dataset_size != int(n_groups * m_per_group):
            raise ValueError(
                "Loaded flat and grouped training tensors disagree on dataset size: "
                f"|X_train|={dataset_size} versus n*m={n_groups * m_per_group}."
            )
        if int(X_train.shape[1]) != dim:
            raise ValueError(
                "Loaded flat and grouped training tensors disagree on feature dimension: "
                f"flat d={int(X_train.shape[1])} versus grouped d={dim}."
            )

        batch_size = int(alg_param["batch_size"])
        client_subset_size = int(alg_param["client_subset_size"])
        if not (1 <= batch_size <= m_per_group):
            raise ValueError(
                f"D-ZeroSARAH expects batch_size in [1, m], got b={batch_size}, m={m_per_group}."
            )
        if not (1 <= client_subset_size <= n_groups):
            raise ValueError(
                "D-ZeroSARAH expects client_subset_size in [1, n], "
                f"got s={client_subset_size}, n={n_groups}."
            )

        alg_param["resolved_device"] = resolve_torch_device(alg_param["device"], alg_param["print_status"])
        alg_param["n_groups"] = n_groups
        alg_param["m_per_group"] = m_per_group
        alg_param["dataset_size"] = dataset_size
        alg_param["batch_size"] = batch_size
        alg_param["client_subset_size"] = client_subset_size
        # The first stochastic step uses nu_0 = 1 so the initialized-memory
        # scheme yields the exact full gradient estimator on the first update.
        alg_param["nu_init"] = 1.0
        # All later steps switch to the theorem-style constant sb / (2nm).
        alg_param["nu_const"] = float(client_subset_size * batch_size) / (2.0 * float(dataset_size))
        alg_param["nu"] = alg_param["nu_const"]
        alg_param["step_size"] = float(
            get_d_zerosarah_stepsize(
                alg_param["L_ij_max_emp"],
                n_groups=n_groups,
                m_per_group=m_per_group,
                client_subset_size=client_subset_size,
                batch_size=batch_size,
                factor=alg_param["factor"],
            )
        )
        return alg_param

    def init_oracles_dict(self, state, oracle, data, alg_param):
        """Create grouped Torch helpers specialized to D-ZeroSARAH updates."""
        regularizer_type = self.arg_values["regularizer_type"]
        lambda_reg = float(alg_param["lambda_reg"])

        def grouped_sample_grads(x, client_indices, local_indices):
            # Gather one arbitrary local minibatch per selected client and
            # evaluate all full regularized component gradients at once.
            return torch_grouped_logreg_grad_tensor(
                x,
                data["X_groups_train"],
                data["y_groups_train"],
                client_indices,
                local_indices,
                lambda_reg,
                regularizer_type,
            )

        def full_grad(x):
            # sqnorm is always based on the exact full objective gradient.
            return torch_logreg_grad(x, data["X_train"], data["y_train"], lambda_reg, regularizer_type)

        return {
            "grouped_sample_grads": grouped_sample_grads,
            "full_grad": full_grad,
        }

    def init_states_dict(self, state, oracle, data, alg_param):
        """Move grouped tensors to device and initialize the full memory table."""
        import torch

        resolved_device = alg_param["resolved_device"]
        data["X_train"] = data["X_train"].to(device=resolved_device, dtype=torch.float64)
        data["y_train"] = data["y_train"].to(device=resolved_device, dtype=torch.int64)
        data["X_groups_train"] = data["X_groups_train"].to(device=resolved_device, dtype=torch.float64)
        data["y_groups_train"] = data["y_groups_train"].to(device=resolved_device, dtype=torch.int64)

        x_0 = torch.as_tensor(state["x"], device=resolved_device, dtype=torch.float64)
        expected_dim = int(data["X_groups_train"].shape[-1])
        if x_0.ndim != 1 or x_0.shape[0] != expected_dim:
            raise ValueError(f"Loaded w_init has shape {tuple(x_0.shape)} but expected ({expected_dim},).")

        n_groups = alg_param["n_groups"]
        m_per_group = alg_param["m_per_group"]
        full_grad_rows = torch_logreg_grad_matrix(
            x_0,
            data["X_train"],
            data["y_train"],
            float(alg_param["lambda_reg"]),
            self.arg_values["regularizer_type"],
        )
        y_memory_groups = full_grad_rows.reshape(n_groups, m_per_group, expected_dim)
        y_groups = y_memory_groups.mean(dim=1)
        y_avg = y_groups.mean(dim=0)

        state["x_prev"] = x_0.detach().clone()
        state["x"] = x_0
        # D-ZeroSARAH follows the pseudocode's v^{-1} = 0 initialization. The
        # exact full gradient still lives in y_avg and is exposed via sqnorm.
        state["v"] = torch.zeros_like(y_avg)
        state["g"] = state["v"]
        state["y_memory_groups"] = y_memory_groups
        state["y_groups"] = y_groups
        state["y_avg"] = y_avg

        alg_param["training_epochs"] = 1.0
        alg_param["latest_sqnorm"] = float(torch.dot(y_avg, y_avg).item())
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
        """Run one grouped D-ZeroSARAH iteration with full regularized memory rows."""
        import torch

        x_t = state["x"]
        x_prev = state["x_prev"]
        v_prev = state["v"]
        y_memory_groups = state["y_memory_groups"]
        y_groups = state["y_groups"]
        y_avg_prev = state["y_avg"]

        n_groups = alg_param["n_groups"]
        m_per_group = alg_param["m_per_group"]
        client_subset_size = alg_param["client_subset_size"]
        batch_size = alg_param["batch_size"]
        dataset_size = float(alg_param["dataset_size"])
        rs_sample = alg_param["rs_sample"]

        selected_clients_np = rs_sample.choice(
            n_groups,
            size=client_subset_size,
            replace=False,
        ).astype(np.int64, copy=False)
        sampled_local_indices_np = np.empty((client_subset_size, batch_size), dtype=np.int64)
        for row in range(client_subset_size):
            sampled_local_indices_np[row] = rs_sample.choice(
                m_per_group,
                size=batch_size,
                replace=False,
            ).astype(np.int64, copy=False)

        selected_clients_t = torch.as_tensor(selected_clients_np, device=x_t.device, dtype=torch.int64)
        sampled_local_indices_t = torch.as_tensor(sampled_local_indices_np, device=x_t.device, dtype=torch.int64)

        grad_curr_rows = oracle["grouped_sample_grads"](x_t, selected_clients_t, sampled_local_indices_t)
        grad_prev_rows = oracle["grouped_sample_grads"](x_prev, selected_clients_t, sampled_local_indices_t)
        old_memory_rows = y_memory_groups[selected_clients_t.unsqueeze(1), sampled_local_indices_t]

        g_curr = grad_curr_rows.mean(dim=1)
        g_prev = grad_prev_rows.mean(dim=1)
        y_prev_sample = old_memory_rows.mean(dim=1)

        # iter == 0 is the first stochastic update, so nu_0 = 1 is applied
        # exactly once before switching to the constant theorem value.
        nu_t = alg_param["nu_init"] if int(alg_param["iter"]) == 0 else alg_param["nu_const"]
        v_t = (
            (g_curr - g_prev).mean(dim=0)
            + (1.0 - nu_t) * v_prev
            + nu_t * ((g_prev - y_prev_sample).mean(dim=0) + y_avg_prev)
        )
        x_next = x_t - alg_param["step_size"] * v_t

        y_memory_groups[selected_clients_t.unsqueeze(1), sampled_local_indices_t] = grad_curr_rows
        client_delta = (grad_curr_rows - old_memory_rows).sum(dim=1) / float(m_per_group)
        y_groups[selected_clients_t] = y_groups[selected_clients_t] + client_delta
        y_avg = y_avg_prev + client_delta.sum(dim=0) / float(n_groups)

        state["x_prev"] = x_t
        state["x"] = x_next
        state["v"] = v_t
        state["g"] = v_t
        state["y_memory_groups"] = y_memory_groups
        state["y_groups"] = y_groups
        state["y_avg"] = y_avg

        epochs_single_iter = (2.0 * float(client_subset_size * batch_size)) / dataset_size
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
    D_ZeroSARAH(sys.argv[1:]).run()
