"""
SILAGE for the m > n synthetic grouped logistic-regression setting.

The implementation keeps the exact finite-sum structure

    (1/n) sum_i (1/m) sum_j f_{i,j}(x) + lambda * R(x)

by treating each local component f_{i,j} as already regularized. The per-group
control variates therefore store full local estimates, and the global estimator
is just the average of those local estimates.
"""

import sys

from src.algorithm import *
from src.oracle_functions import torch_logreg_grad


class SILAGE_m_gt_n(Algorithm):
    def __init__(self, args=None):
        super().__init__(args)

    def script_directory(self):
        return os.path.dirname(os.path.abspath(__file__))

    def fill_alg_params_dict(self, state, oracle, data, alg_param):
        """Fill SILAGE-specific hyperparameters from loaded tensors and diagnostics."""
        if self.arg_values["synthetic_setting"] != "m_gt_n":
            raise ValueError("SILAGE_m>n is only implemented for synthetic_setting='m_gt_n' in this first pass.")
        if self.arg_values["is_grad_comp_init"] != 1:
            raise ValueError("SILAGE_m>n requires --is_grad_comp_init 1 in the current implementation.")

        X_groups = data["X_groups_train"]
        if not hasattr(X_groups, "shape") or len(X_groups.shape) != 3:
            raise ValueError("Expected X_groups_train to be a rank-3 grouped tensor.")

        n_groups = int(X_groups.shape[0])
        m_per_group = int(X_groups.shape[1])
        if n_groups <= 0 or m_per_group <= 0:
            raise ValueError(f"Invalid grouped shape: n_groups={n_groups}, m_per_group={m_per_group}.")

        p = float(n_groups) / float(m_per_group)
        if not (0.0 < p <= 1.0):
            raise ValueError(f"SILAGE_m>n expects p=n/m in (0,1], got p={p} from n={n_groups}, m={m_per_group}.")

        alg_param["resolved_device"] = resolve_torch_device(alg_param["device"], alg_param["print_status"])
        alg_param["n_groups"] = n_groups
        alg_param["m_per_group"] = m_per_group
        alg_param["dataset_size"] = int(n_groups * m_per_group)
        assert alg_param["dataset_size"] == int(data["X_train"].shape[0]), (
            f"Expected dataset_size={alg_param['dataset_size']} to match flattened X_train rows="
            f"{int(data['X_train'].shape[0])}."
        )
        alg_param["p"] = p
        alg_param["step_size"] = float(
            get_silage_m_gt_n_stepsize(
                alg_param["L_global_emp"],
                alg_param["delta2_emp"],
                n_groups=n_groups,
                p=p,
                factor=alg_param["factor"],
            )
        )
        return alg_param

    def init_oracles_dict(self, state, oracle, data, alg_param):
        """Create Torch helpers specialized to grouped synthetic log-reg data."""
        regularizer_type = self.arg_values["regularizer_type"]
        lambda_reg = float(alg_param["lambda_reg"])

        def group_data_grad(x, group_idx):
            # This is the exact gradient of (1/m) sum_j f_{i,j}(x) for one group i,
            # where each f_{i,j} already includes the regularizer term.
            return torch_logreg_grad(
                x,
                data["X_groups_train"][group_idx],
                data["y_groups_train"][group_idx],
                lambda_reg,
                regularizer_type,
            )

        def sample_data_grad(x, group_idx, sample_idx):
            # This is the exact gradient of the single regularized sample f_{i,j}(x)
            # because the helper averages over the provided mini-batch and here the
            # batch has size 1.
            return torch_logreg_grad(
                x,
                data["X_groups_train"][group_idx, sample_idx:sample_idx + 1],
                data["y_groups_train"][group_idx, sample_idx:sample_idx + 1],
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
            "group_data_grad": group_data_grad,
            "sample_data_grad": sample_data_grad,
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
        """Run one SILAGE iteration with regularized grouped control variates."""
        import torch

        x_t = state["x"]
        g_t = state["g"]
        g_groups = state["g_groups"]

        # Line 3: x^{t+1} = x^t - gamma g^t.
        x_next = x_t - alg_param["step_size"] * g_t

        n_groups = alg_param["n_groups"]
        m_per_group = alg_param["m_per_group"]
        dataset_size = float(alg_param["dataset_size"])
        rs_bernoulli = alg_param["rs_bernoulli"]
        rs_sample = alg_param["rs_sample"]

        # Line 4: sample the Bernoulli coin theta^t with probability p.
        theta_t = int(rs_bernoulli.binomial(1, alg_param["p"]))
        refreshed_group = None
        epochs_single_iter = 0.0

        if theta_t == 1:
            # Lines 5-7: refresh one entire group exactly.
            refreshed_group = int(rs_sample.randint(n_groups))
            g_groups[refreshed_group] = oracle["group_data_grad"](x_next, refreshed_group)
            epochs_single_iter += data["X_groups_train"][refreshed_group].shape[0] / dataset_size

        # Lines 9-12: for every remaining group, sample one within-group point and
        # apply the two-point control-variate correction.
        for group_idx in range(n_groups):
            if refreshed_group is not None and group_idx == refreshed_group:
                continue

            sample_idx = int(rs_sample.randint(m_per_group))
            grad_new = oracle["sample_data_grad"](x_next, group_idx, sample_idx)
            grad_old = oracle["sample_data_grad"](x_t, group_idx, sample_idx)
            g_groups[group_idx] = g_groups[group_idx] + grad_new - grad_old
            epochs_single_iter += 2.0 / dataset_size

        # Line 13: average the full local control variates.
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
    SILAGE_m_gt_n(sys.argv[1:]).run()
