from src.utils import *

###############################
## Torch Oracle Helper Stack ##
###############################


def torch_regularizer_value(w, regularizer_type: str):
    """
    Evaluate the configured regularizer in Torch.

    This helper backs the preprocessing-side AdaN+ minimizer.  It lives in the
    active oracle module so the full-batch Torch objective is defined in one
    shared place, while the older NumPy oracle stack stays archived in
    `src/oracle_functions_legacy.py`.
    """
    import torch

    if regularizer_type == "non-cvx":
        return torch.sum((w * w) / (1.0 + w * w))
    if regularizer_type == "cvx":
        return torch.zeros((), dtype=w.dtype, device=w.device)
    if regularizer_type == "str-cvx":
        return torch.sum(w * w)
    raise ValueError(f"Unsupported regularizer_type={regularizer_type}")


def torch_regularizer_grad(w, regularizer_type: str):
    """Return the analytic Torch gradient of the configured regularizer."""
    import torch

    if regularizer_type == "non-cvx":
        return 2.0 * w / (1.0 + w * w) ** 2
    if regularizer_type == "cvx":
        return torch.zeros_like(w)
    if regularizer_type == "str-cvx":
        return 2.0 * w
    raise ValueError(f"Unsupported regularizer_type={regularizer_type}")


def torch_regularizer_hess_diag(w, regularizer_type: str):
    """Return the analytic Torch Hessian diagonal of the configured regularizer."""
    import torch

    if regularizer_type == "non-cvx":
        return 2.0 * (1.0 - 3.0 * w * w) / (1.0 + w * w) ** 3
    if regularizer_type == "cvx":
        return torch.zeros_like(w)
    if regularizer_type == "str-cvx":
        return 2.0 * torch.ones_like(w)
    raise ValueError(f"Unsupported regularizer_type={regularizer_type}")


def torch_logreg_objective(w, X_train_t, y_train_t, lambda_reg: float, regularizer_type: str):
    """
    Evaluate the full regularized logistic objective used by the preprocessing
    AdaN+ minimizer.
    """
    import torch.nn.functional as F

    y_float = y_train_t.to(dtype=w.dtype)
    logits = X_train_t @ w
    data_loss = F.softplus(-y_float * logits).mean()
    return data_loss + lambda_reg * torch_regularizer_value(w, regularizer_type)


def torch_logreg_grad(w, X_train_t, y_train_t, lambda_reg: float, regularizer_type: str):
    """Compute the analytic gradient of the full regularized logistic objective."""
    import torch

    y_float = y_train_t.to(dtype=w.dtype)
    yz = y_float * (X_train_t @ w)
    coeff = y_float * torch.sigmoid(-yz)
    data_grad = -(X_train_t.transpose(0, 1) @ coeff) / X_train_t.shape[0]
    return data_grad + lambda_reg * torch_regularizer_grad(w, regularizer_type)


def torch_logreg_data_grad_matrix(w, X_batch_t, y_batch_t):
    """
    Compute one data-only logistic gradient row per sample in the provided batch.

    The helper is retained for cases where the data term must be separated from
    the regularizer explicitly. Active SILAGE paths can build on top of it when
    they need one full local gradient row per sampled (i, j) pair.
    """
    import torch

    if X_batch_t.ndim != 2:
        raise ValueError(f"Expected X_batch_t to be rank-2, got shape {tuple(X_batch_t.shape)}.")
    y_float = y_batch_t.reshape(-1).to(dtype=w.dtype)
    if X_batch_t.shape[0] != y_float.shape[0]:
        raise ValueError(
            f"Expected matching batch sizes for X_batch_t and y_batch_t, got {X_batch_t.shape[0]} and {y_float.shape[0]}."
        )

    yz = y_float * (X_batch_t @ w)
    coeff = y_float * torch.sigmoid(-yz)
    return -(coeff.unsqueeze(1) * X_batch_t)


def torch_logreg_grad_matrix(w, X_batch_t, y_batch_t, lambda_reg: float, regularizer_type: str):
    """
    Compute one full regularized logistic-gradient row per sample in the batch.

    Each returned row corresponds to the gradient of

        f_{i,j}(w) = log_loss_{i,j}(w) + lambda_reg * R(w),

    so the regularizer gradient is broadcast across all sampled rows.
    """
    import torch

    data_grad_rows = torch_logreg_data_grad_matrix(w, X_batch_t, y_batch_t)
    reg_grad = lambda_reg * torch_regularizer_grad(w, regularizer_type)
    return data_grad_rows + reg_grad.unsqueeze(0)


def torch_grouped_logreg_grad_tensor(
    w,
    X_groups_t,
    y_groups_t,
    group_indices,
    sample_indices,
    lambda_reg: float,
    regularizer_type: str,
):
    """
    Gather arbitrary grouped (client, sample) pairs and return their full
    regularized component gradients with the same broadcasted sampling shape.

    Examples:
        * group_indices.shape == (s,), sample_indices.shape == (s, b)
          -> output.shape == (s, b, d)
        * group_indices.shape == (k,), sample_indices.shape == (k,)
          -> output.shape == (k, d)
    """
    import torch

    group_indices_t = torch.as_tensor(group_indices, device=X_groups_t.device, dtype=torch.int64)
    sample_indices_t = torch.as_tensor(sample_indices, device=X_groups_t.device, dtype=torch.int64)

    while group_indices_t.ndim < sample_indices_t.ndim:
        group_indices_t = group_indices_t.unsqueeze(-1)
    while sample_indices_t.ndim < group_indices_t.ndim:
        sample_indices_t = sample_indices_t.unsqueeze(-1)

    group_indices_t, sample_indices_t = torch.broadcast_tensors(group_indices_t, sample_indices_t)
    gathered_X = X_groups_t[group_indices_t, sample_indices_t]
    gathered_y = y_groups_t[group_indices_t, sample_indices_t]

    leading_shape = tuple(gathered_X.shape[:-1])
    flat_X = gathered_X.reshape(-1, gathered_X.shape[-1])
    flat_y = gathered_y.reshape(-1)
    flat_grads = torch_logreg_grad_matrix(w, flat_X, flat_y, lambda_reg, regularizer_type)
    return flat_grads.reshape(*leading_shape, gathered_X.shape[-1])


def torch_logreg_hess(w, X_train_t, y_train_t, lambda_reg: float, regularizer_type: str):
    """
    Compute the dense analytic Hessian of the full regularized logistic
    objective used by the first Torch AdaN+ implementation.
    """
    import torch

    y_float = y_train_t.to(dtype=w.dtype)
    yz = y_float * (X_train_t @ w)
    coeff = torch.sigmoid(yz) * torch.sigmoid(-yz)
    data_hess = X_train_t.transpose(0, 1) @ (coeff.unsqueeze(1) * X_train_t)
    data_hess = data_hess / X_train_t.shape[0]
    reg_diag = lambda_reg * torch_regularizer_hess_diag(w, regularizer_type)
    data_hess.diagonal().add_(reg_diag)
    return data_hess


# Re-export the archived NumPy oracle stack for the experiment / diagnostics
# code paths, which still import from `src.oracle_functions`.
from src.oracle_functions_legacy import *
