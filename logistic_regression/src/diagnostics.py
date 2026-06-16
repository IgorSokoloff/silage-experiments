import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

EPS = 1e-12


def _progress(iterable, enabled: bool, **kwargs):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _as_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize(v: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    nrm = torch.linalg.norm(v)
    if nrm <= eps:
        return torch.zeros_like(v)
    return v / (nrm + eps)


def _power_iteration_sym(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    device: torch.device,
    n_iters: int,
    tol: float,
    seed: int,
    v0: Optional[torch.Tensor] = None,
) -> Tuple[float, torch.Tensor]:
    if v0 is None:
        gen = torch.Generator(device=device.type)
        gen.manual_seed(seed)
        v = torch.randn(dim, generator=gen, dtype=torch.float32, device=device)
    else:
        v = v0.to(device=device, dtype=torch.float32)

    v = _normalize(v)
    prev_s = None
    s = 0.0

    for _ in range(n_iters):
        u = matvec(v)
        s = float(torch.linalg.norm(u).item())
        if s <= EPS:
            return 0.0, v
        v = u / (s + EPS)

        if prev_s is not None and tol > 0:
            if abs(s - prev_s) <= tol * max(1.0, s):
                break
        prev_s = s

    return float(s), v


def _regularizer_grad(w: torch.Tensor) -> torch.Tensor:
    # d/dw [w^2 / (1+w^2)] = 2w / (1+w^2)^2
    return 2.0 * w / (1.0 + w * w).pow(2)


def _regularizer_hess_diag(w: torch.Tensor) -> torch.Tensor:
    # d/dw [2w/(1+w^2)^2] = 2(1-3w^2)/(1+w^2)^3
    return 2.0 * (1.0 - 3.0 * w * w) / (1.0 + w * w).pow(3)


def _batched_alpha(
    X: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    n = X.shape[0]
    x_dtype = X.dtype
    alpha = torch.empty(n, dtype=x_dtype, device=X.device)
    y_cast = y.to(dtype=x_dtype)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = X[start:end]
        yb = y_cast[start:end]
        z = -yb * (Xb @ w)
        sig = torch.sigmoid(z)
        alpha[start:end] = sig * (1.0 - sig)

    return alpha


def _objective_grad(
    X: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    lambda_reg: float,
    batch_size: int,
) -> torch.Tensor:
    n = X.shape[0]
    grad = torch.zeros_like(w)
    y_cast = y.to(dtype=X.dtype)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        Xb = X[start:end]
        yb = y_cast[start:end]
        z = -yb * (Xb @ w)
        sig = torch.sigmoid(z)
        coeff = -yb * sig
        grad += Xb.t() @ coeff

    grad /= float(n)
    grad += float(lambda_reg) * _regularizer_grad(w)
    return grad


def _build_probe_set(
    d: int,
    seed: int,
    strategy: str,
    probe_T: int,
    probe_path_steps: int,
    probe_path_lr: float,
    X_flat: torch.Tensor,
    y_flat: torch.Tensor,
    lambda_reg: float,
    batch_size: int,
    device: torch.device,
    show_progress: bool = False,
) -> List[torch.Tensor]:
    w0 = torch.zeros(d, dtype=torch.float32, device=device)
    if strategy == "init_only" or probe_T <= 0:
        return [w0]

    probes = [w0]

    if strategy == "init+random":
        gen = torch.Generator(device=device.type)
        gen.manual_seed(seed + 31)
        for _ in range(probe_T):
            wr = torch.randn(d, generator=gen, dtype=torch.float32, device=device)
            probes.append(wr)
        return probes

    if strategy == "init+path":
        traj = [w0.clone()]
        w = w0.clone()

        path_steps = max(1, int(probe_path_steps))
        path_iter = _progress(range(path_steps), show_progress, desc="Diagnostics: probe path", leave=False)
        for _ in path_iter:
            g = _objective_grad(X_flat, y_flat, w, lambda_reg, batch_size)
            w = w - float(probe_path_lr) * g
            traj.append(w.clone())

        idxs = torch.linspace(1, len(traj) - 1, steps=max(1, probe_T)).round().long().tolist()
        for idx in idxs:
            probes.append(traj[idx])
        return probes

    raise ValueError(f"Unsupported probe strategy: {strategy}")


def _global_logistic_mv(X_flat: torch.Tensor, alpha_flat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    n = X_flat.shape[0]
    return X_flat.t() @ (alpha_flat * (X_flat @ v)) / float(n)


def _group_logistic_mv(X_i: torch.Tensor, alpha_i: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    m = X_i.shape[0]
    return X_i.t() @ (alpha_i * (X_i @ v)) / float(m)


def _delta1_from_alpha(
    X_groups: torch.Tensor,
    X_flat: torch.Tensor,
    alpha_groups: torch.Tensor,
    alpha_flat: torch.Tensor,
    power_iters: int,
    power_tol: float,
    seed: int,
    show_progress: bool = False,
) -> Tuple[float, List[float]]:
    n, _, d = X_groups.shape
    device = X_groups.device

    def global_mv(v: torch.Tensor) -> torch.Tensor:
        return _global_logistic_mv(X_flat, alpha_flat, v)

    group_norms = []
    group_iter = _progress(range(n), show_progress, desc="Diagnostics: delta1 groups", leave=False)
    for i in group_iter:
        X_i = X_groups[i]
        alpha_i = alpha_groups[i]

        def diff_mv(v: torch.Tensor, X_i=X_i, alpha_i=alpha_i) -> torch.Tensor:
            return _group_logistic_mv(X_i, alpha_i, v) - global_mv(v)

        norm_i, _ = _power_iteration_sym(
            diff_mv,
            d,
            device,
            power_iters,
            power_tol,
            seed=seed + 1009 * (i + 1),
        )
        group_norms.append(float(norm_i))

    delta1 = math.sqrt(sum(v * v for v in group_norms) / float(n))
    return float(delta1), group_norms


def _representative_direction(
    X_flat: torch.Tensor,
    alpha_flat: torch.Tensor,
    power_iters: int,
    power_tol: float,
    seed: int,
) -> torch.Tensor:
    d = X_flat.shape[1]
    device = X_flat.device

    def global_mv(v: torch.Tensor) -> torch.Tensor:
        return _global_logistic_mv(X_flat, alpha_flat, v)

    _, v = _power_iteration_sym(
        global_mv,
        d,
        device,
        power_iters,
        power_tol,
        seed=seed,
    )
    return v


def _delta2_proxy_from_alpha(
    X_groups: torch.Tensor,
    alpha_groups: torch.Tensor,
    v_rep: torch.Tensor,
    batch_size: int,
) -> float:
    n, m, d = X_groups.shape
    N = n * m
    device = X_groups.device

    # H_i(w) v for all groups, computed in one tensorized pass.
    dot_nm = torch.einsum("nmd,d->nm", X_groups, v_rep)
    coeff_nm = alpha_groups * dot_nm
    hi_v = torch.einsum("nmd,nm->nd", X_groups, coeff_nm) / float(m)

    X_flat = X_groups.reshape(N, d)
    coeff_flat = coeff_nm.reshape(N)
    group_ids = torch.arange(n, device=device).repeat_interleave(m)

    sq_acc = 0.0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Xb = X_flat[start:end]
        cb = coeff_flat[start:end]
        gb = group_ids[start:end]

        hij_v_b = Xb * cb.unsqueeze(1)
        diff_b = hij_v_b - hi_v[gb]
        sq_acc += float(torch.sum(diff_b * diff_b).item())

    return math.sqrt(sq_acc / float(N))


def _delta2_exact_from_alpha(
    X_groups: torch.Tensor,
    alpha_groups: torch.Tensor,
    power_iters: int,
    power_tol: float,
    seed: int,
    show_progress: bool = False,
) -> float:
    n, m, d = X_groups.shape
    device = X_groups.device

    sq_acc = 0.0
    op_id = 0

    outer_iter = _progress(range(n), show_progress, desc="Diagnostics: delta2 exact groups", leave=False)
    for i in outer_iter:
        X_i = X_groups[i]
        alpha_i = alpha_groups[i]

        def hi_mv(v: torch.Tensor, X_i=X_i, alpha_i=alpha_i) -> torch.Tensor:
            return _group_logistic_mv(X_i, alpha_i, v)

        for j in range(m):
            x_ij = X_i[j]
            alpha_ij = alpha_i[j]

            def diff_mv(v: torch.Tensor, x_ij=x_ij, alpha_ij=alpha_ij) -> torch.Tensor:
                hij_v = alpha_ij * x_ij * torch.dot(x_ij, v)
                return hij_v - hi_mv(v)

            v0 = _normalize(x_ij)
            norm_ij, _ = _power_iteration_sym(
                diff_mv,
                d,
                device,
                power_iters,
                power_tol,
                seed=seed + 7919 * (op_id + 1),
                v0=v0,
            )
            sq_acc += float(norm_ij * norm_ij)
            op_id += 1

    return math.sqrt(sq_acc / float(n * m))


def _delta2_clientwise_from_alpha(
    X_groups: torch.Tensor,
    alpha_groups: torch.Tensor,
    power_iters: int,
    power_tol: float,
    seed: int,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Estimate each clientwise sample-to-client similarity constant.

    This mirrors the current delta2_emp exact_per_component path: it enumerates
    all local components but uses power iteration for each operator norm.
    """
    n, m, d = X_groups.shape
    device = X_groups.device
    values = np.zeros(n, dtype=np.float64)

    outer_iter = _progress(range(n), show_progress, desc="Diagnostics: delta2_i groups", leave=False)
    for i in outer_iter:
        X_i = X_groups[i]
        alpha_i = alpha_groups[i]
        sq_acc = 0.0

        def hi_mv(v: torch.Tensor, X_i=X_i, alpha_i=alpha_i) -> torch.Tensor:
            return _group_logistic_mv(X_i, alpha_i, v)

        for j in range(m):
            x_ij = X_i[j]
            alpha_ij = alpha_i[j]

            def diff_mv(v: torch.Tensor, x_ij=x_ij, alpha_ij=alpha_ij) -> torch.Tensor:
                hij_v = alpha_ij * x_ij * torch.dot(x_ij, v)
                return hij_v - hi_mv(v)

            v0 = _normalize(x_ij)
            norm_ij, _ = _power_iteration_sym(
                diff_mv,
                d,
                device,
                power_iters,
                power_tol,
                seed=seed + 7919 * (i * m + j + 1),
                v0=v0,
            )
            sq_acc += float(norm_ij * norm_ij)

        values[i] = math.sqrt(sq_acc / float(m))

    return values


def _dense_logistic_hessian_from_alpha(
    X: torch.Tensor,
    alpha: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Form the exact dense logistic Hessian in float64.

    Under the active regularized-component convention, the regularizer Hessian
    cancels exactly in the delta1/delta2 difference operators because every
    local/global component carries the same regularizer term. So the exact full
    delta path only needs the logistic Hessian pieces here.
    """
    X64 = X.to(dtype=torch.float64)
    alpha64 = alpha.to(dtype=torch.float64)
    weighted_X = X64 * torch.sqrt(alpha64).unsqueeze(1)
    return (weighted_X.t() @ weighted_X) / float(scale)


def _symmetric_operator_norm_exact(mat: torch.Tensor) -> float:
    eigvals = torch.linalg.eigvalsh(mat.to(dtype=torch.float64))
    return float(torch.max(torch.abs(eigvals)).item())


def _delta1_full_from_alpha_exact(
    X_groups: torch.Tensor,
    X_flat: torch.Tensor,
    alpha_groups: torch.Tensor,
    alpha_flat: torch.Tensor,
    show_progress: bool = False,
) -> Tuple[float, List[float]]:
    """
    Exact full-data delta1 using dense float64 Hessians and eigvalsh.
    """
    n, m, _ = X_groups.shape
    H_global = _dense_logistic_hessian_from_alpha(X_flat, alpha_flat, float(n * m))

    sq_acc = 0.0
    group_norms = []
    group_iter = _progress(range(n), show_progress, desc="Diagnostics: delta1_full groups", leave=False)
    for i in group_iter:
        H_i = _dense_logistic_hessian_from_alpha(X_groups[i], alpha_groups[i], float(m))
        norm_i = _symmetric_operator_norm_exact(H_i - H_global)
        group_norms.append(float(norm_i))
        sq_acc += float(norm_i * norm_i)

    return math.sqrt(sq_acc / float(n)), group_norms


def _delta2_full_from_alpha_exact(
    X_groups: torch.Tensor,
    alpha_groups: torch.Tensor,
    show_progress: bool = False,
) -> float:
    """
    Exact full-data delta2 using dense float64 Hessians and eigvalsh.

    This path is intentionally very expensive: it enumerates every component
    (i, j), forms H_ij and H_i exactly, and computes the operator norm of
    H_ij - H_i exactly in float64. Any batching here is only for memory control,
    not data subsampling.
    """
    n, m, _ = X_groups.shape
    sq_acc = 0.0

    outer_iter = _progress(range(n), show_progress, desc="Diagnostics: delta2_full groups", leave=False)
    for i in outer_iter:
        X_i = X_groups[i].to(dtype=torch.float64)
        alpha_i = alpha_groups[i].to(dtype=torch.float64)
        H_i = _dense_logistic_hessian_from_alpha(X_i, alpha_i, float(m))

        for j in range(m):
            x_ij = X_i[j]
            alpha_ij = alpha_i[j]
            H_ij = alpha_ij * torch.outer(x_ij, x_ij)
            norm_ij = _symmetric_operator_norm_exact(H_ij - H_i)
            sq_acc += float(norm_ij * norm_ij)

    return math.sqrt(sq_acc / float(n * m))


def _delta_flat_full_from_alpha_exact(
    X_flat: torch.Tensor,
    alpha_flat: torch.Tensor,
    show_progress: bool = False,
) -> float:
    """
    Exact full-data flat similarity using dense float64 Hessians.

    This enumerates every component r, forms H_r and H exactly, computes
    ||H_r - H||_op exactly with eigvalsh, and aggregates over all components.
    """
    n_samples, _ = X_flat.shape
    H_global = _dense_logistic_hessian_from_alpha(X_flat, alpha_flat, float(n_samples))
    X64 = X_flat.to(dtype=torch.float64)
    alpha64 = alpha_flat.to(dtype=torch.float64)

    sq_acc = 0.0
    sample_iter = _progress(
        range(n_samples),
        show_progress,
        desc="Diagnostics: delta_flat_full samples",
        leave=False,
    )
    for r in sample_iter:
        x_r = X64[r]
        H_r = alpha64[r] * torch.outer(x_r, x_r)
        norm_r = _symmetric_operator_norm_exact(H_r - H_global)
        sq_acc += float(norm_r * norm_r)

    return math.sqrt(sq_acc / float(n_samples))


def _delta2_clientwise_full_from_alpha_exact(
    X_groups: torch.Tensor,
    alpha_groups: torch.Tensor,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Exact full-data clientwise inner similarities.

    For each client i, this enumerates all local components j, forms H_ij and
    H_i exactly, computes ||H_ij - H_i||_op by eigvalsh, and aggregates over j.
    """
    n, m, _ = X_groups.shape
    values = np.zeros(n, dtype=np.float64)

    outer_iter = _progress(range(n), show_progress, desc="Diagnostics: delta2_i_full groups", leave=False)
    for i in outer_iter:
        X_i = X_groups[i].to(dtype=torch.float64)
        alpha_i = alpha_groups[i].to(dtype=torch.float64)
        H_i = _dense_logistic_hessian_from_alpha(X_i, alpha_i, float(m))

        sq_acc = 0.0
        for j in range(m):
            x_ij = X_i[j]
            H_ij = alpha_i[j] * torch.outer(x_ij, x_ij)
            norm_ij = _symmetric_operator_norm_exact(H_ij - H_i)
            sq_acc += float(norm_ij * norm_ij)

        values[i] = math.sqrt(sq_acc / float(m))

    return values


def _delta_flat_proxy_from_alpha(
    X_flat: torch.Tensor,
    alpha_flat: torch.Tensor,
    v_rep: torch.Tensor,
    batch_size: int,
) -> float:
    """
    Approximate the flat similarity constant along one representative
    direction by comparing each sample Hessian H_r to the global Hessian H.

    Under the active regularized-component convention, the regularizer Hessian
    cancels exactly in H_r - H, so this helper only needs the logistic part.
    """
    n_samples, d = X_flat.shape
    hg_v = _global_logistic_mv(X_flat, alpha_flat, v_rep)

    dot_n = X_flat @ v_rep
    coeff_n = alpha_flat * dot_n

    sq_acc = 0.0
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        Xb = X_flat[start:end]
        cb = coeff_n[start:end]

        hr_v_b = Xb * cb.unsqueeze(1)
        diff_b = hr_v_b - hg_v.unsqueeze(0)
        sq_acc += float(torch.sum(diff_b * diff_b).item())

    return math.sqrt(sq_acc / float(n_samples))


def _L_i_empirical_for_probe(
    X_groups: torch.Tensor,
    alpha_groups: torch.Tensor,
    dvec: torch.Tensor,
    lambda_reg: float,
    power_iters: int,
    power_tol: float,
    seed: int,
    show_progress: bool = False,
) -> Tuple[float, List[float]]:
    n, _, d = X_groups.shape
    device = X_groups.device

    norms = []
    group_iter = _progress(range(n), show_progress, desc="Diagnostics: Li groups", leave=False)
    for i in group_iter:
        X_i = X_groups[i]
        alpha_i = alpha_groups[i]

        def full_mv(v: torch.Tensor, X_i=X_i, alpha_i=alpha_i, dvec=dvec) -> torch.Tensor:
            return _group_logistic_mv(X_i, alpha_i, v) + float(lambda_reg) * (dvec * v)

        norm_i, _ = _power_iteration_sym(
            full_mv,
            d,
            device,
            power_iters,
            power_tol,
            seed=seed + 5003 * (i + 1),
        )
        norms.append(float(norm_i))

    return max(norms), norms


def _L_global_empirical_for_probe(
    X_flat: torch.Tensor,
    alpha_flat: torch.Tensor,
    dvec: torch.Tensor,
    lambda_reg: float,
    power_iters: int,
    power_tol: float,
    seed: int,
) -> float:
    d = X_flat.shape[1]
    device = X_flat.device

    def full_mv(v: torch.Tensor) -> torch.Tensor:
        return _global_logistic_mv(X_flat, alpha_flat, v) + float(lambda_reg) * (dvec * v)

    norm_g, _ = _power_iteration_sym(
        full_mv,
        d,
        device,
        power_iters,
        power_tol,
        seed=seed,
    )
    return float(norm_g)


def _L_i_worstcase(
    X_groups: torch.Tensor,
    lambda_reg: float,
    power_iters: int,
    power_tol: float,
    seed: int,
    show_progress: bool = False,
) -> Tuple[float, List[float]]:
    n, _, d = X_groups.shape
    device = X_groups.device

    vals = []
    group_iter = _progress(range(n), show_progress, desc="Diagnostics: Li_wc groups", leave=False)
    for i in group_iter:
        X_i = X_groups[i]
        m = X_i.shape[0]

        def gram_mv(v: torch.Tensor, X_i=X_i, m=m) -> torch.Tensor:
            return X_i.t() @ (X_i @ v) / float(m)

        gram_norm, _ = _power_iteration_sym(
            gram_mv,
            d,
            device,
            power_iters,
            power_tol,
            seed=seed + 3371 * (i + 1),
        )
        vals.append(float(0.25 * gram_norm + 2.0 * float(lambda_reg)))

    return max(vals), vals


def _L_global_worstcase(
    X_flat: torch.Tensor,
    lambda_reg: float,
    power_iters: int,
    power_tol: float,
    seed: int,
) -> float:
    d = X_flat.shape[1]
    device = X_flat.device
    n = X_flat.shape[0]

    def gram_mv(v: torch.Tensor) -> torch.Tensor:
        return X_flat.t() @ (X_flat @ v) / float(n)

    gram_norm, _ = _power_iteration_sym(
        gram_mv,
        d,
        device,
        power_iters,
        power_tol,
        seed=seed,
    )
    return float(0.25 * gram_norm + 2.0 * float(lambda_reg))


@torch.no_grad()
def estimate_logreg_noncvx_diagnostics(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    group_indices: torch.Tensor,
    lambda_reg: float,
    compute_deltas: bool,
    compute_L_bounds: bool,
    compute_worstcase: bool,
    probe_T: int,
    probe_strategy: str,
    probe_path_steps: int,
    probe_path_lr: float,
    delta_power_iters: int,
    delta_power_tol: float,
    delta2_mode: str,
    delta2_power_iters: int,
    delta_batch_size: int,
    L_power_iters: int,
    L_power_tol: float,
    L_batch_size: int,
    device: Union[str, torch.device],
    seed: int,
    compute_delta1_full: bool = False,
    compute_delta2_full: bool = False,
    probe_path_lr_rule: Optional[str] = None,
    delta_full_method: Optional[str] = None,
    show_progress: bool = False,
) -> Dict[str, object]:
    device = _as_device(device)
    _set_seed(seed)

    X_train = X_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device, dtype=torch.int64)
    group_indices = group_indices.to(device=device, dtype=torch.int64)

    if not torch.all((y_train == -1) | (y_train == 1)):
        raise ValueError("Expected labels in {-1, +1} for diagnostics.")

    y_train_f = y_train.to(torch.float32)

    X_groups = X_train[group_indices]
    y_groups = y_train_f[group_indices]

    n, m, d = X_groups.shape
    X_flat = X_groups.reshape(n * m, d).contiguous()
    y_flat = y_groups.reshape(n * m).contiguous()
    any_full_delta = bool(compute_delta1_full or compute_delta2_full)

    probes = _build_probe_set(
        d=d,
        seed=seed,
        strategy=probe_strategy,
        probe_T=probe_T,
        probe_path_steps=probe_path_steps,
        probe_path_lr=probe_path_lr,
        X_flat=X_flat,
        y_flat=y_flat,
        lambda_reg=lambda_reg,
        batch_size=L_batch_size,
        device=device,
        show_progress=show_progress,
    )

    alpha_cache = []
    alpha_cache_exact = []
    dvec_cache = []
    max_abs_d_cache = []
    if compute_deltas or compute_L_bounds or any_full_delta:
        alpha_bs = max(1, min(delta_batch_size, L_batch_size))
        if any_full_delta:
            X_groups_exact = X_groups.to(dtype=torch.float64)
            X_flat_exact = X_groups_exact.reshape(n * m, d).contiguous()
            y_flat_exact = y_flat.to(dtype=torch.float64)
        probe_cache_iter = _progress(
            probes,
            show_progress,
            total=len(probes),
            desc="Diagnostics: alpha cache",
            leave=False,
        )
        for w in probe_cache_iter:
            if compute_deltas or compute_L_bounds:
                alpha_flat = _batched_alpha(X_flat, y_flat, w, alpha_bs)
                alpha_groups = alpha_flat.view(n, m)
                dvec = _regularizer_hess_diag(w)
                alpha_cache.append((alpha_flat, alpha_groups))
                dvec_cache.append(dvec)
                max_abs_d_cache.append(float(torch.max(torch.abs(dvec)).item()))

            if any_full_delta:
                w_exact = w.to(device=device, dtype=torch.float64)
                alpha_flat_exact = _batched_alpha(X_flat_exact, y_flat_exact, w_exact, alpha_bs)
                alpha_groups_exact = alpha_flat_exact.view(n, m)
                alpha_cache_exact.append((alpha_flat_exact, alpha_groups_exact))

    diagnostics: Dict[str, object] = {
        "delta1_emp": None,
        "delta2_emp": None,
        "delta1_emp_full": None,
        "delta2_emp_full": None,
        "delta1_wc": None,
        "delta2_wc": None,
        "L_ij_max_emp": None,
        "L_i_max_emp": None,
        "L_global_emp": None,
        "L_ij_max_wc": None,
        "L_i_max_wc": None,
        "L_global_wc": None,
        "delta2_mode": delta2_mode,
        "num_probes": len(probes),
        "probe_strategy": probe_strategy,
        "probe_T": int(probe_T),
        "probe_path_steps": int(probe_path_steps),
        "probe_path_lr_used": float(probe_path_lr),
        "probe_path_lr_rule": probe_path_lr_rule,
        "delta_full_method": delta_full_method if any_full_delta else None,
        "settings": {
            "lambda_reg": float(lambda_reg),
            "compute_deltas": bool(compute_deltas),
            "compute_delta1_full": bool(compute_delta1_full),
            "compute_delta2_full": bool(compute_delta2_full),
            "compute_L_bounds": bool(compute_L_bounds),
            "compute_worstcase": bool(compute_worstcase),
            "probe_T": int(probe_T),
            "probe_strategy": probe_strategy,
            "probe_path_steps": int(probe_path_steps),
            "probe_path_lr": float(probe_path_lr),
            "probe_path_lr_used": float(probe_path_lr),
            "probe_path_lr_rule": probe_path_lr_rule,
            "delta_power_iters": int(delta_power_iters),
            "delta_power_tol": float(delta_power_tol),
            "delta2_power_iters": int(delta2_power_iters),
            "delta_batch_size": int(delta_batch_size),
            "delta_full_method": delta_full_method if any_full_delta else None,
            "L_power_iters": int(L_power_iters),
            "L_power_tol": float(L_power_tol),
            "L_batch_size": int(L_batch_size),
            "seed": int(seed),
            "device": str(device),
            "n_groups": int(n),
            "m_per_group": int(m),
            "dim": int(d),
            "num_partition_samples": int(n * m),
        },
    }

    if compute_deltas:
        delta1_per_probe = []
        delta2_per_probe = []

        delta_probe_iter = _progress(
            range(len(alpha_cache)),
            show_progress,
            total=len(alpha_cache),
            desc="Diagnostics: deltas",
            leave=False,
        )
        for probe_id in delta_probe_iter:
            alpha_flat, alpha_groups = alpha_cache[probe_id]
            delta1_w, _ = _delta1_from_alpha(
                X_groups,
                X_flat,
                alpha_groups,
                alpha_flat,
                power_iters=delta_power_iters,
                power_tol=delta_power_tol,
                seed=seed + 100_000 + probe_id,
                show_progress=show_progress and n >= 100,
            )

            v_rep = _representative_direction(
                X_flat,
                alpha_flat,
                power_iters=delta_power_iters,
                power_tol=delta_power_tol,
                seed=seed + 200_000 + probe_id,
            )

            if delta2_mode == "approx_shared_vector":
                delta2_w = _delta2_proxy_from_alpha(
                    X_groups,
                    alpha_groups,
                    v_rep=v_rep,
                    batch_size=delta_batch_size,
                )
            else:
                delta2_w = _delta2_exact_from_alpha(
                    X_groups,
                    alpha_groups,
                    power_iters=delta2_power_iters,
                    power_tol=delta_power_tol,
                    seed=seed + 300_000 + probe_id,
                    show_progress=show_progress and n >= 100,
                )

            delta1_per_probe.append(float(delta1_w))
            delta2_per_probe.append(float(delta2_w))

        diagnostics["delta1_emp"] = float(max(delta1_per_probe)) if delta1_per_probe else None
        diagnostics["delta2_emp"] = float(max(delta2_per_probe)) if delta2_per_probe else None
        diagnostics["delta1_per_probe"] = delta1_per_probe
        diagnostics["delta2_per_probe"] = delta2_per_probe

        if compute_worstcase:
            alpha_wc_groups = torch.full((n, m), 0.25, dtype=torch.float32, device=device)
            alpha_wc_flat = alpha_wc_groups.reshape(n * m)

            delta1_wc, _ = _delta1_from_alpha(
                X_groups,
                X_flat,
                alpha_wc_groups,
                alpha_wc_flat,
                power_iters=delta_power_iters,
                power_tol=delta_power_tol,
                seed=seed + 400_000,
                show_progress=show_progress and n >= 100,
            )

            v_wc = _representative_direction(
                X_flat,
                alpha_wc_flat,
                power_iters=delta_power_iters,
                power_tol=delta_power_tol,
                seed=seed + 500_000,
            )

            if delta2_mode == "approx_shared_vector":
                delta2_wc = _delta2_proxy_from_alpha(
                    X_groups,
                    alpha_wc_groups,
                    v_rep=v_wc,
                    batch_size=delta_batch_size,
                )
            else:
                delta2_wc = _delta2_exact_from_alpha(
                    X_groups,
                    alpha_wc_groups,
                    power_iters=delta2_power_iters,
                    power_tol=delta_power_tol,
                    seed=seed + 600_000,
                    show_progress=show_progress and n >= 100,
                )

            diagnostics["delta1_wc"] = float(delta1_wc)
            diagnostics["delta2_wc"] = float(delta2_wc)

    if any_full_delta:
        delta1_full_per_probe = []
        delta2_full_per_probe = []

        full_probe_iter = _progress(
            range(len(alpha_cache_exact)),
            show_progress,
            total=len(alpha_cache_exact),
            desc="Diagnostics: full exact deltas",
            leave=False,
        )
        for probe_id in full_probe_iter:
            alpha_flat_exact, alpha_groups_exact = alpha_cache_exact[probe_id]

            if compute_delta1_full:
                delta1_full_w, _ = _delta1_full_from_alpha_exact(
                    X_groups_exact,
                    X_flat_exact,
                    alpha_groups_exact,
                    alpha_flat_exact,
                    show_progress=show_progress and n >= 100,
                )
                delta1_full_per_probe.append(float(delta1_full_w))

            if compute_delta2_full:
                delta2_full_w = _delta2_full_from_alpha_exact(
                    X_groups_exact,
                    alpha_groups_exact,
                    show_progress=show_progress and n >= 100,
                )
                delta2_full_per_probe.append(float(delta2_full_w))

        diagnostics["delta1_emp_full"] = (
            float(max(delta1_full_per_probe)) if compute_delta1_full and delta1_full_per_probe else None
        )
        diagnostics["delta2_emp_full"] = (
            float(max(delta2_full_per_probe)) if compute_delta2_full and delta2_full_per_probe else None
        )
        diagnostics["delta1_emp_full_per_probe"] = delta1_full_per_probe
        diagnostics["delta2_emp_full_per_probe"] = delta2_full_per_probe

    if compute_L_bounds:
        a_sq = torch.sum(X_flat * X_flat, dim=1)

        L_ij_emp = -float("inf")
        L_i_emp = -float("inf")
        L_global_emp = -float("inf")

        L_i_per_probe = []
        L_global_per_probe = []

        L_probe_iter = _progress(
            range(len(alpha_cache)),
            show_progress,
            total=len(alpha_cache),
            desc="Diagnostics: L bounds",
            leave=False,
        )
        for probe_id in L_probe_iter:
            (alpha_flat, alpha_groups) = alpha_cache[probe_id]
            dvec = dvec_cache[probe_id]
            dmax = max_abs_d_cache[probe_id]
            cur_L_ij = torch.max(alpha_flat * a_sq + float(lambda_reg) * float(dmax)).item()
            L_ij_emp = max(L_ij_emp, float(cur_L_ij))

            Li_probe, _ = _L_i_empirical_for_probe(
                X_groups,
                alpha_groups,
                dvec=dvec,
                lambda_reg=lambda_reg,
                power_iters=L_power_iters,
                power_tol=L_power_tol,
                seed=seed + 700_000 + probe_id,
                show_progress=show_progress and n >= 100,
            )
            L_i_emp = max(L_i_emp, float(Li_probe))
            L_i_per_probe.append(float(Li_probe))

            Lg_probe = _L_global_empirical_for_probe(
                X_flat,
                alpha_flat,
                dvec=dvec,
                lambda_reg=lambda_reg,
                power_iters=L_power_iters,
                power_tol=L_power_tol,
                seed=seed + 800_000 + probe_id,
            )
            L_global_emp = max(L_global_emp, float(Lg_probe))
            L_global_per_probe.append(float(Lg_probe))

        diagnostics["L_ij_max_emp"] = float(L_ij_emp)
        diagnostics["L_i_max_emp"] = float(L_i_emp)
        diagnostics["L_global_emp"] = float(L_global_emp)
        diagnostics["L_i_per_probe_max"] = L_i_per_probe
        diagnostics["L_global_per_probe"] = L_global_per_probe

        if compute_worstcase:
            diagnostics["L_ij_max_wc"] = float(torch.max(0.25 * a_sq + 2.0 * float(lambda_reg)).item())

            Li_wc, _ = _L_i_worstcase(
                X_groups,
                lambda_reg=lambda_reg,
                power_iters=L_power_iters,
                power_tol=L_power_tol,
                seed=seed + 900_000,
                show_progress=show_progress and n >= 100,
            )
            diagnostics["L_i_max_wc"] = float(Li_wc)

            Lg_wc = _L_global_worstcase(
                X_flat,
                lambda_reg=lambda_reg,
                power_iters=L_power_iters,
                power_tol=L_power_tol,
                seed=seed + 950_000,
            )
            diagnostics["L_global_wc"] = float(Lg_wc)

    return diagnostics


@torch.no_grad()
def estimate_logreg_noncvx_flat_delta(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    lambda_reg: float,
    probe_T: int,
    probe_strategy: str,
    probe_path_steps: int,
    probe_path_lr: float,
    delta_power_iters: int,
    delta_power_tol: float,
    delta_batch_size: int,
    device: Union[str, torch.device],
    seed: int,
    probe_path_lr_rule: Optional[str] = None,
    show_progress: bool = False,
) -> Dict[str, object]:
    """
    Estimate the flat similarity constant delta_flat_emp over the
    flattened synthetic dataset.
    """
    device = _as_device(device)
    _set_seed(seed)

    X_train = X_train.to(device=device, dtype=torch.float32)
    y_train = y_train.to(device=device, dtype=torch.int64).flatten()

    if X_train.ndim != 2:
        raise ValueError(f"Expected X_train to be rank-2, got shape {tuple(X_train.shape)}.")
    if y_train.ndim != 1:
        raise ValueError(f"Expected y_train to be rank-1, got shape {tuple(y_train.shape)}.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Expected X_train and y_train to have matching lengths, got {X_train.shape[0]} and {y_train.shape[0]}."
        )
    if not torch.all((y_train == -1) | (y_train == 1)):
        raise ValueError("Expected labels in {-1, +1} for flat diagnostics.")

    y_flat = y_train.to(torch.float32)
    n_samples, d = X_train.shape

    probes = _build_probe_set(
        d=int(d),
        seed=seed,
        strategy=probe_strategy,
        probe_T=probe_T,
        probe_path_steps=probe_path_steps,
        probe_path_lr=probe_path_lr,
        X_flat=X_train,
        y_flat=y_flat,
        lambda_reg=lambda_reg,
        batch_size=delta_batch_size,
        device=device,
        show_progress=show_progress,
    )

    alpha_cache = []
    probe_cache_iter = _progress(
        probes,
        show_progress,
        total=len(probes),
        desc="Diagnostics: alpha cache",
        leave=False,
    )
    for w in probe_cache_iter:
        alpha_flat = _batched_alpha(X_train, y_flat, w, delta_batch_size)
        alpha_cache.append(alpha_flat)

    delta_flat_per_probe = []
    delta_probe_iter = _progress(
        range(len(alpha_cache)),
        show_progress,
        total=len(alpha_cache),
        desc="Diagnostics: delta_flat",
        leave=False,
    )
    for probe_id in delta_probe_iter:
        alpha_flat = alpha_cache[probe_id]
        v_rep = _representative_direction(
            X_train,
            alpha_flat,
            power_iters=delta_power_iters,
            power_tol=delta_power_tol,
            seed=seed + 1_200_000 + probe_id,
        )
        delta_flat_w = _delta_flat_proxy_from_alpha(
            X_train,
            alpha_flat,
            v_rep=v_rep,
            batch_size=delta_batch_size,
        )
        delta_flat_per_probe.append(float(delta_flat_w))

    return {
        "delta_flat_emp": float(max(delta_flat_per_probe)) if delta_flat_per_probe else None,
        "delta_flat_per_probe": delta_flat_per_probe,
        "num_probes": len(probes),
        "probe_path_lr_used": float(probe_path_lr),
        "probe_path_lr_rule": probe_path_lr_rule,
        "settings": {
            "lambda_reg": float(lambda_reg),
            "probe_T": int(probe_T),
            "probe_strategy": probe_strategy,
            "probe_path_steps": int(probe_path_steps),
            "probe_path_lr": float(probe_path_lr),
            "probe_path_lr_used": float(probe_path_lr),
            "probe_path_lr_rule": probe_path_lr_rule,
            "delta_power_iters": int(delta_power_iters),
            "delta_power_tol": float(delta_power_tol),
            "delta_batch_size": int(delta_batch_size),
            "seed": int(seed),
            "device": str(device),
            "dim": int(d),
            "dataset_size": int(n_samples),
        },
    }


@torch.no_grad()
def estimate_logreg_noncvx_flat_delta_full(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    lambda_reg: float,
    probe_T: int,
    probe_strategy: str,
    probe_path_steps: int,
    probe_path_lr: float,
    delta_batch_size: int,
    device: Union[str, torch.device],
    seed: int,
    probe_path_lr_rule: Optional[str] = None,
    show_progress: bool = False,
) -> Dict[str, object]:
    """
    Compute the flat similarity constant with exact dense float64 eigensolves.

    The probe path is still generated in the same way as the other diagnostics,
    but the Hessian-difference norms are exact for each saved probe.
    """
    device = _as_device(device)
    _set_seed(seed)

    X_probe = X_train.to(device=device, dtype=torch.float32)
    y_probe = y_train.to(device=device, dtype=torch.int64).flatten()

    if X_probe.ndim != 2:
        raise ValueError(f"Expected X_train to be rank-2, got shape {tuple(X_probe.shape)}.")
    if y_probe.ndim != 1:
        raise ValueError(f"Expected y_train to be rank-1, got shape {tuple(y_probe.shape)}.")
    if X_probe.shape[0] != y_probe.shape[0]:
        raise ValueError(
            f"Expected X_train and y_train to have matching lengths, got {X_probe.shape[0]} and {y_probe.shape[0]}."
        )
    if not torch.all((y_probe == -1) | (y_probe == 1)):
        raise ValueError("Expected labels in {-1, +1} for flat diagnostics.")

    n_samples, d = X_probe.shape
    y_probe_f = y_probe.to(torch.float32)

    probes = _build_probe_set(
        d=int(d),
        seed=seed,
        strategy=probe_strategy,
        probe_T=probe_T,
        probe_path_steps=probe_path_steps,
        probe_path_lr=probe_path_lr,
        X_flat=X_probe,
        y_flat=y_probe_f,
        lambda_reg=lambda_reg,
        batch_size=delta_batch_size,
        device=device,
        show_progress=show_progress,
    )

    X_exact = X_probe.to(dtype=torch.float64)
    y_exact = y_probe.to(dtype=torch.float64)

    delta_flat_per_probe = []
    probe_iter = _progress(
        probes,
        show_progress,
        total=len(probes),
        desc="Diagnostics: delta_flat_full",
        leave=False,
    )
    for w in probe_iter:
        w_exact = w.to(device=device, dtype=torch.float64)
        alpha_flat_exact = _batched_alpha(X_exact, y_exact, w_exact, delta_batch_size)
        delta_flat_w = _delta_flat_full_from_alpha_exact(
            X_exact,
            alpha_flat_exact,
            show_progress=show_progress,
        )
        delta_flat_per_probe.append(float(delta_flat_w))

    return {
        "delta_flat_emp_full": float(max(delta_flat_per_probe)) if delta_flat_per_probe else None,
        "delta_flat_emp_full_per_probe": delta_flat_per_probe,
        "num_probes": len(probes),
        "probe_path_lr_used": float(probe_path_lr),
        "probe_path_lr_rule": probe_path_lr_rule,
        "delta_full_method": "exact_eigvalsh_float64",
        "settings": {
            "lambda_reg": float(lambda_reg),
            "probe_T": int(probe_T),
            "probe_strategy": probe_strategy,
            "probe_path_steps": int(probe_path_steps),
            "probe_path_lr": float(probe_path_lr),
            "probe_path_lr_used": float(probe_path_lr),
            "probe_path_lr_rule": probe_path_lr_rule,
            "delta_batch_size": int(delta_batch_size),
            "seed": int(seed),
            "device": str(device),
            "dim": int(d),
            "dataset_size": int(n_samples),
            "method": "exact_eigvalsh_float64",
        },
    }


@torch.no_grad()
def estimate_logreg_noncvx_client_delta2_i(
    X_groups_train: torch.Tensor,
    y_groups_train: torch.Tensor,
    lambda_reg: float,
    probe_T: int,
    probe_strategy: str,
    probe_path_steps: int,
    probe_path_lr: float,
    delta_power_iters: int,
    delta_power_tol: float,
    delta_batch_size: int,
    device: Union[str, torch.device],
    seed: int,
    probe_path_lr_rule: Optional[str] = None,
    show_progress: bool = False,
) -> Dict[str, object]:
    """
    Estimate clientwise inner similarity constants delta2_i_emp.

    This follows the existing delta2_emp exact_per_component style: all local
    components are enumerated, while each operator norm is approximated with
    power iteration. The common regularizer Hessian cancels in H_ij - H_i under
    the active regularized-component convention.
    """
    device = _as_device(device)
    _set_seed(seed)

    X_groups = X_groups_train.to(device=device, dtype=torch.float32)
    y_groups = y_groups_train.to(device=device, dtype=torch.int64)

    if X_groups.ndim != 3:
        raise ValueError(f"Expected X_groups_train to be rank-3, got shape {tuple(X_groups.shape)}.")
    if y_groups.ndim != 2:
        raise ValueError(f"Expected y_groups_train to be rank-2, got shape {tuple(y_groups.shape)}.")
    if X_groups.shape[:2] != y_groups.shape:
        raise ValueError(
            "Expected X_groups_train and y_groups_train to agree on (n, m), "
            f"got {tuple(X_groups.shape[:2])} and {tuple(y_groups.shape)}."
        )
    if not torch.all((y_groups == -1) | (y_groups == 1)):
        raise ValueError("Expected grouped labels in {-1, +1} for clientwise diagnostics.")

    n, m, d = X_groups.shape
    X_flat = X_groups.reshape(n * m, d).contiguous()
    y_flat = y_groups.to(dtype=torch.float32).reshape(n * m).contiguous()

    probes = _build_probe_set(
        d=int(d),
        seed=seed,
        strategy=probe_strategy,
        probe_T=probe_T,
        probe_path_steps=probe_path_steps,
        probe_path_lr=probe_path_lr,
        X_flat=X_flat,
        y_flat=y_flat,
        lambda_reg=lambda_reg,
        batch_size=delta_batch_size,
        device=device,
        show_progress=show_progress,
    )

    delta2_i_per_probe = []
    probe_iter = _progress(
        probes,
        show_progress,
        total=len(probes),
        desc="Diagnostics: delta2_i",
        leave=False,
    )
    for probe_id, w in enumerate(probe_iter):
        alpha_flat = _batched_alpha(X_flat, y_flat, w, delta_batch_size)
        alpha_groups = alpha_flat.view(n, m)
        values_i = _delta2_clientwise_from_alpha(
            X_groups,
            alpha_groups,
            power_iters=delta_power_iters,
            power_tol=delta_power_tol,
            seed=seed + 1_300_000 + probe_id,
            show_progress=show_progress and n >= 100,
        )
        delta2_i_per_probe.append(values_i)

    if not delta2_i_per_probe:
        raise RuntimeError("No probe values were computed for delta2_i_emp.")

    stacked = np.stack(delta2_i_per_probe, axis=0)
    delta2_i_emp = np.max(stacked, axis=0).astype(np.float64, copy=False)
    delta2_bar_sq_emp = float(np.mean(delta2_i_emp * delta2_i_emp))

    return {
        "delta2_i_emp": delta2_i_emp,
        "delta2_i_per_probe": stacked,
        "delta2_bar_sq_emp": delta2_bar_sq_emp,
        "num_probes": len(probes),
        "probe_path_lr_used": float(probe_path_lr),
        "probe_path_lr_rule": probe_path_lr_rule,
        "settings": {
            "lambda_reg": float(lambda_reg),
            "probe_T": int(probe_T),
            "probe_strategy": probe_strategy,
            "probe_path_steps": int(probe_path_steps),
            "probe_path_lr": float(probe_path_lr),
            "probe_path_lr_used": float(probe_path_lr),
            "probe_path_lr_rule": probe_path_lr_rule,
            "delta_power_iters": int(delta_power_iters),
            "delta_power_tol": float(delta_power_tol),
            "delta_batch_size": int(delta_batch_size),
            "seed": int(seed),
            "device": str(device),
            "n_groups": int(n),
            "m_per_group": int(m),
            "dim": int(d),
            "method": "exact_components_power_iteration",
        },
    }


@torch.no_grad()
def estimate_logreg_noncvx_client_delta2_i_full(
    X_groups_train: torch.Tensor,
    y_groups_train: torch.Tensor,
    lambda_reg: float,
    probe_T: int,
    probe_strategy: str,
    probe_path_steps: int,
    probe_path_lr: float,
    delta_batch_size: int,
    device: Union[str, torch.device],
    seed: int,
    probe_path_lr_rule: Optional[str] = None,
    show_progress: bool = False,
) -> Dict[str, object]:
    """
    Compute clientwise inner similarities with exact dense eigensolves.
    """
    device = _as_device(device)
    _set_seed(seed)

    X_groups_probe = X_groups_train.to(device=device, dtype=torch.float32)
    y_groups_probe = y_groups_train.to(device=device, dtype=torch.int64)

    if X_groups_probe.ndim != 3:
        raise ValueError(f"Expected X_groups_train to be rank-3, got shape {tuple(X_groups_probe.shape)}.")
    if y_groups_probe.ndim != 2:
        raise ValueError(f"Expected y_groups_train to be rank-2, got shape {tuple(y_groups_probe.shape)}.")
    if X_groups_probe.shape[:2] != y_groups_probe.shape:
        raise ValueError(
            "Expected X_groups_train and y_groups_train to agree on (n, m), "
            f"got {tuple(X_groups_probe.shape[:2])} and {tuple(y_groups_probe.shape)}."
        )
    if not torch.all((y_groups_probe == -1) | (y_groups_probe == 1)):
        raise ValueError("Expected grouped labels in {-1, +1} for clientwise diagnostics.")

    n, m, d = X_groups_probe.shape
    X_flat_probe = X_groups_probe.reshape(n * m, d).contiguous()
    y_flat_probe = y_groups_probe.to(dtype=torch.float32).reshape(n * m).contiguous()

    probes = _build_probe_set(
        d=int(d),
        seed=seed,
        strategy=probe_strategy,
        probe_T=probe_T,
        probe_path_steps=probe_path_steps,
        probe_path_lr=probe_path_lr,
        X_flat=X_flat_probe,
        y_flat=y_flat_probe,
        lambda_reg=lambda_reg,
        batch_size=delta_batch_size,
        device=device,
        show_progress=show_progress,
    )

    X_groups_exact = X_groups_probe.to(dtype=torch.float64)
    X_flat_exact = X_groups_exact.reshape(n * m, d).contiguous()
    y_flat_exact = y_groups_probe.to(dtype=torch.float64).reshape(n * m).contiguous()

    delta2_i_per_probe = []
    probe_iter = _progress(
        probes,
        show_progress,
        total=len(probes),
        desc="Diagnostics: delta2_i_full",
        leave=False,
    )
    for w in probe_iter:
        w_exact = w.to(device=device, dtype=torch.float64)
        alpha_flat_exact = _batched_alpha(X_flat_exact, y_flat_exact, w_exact, delta_batch_size)
        alpha_groups_exact = alpha_flat_exact.view(n, m)
        values_i = _delta2_clientwise_full_from_alpha_exact(
            X_groups_exact,
            alpha_groups_exact,
            show_progress=show_progress and n >= 100,
        )
        delta2_i_per_probe.append(values_i)

    if not delta2_i_per_probe:
        raise RuntimeError("No probe values were computed for delta2_i_emp_full.")

    stacked = np.stack(delta2_i_per_probe, axis=0)
    delta2_i_emp_full = np.max(stacked, axis=0).astype(np.float64, copy=False)
    delta2_bar_sq_emp_full = float(np.mean(delta2_i_emp_full * delta2_i_emp_full))

    return {
        "delta2_i_emp_full": delta2_i_emp_full,
        "delta2_i_emp_full_per_probe": stacked,
        "delta2_bar_sq_emp_full": delta2_bar_sq_emp_full,
        "num_probes": len(probes),
        "probe_path_lr_used": float(probe_path_lr),
        "probe_path_lr_rule": probe_path_lr_rule,
        "delta_full_method": "exact_eigvalsh_float64",
        "settings": {
            "lambda_reg": float(lambda_reg),
            "probe_T": int(probe_T),
            "probe_strategy": probe_strategy,
            "probe_path_steps": int(probe_path_steps),
            "probe_path_lr": float(probe_path_lr),
            "probe_path_lr_used": float(probe_path_lr),
            "probe_path_lr_rule": probe_path_lr_rule,
            "delta_batch_size": int(delta_batch_size),
            "seed": int(seed),
            "device": str(device),
            "n_groups": int(n),
            "m_per_group": int(m),
            "dim": int(d),
            "method": "exact_eigvalsh_float64",
        },
    }
