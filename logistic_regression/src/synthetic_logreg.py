"""Synthetic grouped data generation for binary logistic regression."""

from typing import Dict, Tuple

import torch


# Public dataset name used by the preprocessing CLI.
SYNTHETIC_DIRICHLET_LOGREG_DATASET = "synthetic_dirichlet_logreg"

# Zero-based setting presets requested in the specification.
SYNTHETIC_SETTING_TO_NM = {
    "m_gt_n": (50, 250),
    "n_gt_m": (250, 50),
}

# The four requested regime names are kept as string constants for validation.
SYNTHETIC_REGIMES = {
    "d1_small_d2_small",
    "d1_small_d2_large",
    "d1_large_d2_small",
    "d1_large_d2_large",
}


# The synthetic grouped pipeline now hardcodes regime-specific parameters here so
# the generator is the single source of truth rather than the launcher config.
SYNTHETIC_REGIME_PARAMETERS = {
    "m_gt_n": {
        "d1_small_d2_small": {
            "K": 32,
            "T": 4,
            "alpha_dir": 2000.0,
            "eps_dir": 1e-4,
            "sigma_obs": 0.01,
            "r_inter": 8.0,
            "r_intra": 0.02,
            "lambda_reg": 200.0,
        },
        "d1_small_d2_large": {
            "K": 32,
            "T": 4,
            "alpha_dir": 1000.0,
            "eps_dir": 1e-3,
            "sigma_obs": 0.02,
            "r_inter": 25.0,
            "r_intra": 0.5,
            "lambda_reg": 200.0,
        },
        "d1_large_d2_small": {
            "K": 32,
            "T": 4,
            "alpha_dir": 1000.0,
            "eps_dir": 1e-4,
            "sigma_obs": 0.01,
            "r_inter": 25.0,
            "r_intra": 0.02,
            "lambda_reg": 200.0,
        },
        "d1_large_d2_large": {
            "K": 64,
            "T": 8,
            "alpha_dir": 1000.0,
            "eps_dir": 1e-3,
            "sigma_obs": 0.02,
            "r_inter": 30.0,
            "r_intra": 0.5,
            "lambda_reg": 200.0,
        },
    },
    "n_gt_m": {
        "d1_small_d2_small": {
            "K": 32,
            "T": 4,
            "alpha_dir": 4000.0,
            "eps_dir": 1e-4,
            "sigma_obs": 0.005,
            "r_inter": 8.0,
            "r_intra": 0.01,
            "lambda_reg": 200.0,
        },
        "d1_small_d2_large": {
            "K": 32,
            "T": 4,
            "alpha_dir": 1500.0,
            "eps_dir": 1e-3,
            "sigma_obs": 0.02,
            "r_inter": 30.0,
            "r_intra": 0.5,
            "lambda_reg": 200.0,
        },
        "d1_large_d2_small": {
            "K": 32,
            "T": 4,
            "alpha_dir": 1500.0,
            "eps_dir": 1e-4,
            "sigma_obs": 0.005,
            "r_inter": 30.0,
            "r_intra": 0.01,
            "lambda_reg": 200.0,
        },
        "d1_large_d2_large": {
            "K": 64,
            "T": 8,
            "alpha_dir": 1500.0,
            "eps_dir": 1e-3,
            "sigma_obs": 0.02,
            "r_inter": 30.0,
            "r_intra": 0.5,
            "lambda_reg": 200.0,
        },
    },
}


def synthetic_logreg_setting_to_nm(setting: str) -> Tuple[int, int]:
    """Resolve the named synthetic setting into the requested zero-based (n, m) pair."""
    if setting not in SYNTHETIC_SETTING_TO_NM:
        raise ValueError(f"Unknown synthetic setting: {setting}")
    return SYNTHETIC_SETTING_TO_NM[setting]


def synthetic_logreg_regime_params(setting: str, regime: str) -> Dict[str, float]:
    """Return a copy of the hardcoded synthetic parameters for one (setting, regime) pair."""
    if setting not in SYNTHETIC_REGIME_PARAMETERS:
        raise ValueError(f"Unknown synthetic setting: {setting}")
    if regime not in SYNTHETIC_REGIME_PARAMETERS[setting]:
        raise ValueError(f"Unknown synthetic regime '{regime}' for setting '{setting}'.")
    return dict(SYNTHETIC_REGIME_PARAMETERS[setting][regime])


def _format_suffix_float(value: float) -> str:
    """Format floating-point suffix fields in the same comma-separated style used elsewhere."""
    return repr(round(float(value), 8)).replace(".", ",")


def synthetic_logreg_extension_suffix(
    setting: str,
    regime: str,
    n_groups: int,
    m_per_group: int,
    d: int,
    K: int,
    T: int,
) -> str:
    """Build a synthetic-specific dataset suffix.

    Most regimes are identified only by the stable structural fields. For the
    redesigned d1_large_d2_large regime, we also append a compact hardcoded
    configuration signature so this stronger macro-separated construction does
    not silently collide with older K=64, T=8 artifacts.
    """
    suffix = (
        f"_ss-{setting}"
        f"_sr-{regime}"
        f"_n{n_groups}"
        f"_m{m_per_group}"
        f"_d{d}"
        f"_K{K}"
        f"_T{T}"
    )
    if regime == "d1_large_d2_large":
        regime_params = synthetic_logreg_regime_params(setting, regime)
        suffix += (
            f"_ad{_format_suffix_float(regime_params['alpha_dir'])}"
            f"_ed{_format_suffix_float(regime_params['eps_dir'])}"
            f"_so{_format_suffix_float(regime_params['sigma_obs'])}"
            f"_ri{_format_suffix_float(regime_params['r_inter'])}"
            f"_ra{_format_suffix_float(regime_params['r_intra'])}"
            f"_lr{_format_suffix_float(regime_params['lambda_reg'])}"
        )
    return suffix


def _orthonormal_row_directions(num_rows: int, d: int, dtype: torch.dtype) -> torch.Tensor:
    """Sample `num_rows` orthonormal row directions in R^d using a QR factorization."""
    # We first sample the matrix exactly as described in the math note.
    raw_rows = torch.randn((num_rows, d), dtype=dtype)

    # QR is numerically stable on the transpose because it orthonormalizes columns.
    q_cols, _ = torch.linalg.qr(raw_rows.transpose(0, 1), mode="reduced")

    # Transposing back gives us orthonormal row directions in the original ambient space.
    return q_cols.transpose(0, 1).contiguous()


def _standard_basis(K: int, index: int, dtype: torch.dtype) -> torch.Tensor:
    """Return the zero-based standard basis vector e_index in R^K."""
    basis = torch.zeros(K, dtype=dtype)
    basis[index] = 1.0
    return basis


def _build_beta_groups(
    n_groups: int,
    K: int,
    T: int,
    q: int,
    regime: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
    """Construct the prototype mixture beta_i for every group using zero-based indices."""
    # The first component index in family r is a_r = r * q.
    family_starts = torch.arange(T, dtype=torch.int64) * int(q)

    # Regime 1 uses the same concentrated prototype for every group.
    if regime == "d1_small_d2_small":
        prototype = 0.95 * _standard_basis(K, 0, dtype) + 0.05 * _standard_basis(K, 1, dtype)
        beta_groups = prototype.unsqueeze(0).repeat(n_groups, 1)
        description = "For every group i, beta_i = 0.95 * e_0 + 0.05 * e_1."
        return beta_groups, {"shared": prototype}, description

    # Regime 2 uses a broad shared prototype with one component per macro-family.
    if regime == "d1_small_d2_large":
        prototype = torch.zeros(K, dtype=dtype)
        prototype[family_starts] = 1.0 / float(T)
        beta_groups = prototype.unsqueeze(0).repeat(n_groups, 1)
        description = (
            "For every group i, beta_i = (1/T) * sum_{r=0}^{T-1} e_{r*q}; "
            "for the default T=4, q=8 this is 0.25 * (e_0 + e_8 + e_16 + e_24)."
        )
        return beta_groups, {"shared": prototype}, description

    # Regime 3 varies the active family with i % T while staying concentrated within that family.
    if regime == "d1_large_d2_small":
        if q < 2:
            raise ValueError("Regime d1_large_d2_small requires q >= 2 so that a_r + 1 exists.")
        beta_groups = torch.zeros((n_groups, K), dtype=dtype)
        prototypes = {}
        for r in range(T):
            start = int(family_starts[r].item())
            prototype = 0.95 * _standard_basis(K, start, dtype) + 0.05 * _standard_basis(K, start + 1, dtype)
            prototypes[f"family_{r}"] = prototype
        for i in range(n_groups):
            family_idx = i % T
            beta_groups[i] = prototypes[f"family_{family_idx}"]
        description = (
            "For group i, let r = i % T and a_r = r * q; then beta_i = 0.95 * e_{a_r} + 0.05 * e_{a_r + 1}."
        )
        return beta_groups, prototypes, description

    # Regime 4 now gives each group its own family-anchored identity while still
    # keeping the group's mixture broad. This is the intended compromise:
    # delta_1 should grow because different groups are anchored to different
    # macro-families, while delta_2 should stay large because each group still
    # spreads mass across two macro-families and two coordinates per family.
    if regime == "d1_large_d2_large":
        if T < 8:
            raise ValueError("Regime d1_large_d2_large expects at least 8 macro-families.")
        if q < 5:
            raise ValueError("Regime d1_large_d2_large requires q >= 5 so that a_r + 4 exists.")
        beta_groups = torch.zeros((n_groups, K), dtype=dtype)
        prototypes = {}

        # gamma_r is the within-family broad pair for family r. The first active
        # coordinate is a_r = r * q, and the second is a_r + 4, so gamma_r stays
        # entirely inside family r but is not concentrated on a single basis
        # vector. That internal two-coordinate spread is one of the reasons
        # delta_2 can remain large in this regime.
        gamma_by_family = {}
        for r in range(T):
            start = int(family_starts[r].item())
            gamma_r = 0.5 * _standard_basis(K, start, dtype) + 0.5 * _standard_basis(K, start + 4, dtype)
            gamma_by_family[f"family_{r}"] = gamma_r

        # beta^(r) is anchored to family r, but it also mixes in the neighboring
        # family (r + 1) % T. The 0.7 / 0.3 split makes family r the dominant
        # identity of the group, which is the intended lever for increasing
        # delta_1, while the extra 0.3 mass on the neighbor keeps the group broad
        # across macro-families so delta_2 does not collapse.
        for r in range(T):
            next_family = (r + 1) % T
            prototype = 0.7 * gamma_by_family[f"family_{r}"] + 0.3 * gamma_by_family[f"family_{next_family}"]
            prototypes[f"family_anchor_{r}"] = prototype

        # Group i receives beta^(i % T). Using i % T gives T distinct zero-based
        # group types, one for each macro-family anchor, instead of only a couple
        # of symmetric templates. That broader collection of group identities is
        # exactly what should make between-group heterogeneity larger.
        for i in range(n_groups):
            family_idx = i % T
            beta_groups[i] = prototypes[f"family_anchor_{family_idx}"]
        description = (
            "Let a_r = r*q with q=K/T=8 and gamma_r = 0.5*e_{a_r} + 0.5*e_{a_r+4}. "
            "For each r in {0,...,T-1}, define beta^(r) = 0.7*gamma_r + 0.3*gamma_{(r+1) mod T}. "
            "For group i, beta_i = beta^(i mod T)."
        )
        return beta_groups, prototypes, description

    raise ValueError(f"Unknown synthetic regime: {regime}")


def _sample_balanced_labels(
    X_flat: torch.Tensor,
    low: float = 0.3,
    high: float = 0.7,
    max_attempts: int = 32,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sample labels from a normalized teacher and optionally resample until the class balance is acceptable."""
    d = X_flat.shape[1]
    chosen_payload = None

    # We keep resampling the teacher direction, not the features, exactly as requested.
    for attempt in range(max_attempts):
        # Draw the raw teacher g ~ N(0, I_d).
        g = torch.randn(d, dtype=X_flat.dtype)

        # Normalize g to obtain the unit direction w_raw.
        w_raw = g / (torch.linalg.norm(g) + 1e-12)

        # Compute the median absolute raw logit used for adaptive scaling.
        raw_logits = X_flat @ w_raw
        median_abs_logit = torch.median(torch.abs(raw_logits))

        # Scale the teacher so the logits live in a numerically useful range.
        tau = 1.0 / (median_abs_logit + 1e-12)
        w_star = tau * w_raw

        # Convert logits to Bernoulli probabilities for binary labels.
        probs = torch.sigmoid(X_flat @ w_star)

        # Sample zero-based Bernoulli outcomes and then map them to {-1, +1}.
        y01 = torch.bernoulli(probs).to(torch.int64)
        positive_fraction = float(y01.to(torch.float32).mean().item())
        y_signed = 2 * y01 - 1

        chosen_payload = {
            "y_flat": y_signed.to(torch.int64),
            "positive_fraction": positive_fraction,
            "tau": float(tau.item()),
            "median_abs_raw_logit": float(median_abs_logit.item()),
            "teacher_norm": float(torch.linalg.norm(w_star).item()),
            "attempts_used": attempt + 1,
            "accepted": low <= positive_fraction <= high,
        }

        # Stop early as soon as the empirical class balance falls in the target interval.
        if low <= positive_fraction <= high:
            break

    return chosen_payload["y_flat"], {
        "positive_fraction": chosen_payload["positive_fraction"],
        "tau": chosen_payload["tau"],
        "median_abs_raw_logit": chosen_payload["median_abs_raw_logit"],
        "teacher_norm": chosen_payload["teacher_norm"],
        "label_resample_attempts": chosen_payload["attempts_used"],
        "label_balance_accepted": chosen_payload["accepted"],
        "positive_fraction_interval": [float(low), float(high)],
    }


def generate_synthetic_dirichlet_logreg(
    synthetic_setting: str,
    synthetic_regime: str,
    d: int,
    seed: int,
) -> Dict[str, object]:
    """Generate grouped synthetic data in the exact tensor layout expected by the diagnostics pipeline."""
    # All synthetic tensors are generated on CPU so saving and diagnostics handoff stay simple and deterministic.
    dtype = torch.float32
    torch.manual_seed(int(seed))

    # The regime table is the source of truth for synthetic generation parameters.
    regime_params = synthetic_logreg_regime_params(synthetic_setting, synthetic_regime)
    K = int(regime_params["K"])
    T = int(regime_params["T"])
    alpha_dir = float(regime_params["alpha_dir"])
    eps_dir = float(regime_params["eps_dir"])
    sigma_obs = float(regime_params["sigma_obs"])
    r_inter = float(regime_params["r_inter"])
    r_intra = float(regime_params["r_intra"])

    # Resolve the requested grouped setting into the exact (n, m) pair.
    n_groups, m_per_group = synthetic_logreg_setting_to_nm(synthetic_setting)

    # The model requires K to split evenly into T macro-families.
    if K % T != 0:
        raise ValueError(f"Expected K % T == 0, got K={K}, T={T}.")
    q = K // T

    # The QR-based construction needs enough ambient dimensions for T + K orthonormal rows.
    if d < T + K:
        raise ValueError(f"Expected d >= T + K, got d={d}, T + K={T + K}.")

    # Draw T macro-family directions and K local component directions in a stable way.
    row_dirs = _orthonormal_row_directions(T + K, d, dtype=dtype)
    macro_dirs = row_dirs[:T]
    local_dirs = row_dirs[T:T + K]

    # Map each zero-based component index k to its macro-family t(k) = k // q.
    family_of_component = torch.arange(K, dtype=torch.int64) // int(q)

    # Build the separated macro-centers c_r = r_inter * a_r.
    macro_centers = float(r_inter) * macro_dirs

    # Build the nearby component means mu_k = c_{t(k)} + r_intra * u_k.
    component_means = macro_centers[family_of_component] + float(r_intra) * local_dirs

    # Construct the regime-specific prototype mixtures beta_i.
    beta_groups, beta_prototypes, beta_rule_description = _build_beta_groups(
        n_groups=n_groups,
        K=K,
        T=T,
        q=q,
        regime=synthetic_regime,
        dtype=dtype,
    )

    # Apply the requested Dirichlet smoothing to avoid exact zero concentrations.
    uniform_beta = torch.full((1, K), 1.0 / float(K), dtype=dtype)
    beta_tilde_groups = (1.0 - float(eps_dir)) * beta_groups + float(eps_dir) * uniform_beta

    # Sample the actual group mixtures pi_i ~ Dirichlet(alpha_dir * beta_i_tilde).
    dirichlet = torch.distributions.Dirichlet(float(alpha_dir) * beta_tilde_groups)
    pi_groups = dirichlet.sample()

    # Sample latent assignments Z_{ij} from the zero-based categorical distributions pi_i.
    categorical = torch.distributions.Categorical(probs=pi_groups)
    latent_assignments = categorical.sample((m_per_group,)).transpose(0, 1).contiguous()

    # Gather the component means mu_{Z_{ij}} for every zero-based pair (i, j).
    mean_groups = component_means[latent_assignments]

    # Add isotropic Gaussian observation noise to obtain X_{ij}.
    noise = float(sigma_obs) * torch.randn((n_groups, m_per_group, d), dtype=dtype)
    X_groups_train = mean_groups + noise

    # Flatten grouped features into the shape expected by the diagnostics pipeline.
    X_train = X_groups_train.reshape(n_groups * m_per_group, d).contiguous()

    # Sample labels using the rescaled teacher model and map them to {-1, +1}.
    y_train, label_stats = _sample_balanced_labels(X_train)

    # Restore the grouped label tensor so every sample still has coordinates (i, j).
    y_groups_train = y_train.reshape(n_groups, m_per_group).contiguous()

    # Define the canonical zero-based flat-to-group index map requested in the spec.
    group_indices_train = torch.arange(n_groups * m_per_group, dtype=torch.int64).reshape(n_groups, m_per_group)

    # Build the metadata payload used for auditability and downstream inspection.
    metadata = {
        "dataset": SYNTHETIC_DIRICHLET_LOGREG_DATASET,
        "synthetic_setting": synthetic_setting,
        "synthetic_regime": synthetic_regime,
        "n_groups": int(n_groups),
        "m_per_group": int(m_per_group),
        "d": int(d),
        "K": int(K),
        "T": int(T),
        "q": int(q),
        "hardcoded_regime_parameters": {
            **regime_params,
            "K": int(regime_params["K"]),
            "T": int(regime_params["T"]),
        },
        "alpha_dir": float(alpha_dir),
        "eps_dir": float(eps_dir),
        "sigma_obs": float(sigma_obs),
        "r_inter": float(r_inter),
        "r_intra": float(r_intra),
        "seed": int(seed),
        "n_train": int(n_groups * m_per_group),
        "feature_dim": int(d),
        "group_indices_rule": "group_indices_train = torch.arange(n*m).reshape(n, m) with zero-based indexing.",
        "beta_rule_description": beta_rule_description,
        "beta_groups": beta_groups.tolist(),
        "beta_tilde_groups": beta_tilde_groups.tolist(),
        "beta_prototypes": {key: value.tolist() for key, value in beta_prototypes.items()},
        "family_of_component": family_of_component.tolist(),
        "macro_center_norms": torch.linalg.norm(macro_centers, dim=1).tolist(),
        "local_direction_norms": torch.linalg.norm(local_dirs, dim=1).tolist(),
        "component_mean_norms": torch.linalg.norm(component_means, dim=1).tolist(),
        "latent_component_counts": torch.bincount(latent_assignments.reshape(-1), minlength=K).tolist(),
        "pi_group_argmax": torch.argmax(pi_groups, dim=1).tolist(),
        "pi_group_entropy": (-pi_groups * torch.log(pi_groups + 1e-12)).sum(dim=1).tolist(),
        **label_stats,
    }

    return {
        "X_train": X_train.to(torch.float32),
        "y_train": y_train.to(torch.int64),
        "X_groups_train": X_groups_train.to(torch.float32),
        "y_groups_train": y_groups_train.to(torch.int64),
        "group_indices_train": group_indices_train.to(torch.int64),
        "metadata": metadata,
    }
