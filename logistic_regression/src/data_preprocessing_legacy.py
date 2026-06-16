"""
Legacy preprocessing routines kept outside the main logreg-focused class.
"""

from src.experiment import *


class LegacyDataPreprocessingMixin:
    def l1_norm_asserts(self):
        assert self.arg_values["loss_func"] == "l1_norm"
        assert self.arg_values["dataset"] in L1_NORM_DATASETS
        assert self.arg_values["dim"] >= 0
        assert self.arg_values["num_samples"] > 0
        assert self.arg_values["noise_scale"] >= 0
        assert self.arg_values["batchsize"] > 0

    def linreg_asserts(self):
        assert self.arg_values["loss_func"] == "lin-reg"
        assert self.arg_values["dataset"] in LINREG_DATASETS
        assert self.arg_values["dim"] > 0
        assert self.arg_values["num_samples"] > 0

    def quartic_asserts(self):
        assert self.arg_values["loss_func"] == "quartic"
        assert self.arg_values["dataset"] in QUARTIC_DATASETS
        assert self.arg_values["generate_dataset"] == 1
        assert self.arg_values["dim"] > 0

    def generate_quartic_synthetic(self, seed=42):
        """
        Generate a synthetic SPD matrix X (dim×dim) with spectrum spanning [cond_number, 1],
        compute L0, mu0, save X_0 and y_0, track memory and set x_star, f_star=0.
        """
        self.dataset_path = self.data_path
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        dim = self.arg_values["dim"]
        cond = self.arg_values["cond_number"]
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        eigvals = np.linspace(cond, 1.0, dim, dtype=np.float64) / cond
        D = np.diag(eigvals)

        rs = RandomState(seed)
        M = rs.randn(dim, dim)
        Q, _ = np.linalg.qr(M)

        X0 = Q @ D @ Q.T
        y0 = np.zeros(dim, dtype=np.float64)

        data_to_save = {"X_0": X0, "y_0": y0}
        self.save_dataset(data_to_save)

        self.x_0 = rs.normal(loc=0.0, scale=1.0, size=dim)
        self.save_w_init()

        arrays_memory = 0
        for value in data_to_save.values():
            if isinstance(value, np.ndarray):
                arrays_memory += value.nbytes
        final_memory = process.memory_info().rss
        total_memory = final_memory - initial_memory
        print(f"Memory used by NumPy arrays: {arrays_memory / (1024**3):.2f} GB")
        print(f"Total memory used during generation: {total_memory / (1024**3):.2f} GB")

        self.comp_params_dict["la"] = 0
        self.comp_params_dict["L_0"] = second_matrix_norm(X0)
        self.comp_params_dict["mu_0"] = min_eigval(X0)

        self.comp_params_dict["x_star"] = np.zeros(dim, dtype=np.float64)
        self.comp_params_dict["f_star"] = 0.0

    def generate_linreg_synthetic_dense(self, seed):
        num_samples = self.arg_values["num_samples"]
        num_workers = self.arg_values["num_workers"]
        dim = self.arg_values["dim"]
        assert num_workers == 1
        pretrained_fraction = 0.9
        num_samples_pre = int(num_samples * pretrained_fraction)
        num_samples_ft = num_samples - num_samples_pre
        self.comp_params_dict["num_samples_pre"] = num_samples_pre
        self.comp_params_dict["num_samples_ft"] = num_samples_ft

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        A_pre, b_pre = make_regression(
            n_samples=num_samples_pre,
            n_features=dim,
            n_informative=dim,
            noise=20.0,
            bias=0.0,
            tail_strength=0.8,
            effective_rank=64,
            random_state=42,
        )
        A_ft, b_ft = make_regression(
            n_samples=num_samples_ft,
            n_features=dim,
            n_informative=dim // 2,
            noise=50.0,
            bias=10.0,
            tail_strength=0.9,
            effective_rank=32,
            random_state=84,
        )
        A_pre_scaled, b_pre_scaled = scale_arrays(A_pre, b_pre)
        A_ft_scaled, b_ft_scaled = scale_arrays_custom(A_ft, b_ft, target_mean=1.0, target_std=2.0)

        self.A_pre, self.b_pre = A_pre_scaled.copy(), b_pre_scaled.copy()
        self.X_pre, self.y_pre, self.c_pre = transform_to_quadratic_form(A_pre_scaled, b_pre_scaled)

        self.A_ft, self.b_ft = A_ft_scaled.copy(), b_ft_scaled.copy()
        self.X_ft, self.y_ft, self.c_ft = transform_to_quadratic_form(A_ft_scaled, b_ft_scaled)
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        data_to_save = {
            "X_pre": self.X_pre,
            "y_pre": self.y_pre,
            "c_pre": self.c_pre,
            "X_ft": self.X_ft,
            "y_ft": self.y_ft,
            "c_ft": self.c_ft,
            "A_pre": self.A_pre,
            "b_pre": self.b_pre,
            "A_ft": self.A_ft,
            "b_ft": self.b_ft,
        }
        self.save_dataset(data_to_save)

        if self.arg_values["loss_func"] == "lin-reg" and self.arg_values["regularizer_type"] == "str-cvx":
            raise NotImplementedError("str-cvx regularizer is not yet implemented")
        elif self.arg_values["loss_func"] == "lin-reg" and self.arg_values["regularizer_type"] == "cvx":
            raise NotImplementedError("cvx regularizer option is not yet implemented")
        elif self.arg_values["loss_func"] == "lin-reg" and self.arg_values["regularizer_type"] == "non-cvx":
            L_pre_non_reg = second_matrix_norm(
                self.oracle_dict["hess_bound"](
                    np.zeros(dim),
                    self.X_pre,
                    self.y_pre,
                    {"la": 0, "n_samples": self.comp_params_dict["num_samples_pre"]},
                )
            )
            self.comp_params_dict["la_pre"] = L_pre_non_reg
            self.comp_params_dict["L_0_pre"] = second_matrix_norm(
                self.oracle_dict["hess_bound"](
                    np.zeros(dim),
                    self.X_pre,
                    self.y_pre,
                    {"la": self.comp_params_dict["la_pre"], "n_samples": self.comp_params_dict["num_samples_pre"]},
                )
            )
            L_ft_non_reg = second_matrix_norm(
                self.oracle_dict["hess_bound"](
                    np.zeros(dim),
                    self.X_ft,
                    self.y_ft,
                    {"la": 0, "n_samples": self.comp_params_dict["num_samples_ft"]},
                )
            )
            self.comp_params_dict["la_ft"] = L_ft_non_reg
            self.comp_params_dict["L_0_ft"] = second_matrix_norm(
                self.oracle_dict["hess_bound"](
                    np.zeros(dim),
                    self.X_ft,
                    self.y_ft,
                    {"la": self.comp_params_dict["la_ft"], "n_samples": self.comp_params_dict["num_samples_ft"]},
                )
            )

            x_star, f_star = self.minimize(
                self.X_pre,
                self.y_pre,
                self.c_pre,
                {
                    "la": self.comp_params_dict["la_pre"],
                    "n_samples": self.comp_params_dict["num_samples_pre"],
                    "stepsize": 1 / self.comp_params_dict["L_0_pre"],
                },
                seed,
            )
            self.comp_params_dict["x_star_pre"] = x_star.copy()
            self.comp_params_dict["f_star_pre"] = f_star
        else:
            raise ValueError("regularizer_type is not supported")

        self.x_0 = self.comp_params_dict["x_star_pre"].copy()
        self.save_w_init()

        final_memory = process.memory_info().rss
        total_memory_used = final_memory - initial_memory
        arrays_memory = 0
        for value in data_to_save.values():
            if isinstance(value, np.ndarray):
                arrays_memory += value.nbytes

        print(f"Memory used by NumPy arrays: {arrays_memory / (1024 ** 3):.2f} GB")
        print(f"Total memory used during the generation: {total_memory_used / (1024 ** 3):.2f} GB")

    def gd_minimize(self, f_d, grad_d, x_0, gamma=1e-3, max_iter=10000, tol=1e-24):
        """
        Legacy NumPy gradient-descent helper retained outside the active synthetic
        log-reg minimization path.
        """
        x_t = np.copy(x_0)
        t = 0

        grad_x_t = grad_d(x_t).copy()
        sqn_grad_x_t = np.dot(grad_x_t, grad_x_t)

        with tqdm(total=max_iter) as pbar:
            while t < max_iter and sqn_grad_x_t > tol:
                pbar.set_description(f"Iter {t}, sqnorm(grad_x_t)={sqn_grad_x_t:e}, f(x_t)={f_d(x_t):e}")
                x_t -= gamma * grad_x_t
                grad_x_t = grad_d(x_t).copy()
                sqn_grad_x_t = np.dot(grad_x_t, grad_x_t)
                t += 1
                pbar.update(1)

        x_star = x_t
        f_star = f_d(x_star)

        tqdm.write(f"Stopped at iteration {t}, sqnorm(grad_x_t)={sqn_grad_x_t:e}")
        return x_star, f_star
