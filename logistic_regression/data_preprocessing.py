"""
Adam-norm project
"""

"""
A script for the data preprocessing.
It takes the given dataset and outomes the partition.

Adaptation for project hfh.
Note for me: If I need a removed commmented code, I can find in similar script for the MSPPM projects 
"""

from src.experiment import *
from src.experiment_legacy import LegacyExperimentOracleMixin
from src.data_preprocessing_legacy import LegacyDataPreprocessingMixin
from src.synthetic_logreg import (
    SYNTHETIC_DIRICHLET_LOGREG_DATASET,
    SYNTHETIC_REGIMES,
    SYNTHETIC_SETTING_TO_NM,
    generate_synthetic_dirichlet_logreg,
    synthetic_logreg_regime_params,
    synthetic_logreg_setting_to_nm,
)
from src.oracle_functions import (
    torch_logreg_grad,
    torch_logreg_hess,
    torch_logreg_objective,
)


class DataPreprocessing(LegacyDataPreprocessingMixin, LegacyExperimentOracleMixin, Experiment):
    def __init__(self, args):
        # Create a dictionary of arguments and their values
        self.arg_values = vars(self.argument_parser())
        # Resolve dataset-specific defaults before we print or build any path suffixes.
        self._apply_dataset_specific_defaults()
        # Print the input arguments
        print("-"*80)
        my_print("Input arguments:", self.arg_values["print_args"])
        for key, value in self.arg_values.items():
            my_print(f"{key}: {value}", self.arg_values["print_args"])

        self.general_asserts()
        # Canonicalize cluster representation once (supports deprecated --repr fallback).
        self.arg_values["cluster_repr"] = self._cluster_repr_mode()
        if self.arg_values["loss_func"]=="l1_norm":
            self.l1_norm_asserts()
        elif self.arg_values["loss_func"]=="lin-reg":
            self.linreg_asserts()
        elif self.arg_values["loss_func"]=="log-reg":
            self.logreg_asserts()
        elif self.arg_values["loss_func"]=="quartic":
            self.quartic_asserts()
        else:
            raise ValueError("loss_func is not supported")
        
        print_time(self.arg_values["print_status"])

        self.path_initialisation()
        self.init_comp_params_dict()
        self.requested_comp_params_set = set(self.comp_params_dict.keys())
        self.init_load_params_dict()
        self.init_data_dict()
        self.init_alg_params_dict()
        #self.load_params_dict = parse_params_to_dict(self.arg_values['loadable_params'], ALLOWABLE_PARAMS)

        self.init_exp_param_extension()
        self.init_exp_data_extension()
        self.init_dataset_path()
        self.init_w_init_extension()
        # Keep one legacy bootstrap call for the older preprocessing code paths
        # that still use self.oracle_dict["hess_bound"]. The active SILAGE
        # algorithm path no longer depends on this setup.
        self.init_oracles()
        
        # a flag indicating that dataset is loaded
        # self.is_dataset_loaded = 0

    def init_data_dict(self):
        """Mirror the algorithm-side prepared-data contract for preprocessing."""
        self.data_dict = parse_params_to_dict(self.arg_values['loadable_datasets'], ALLOWABLE_DATASETS)
        assert set(self.data_dict.keys()).issubset(ALLOWABLE_DATASETS)

    def _copy_comp_param_value(self, value):
        return value.copy() if isinstance(value, np.ndarray) else value

    def _build_requested_comp_params_payload(self):
        """Build the save payload from the existing bundle plus requested outputs."""
        self.comp_params_path = self.data_path + 'comp_params' + self.exp_params_extension + "/"
        existing_bundle = load_comp_params_bundle(self.comp_params_path, None, self.arg_values["print_status"])

        if not self.requested_comp_params_set:
            return None

        unresolved_requested = [
            key for key in sorted(self.requested_comp_params_set)
            if self.comp_params_dict.get(key) is None
        ]
        if unresolved_requested:
            raise ValueError(
                "Cannot save requested comp params because the following requested keys are still unset: "
                f"{unresolved_requested}"
            )

        payload = {
            key: self._copy_comp_param_value(value)
            for key, value in existing_bundle.items()
        }
        for key in sorted(self.requested_comp_params_set):
            payload[key] = self._copy_comp_param_value(self.comp_params_dict[key])
        return payload

    def save_comp_params(self):
        self.comp_params_path = self.data_path + 'comp_params' + self.exp_params_extension + "/"
        if not os.path.exists(self.comp_params_path):
            os.makedirs(self.comp_params_path)

        payload = self._build_requested_comp_params_payload()
        if payload is None:
            my_print(
                "No computable_params were requested; skipped comp-params bundle save.",
                self.arg_values["print_status"],
            )
            return

        save_comp_params_bundle(self.comp_params_path, payload, self.arg_values["print_status"])
        
    @staticmethod
    def argument_parser() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')
        #at this point we assume that the dataset is already downloaded and is in the data folder
        parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mnist01', help='The name of the dataset')
        parser.add_argument('--loss_func', action='store', dest='loss_func', type=str, default="log-reg", help='log-reg or l1_norm')
        parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')
        parser.add_argument('--hetero', action='store', dest='hetero', type=int, default=0, help='hetero setting')
        parser.add_argument('--is_minimize', action='store', dest='is_minimize', type=int, default=0, help='minimize or not')
        parser.add_argument('--regularizer_type', action='store', dest='regularizer_type', type=str, default='non-cvx', help='str-cvx or non-cvx')
        parser.add_argument('--lambda_reg', action='store', dest='lambda_reg', type=float, default=1e-3, help='Regularization coefficient lambda')
        parser.add_argument('--generate_dataset', action='store', dest='generate_dataset', type=int, default=0, help='generate_dataset')
        parser.add_argument('--load_raw_dataset', action='store', dest='load_raw_dataset', type=int, default=0, help='load_raw_dataset')
        parser.add_argument('--load_prepared_dataset', action='store', dest='load_prepared_dataset', type=int, default=0, help='load_prepared_dataset')
        parser.add_argument('--is_sparse_dataset', action='store', dest='is_sparse_dataset', type=int, default=0, help='sparse dataset or not')
        parser.add_argument('--print_args', action='store', dest='print_args', type=int, default=1, help='print_args')
        parser.add_argument('--print_status', action='store', dest='print_status', type=int, default=0, help='print_status')
        parser.add_argument('--computable_params', action='store', dest='comp_params_str', type=str, default="['lambda_reg']", help='list of computable params')
        parser.add_argument('--loadable_params', action='store', dest='loadable_params', type=str, default="[]", help='list of loadable params')
        parser.add_argument('--loadable_datasets', action='store', dest='loadable_datasets', type=str, default="[]", help='list of loadable data arrays')
        parser.add_argument('--batchsize', action='store', dest='batchsize', type=int, default=1, help='batchsize')
        parser.add_argument('--use_ray', action='store', dest='use_ray', type=int, default=1, help='use ray or not')
        #parser.add_argument('--cond_number', action='store', dest='cond_number', type=int, default=1000, help='condition number')

        # SILAGE-style finite-sum partitioning arguments
        parser.add_argument('--n_groups', action='store', dest='n_groups', type=int, default=128, help='Number of groups n')
        parser.add_argument('--m_per_group', action='store', dest='m_per_group', type=int, default=32, help='Points per group m')
        parser.add_argument('--partition_regime', action='store', dest='partition_regime', type=str, default='random_d1s_d2l', help='random_d1s_d2l|clustered_d1l_d2s|intermediate_d1m_d2m|stratified_d1s_d2s|score_mix_d1l_d2l')
        parser.add_argument('--cluster_repr', action='store', dest='cluster_repr', type=str, default=None, help='raw|pca representation used only for partitioning')
        # Deprecated alias retained for backward compatibility.
        parser.add_argument('--repr', action='store', dest='repr', type=str, default='raw', help=argparse.SUPPRESS)
        parser.add_argument('--pca_dim', action='store', dest='pca_dim', type=int, default=64, help='PCA target dimension')
        parser.add_argument('--kmeans_K', action='store', dest='kmeans_K', type=int, default=16, help='Number of clusters for intermediate_d1m_d2m/stratified_d1s_d2s regimes')
        parser.add_argument('--score_type', action='store', dest='score_type', type=str, default='mean_intensity', help='mean_intensity|l1')
        parser.add_argument('--device', action='store', dest='device', type=str, default='cpu', help='cpu|cuda|cuda:N')
        parser.add_argument('--seed', action='store', dest='seed', type=int, default=42, help='Global seed')

        # Controlled synthetic grouped-data arguments for binary logistic regression.
        # For synthetic_dirichlet_logreg, the regime table overrides K/T and the
        # generator hyperparameters; these flags remain for compatibility only.
        parser.add_argument('--synthetic_setting', action='store', dest='synthetic_setting', type=str, default='m_gt_n', choices=sorted(SYNTHETIC_SETTING_TO_NM.keys()), help='Synthetic grouped setting')
        parser.add_argument('--synthetic_regime', action='store', dest='synthetic_regime', type=str, default='d1_small_d2_small', choices=sorted(SYNTHETIC_REGIMES), help='Synthetic beta_i regime')
        
        parser.add_argument('--d', action='store', dest='synthetic_d', type=int, default=1000, help='Synthetic feature dimension d')
        parser.add_argument('--K', action='store', dest='synthetic_K', type=int, default=32, help='Synthetic latent component count K')
        parser.add_argument('--T', action='store', dest='synthetic_T', type=int, default=4, help='Synthetic macro-family count T')
        
        parser.add_argument('--alpha_dir', action='store', dest='alpha_dir', type=float, default=200.0, help='Dirichlet concentration scale')
        parser.add_argument('--eps_dir', action='store', dest='eps_dir', type=float, default=1e-3, help='Dirichlet smoothing weight')
        parser.add_argument('--sigma_obs', action='store', dest='sigma_obs', type=float, default=0.0632455532, help='Observation noise scale')
        parser.add_argument('--r_inter', action='store', dest='r_inter', type=float, default=6.0, help='Macro-center separation radius')
        parser.add_argument('--r_intra', action='store', dest='r_intra', type=float, default=0.5, help='Within-family component radius')

        # Diagnostics (delta/L) settings
        parser.add_argument('--estimate_diagnostics', action='store', dest='estimate_diagnostics', type=int, default=0, help='Enable diagnostics estimation')
        parser.add_argument('--save_diagnostics', action='store', dest='save_diagnostics', type=int, default=1, help='Persist diagnostics JSON and comp-params updates')
        parser.add_argument('--compute_deltas', action='store', dest='compute_deltas', type=int, default=1, help='Compute delta1/delta2 diagnostics')
        parser.add_argument('--compute_L_bounds', action='store', dest='compute_L_bounds', type=int, default=1, help='Compute L-bound diagnostics')
        parser.add_argument('--compute_worstcase', action='store', dest='compute_worstcase', type=int, default=1, help='Compute worst-case diagnostics')

        parser.add_argument('--probe_T', action='store', dest='probe_T', type=int, default=0, help='Number of extra probe points')
        parser.add_argument('--probe_strategy', action='store', dest='probe_strategy', type=str, default='init_only', choices=['init_only', 'init+random', 'init+path'], help='Probe strategy')
        parser.add_argument('--probe_path_steps', action='store', dest='probe_path_steps', type=int, default=50, help='Path probe steps')
        parser.add_argument('--probe_path_lr', action='store', dest='probe_path_lr', type=float, default=0.1, help='Manual path probe learning rate')
        parser.add_argument(
            '--probe_path_lr_mode',
            action='store',
            dest='probe_path_lr_mode',
            type=str,
            default='manual',
            choices=['manual', 'adaptive_global_wc'],
            help='Use manual --probe_path_lr or adaptive 1/(2*L_global_wc) for prepared-data delta diagnostics',
        )

        parser.add_argument('--delta_power_iters', action='store', dest='delta_power_iters', type=int, default=30, help='Power iterations for delta1/shared direction')
        parser.add_argument('--delta_power_tol', action='store', dest='delta_power_tol', type=float, default=1e-6, help='Power method tolerance for deltas')
        parser.add_argument('--delta2_mode', action='store', dest='delta2_mode', type=str, default='approx_shared_vector', choices=['approx_shared_vector', 'exact_per_component'], help='delta2 estimation mode')
        parser.add_argument('--delta2_power_iters', action='store', dest='delta2_power_iters', type=int, default=8, help='Power iterations for exact delta2 mode')
        parser.add_argument('--delta_batch_size', action='store', dest='delta_batch_size', type=int, default=2048, help='Chunk size for delta computations (memory control only, not data subsampling)')

        parser.add_argument('--L_power_iters', action='store', dest='L_power_iters', type=int, default=30, help='Power iterations for L-bounds')
        parser.add_argument('--L_power_tol', action='store', dest='L_power_tol', type=float, default=1e-6, help='Power method tolerance for L-bounds')
        parser.add_argument('--L_batch_size', action='store', dest='L_batch_size', type=int, default=2048, help='Batch size for L computations')
        
        
        # Legacy-only parameters kept for backward compatibility.
        legacy_group = parser.add_argument_group("legacy (inactive in current project)")
        legacy_group.add_argument('--noise_scale', action='store', dest='noise_scale', type=float, default=None, help=argparse.SUPPRESS)
        legacy_group.add_argument('--dim', action='store', dest='dim', type=int, default=None, help=argparse.SUPPRESS)
        legacy_group.add_argument('--num_samples', action='store', dest='num_samples', type=int, default=None, help=argparse.SUPPRESS)
        return parser.parse_args()

    def general_asserts(self):
        assert self.arg_values["loss_func"] in SUPPORTED_LOSS_FUNCS, f"loss_func={self.arg_values['loss_func']} is not supported"
        #assert set(self.arg_values["loadable_params"]).issubset(ALLOWABLE_PARAMS), f"loadable_params={self.arg_values['loadable_params']} contains not allowable params"
        assert self.arg_values["num_workers"] > 0
        assert self.arg_values["hetero"] in [0,1]
        assert self.arg_values["use_ray"] in [0,1]
        assert self.arg_values["is_minimize"] in [0,1] 
        assert self.arg_values["regularizer_type"] in SUPPORTED_REGULARIZERS
        assert self.arg_values["generate_dataset"] in [0,1]
        assert self.arg_values["load_prepared_dataset"] in [0,1]
        assert self.arg_values["load_raw_dataset"] in [0,1]
        assert self.arg_values["load_prepared_dataset"]*self.arg_values["generate_dataset"]==0 #only one of them can be 1
        assert self.arg_values["load_prepared_dataset"]*self.arg_values["load_raw_dataset"]==0 #only one of them can be 1
        assert self.arg_values["load_raw_dataset"]*self.arg_values["generate_dataset"]==0 #only one of them can be 1
        assert self.arg_values["is_sparse_dataset"] in [0,1]
        assert self.arg_values["print_args"] in [0,1]
        assert self.arg_values["print_status"] in [0,1]
        assert self.arg_values["n_groups"] > 0
        assert self.arg_values["m_per_group"] > 0
        assert self.arg_values["partition_regime"] in {
            "random_d1s_d2l",
            "clustered_d1l_d2s",
            "intermediate_d1m_d2m",
            "stratified_d1s_d2s",
            "score_mix_d1l_d2l",
        }
        assert self._cluster_repr_mode() in {"raw", "pca"}
        assert self.arg_values["pca_dim"] > 0
        assert self.arg_values["kmeans_K"] > 1
        assert self.arg_values["score_type"] in {"mean_intensity", "l1"}
        device_str = self.arg_values["device"]
        assert device_str == "cpu" or device_str.startswith("cuda"), "device must be 'cpu', 'cuda', or 'cuda:N'"
        assert self.arg_values["estimate_diagnostics"] in [0, 1]
        assert self.arg_values["save_diagnostics"] in [0, 1]
        assert self.arg_values["compute_deltas"] in [0, 1]
        assert self.arg_values["compute_L_bounds"] in [0, 1]
        assert self.arg_values["compute_worstcase"] in [0, 1]
        assert self.arg_values["probe_T"] >= 0
        assert self.arg_values["probe_path_steps"] >= 0
        assert self.arg_values["probe_path_lr"] > 0
        assert self.arg_values["probe_path_lr_mode"] in {"manual", "adaptive_global_wc"}
        assert self.arg_values["delta_power_iters"] > 0
        assert self.arg_values["delta_power_tol"] >= 0
        assert self.arg_values["delta2_power_iters"] > 0
        assert self.arg_values["delta_batch_size"] > 0
        assert self.arg_values["lambda_reg"] >= 0
        assert self.arg_values["L_power_iters"] > 0
        assert self.arg_values["L_power_tol"] >= 0
        assert self.arg_values["L_batch_size"] > 0
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['comp_params_str'])), "Not all elements are strings."

    def logreg_asserts(self):
        assert self.arg_values["loss_func"]=="log-reg"
        assert self.arg_values["dataset"] in LOGREG_DATASETS
        if self.arg_values["dataset"] == SYNTHETIC_DIRICHLET_LOGREG_DATASET:
            self.synthetic_logreg_asserts()

    def synthetic_logreg_asserts(self):
        # These assertions keep the synthetic grouped construction mathematically valid.
        assert self.arg_values["synthetic_setting"] in SYNTHETIC_SETTING_TO_NM
        assert self.arg_values["synthetic_regime"] in SYNTHETIC_REGIMES
        assert self.arg_values["synthetic_d"] > 0
        assert self.arg_values["synthetic_K"] > 0
        assert self.arg_values["synthetic_T"] > 0
        assert self.arg_values["synthetic_K"] % self.arg_values["synthetic_T"] == 0
        assert self.arg_values["synthetic_d"] >= self.arg_values["synthetic_K"] + self.arg_values["synthetic_T"]
        assert self.arg_values["alpha_dir"] > 0
        assert 0 <= self.arg_values["eps_dir"] < 1
        assert self.arg_values["sigma_obs"] > 0
        assert self.arg_values["r_inter"] > 0
        assert self.arg_values["r_intra"] >= 0

        # The regime definitions use component offsets inside each family, so q must be large enough.
        q = self.arg_values["synthetic_K"] // self.arg_values["synthetic_T"]
        if self.arg_values["synthetic_regime"] == "d1_small_d2_small":
            assert self.arg_values["synthetic_K"] >= 2
        if self.arg_values["synthetic_regime"] == "d1_large_d2_small":
            assert q >= 2
        if self.arg_values["synthetic_regime"] == "d1_large_d2_large":
            assert self.arg_values["synthetic_T"] >= 8
            assert q >= 5

    # consider mooving to the class Experiment
    def path_initialisation(self):
        self.data_name = self.arg_values["dataset"] + ".txt"
        # Path to the directory of the script that is running
        script_directory = os.path.dirname(os.path.abspath(__file__))
        self.raw_data_path = script_directory +'/data/'
        self.project_path = script_directory + "/"
        self.data_path = self.project_path + "data_{0}/".format(self.arg_values["dataset"])
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

    def save_w_init(self):
        if not hasattr(self, "x_0"):
            raise ValueError("x_0 is not initialized; call dataset loader/generator before save_w_init().")
        np.save(self.data_path + 'w_init' + self.w_init_extension, self.x_0)
    
    def save_dataset(self, data_to_save):
        import json
        try:
            import torch
        except ImportError:
            torch = None

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        for key, value in data_to_save.items():
            if key == "metadata":
                metadata_path = os.path.join(self.dataset_path, "metadata.json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(value, f, indent=2)
                my_print("metadata is saved.", self.arg_values["print_status"])
            elif torch is not None and isinstance(value, torch.Tensor):
                tensor_path = os.path.join(self.dataset_path, key if key.endswith(".pt") else f"{key}.pt")
                torch.save(value.detach().cpu(), tensor_path)
                my_print(f"{key} tensor is saved.", self.arg_values["print_status"])
            elif key=="X":
                if self.arg_values["is_sparse_dataset"]:
                    file_path = self.dataset_path + 'X' + self.exp_data_extension + '.h5'
                    with h5py.File(file_path, 'w', track_order=True) as f:
                        for i in range(len(self.X)):
                            matrix = self.X[i]
                            grp = f.create_group(f'matrix_{i}')
                            grp.create_dataset('data', data=matrix.data)
                            grp.create_dataset('indices', data=matrix.indices)
                            grp.create_dataset('indptr', data=matrix.indptr)
                            grp.create_dataset('shape', data=matrix.shape)
                    my_print("Sparse datasets are saved.", self.arg_values["print_status"])
                else:
                    np.save(self.dataset_path + 'X' + self.exp_data_extension, self.X)
                    my_print("Dense datasets are saved.", self.arg_values["print_status"])
            else:
                np.save(self.dataset_path + key + self.exp_data_extension, value)
                my_print(f"{key} dataset is saved.", self.arg_values["print_status"])

    def _apply_dataset_specific_defaults(self):
        # The synthetic grouped pathway derives n and m from the named setting.
        if self.arg_values["dataset"] != SYNTHETIC_DIRICHLET_LOGREG_DATASET:
            return

        # Resolve the requested setting into the exact grouped dimensions.
        n_groups, m_per_group = synthetic_logreg_setting_to_nm(self.arg_values["synthetic_setting"])

        # Overwrite the generic CLI placeholders so all downstream naming uses the true grouped shape.
        self.arg_values["n_groups"] = int(n_groups)
        self.arg_values["m_per_group"] = int(m_per_group)

        # The synthetic generator is now the source of truth for regime-specific
        # K/T, Dirichlet/noise/radius parameters, and lambda_reg.
        regime_params = synthetic_logreg_regime_params(
            self.arg_values["synthetic_setting"],
            self.arg_values["synthetic_regime"],
        )
        self.arg_values["synthetic_K"] = int(regime_params["K"])
        self.arg_values["synthetic_T"] = int(regime_params["T"])
        self.arg_values["alpha_dir"] = float(regime_params["alpha_dir"])
        self.arg_values["eps_dir"] = float(regime_params["eps_dir"])
        self.arg_values["sigma_obs"] = float(regime_params["sigma_obs"])
        self.arg_values["r_inter"] = float(regime_params["r_inter"])
        self.arg_values["r_intra"] = float(regime_params["r_intra"])
        self.arg_values["lambda_reg"] = float(regime_params["lambda_reg"])

        # Synthetic grouped data does not use a clustering representation, so keep the placeholder canonical.
        if self.arg_values.get("cluster_repr") is None:
            self.arg_values["cluster_repr"] = "raw"
    
    def _resolve_torch_device(self):
        import torch
        requested_device = str(self.arg_values["device"])
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            my_print("CUDA requested but unavailable. Falling back to CPU.", self.arg_values["print_status"])
            return torch.device("cpu")
        if requested_device.startswith("cuda:"):
            try:
                device_id = int(requested_device.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device string: {requested_device}") from exc
            if device_id < 0 or device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"Requested CUDA device {requested_device} is out of range. "
                    f"Available devices: 0..{max(torch.cuda.device_count()-1, 0)}"
                )
        return torch.device(requested_device)

    def _compute_scores(self, X_np):
        if self.arg_values["score_type"] == "mean_intensity":
            return X_np.mean(axis=1)
        if self.arg_values["score_type"] == "l1":
            return np.abs(X_np).sum(axis=1)
        raise ValueError("score_type is not supported")

    def _cluster_repr_mode(self):
        cluster_repr = self.arg_values.get("cluster_repr")
        if cluster_repr is None:
            cluster_repr = self.arg_values.get("repr", "raw")
        return cluster_repr

    def _diagnostics_lambda_reg(self):
        return float(self.arg_values["lambda_reg"])

    def _ensure_prepared_group_indices_loaded(self):
        """Auto-load group membership for prepared grouped diagnostics when needed."""
        import torch

        group_indices = self.data_dict.get("group_indices_train")
        if group_indices is not None:
            return group_indices

        group_indices_path = os.path.join(self.dataset_path, "group_indices_train.pt")
        if not os.path.exists(group_indices_path):
            raise FileNotFoundError(
                "Prepared grouped diagnostics require group_indices_train.pt in the prepared dataset folder, "
                f"but it was not found at: {group_indices_path}"
            )

        group_indices = torch.load(group_indices_path, map_location="cpu")
        self.data_dict["group_indices_train"] = group_indices
        my_print(
            f"Auto-loaded group_indices_train from {group_indices_path}",
            self.arg_values["print_status"],
        )
        return group_indices

    def _estimate_and_save_diagnostics(
        self,
        X_train_np,
        y_train_np,
        group_indices_np,
        resolved_device=None,
        probe_path_lr_override=None,
        probe_path_lr_rule=None,
        compute_delta1_full=False,
        compute_delta2_full=False,
        compute_delta_flat=False,
        compute_delta2_i=False,
        compute_deltas_override=None,
        diagnostics_filename=None,
    ):
        if not self.arg_values["estimate_diagnostics"]:
            return None

        import json
        import torch
        from src.diagnostics import (
            estimate_logreg_noncvx_client_delta2_i,
            estimate_logreg_noncvx_diagnostics,
            estimate_logreg_noncvx_flat_delta,
        )

        device = self._resolve_torch_device() if resolved_device is None else resolved_device
        lambda_reg = self._diagnostics_lambda_reg()
        probe_path_lr_used = (
            float(probe_path_lr_override)
            if probe_path_lr_override is not None
            else float(self.arg_values["probe_path_lr"])
        )

        X_train_t = X_train_np if isinstance(X_train_np, torch.Tensor) else torch.as_tensor(X_train_np)
        y_train_t = y_train_np if isinstance(y_train_np, torch.Tensor) else torch.as_tensor(y_train_np)
        group_indices_t = group_indices_np if isinstance(group_indices_np, torch.Tensor) else torch.as_tensor(group_indices_np)

        compute_standard_deltas = (
            bool(self.arg_values["compute_deltas"])
            if compute_deltas_override is None
            else bool(compute_deltas_override)
        )
        needs_generic_diagnostics = (
            compute_standard_deltas
            or bool(compute_delta1_full)
            or bool(compute_delta2_full)
            or bool(self.arg_values["compute_L_bounds"])
            or bool(self.arg_values["compute_worstcase"])
        )

        diagnostics = {}
        if needs_generic_diagnostics:
            diagnostics.update(
                estimate_logreg_noncvx_diagnostics(
                    X_train=X_train_t.to(torch.float32),
                    y_train=y_train_t.to(torch.int64),
                    group_indices=group_indices_t.to(torch.int64),
                    lambda_reg=lambda_reg,
                    compute_deltas=compute_standard_deltas,
                    compute_delta1_full=bool(compute_delta1_full),
                    compute_delta2_full=bool(compute_delta2_full),
                    compute_L_bounds=bool(self.arg_values["compute_L_bounds"]),
                    compute_worstcase=bool(self.arg_values["compute_worstcase"]),
                    probe_T=int(self.arg_values["probe_T"]),
                    probe_strategy=self.arg_values["probe_strategy"],
                    probe_path_steps=int(self.arg_values["probe_path_steps"]),
                    probe_path_lr=probe_path_lr_used,
                    delta_power_iters=int(self.arg_values["delta_power_iters"]),
                    delta_power_tol=float(self.arg_values["delta_power_tol"]),
                    delta2_mode=self.arg_values["delta2_mode"],
                    delta2_power_iters=int(self.arg_values["delta2_power_iters"]),
                    delta_batch_size=int(self.arg_values["delta_batch_size"]),
                    L_power_iters=int(self.arg_values["L_power_iters"]),
                    L_power_tol=float(self.arg_values["L_power_tol"]),
                    L_batch_size=int(self.arg_values["L_batch_size"]),
                    device=device,
                    seed=int(self.arg_values["seed"]),
                    probe_path_lr_rule=probe_path_lr_rule,
                    delta_full_method="exact_eigvalsh_float64" if (compute_delta1_full or compute_delta2_full) else None,
                    show_progress=bool(self.arg_values["print_status"]),
                )
            )

        if compute_delta_flat:
            flat_diagnostics = estimate_logreg_noncvx_flat_delta(
                X_train=X_train_t,
                y_train=y_train_t,
                lambda_reg=lambda_reg,
                probe_T=int(self.arg_values["probe_T"]),
                probe_strategy=self.arg_values["probe_strategy"],
                probe_path_steps=int(self.arg_values["probe_path_steps"]),
                probe_path_lr=probe_path_lr_used,
                delta_power_iters=int(self.arg_values["delta_power_iters"]),
                delta_power_tol=float(self.arg_values["delta_power_tol"]),
                delta_batch_size=int(self.arg_values["delta_batch_size"]),
                device=device,
                seed=int(self.arg_values["seed"]),
                probe_path_lr_rule=probe_path_lr_rule,
                show_progress=bool(self.arg_values["print_status"]),
            )
            diagnostics.update(flat_diagnostics)

        if compute_delta2_i:
            group_indices_device = group_indices_t.to(device=X_train_t.device, dtype=torch.int64)
            X_groups_t = X_train_t[group_indices_device]
            y_groups_t = y_train_t[group_indices_device]
            client_diagnostics = estimate_logreg_noncvx_client_delta2_i(
                X_groups_train=X_groups_t,
                y_groups_train=y_groups_t,
                lambda_reg=lambda_reg,
                probe_T=int(self.arg_values["probe_T"]),
                probe_strategy=self.arg_values["probe_strategy"],
                probe_path_steps=int(self.arg_values["probe_path_steps"]),
                probe_path_lr=probe_path_lr_used,
                delta_power_iters=int(self.arg_values["delta_power_iters"]),
                delta_power_tol=float(self.arg_values["delta_power_tol"]),
                delta_batch_size=int(self.arg_values["delta_batch_size"]),
                device=device,
                seed=int(self.arg_values["seed"]),
                probe_path_lr_rule=probe_path_lr_rule,
                show_progress=bool(self.arg_values["print_status"]),
            )
            diagnostics.update(client_diagnostics)

        def _json_safe(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {key: _json_safe(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_json_safe(item) for item in value]
            if isinstance(value, tuple):
                return [_json_safe(item) for item in value]
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                return value.item()
            return value

        if self.arg_values["save_diagnostics"]:
            diagnostics_basename = diagnostics_filename or "diagnostics.json"
            diagnostics_path = os.path.join(self.dataset_path, diagnostics_basename)
            with open(diagnostics_path, "w", encoding="utf-8") as f:
                json.dump(_json_safe(diagnostics), f, indent=2)
            my_print(f"Saved diagnostics to {diagnostics_path}", self.arg_values["print_status"])
        else:
            my_print("save_diagnostics=0; diagnostics were not written to disk.", self.arg_values["print_status"])

        # Print dataset-specific context so synthetic runs do not inherit the MNIST partition label.
        if self.arg_values["dataset"] == SYNTHETIC_DIRICHLET_LOGREG_DATASET:
            diagnostics_context = (
                f"Diagnostics context: synthetic_setting={self.arg_values['synthetic_setting']}, "
                f"synthetic_regime={self.arg_values['synthetic_regime']}, "
                f"n={self.arg_values['n_groups']}, m={self.arg_values['m_per_group']}"
            )
        else:
            diagnostics_context = (
                f"Diagnostics context: regime={self.arg_values['partition_regime']}, "
                f"n={self.arg_values['n_groups']}, m={self.arg_values['m_per_group']}, "
                f"cluster_repr={self.arg_values['cluster_repr']}"
            )
        my_print(
            diagnostics_context,
            self.arg_values["print_status"],
        )
        my_print(
            f"probe_path_lr_used: {probe_path_lr_used}, probe_path_lr_rule: {probe_path_lr_rule or 'manual'}",
            self.arg_values["print_status"],
        )

        # Optional propagation to comp_params (if user requested these keys).
        diag_scalar_keys = [
            "delta1_emp_full", "delta2_emp_full",
            "delta_flat_emp", "delta2_i_emp", "delta2_bar_sq_emp",
            "delta1_emp", "delta2_emp", "delta1_wc", "delta2_wc",
            "L_ij_max_emp", "L_i_max_emp", "L_global_emp",
            "L_ij_max_wc", "L_i_max_wc", "L_global_wc",
        ]
        for key in diag_scalar_keys:
            if key not in diagnostics:
                continue
            is_requested_key = key in self.requested_comp_params_set
            has_real_value = diagnostics[key] is not None
            if self.arg_values["save_diagnostics"] and is_requested_key and has_real_value:
                self.comp_params_dict[key] = diagnostics[key]
            if has_real_value or is_requested_key:
                my_print(f"{key}: {diagnostics[key]}", self.arg_values["print_status"])

        return diagnostics

    def _partition_train_indices(self, X_np, y_np):
        from sklearn.cluster import KMeans

        n_groups = self.arg_values["n_groups"]
        m_per_group = self.arg_values["m_per_group"]
        regime = self.arg_values["partition_regime"]
        seed = self.arg_values["seed"]

        N = X_np.shape[0]
        total_needed = n_groups * m_per_group
        if total_needed > N:
            raise ValueError(f"Requested n_groups*m_per_group={total_needed} exceeds available samples={N}")

        rs = RandomState(seed)

        def _rebalance_exact_m(groups):
            moves = 0
            donors = [i for i in range(n_groups) if len(groups[i]) > m_per_group]
            receivers = [i for i in range(n_groups) if len(groups[i]) < m_per_group]
            while donors and receivers:
                d = donors[0]
                r = receivers[0]
                can_give = len(groups[d]) - m_per_group
                need = m_per_group - len(groups[r])
                take = min(can_give, need)
                moved = groups[d][-take:]
                del groups[d][-take:]
                groups[r].extend(moved)
                moves += take
                if len(groups[d]) == m_per_group:
                    donors.pop(0)
                if len(groups[r]) == m_per_group:
                    receivers.pop(0)
            if any(len(g) != m_per_group for g in groups):
                raise RuntimeError("Rebalancing failed to enforce exact |S_i| = m for all groups.")
            return moves

        partition_meta = {
            "regime": regime,
            "n_groups": int(n_groups),
            "m_per_group": int(m_per_group),
            "nm": int(total_needed),
            "seed": int(seed),
        }

        all_indices = np.arange(N, dtype=np.int64)
        base_indices = rs.permutation(N)[:total_needed]

        if regime == "random_d1s_d2l":
            group_indices = base_indices.reshape(n_groups, m_per_group)
            partition_meta["selection_rule"] = "single seeded shuffle of N_train and reshape to [n,m]"
        elif regime == "clustered_d1l_d2s":
            kmeans = KMeans(n_clusters=n_groups, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(X_np[base_indices])
            centroids = kmeans.cluster_centers_
            centroid_dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)

            available = {}
            for cid in range(n_groups):
                members = base_indices[labels == cid]
                members = members[rs.permutation(len(members))]
                members = np.sort(members).tolist()
                available[cid] = members

            groups = [[] for _ in range(n_groups)]
            for cid in range(n_groups):
                take = min(m_per_group, len(available[cid]))
                groups[cid].extend(available[cid][:take])
                del available[cid][:take]

            topup_counts = [0 for _ in range(n_groups)]
            for cid in range(n_groups):
                while len(groups[cid]) < m_per_group:
                    neighbors = np.argsort(centroid_dists[cid], kind="stable")
                    filled = False
                    for n_cid in neighbors:
                        if n_cid == cid:
                            continue
                        if available[n_cid]:
                            groups[cid].append(available[n_cid].pop(0))
                            topup_counts[cid] += 1
                            filled = True
                            break
                    if not filled:
                        remaining = [x for pool in available.values() for x in pool]
                        if not remaining:
                            raise RuntimeError("Insufficient residual samples while filling clustered groups.")
                        fallback = remaining[0]
                        groups[cid].append(fallback)
                        for pool in available.values():
                            if fallback in pool:
                                pool.remove(fallback)
                                break

            group_indices = np.array(groups, dtype=np.int64)
            partition_meta["clustered_fill_rule"] = (
                "Own cluster first; if |C_i|<m, top-up from nearest clusters by centroid distance without reuse."
            )
            partition_meta["clustered_topup_counts"] = [int(x) for x in topup_counts]
        elif regime == "intermediate_d1m_d2m":
            K = min(max(self.arg_values["kmeans_K"], n_groups), total_needed)
            kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(X_np[base_indices])

            micro_clusters = {}
            for cid in range(K):
                members = base_indices[labels == cid]
                members = members[rs.permutation(len(members))]
                members = np.sort(members).tolist()
                micro_clusters[cid] = members

            groups = [[] for _ in range(n_groups)]
            for cid in range(K):
                gid = cid % n_groups
                for idx in micro_clusters[cid]:
                    if len(groups[gid]) < m_per_group:
                        groups[gid].append(idx)

            used = set(x for g in groups for x in g)
            leftover = [idx for idx in base_indices if idx not in used]
            leftover = [leftover[i] for i in rs.permutation(len(leftover))]
            p = 0
            topup_counts = [0 for _ in range(n_groups)]
            for gid in range(n_groups):
                need = m_per_group - len(groups[gid])
                if need > 0:
                    take_slice = leftover[p:p + need]
                    if len(take_slice) < need:
                        raise RuntimeError("Insufficient leftover samples for intermediate top-up.")
                    groups[gid].extend(take_slice)
                    topup_counts[gid] = need
                    p += need

            group_indices = np.array(groups, dtype=np.int64)
            partition_meta["intermediate_K"] = int(K)
            partition_meta["intermediate_topup_counts"] = [int(x) for x in topup_counts]
            partition_meta["intermediate_rule"] = (
                "K micro-clusters assigned by cid mod n; top-up from remaining seeded-random pool."
            )
        elif regime == "stratified_d1s_d2s":
            # For stratified regime, K should stay small enough that each group
            # can represent strata composition when m is small.
            K_req = int(self.arg_values["kmeans_K"])
            K_max_by_m = max(2, min(m_per_group, 64))
            K = min(max(2, K_req), K_max_by_m, N)
            kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
            labels_all = kmeans.fit_predict(X_np[all_indices])
            counts = np.bincount(labels_all, minlength=K).astype(np.int64)

            targets = (total_needed / N) * counts.astype(np.float64)
            quotas = np.floor(targets).astype(np.int64)
            residual = int(total_needed - quotas.sum())
            frac = targets - quotas
            order = np.argsort(-frac, kind="stable")
            for cid in order:
                if residual <= 0:
                    break
                if quotas[cid] < counts[cid]:
                    quotas[cid] += 1
                    residual -= 1
            if residual > 0:
                for cid in range(K):
                    if residual <= 0:
                        break
                    if quotas[cid] < counts[cid]:
                        quotas[cid] += 1
                        residual -= 1
            if quotas.sum() != total_needed:
                raise RuntimeError("Failed to build stratified quotas with sum M_k = nm.")

            groups = [[] for _ in range(n_groups)]
            for cid in range(K):
                cluster_idx = np.where(labels_all == cid)[0]
                cluster_idx = cluster_idx[rs.permutation(len(cluster_idx))]
                selected = cluster_idx[:int(quotas[cid])]
                slices = np.array_split(selected, n_groups)
                for gid in range(n_groups):
                    groups[gid].extend(slices[gid].tolist())

            rebalance_moves = _rebalance_exact_m(groups)
            for gid in range(n_groups):
                groups[gid] = np.array(groups[gid], dtype=np.int64)
                groups[gid] = groups[gid][np.argsort(groups[gid], kind="stable")].tolist()

            group_indices = np.array(groups, dtype=np.int64)
            partition_meta["stratified_K"] = int(K)
            partition_meta["stratified_K_requested"] = int(K_req)
            partition_meta["stratified_K_max_by_m"] = int(K_max_by_m)
            partition_meta["stratified_rebalance_moves"] = int(rebalance_moves)
            partition_meta["stratified_quota_sum"] = int(quotas.sum())
            partition_meta["stratified_rule"] = (
                "M_k=floor((nm/N)N_k)+deterministic remainder fix; micro-cluster slices split across groups; minimal rebalance."
            )
        elif regime == "score_mix_d1l_d2l":
            scores = self._compute_scores(X_np)
            sorted_indices = np.argsort(scores, kind="stable")
            B_nominal = max(4 * m_per_group, 2 * m_per_group)
            B_max = N // n_groups
            if B_max < 2 * m_per_group:
                raise ValueError("Not enough samples to satisfy score_mix requirement B >= 2m for all groups.")
            B = min(B_nominal, B_max)
            total_band = n_groups * B
            offset = (N - total_band) // 2
            window = sorted_indices[offset:offset + total_band]
            bands = window.reshape(n_groups, B)

            low_take = m_per_group // 2
            high_take = m_per_group - low_take
            groups = []
            for gid in range(n_groups):
                band = bands[gid]
                low = band[:low_take]
                high = band[-high_take:]
                group = np.concatenate([low, high]).astype(np.int64)
                groups.append(group)
            group_indices = np.vstack(groups)
            partition_meta["score_band_size_B"] = int(B)
            partition_meta["score_band_offset"] = int(offset)
            partition_meta["score_rule"] = (
                "Disjoint consecutive score bands; each group takes low/high extremes within its own band."
            )
        else:
            raise ValueError(
                "partition_regime is not supported. Use one of: "
                "random_d1s_d2l, clustered_d1l_d2s, intermediate_d1m_d2m, "
                "stratified_d1s_d2s, score_mix_d1l_d2l"
            )

        flat = group_indices.reshape(-1)
        if flat.shape[0] != total_needed:
            raise RuntimeError("Partition output shape mismatch.")
        if np.unique(flat).shape[0] != total_needed:
            raise RuntimeError("Partition must not reuse any sample index.")

        partition_meta["unique_indices"] = int(np.unique(flat).shape[0])
        return group_indices.astype(np.int64), partition_meta

    def _load_raw_mnist01(self):
        import datetime
        import torch
        from torchvision.datasets import MNIST
        from sklearn.decomposition import PCA

        seed = self.arg_values["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        device = self._resolve_torch_device()
        mnist_root = os.path.join(self.raw_data_path, "torchvision_mnist")
        train_ds = MNIST(root=mnist_root, train=True, download=True)
        test_ds = MNIST(root=mnist_root, train=False, download=True)

        X_train = (train_ds.data.float() / 255.0).view(-1, 28 * 28)
        y_train = train_ds.targets.long()
        X_test = (test_ds.data.float() / 255.0).view(-1, 28 * 28)
        y_test = test_ds.targets.long()

        train_mask = (y_train == 0) | (y_train == 1)
        test_mask = (y_test == 0) | (y_test == 1)

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        y_train = torch.where(y_train == 0, -torch.ones_like(y_train), torch.ones_like(y_train)).to(torch.int64)
        y_test = torch.where(y_test == 0, -torch.ones_like(y_test), torch.ones_like(y_test)).to(torch.int64)

        X_train_raw_np = X_train.cpu().numpy().astype(np.float32)
        X_test_raw_np = X_test.cpu().numpy().astype(np.float32)
        y_train_np = y_train.cpu().numpy().astype(np.int64)

        cluster_repr_requested = self._cluster_repr_mode()
        regime = self.arg_values["partition_regime"]
        pca_partition_regimes = {"clustered_d1l_d2s", "intermediate_d1m_d2m", "stratified_d1s_d2s"}
        use_pca_for_partition = (cluster_repr_requested == "pca") and (regime in pca_partition_regimes)

        cluster_repr_used = "pca" if use_pca_for_partition else "raw"
        X_cluster_np = X_train_raw_np
        pca_info = {}
        if use_pca_for_partition:
            target_dim = min(self.arg_values["pca_dim"], X_train_raw_np.shape[1])
            pca = PCA(n_components=target_dim, random_state=seed)
            X_cluster_np = pca.fit_transform(X_train_raw_np).astype(np.float32)
            np.save(os.path.join(self.dataset_path, "cluster_pca_components.npy"), pca.components_.astype(np.float32))
            np.save(os.path.join(self.dataset_path, "cluster_pca_mean.npy"), pca.mean_.astype(np.float32))
            pca_info = {
                "cluster_pca_dim": int(target_dim),
                "cluster_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
                "cluster_pca_saved_files": ["cluster_pca_components.npy", "cluster_pca_mean.npy"],
            }

        grouped_indices, partition_meta = self._partition_train_indices(X_cluster_np, y_train_np)
        flat_idx = grouped_indices.reshape(-1)
        X_groups_np = X_train_raw_np[flat_idx].reshape(self.arg_values["n_groups"], self.arg_values["m_per_group"], -1)
        y_groups_np = y_train_np[flat_idx].reshape(self.arg_values["n_groups"], self.arg_values["m_per_group"])

        X_train_t = torch.from_numpy(X_train_raw_np).to(torch.float32).to(device)
        y_train_t = torch.from_numpy(y_train_np).to(torch.int64).to(device)
        X_test_t = torch.from_numpy(X_test_raw_np).to(torch.float32).to(device)
        y_test_t = torch.from_numpy(y_test.cpu().numpy().astype(np.int64)).to(torch.int64).to(device)
        X_groups_t = torch.from_numpy(X_groups_np).to(torch.float32).to(device)
        y_groups_t = torch.from_numpy(y_groups_np).to(torch.int64).to(device)
        group_indices_t = torch.from_numpy(grouped_indices).to(torch.int64).to(device)
        resolved_device_str = str(device)

        self.save_dataset({
            "X_train": X_train_t,
            "y_train": y_train_t,
            "X_test": X_test_t,
            "y_test": y_test_t,
            "X_groups_train": X_groups_t,
            "y_groups_train": y_groups_t,
            "group_indices_train": group_indices_t,
            "metadata": {
                "dataset": "mnist01",
                "classes": [0, 1],
                "label_mapping": {"0": -1, "1": 1},
                "loss_func": self.arg_values["loss_func"],
                "partition_regime": self.arg_values["partition_regime"],
                "n_groups": int(self.arg_values["n_groups"]),
                "m_per_group": int(self.arg_values["m_per_group"]),
                "training_repr": "raw",
                "cluster_repr_requested": cluster_repr_requested,
                "cluster_repr_used": cluster_repr_used,
                "score_type": self.arg_values["score_type"],
                "kmeans_K": int(self.arg_values["kmeans_K"]),
                "seed": int(seed),
                "feature_dim": int(X_train_raw_np.shape[1]),
                "cluster_feature_dim": int(X_cluster_np.shape[1]),
                "n_train": int(X_train_raw_np.shape[0]),
                "n_test": int(X_test_raw_np.shape[0]),
                "device": self.arg_values["device"],
                "requested_device": str(self.arg_values["device"]),
                "resolved_device": resolved_device_str,
                "group_membership_note": "delta1/delta2 depend on set membership S_i; within-group order j is not semantically important.",
                "partition_details": partition_meta,
                "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                **pca_info,
            },
        })

        diagnostics = self._estimate_and_save_diagnostics(
            X_train_np=X_train_raw_np,
            y_train_np=y_train_np,
            group_indices_np=grouped_indices,
            resolved_device=device,
        )
        if diagnostics is not None:
            my_print(
                (
                    f"Diagnostics summary: delta1_emp={diagnostics.get('delta1_emp', 'na')}, "
                    f"delta2_emp={diagnostics.get('delta2_emp', 'na')}, "
                    f"L_global_emp={diagnostics.get('L_global_emp', 'na')}, "
                    f"L_global_wc={diagnostics.get('L_global_wc', 'na')}"
                ),
                self.arg_values["print_status"],
            )

        # Initialize and persist a deterministic starting point for downstream algorithm runs.
        rs_w_init = RandomState(seed)
        self.x_0 = rs_w_init.normal(loc=0.0, scale=1.0, size=int(X_train_raw_np.shape[1])).astype(np.float64)
        self.save_w_init()

        self.arg_values["dim"] = X_train_raw_np.shape[1]
        self.comp_params_dict["lambda_reg"] = float(self.arg_values["lambda_reg"])
        self.comp_params_dict["dim"] = int(X_train_raw_np.shape[1])
        self.comp_params_dict["n_groups"] = int(self.arg_values["n_groups"])
        self.comp_params_dict["m_per_group"] = int(self.arg_values["m_per_group"])

    def _generate_synthetic_dirichlet_logreg(self):
        import datetime
        import torch

        # Keep the generated tensors on CPU; the diagnostics routine will move them as needed.
        generated = generate_synthetic_dirichlet_logreg(
            synthetic_setting=self.arg_values["synthetic_setting"],
            synthetic_regime=self.arg_values["synthetic_regime"],
            d=int(self.arg_values["synthetic_d"]),
            seed=int(self.arg_values["seed"]),
        )

        # Extract the grouped tensors in the exact format expected by the current diagnostics code.
        X_train_t = generated["X_train"].to(torch.float32)
        y_train_t = generated["y_train"].to(torch.int64)
        X_groups_t = generated["X_groups_train"].to(torch.float32)
        y_groups_t = generated["y_groups_train"].to(torch.int64)
        group_indices_t = generated["group_indices_train"].to(torch.int64)
        resolved_device = self._resolve_torch_device()
        resolved_device_str = str(resolved_device)

        # Save the synthetic dataset using the same tensor file names as the MNIST pathway.
        self.save_dataset({
            "X_train": X_train_t,
            "y_train": y_train_t,
            "X_groups_train": X_groups_t,
            "y_groups_train": y_groups_t,
            "group_indices_train": group_indices_t,
            "metadata": {
                **generated["metadata"],
                "loss_func": self.arg_values["loss_func"],
                "regularizer_type": self.arg_values["regularizer_type"],
                "lambda_reg": float(self.arg_values["lambda_reg"]),
                "device": self.arg_values["device"],
                "requested_device": str(self.arg_values["device"]),
                "resolved_device": resolved_device_str,
                "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            },
        })

        # Reuse the existing diagnostics estimator so synthetic and MNIST use identical measurements.
        requested_keys = set(self.requested_comp_params_set)
        diagnostics = self._estimate_and_save_diagnostics(
            X_train_np=X_train_t.cpu().numpy().astype(np.float32),
            y_train_np=y_train_t.cpu().numpy().astype(np.int64),
            group_indices_np=group_indices_t.cpu().numpy().astype(np.int64),
            resolved_device=resolved_device,
            compute_delta1_full="delta1_emp_full" in requested_keys,
            compute_delta2_full="delta2_emp_full" in requested_keys,
            compute_delta_flat="delta_flat_emp" in requested_keys,
            compute_delta2_i=(
                "delta2_i_emp" in requested_keys
                or "delta2_bar_sq_emp" in requested_keys
            ),
        )
        if diagnostics is not None:
            my_print(
                (
                    f"Diagnostics summary: delta1_emp={diagnostics.get('delta1_emp', 'na')}, "
                    f"delta2_emp={diagnostics.get('delta2_emp', 'na')}, "
                    f"L_global_emp={diagnostics.get('L_global_emp', 'na')}, "
                    f"L_global_wc={diagnostics.get('L_global_wc', 'na')}"
                ),
                self.arg_values["print_status"],
            )

        # Keep the initial point convention identical to the current MNIST preprocessing pathway.
        rs_w_init = RandomState(int(self.arg_values["seed"]))
        self.x_0 = rs_w_init.normal(
            loc=0.0,
            scale=1.0,
            size=int(self.arg_values["synthetic_d"]),
        ).astype(np.float64)
        self.save_w_init()

        # Populate the scalar comp_params keys expected by the existing bundle-saving logic.
        self.arg_values["dim"] = int(self.arg_values["synthetic_d"])
        self.comp_params_dict["lambda_reg"] = float(self.arg_values["lambda_reg"])
        self.comp_params_dict["dim"] = int(self.arg_values["synthetic_d"])
        self.comp_params_dict["n_groups"] = int(self.arg_values["n_groups"])
        self.comp_params_dict["m_per_group"] = int(self.arg_values["m_per_group"])

    def generation(self, seed=None):
        seed = int(self.arg_values["seed"] if seed is None else seed)

        if self.arg_values["dataset"] == SYNTHETIC_DIRICHLET_LOGREG_DATASET:
            if self.arg_values["loss_func"] != "log-reg":
                raise ValueError("Synthetic Dirichlet generation is only supported for log-reg.")
            self._generate_synthetic_dirichlet_logreg()
            return

        if self.arg_values["loss_func"] == "quartic" and self.arg_values["dataset"] == "synthetic":
            self.generate_quartic_synthetic(seed)
            return

        raise ValueError(
            f"generate_dataset is not supported for dataset={self.arg_values['dataset']} "
            f"and loss_func={self.arg_values['loss_func']}"
        )

    def load_raw_dataset(self):
        if self.arg_values["loss_func"]=="l1_norm":
            raise ValueError("l1_norm loss_func temporarily is not supported for load_raw_dataset")
        if self.arg_values["loss_func"] != "log-reg":
            raise ValueError("loss_func is not supported")

        if self.arg_values["dataset"] == SYNTHETIC_DIRICHLET_LOGREG_DATASET:
            raise ValueError(
                "synthetic_dirichlet_logreg is generated data; use --generate_dataset 1 "
                "instead of --load_raw_dataset 1."
            )
        if self.arg_values["dataset"] == "mnist01":
            self._load_raw_mnist01()
            return
        self.load_raw_log_reg(seed=self.arg_values["seed"])
            
    #consider mooving to the class Experiment    
    def load_raw_log_reg(self, seed=42):
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        if self.arg_values["is_sparse_dataset"]:
            raise NotImplementedError("sparse datasets are not supported at this point")

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
            
        data, enc_labels = load_svmlight_dataset(self.raw_data_path + self.data_name, self.arg_values["is_sparse_dataset"])  
        assert (type(enc_labels) == np.ndarray)
        if np.sum(np.isnan(enc_labels)) > 0:
            raise ValueError("nan values of labels")
        if np.sum(np.isnan(data)) > 0:
            raise ValueError("nan values in data matrix")
        my_print (f"Data shape: {data.shape}", self.arg_values["print_status"])
        
        self.X_0 = np.float64(data)
        self.y_0 = enc_labels
        assert len(self.X_0.shape) == 2
        assert len(self.y_0.shape) == 1
        nan_check([self.X_0, self.y_0])
        dim = self.X_0.shape[1]
        
        self.any_vector = np.zeros(dim)
        rs_w_init = RandomState(seed)
        self.x_0 = rs_w_init.normal(loc=0.0, scale=1.0, size=dim)

        data_to_save = {'X_0': self.X_0, 'y_0': self.y_0}
        self.save_dataset(data_to_save)
        self.save_w_init()

        lambda_reg = float(self.arg_values["lambda_reg"])
        self.comp_params_dict["lambda_reg"] = lambda_reg
        self.comp_params_dict['L_0'] = second_matrix_norm(
            self.oracle_dict["hess_bound"](np.zeros(dim), self.X_0, {"lambda_reg": lambda_reg})
        )
        self.arg_values["dim"] = dim
        
        final_memory = process.memory_info().rss
        total_memory_used = final_memory - initial_memory
        arrays_memory = 0
        for value in data_to_save.values():
            if isinstance(value, np.ndarray):
                arrays_memory += value.nbytes

        print(f"Memory used by NumPy arrays: {arrays_memory / (1024 ** 3):.2f} GB")
        print(f"Total memory used during the generation: {total_memory_used / (1024 ** 3):.2f} GB")
    
    def load_precomputed_params(self):
        """Load the requested precomputed scalar parameters from the existing comp-params bundle."""
        if len(self.loadable_params_list) > 0:
            self.comp_params_path = self.data_path + 'comp_params' + self.exp_params_extension + "/"
            loaded_bundle = load_comp_params_bundle(self.comp_params_path, self.loadable_params_list, self.arg_values["print_status"])
            for param in self.loadable_params_list:
                if param in loaded_bundle:
                    self.alg_params_dict[param] = loaded_bundle[param]
                    continue
                param_path = self.comp_params_path + param + self.exp_params_extension
                self.alg_params_dict[param] = load_param(param_path, param, self.arg_values["print_status"])

    def _run_prepared_diagnostics_only(self):
        """
        Prepared-data diagnostics-only path for synthetic delta/L runs.

        This path is active when:
        - load_prepared_dataset = 1
        - estimate_diagnostics = 1
        - is_minimize = 0
        """
        if self.arg_values["load_prepared_dataset"] != 1:
            return None
        if self.arg_values["estimate_diagnostics"] != 1:
            return None
        if self.arg_values["is_minimize"] != 0:
            return None
        if self.arg_values["loss_func"] != "log-reg":
            raise NotImplementedError("Prepared-data diagnostics-only mode is implemented only for log-reg.")

        requested_full_delta_keys = {
            "delta1_emp_full": "delta1_emp_full" in self.comp_params_dict,
            "delta2_emp_full": "delta2_emp_full" in self.comp_params_dict,
        }
        requested_lightweight_keys = {
            "delta_flat_emp": "delta_flat_emp" in self.comp_params_dict,
            "delta2_i_emp": "delta2_i_emp" in self.comp_params_dict,
            "delta2_bar_sq_emp": "delta2_bar_sq_emp" in self.comp_params_dict,
        }

        stage_requested_keys = set(self.requested_comp_params_set)
        legacy_delta_keys = {"delta1_emp", "delta2_emp", "delta1_wc", "delta2_wc"}
        any_full_delta = any(requested_full_delta_keys.values())
        any_lightweight_delta = any(requested_lightweight_keys.values())
        requested_generic_delta_keys = any(key in stage_requested_keys for key in legacy_delta_keys)
        any_requested_delta = any_full_delta or any_lightweight_delta or requested_generic_delta_keys

        X_train_t, y_train_t, resolved_device = self._canonicalize_loaded_train_tensors()
        group_indices_t = self._ensure_prepared_group_indices_loaded()
        if not hasattr(group_indices_t, "shape") or len(group_indices_t.shape) != 2:
            raise ValueError(
                f"Expected group_indices_train to have shape (n_groups, m_per_group), got {getattr(group_indices_t, 'shape', None)}."
            )

        existing_bundle = self._merge_existing_comp_params_bundle(
            stage_computed_keys=stage_requested_keys,
            include_all_existing=True,
        )

        probe_path_lr_override = None
        probe_path_lr_rule = "manual"
        if any_requested_delta and self.arg_values["probe_path_lr_mode"] == "adaptive_global_wc":
            L_global_wc = existing_bundle.get("L_global_wc")
            if L_global_wc is None:
                raise ValueError(
                    "Adaptive diagnostics require an existing L_global_wc in the comp_params bundle "
                    "to define probe_path_lr = 1/(2*L_global_wc)."
                )
            L_global_wc = float(L_global_wc)
            if not np.isfinite(L_global_wc) or L_global_wc <= 0:
                raise ValueError(f"Loaded L_global_wc must be positive and finite, got {L_global_wc}.")

            probe_path_lr_override = 1.0 / (2.0 * L_global_wc)
            probe_path_lr_rule = "1/(2*L_global_wc)"

        # Keep scalar metadata coherent if the user asked to save them in this stage.
        if "dim" in self.comp_params_dict:
            self.comp_params_dict["dim"] = int(X_train_t.shape[1])
        if "n_groups" in self.comp_params_dict:
            self.comp_params_dict["n_groups"] = int(group_indices_t.shape[0])
        if "m_per_group" in self.comp_params_dict:
            self.comp_params_dict["m_per_group"] = int(group_indices_t.shape[1])
        if "lambda_reg" in self.comp_params_dict:
            self.comp_params_dict["lambda_reg"] = float(self.arg_values["lambda_reg"])

        requested_keys_token = "_".join(sorted(stage_requested_keys))
        safe_token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in requested_keys_token)
        diagnostics_filename = f"diagnostics_{safe_token}.json" if safe_token else "diagnostics.json"

        return self._estimate_and_save_diagnostics(
            X_train_np=X_train_t,
            y_train_np=y_train_t,
            group_indices_np=group_indices_t,
            resolved_device=resolved_device,
            probe_path_lr_override=probe_path_lr_override,
            probe_path_lr_rule=probe_path_lr_rule,
            compute_delta1_full=requested_full_delta_keys["delta1_emp_full"],
            compute_delta2_full=requested_full_delta_keys["delta2_emp_full"],
            compute_delta_flat=requested_lightweight_keys["delta_flat_emp"],
            compute_delta2_i=(
                requested_lightweight_keys["delta2_i_emp"]
                or requested_lightweight_keys["delta2_bar_sq_emp"]
            ),
            compute_deltas_override=bool(requested_generic_delta_keys),
            diagnostics_filename=diagnostics_filename,
        )

    def _canonicalize_loaded_train_tensors(self):
        """Validate the prepared synthetic tensors and move them to the minimization device."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Torch is required for prepared synthetic minimization.") from exc

        required_keys = ["X_train", "y_train"]
        missing_keys = [key for key in required_keys if key not in self.data_dict or self.data_dict[key] is None]
        if missing_keys:
            raise ValueError(
                "Prepared-data minimization requires the following loadable_datasets: "
                + ", ".join(required_keys)
                + f". Missing: {missing_keys}"
            )

        X_train = self.data_dict["X_train"]
        y_train = self.data_dict["y_train"]
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.as_tensor(X_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.as_tensor(y_train)

        if X_train.ndim != 2:
            raise ValueError(f"Expected X_train to be rank-2, got shape {tuple(X_train.shape)}.")
        if y_train.ndim != 1:
            raise ValueError(f"Expected y_train to be rank-1, got shape {tuple(y_train.shape)}.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"Prepared X_train/y_train have inconsistent lengths: {X_train.shape[0]} vs {y_train.shape[0]}."
            )

        resolved_device = self._resolve_torch_device()
        X_train_t = X_train.to(device=resolved_device, dtype=torch.float64)
        y_train_t = y_train.to(device=resolved_device, dtype=torch.int64).flatten()
        unique_labels = set(torch.unique(y_train_t).detach().cpu().tolist())
        if not unique_labels.issubset({-1, 1}):
            raise ValueError(f"Prepared synthetic labels must lie in {{-1, +1}}, got {sorted(unique_labels)}.")

        self.arg_values["dim"] = int(X_train_t.shape[1])
        self.loaded_train_tensors = {
            "X_train": X_train_t,
            "y_train": y_train_t,
            "device": resolved_device,
        }
        return X_train_t, y_train_t, resolved_device

    def _merge_existing_comp_params_bundle(self, stage_computed_keys=None, include_all_existing=False):
        """Preserve previously saved bundle contents while updating only the current stage outputs."""
        self.comp_params_path = self.data_path + 'comp_params' + self.exp_params_extension + "/"
        existing_bundle = load_comp_params_bundle(self.comp_params_path, None, self.arg_values["print_status"])

        expected_lambda = float(self.arg_values["lambda_reg"])
        loaded_lambda = existing_bundle.get("lambda_reg", self.alg_params_dict.get("lambda_reg"))
        if loaded_lambda is not None and not np.isclose(float(loaded_lambda), expected_lambda, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Loaded lambda_reg={float(loaded_lambda)} does not match the current synthetic regime value {expected_lambda}."
            )

        if include_all_existing:
            for key, loaded_value in existing_bundle.items():
                if key not in self.comp_params_dict:
                    self.comp_params_dict[key] = loaded_value.copy() if isinstance(loaded_value, np.ndarray) else loaded_value

        for key in self.comp_params_dict.keys():
            if self.comp_params_dict[key] is None and key in existing_bundle:
                loaded_value = existing_bundle[key]
                self.comp_params_dict[key] = loaded_value.copy() if isinstance(loaded_value, np.ndarray) else loaded_value

        scalar_updates = {
            "lambda_reg": expected_lambda,
            "dim": int(self.arg_values["dim"]),
            "n_groups": int(self.arg_values["n_groups"]),
            "m_per_group": int(self.arg_values["m_per_group"]),
        }
        for key, value in scalar_updates.items():
            if key in self.comp_params_dict:
                self.comp_params_dict[key] = value

        if stage_computed_keys is None:
            stage_computed_keys = {"lambda_reg", "dim", "n_groups", "m_per_group", "x_star", "f_star"}
        missing_preserved_keys = [
            key for key, value in self.comp_params_dict.items()
            if value is None and key not in stage_computed_keys
        ]
        if missing_preserved_keys:
            raise ValueError(
                "Cannot run the minimization stage because the existing comp_params bundle is missing "
                f"the preserved keys: {missing_preserved_keys}"
            )
        return existing_bundle

    def adan_plus_minimize(self, X_train_t, y_train_t, lambda_reg, x_0_t, max_iter=1_000_000, tol=1e-18, gd_init_step=1e-2, log_freq=100):
        """Run the Torch AdaN+ solver on the loaded full synthetic log-reg objective."""
        import torch

        eps = 1e-12
        if log_freq <= 0:
            raise ValueError(f"log_freq must be positive, got {log_freq}.")
        eye = torch.eye(X_train_t.shape[1], dtype=X_train_t.dtype, device=X_train_t.device)
        regularizer_type = self.arg_values["regularizer_type"]

        f_d = lambda w: torch_logreg_objective(w, X_train_t, y_train_t, lambda_reg, regularizer_type)
        grad_d = lambda w: torch_logreg_grad(w, X_train_t, y_train_t, lambda_reg, regularizer_type)
        hess_d = lambda w: torch_logreg_hess(w, X_train_t, y_train_t, lambda_reg, regularizer_type)

        # AdaN+ line 1: start from x^0 and construct x^1 via one gradient step so x^0 != x^1.
        x_km1 = x_0_t.detach().clone()
        g_km1 = grad_d(x_km1)
        x_k = x_km1 - gd_init_step * g_km1
        g_k = grad_d(x_k)
        B_km1 = hess_d(x_km1)

        # AdaN+ line 2: initialize H_0 from the residual of the local quadratic model at x^0.
        s_0 = x_k - x_km1
        r_0 = g_k - g_km1 - B_km1 @ s_0
        denom_0 = max(float(torch.dot(s_0, s_0).item()), eps)
        H_km1 = max(float(torch.linalg.vector_norm(r_0).item()) / denom_0, eps)

        k = 1
        disable_progress = not bool(self.arg_values["print_status"])
        with tqdm(total=max_iter, disable=disable_progress, miniters=log_freq) as pbar:
            while k < max_iter and float(torch.dot(g_k, g_k).item()) > tol:
                # AdaN+ line 4: local cubic residual ratio M_k.
                s_km1 = x_k - x_km1
                r_k = g_k - g_km1 - B_km1 @ s_km1
                denom_k = max(float(torch.dot(s_km1, s_km1).item()), eps)
                M_k = float(torch.linalg.vector_norm(r_k).item()) / denom_k

                # AdaN+ lines 5-6: update H_k and lambda_k from the model mismatch and current gradient norm.
                H_k = max(M_k, 0.5 * H_km1, eps)
                g_k_norm = max(float(torch.linalg.vector_norm(g_k).item()), eps)
                lambda_k = max(np.sqrt(H_k * g_k_norm), eps)

                # AdaN+ line 7: solve the damped Newton system for the next iterate.
                B_k = hess_d(x_k)
                try:
                    d_k = torch.linalg.solve(B_k + lambda_k * eye, g_k)
                except RuntimeError:
                    if not disable_progress:
                        tqdm.write("Torch solve failed in AdaN+; falling back to a scaled gradient step.")
                    d_k = g_k / lambda_k
                x_kp1 = x_k - d_k

                # Shift the two-step state used by the AdaN+ recurrence.
                x_km1, x_k = x_k, x_kp1
                g_km1, g_k = g_k, grad_d(x_k)
                B_km1 = B_k
                H_km1 = H_k

                sqnorm_grad = float(torch.dot(g_k, g_k).item())
                if not disable_progress and k % log_freq == 0:
                    pbar.set_description(f"k={k}, sqnorm(grad)={sqnorm_grad:.2e}, f={float(f_d(x_k).item()):.2e}")
                pbar.update(1)
                k += 1

        x_star_t = x_k.detach().clone()
        f_star = float(f_d(x_star_t).item())
        if not disable_progress:
            tqdm.write(f"Stopped at k={k}, sqnorm(grad_xk)={float(torch.dot(g_k, g_k).item()):e}")
        return x_star_t, f_star

    def minimize(self, X_train_t=None, y_train_t=None, x_0_t=None, max_iter=1_000_000, tol=1e-16, gd_init_step=1e-2):
        """Minimize the prepared synthetic log-reg objective from the saved per-dataset initializer."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Torch is required for prepared synthetic minimization.") from exc

        if self.arg_values["dataset"] != SYNTHETIC_DIRICHLET_LOGREG_DATASET or self.arg_values["loss_func"] != "log-reg":
            raise NotImplementedError(
                "The current preprocessing minimization path is implemented only for synthetic_dirichlet_logreg with log-reg loss."
            )

        if X_train_t is None or y_train_t is None:
            cached_tensors = getattr(self, "loaded_train_tensors", None)
            if cached_tensors is not None:
                X_train_t = cached_tensors["X_train"]
                y_train_t = cached_tensors["y_train"]
            else:
                X_train_t, y_train_t, _ = self._canonicalize_loaded_train_tensors()
        if x_0_t is None:
            if not hasattr(self, "x_0"):
                raise ValueError("x_0 is not initialized; call load_w_init() before minimize().")
            x_0_t = torch.from_numpy(np.asarray(self.x_0, dtype=np.float64)).to(
                device=X_train_t.device,
                dtype=torch.float64,
            )

        if x_0_t.ndim != 1 or x_0_t.shape[0] != X_train_t.shape[1]:
            raise ValueError(
                f"Saved w_init has shape {tuple(x_0_t.shape)}, expected ({X_train_t.shape[1]},)."
            )

        lambda_reg = float(self.arg_values["lambda_reg"])
        x_star_t, f_star = self.adan_plus_minimize(
            X_train_t,
            y_train_t,
            lambda_reg,
            x_0_t,
            max_iter=max_iter,
            tol=tol,
            gd_init_step=gd_init_step,
        )
        self.comp_params_dict["x_star"] = x_star_t.detach().cpu().numpy().astype(np.float64, copy=True)
        self.comp_params_dict["f_star"] = float(f_star)
        return self.comp_params_dict["x_star"].copy(), self.comp_params_dict["f_star"]

if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(sys.argv[1:])
    diagnostics_only_ran = False
    if data_preprocessing.arg_values["load_prepared_dataset"]==1:
        data_preprocessing.load_prepared_datasets()
    if data_preprocessing.arg_values["load_raw_dataset"]==1:
        data_preprocessing.load_raw_dataset()
    if data_preprocessing.arg_values["generate_dataset"]==1:
        data_preprocessing.generation()
    data_preprocessing.load_precomputed_params()
    if (
        data_preprocessing.arg_values["load_prepared_dataset"] == 1
        and data_preprocessing.arg_values["estimate_diagnostics"] == 1
        and data_preprocessing.arg_values["is_minimize"] == 0
    ):
        diagnostics_only_ran = True
        data_preprocessing._run_prepared_diagnostics_only()
    
    if data_preprocessing.arg_values["is_minimize"]==1:
        if data_preprocessing.arg_values["load_prepared_dataset"] != 1:
            raise ValueError("Prepared-data minimization requires --load_prepared_dataset 1.")
        data_preprocessing.load_w_init()
        data_preprocessing._canonicalize_loaded_train_tensors()
        data_preprocessing._merge_existing_comp_params_bundle(include_all_existing=True)
        data_preprocessing.minimize()
    if diagnostics_only_ran and data_preprocessing.arg_values["save_diagnostics"] == 0:
        my_print(
            "save_diagnostics=0; skipped comp-params bundle save for diagnostics-only run.",
            data_preprocessing.arg_values["print_status"],
        )
    else:
        data_preprocessing.save_comp_params()
    data_preprocessing.log_peak_memory_usage()

    
