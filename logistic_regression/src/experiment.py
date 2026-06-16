from src.utils import *
from src.oracle_functions import *
from src.synthetic_logreg import (
    SYNTHETIC_DIRICHLET_LOGREG_DATASET,
    synthetic_logreg_extension_suffix,
    synthetic_logreg_regime_params,
    synthetic_logreg_setting_to_nm,
)

###################################################
######### Global project-dependend params #########
###################################################



ALLOWABLE_EXPERIMENTS = ['SILAGE']

ALLOWABLE_ALGORITHMS = ['SILAGE_m>n', 'SILAGE_n>m', 'ZeroSARAH', 'D-ZeroSARAH', 'SILVER']
ALLOWABLE_SAMPLINGS = ['full']

ALLOWABLE_COMPRESSORS = ['sameRandK', 'indRandK', 'PermK', 'TopK', 'I', 'sameRandK-TopK', 'indRandK-TopK', 'PermRandK-TopK']
ALLOWABLE_STOP_CRITERIA = ['epochs', 'bits', 'comms', 'iters', 'arg_res', 'func_diff', 'sqnorm']

ALLOWABLE_COLLECTABLE_METRICS = ['epochs', 'bits', 'comms', 'iters', 'arg_res', 'func_diff', 'sqnorm']


#ALLOWABLE_COLLECTABLE_METRICS += [item+"_grad_comp" for item in ALLOWABLE_COLLECTABLE_METRICS] # when full grad is computed  
ALLOWABLE_AVG_KEYS = ['func_diff', 'arg_res', 'sqnorm']


#ALLOWABLE_AVG_KEYS += [item+"_grad_comp" for item in ALLOWABLE_AVG_KEYS] # when full grad is computed
ALLOWABLE_NON_AVG_KEYS = list(set(ALLOWABLE_COLLECTABLE_METRICS) - set(ALLOWABLE_AVG_KEYS))

ALLOWABLE_DATASETS = set([
    'X_0', 'y_0',
    'X_train', 'y_train', 'X_test', 'y_test',
    'X_groups_train', 'y_groups_train',
    'group_indices_train',
])


ALLOWABLE_STOP_CRITERIA_CONDITIONS = {'epochs': lambda cur_epoch_number, max_epochs: cur_epoch_number < max_epochs,
                                    'bits': lambda cur_bits_number, max_bits: cur_bits_number < max_bits,
                                    'comms': lambda cur_comms_number, max_comms: cur_comms_number < max_comms,
                                    'iters': lambda cur_iters_number, max_iters: cur_iters_number < max_iters,
                                    'arg_res': lambda cur_arg_res, tol: cur_arg_res > tol,
                                    'func_diff': lambda cur_func_diff, tol: cur_func_diff > tol,
                                    'sqnorm': lambda cur_sqnorm, tol: cur_sqnorm > tol
                                    }
#'L_0,i', 'wtL_0', 'L_pm', 'L_0,pm', 

ALLOWABLE_PARAMS = set([
    'L_0', 'lambda_reg', 'x_star', 'f_star', 'dim', 'n_groups', 'm_per_group',
    'optimal_b',
    'optimal_b_silver',
    'delta_flat_emp', 'delta_flat_emp_full',
    'delta1_emp_full', 'delta2_emp_full',
    'delta2_i_emp', 'delta2_bar_sq_emp',
    'delta2_i_emp_full', 'delta2_bar_sq_emp_full',
    'delta1_emp', 'delta2_emp', 'delta1_wc', 'delta2_wc',
    'L_ij_max_emp', 'L_i_max_emp', 'L_global_emp',
    'L_ij_max_wc', 'L_i_max_wc', 'L_global_wc',
])


LIBSVM_CLASSIFICATION_DATASETS = set([
    'a9a', 'w8a', 'mushrooms', 'ijcnn1', 'covtype', 'phishing', 'rcv1',
    'real-sim', 'news20.binary', 'cod-rna', 'dna', 'svmguide3',
    'svmguide1', 'svmguide2', 'splice', 'madelon', 'gisette', 'dexter',
    'dorothea', 'colon-cancer', 'leukemia', 'lung-cancer', 'rcv1.binary',
    'sector', 'usps', 'mnist'
])

SUPPORTED_LOSS_FUNCS = set([
    'log-reg',
    'quadratic',
    'l1_norm',
    'lin-reg',
    'quartic'
])

SUPPORTED_REGULARIZERS = ['str-cvx', 'cvx', 'non-cvx']

# Specific datasets for archived non-logistic-regression experiments
QUADRATIC_DATASETS = set(['synthetic_dense', 'synthetic_sparse', 'synthetic_sparse_zero'])
L1_NORM_DATASETS = set(['synthetic_dense', 'synthetic_sparse', 'synthetic_sparse_zero'])
LINREG_DATASETS = set(['synthetic_dense'])
QUARTIC_DATASETS = set(['synthetic'])

LOGREG_DATASETS = LIBSVM_CLASSIFICATION_DATASETS.copy() | {'mnist01', SYNTHETIC_DIRICHLET_LOGREG_DATASET}

ALLOWABLE_PLOT_FAMILIES = ['ALL', 'SINGLE_RELEASE']


# Ray parameter
NUM_CORES = 48



####################################################################################
class Experiment():
    def __init__(self):
        pass

    def load_prepared_datasets(self):
        try:
            import torch
        except ImportError:
            torch = None

        for data_part_name in self.data_dict.keys():
            pt_path = os.path.join(self.dataset_path, f"{data_part_name}.pt")
            if os.path.exists(pt_path):
                if torch is None:
                    raise ImportError(f"Found {pt_path} but torch is not installed.")
                self.data_dict[data_part_name] = torch.load(pt_path, map_location="cpu")
                my_print(f"Loaded {data_part_name} from {pt_path}", self.arg_values["print_status"])
                continue

            path = self.dataset_path + data_part_name + self.exp_data_extension
            self.data_dict[data_part_name] = load_param(path, data_part_name, self.arg_values["print_status"])
    
    def get_part_dataset(self, data_part_name, inds):
        # If there are bugs, see implementation in the source code for the hfh project
        path = self.dataset_path + data_part_name + self.exp_data_extension
        return load_selected_sparse_matrices(path, data_part_name, inds, self.arg_values["print_status"])
        
    def load_w_init(self):
        w_init_path = self.data_path + 'w_init' + self.w_init_extension + '.npy'
        if os.path.exists(w_init_path):
            self.x_0 = np.array(np.load(w_init_path), dtype=np.float64)
            return

        if self.arg_values["dataset"] == SYNTHETIC_DIRICHLET_LOGREG_DATASET:
            legacy_extension = f'_{self.arg_values["loss_func"]}_{self.arg_values["dataset"]}'
            legacy_w_init_path = self.data_path + 'w_init' + legacy_extension + '.npy'
            if os.path.exists(legacy_w_init_path):
                my_print(
                    "Loaded legacy shared synthetic w_init; regenerate preprocessing to save a per-dataset initializer.",
                    self.arg_values["print_status"],
                )
                self.x_0 = np.array(np.load(legacy_w_init_path), dtype=np.float64)
                return

        raise FileNotFoundError(w_init_path)

    def load_parameters(self):
        self.comp_params_path = self.data_path + 'comp_params' + self.exp_params_extension + "/"
        requested_params = list(self.alg_params_dict.keys())
        if not requested_params:
            return

        # New format: single NPZ bundle.
        loaded_bundle = load_comp_params_bundle(self.comp_params_path, requested_params, self.arg_values["print_status"])

        # Legacy fallback: per-parameter files.
        for param in requested_params:
            if param in loaded_bundle:
                self.alg_params_dict[param] = loaded_bundle[param]
                continue
            param_path = self.comp_params_path + param + self.exp_params_extension
            try:
                self.alg_params_dict[param] = load_param(param_path, param, self.arg_values["print_status"])
            except FileNotFoundError:
                if param == "lambda_reg":
                    legacy_param_path = self.comp_params_path + "la" + self.exp_params_extension
                    try:
                        self.alg_params_dict[param] = load_param(legacy_param_path, "la", self.arg_values["print_status"])
                        my_print("Loaded legacy 'la' and mapped it to 'lambda_reg'.", self.arg_values["print_status"])
                    except FileNotFoundError:
                        if "lambda_reg" in self.arg_values:
                            self.alg_params_dict[param] = float(self.arg_values["lambda_reg"])
                            my_print("lambda_reg not found in comp_params; using CLI/default value.", self.arg_values["print_status"])
                        else:
                            raise
                else:
                    raise
    
    def init_comp_params_dict(self):
        try: 
            comp_params_list = ast.literal_eval(self.arg_values["comp_params_str"])
        except ValueError:
            print("The string is not a valid list representation.")
        
        if isinstance(comp_params_list, list) and all(isinstance(item, str) for item in comp_params_list):
            self.comp_params_dict = {key: None for key in comp_params_list}
        else:
            print("The list does not contain only string elements.")

        self.comp_params_set = set(comp_params_list)
        assert(self.comp_params_set.issubset(ALLOWABLE_PARAMS))        

    def _require_arg_value(self, key):
        """Return a required argument value or raise a clear error if it is missing."""
        value = self.arg_values.get(key)
        if value is None:
            raise ValueError(f"Missing required argument '{key}' for this code path.")
        return value
    
    # Project dependent functions
    def init_exp_param_extension(self): 
        loss_func = self.arg_values["loss_func"]
        dataset = self.arg_values["dataset"]
        # Synthetic grouped data uses a dedicated suffix because there is no MNIST-style partition regime/repr.
        if dataset == SYNTHETIC_DIRICHLET_LOGREG_DATASET:
            # Synthetic suffixes now depend only on stable dataset identity fields.
            synthetic_setting = self._require_arg_value("synthetic_setting")
            synthetic_regime = self._require_arg_value("synthetic_regime")
            n_groups = self.arg_values.get("n_groups")
            m_per_group = self.arg_values.get("m_per_group")
            if n_groups in (None, "na") or m_per_group in (None, "na"):
                n_groups, m_per_group = synthetic_logreg_setting_to_nm(synthetic_setting)
            regime_params = synthetic_logreg_regime_params(
                synthetic_setting,
                synthetic_regime,
            )
            self.exp_params_extension = (
                f'_{loss_func}_{dataset}'
                + synthetic_logreg_extension_suffix(
                    setting=synthetic_setting,
                    regime=synthetic_regime,
                    n_groups=int(n_groups),
                    m_per_group=int(m_per_group),
                    d=int(self.arg_values.get("synthetic_d", self.arg_values.get("dim", 1000))),
                    K=int(regime_params["K"]),
                    T=int(regime_params["T"]),
                )
            )
            return
        partition_regime = self.arg_values.get("partition_regime", "none")
        n_groups = self.arg_values.get("n_groups", "na")
        m_per_group = self.arg_values.get("m_per_group", "na")
        repr_mode = self.arg_values.get("cluster_repr", self.arg_values.get("repr", "raw"))
        self.exp_params_extension = (
            f'_{loss_func}_{dataset}'
            f'_pr-{partition_regime}_n{n_groups}_m{m_per_group}_repr-{repr_mode}'
        )

    # Project dependent functions
    def init_exp_data_extension(self):    
        loss_func = self.arg_values["loss_func"]
        dataset = self.arg_values["dataset"]
        # Synthetic grouped data uses a dedicated suffix because there is no MNIST-style partition regime/repr.
        if dataset == SYNTHETIC_DIRICHLET_LOGREG_DATASET:
            # Synthetic suffixes now depend only on stable dataset identity fields.
            synthetic_setting = self._require_arg_value("synthetic_setting")
            synthetic_regime = self._require_arg_value("synthetic_regime")
            n_groups = self.arg_values.get("n_groups")
            m_per_group = self.arg_values.get("m_per_group")
            if n_groups is None or m_per_group is None:
                n_groups, m_per_group = synthetic_logreg_setting_to_nm(synthetic_setting)
            regime_params = synthetic_logreg_regime_params(
                synthetic_setting,
                synthetic_regime,
            )
            self.exp_data_extension = (
                f'_{loss_func}_{dataset}'
                + synthetic_logreg_extension_suffix(
                    setting=synthetic_setting,
                    regime=synthetic_regime,
                    n_groups=int(n_groups),
                    m_per_group=int(m_per_group),
                    d=int(self.arg_values.get("synthetic_d", self.arg_values.get("dim", 1000))),
                    K=int(regime_params["K"]),
                    T=int(regime_params["T"]),
                )
            )
            return
        partition_regime = self.arg_values.get("partition_regime")
        n_groups = self.arg_values.get("n_groups")
        m_per_group = self.arg_values.get("m_per_group")
        repr_mode = self.arg_values.get("cluster_repr", self.arg_values.get("repr"))
        self.exp_data_extension = f'_{loss_func}_{dataset}'
        if partition_regime is not None and n_groups is not None and m_per_group is not None and repr_mode is not None:
            self.exp_data_extension += f'_pr-{partition_regime}_n{n_groups}_m{m_per_group}_repr-{repr_mode}'

    # Project dependent functions
    def init_w_init_extension(self):      
        dataset = self.arg_values["dataset"]
        if dataset == SYNTHETIC_DIRICHLET_LOGREG_DATASET:
            # Keep one initializer per saved synthetic dataset by keying it off
            # the same fully resolved suffix as the dataset folder itself.
            self.w_init_extension = self.exp_data_extension
            return

        loss_func = self.arg_values["loss_func"]
        self.w_init_extension = f'_{loss_func}_{dataset}'

    #Project dependend functions
    def init_exp_name_extension(self):
        exp_name = self.arg_values.get('exp_name', 'SILAGE') or 'SILAGE'
        alg_name = self.arg_values.get('alg_name', '')
        if exp_name not in ALLOWABLE_EXPERIMENTS:
            raise ValueError("other options are not supported")
        if alg_name and alg_name not in ALLOWABLE_ALGORITHMS:
            raise ValueError("algorithm is not supported for selected experiment")
        alg_suffix = alg_name if alg_name else ALLOWABLE_ALGORITHMS[0]
        batch_mode_suffix = ""
        # Batch-sensitive algorithms must encode their oracle budget so runs with
        # different sampling costs do not collide in logs / metric folders.
        if alg_suffix == "SILAGE_n>m":
            uses_optimal_b = hasattr(self, "loadable_params_set") and "optimal_b" in self.loadable_params_set
            if uses_optimal_b:
                batch_mode_suffix = "_optimal_b"
            else:
                batch_size = self.arg_values.get("batch_size")
                if batch_size is None:
                    raise ValueError("batch_size must be provided when optimal_b is not used")
                batch_mode_suffix = f"_b{int(batch_size)}"
        elif alg_suffix == "ZeroSARAH":
            batch_size = self.arg_values.get("batch_size")
            if batch_size is None:
                raise ValueError("batch_size must be provided for ZeroSARAH experiments")
            batch_mode_suffix = f"_b{int(batch_size)}"
        elif alg_suffix == "SILVER":
            batch_size = self.arg_values.get("batch_size")
            if batch_size is None:
                raise ValueError("batch_size must be provided for SILVER experiments")
            batch_mode_suffix = f"_b{int(batch_size)}"
        elif alg_suffix == "D-ZeroSARAH":
            batch_size = self.arg_values.get("batch_size")
            client_subset_size = self.arg_values.get("client_subset_size")
            if batch_size is None or client_subset_size is None:
                raise ValueError("batch_size and client_subset_size must be provided for D-ZeroSARAH experiments")
            batch_mode_suffix = f"_s{int(client_subset_size)}_b{int(batch_size)}"
        self.exp_name_extension = self.exp_params_extension + f"_{alg_suffix}{batch_mode_suffix}"
    
    #Project dependend functions
    def init_dataset_path(self):
        self.dataset_path = self.data_path + 'data' + self.exp_data_extension + "/"
    
    #Project dependend functions
    #legacy code from previous projects
    def extract_str_from_param(self, str):
        str_list = str_filter(extract_str_multiple(self.alg_params_dict.keys(), [str, "_"+self.arg_values['exp_name']]), "_func_opt")
        assert len(str_list)>0
        
        if len(str_list)==1:
            extracted_str = str_list[0]
        else:
            if self.arg_values['sampling'] == "NICE":
                str_list = str_filter(str_list, "imp")
            elif "imp" in self.arg_values['sampling']:
                str_list = str_filter(str_list, "NICE")
            extracted_str = str_list[0]
        assert len(str_list)==1
        return extracted_str
    
    def init_alg_params_dict(self):
        self.alg_params_dict = parse_params_to_dict(self.arg_values['loadable_params'], ALLOWABLE_PARAMS)
        assert set(self.alg_params_dict.keys()).issubset(ALLOWABLE_PARAMS)
        
    def init_load_params_dict(self):
        try: 
            load_params_list = ast.literal_eval(self.arg_values["loadable_params"])
        except ValueError:
            print("The string is not a valid list representation.")
        
        if isinstance(load_params_list, list) and all(isinstance(item, str) for item in load_params_list):
            self.load_params_dict = {key: None for key in load_params_list}
        else:
            print("The list does not contain only string elements.")
        
        self.loadable_params_list = load_params_list
        self.loadable_params_set = set(load_params_list)
        assert(self.loadable_params_set.issubset(ALLOWABLE_PARAMS))
        
    def save_comp_params(self):
        self.comp_params_path = self.data_path + 'comp_params' + self.exp_params_extension + "/"
        if not os.path.exists(self.comp_params_path):
            os.makedirs(self.comp_params_path)
        save_comp_params_bundle(self.comp_params_path, self.comp_params_dict, self.arg_values["print_status"])
            
    def log_peak_memory_usage(self):
        peak_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        my_print(f"Peak Memory Usage: {peak_memory_usage / 1024} MB", self.arg_values["print_status"])
