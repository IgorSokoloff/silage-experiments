from src.experiment import *
from src.synthetic_logreg import SYNTHETIC_REGIMES, SYNTHETIC_SETTING_TO_NM


def _clone_value(value):
    """Clone mutable launch state while preserving scalars by reference."""
    try:
        import torch
    except ImportError:
        torch = None

    if isinstance(value, np.ndarray):
        return value.copy()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return value


def _current_metric_value(metric, collectable_metrics_dict, alg_params_dict):
    """Read the latest runtime metric, falling back to stored metric histories."""
    runtime_key = {
        "iters": "iter",
        "epochs": "training_epochs",
        "sqnorm": "latest_sqnorm",
    }.get(metric)

    if runtime_key is not None:
        value = alg_params_dict.get(runtime_key)
        if value is not None:
            return value
        if metric == "sqnorm":
            return float("inf")

    metric_history = collectable_metrics_dict.get(metric)
    if isinstance(metric_history, list) and metric_history:
        return metric_history[-1]
    return float("inf")


def stopping_criterion(stop_criteruim_values_dict, stop_criteria_conditions_dict, collectable_metrics_dict, alg_params_dict):
    checks = []
    for metric, max_value in stop_criteruim_values_dict.items():
        cur_value = _current_metric_value(metric, collectable_metrics_dict, alg_params_dict)
        checks.append(stop_criteria_conditions_dict[metric](cur_value, max_value))
    return all(checks)


def save_data(collectable_metrics_dict, logs_path, experiment_str, is_print=0):
    """Persist final metric histories in both bundle and array form."""
    save_metrics_bundle(
        logs_path,
        collectable_metrics_dict,
        experiment_str,
        is_print=is_print,
        bundle_name=f"metrics_final_{experiment_str}.npz",
    )
    save_metrics_arrays(logs_path, collectable_metrics_dict, experiment_str, is_print=is_print)


def print_last_point_metrics(collectable_metrics_dict):
    print("---------------------------------------------")
    print("Last point metrics:")
    for metric, value in collectable_metrics_dict.items():
        if value is None:
            continue
        if isinstance(value, list):
            out_value = value[-1] if len(value) > 0 else "N/A"
        elif isinstance(value, np.ndarray) and value.ndim > 0:
            out_value = value[-1]
        else:
            out_value = value
        print(f"{metric}: {out_value}")
    print("---------------------------------------------")


def clone_state_for_launch(state_dict):
    return {key: _clone_value(value) for key, value in state_dict.items()}


def clone_alg_params_for_launch(alg_params_dict):
    return {key: _clone_value(value) for key, value in alg_params_dict.items()}


def clone_metrics_for_launch(metrics_dict):
    return {key: _clone_value(value) for key, value in metrics_dict.items()}


def run_single_launch(
    seed,
    states_dict,
    data_dict,
    stop_criteruim_values_dict,
    stop_criteria_conditions_dict,
    alg_params_dict,
    collectable_metrics_dict,
    oracle_dict,
    update,
    init_collectable_metrics_dict,
    update_collectable_metrics_dict,
    init_states_dict,
    init_oracles_dict,
    fill_alg_params_dict,
):
    """Run a single stochastic launch on an already loaded synthetic dataset."""
    alg_params_dict["rs_bernoulli"] = RandomState(seed)
    alg_params_dict["rs_sample"] = RandomState(3 * alg_params_dict["NUM_LAUNCHES"] + seed)

    oracle_dict = init_oracles_dict(states_dict, oracle_dict, data_dict, alg_params_dict)
    alg_params_dict = fill_alg_params_dict(states_dict, oracle_dict, data_dict, alg_params_dict)
    states_dict, alg_params_dict = init_states_dict(states_dict, oracle_dict, data_dict, alg_params_dict)
    collectable_metrics_dict = init_collectable_metrics_dict(
        states_dict,
        collectable_metrics_dict,
        alg_params_dict,
        oracle_dict,
        data_dict,
    )

    while stopping_criterion(stop_criteruim_values_dict, stop_criteria_conditions_dict, collectable_metrics_dict, alg_params_dict):
        states_dict, collectable_metrics_dict, alg_params_dict = update(
            states_dict,
            data_dict,
            collectable_metrics_dict,
            alg_params_dict,
            oracle_dict,
            update_collectable_metrics_dict,
        )

        cur_iter = int(alg_params_dict["iter"])
        if cur_iter % alg_params_dict["PRINT_EVERY"] == 0:
            epoch = float(alg_params_dict.get("training_epochs", 0.0))
            sqnorm = alg_params_dict.get("latest_sqnorm")
            sqnorm_str = f"{float(sqnorm):.6e}" if sqnorm is not None else "na"
            print(f"iter={cur_iter} epoch={epoch:.6f} sqnorm={sqnorm_str}")

        if cur_iter % alg_params_dict["SAVE_EVERY"] == 0 and alg_params_dict["NUM_LAUNCHES"] == 1:
            save_metrics_bundle(
                alg_params_dict["logs_path"],
                collectable_metrics_dict,
                alg_params_dict["experiment_str"],
                is_print=alg_params_dict["print_status"],
                bundle_name=f"metrics_checkpoint_{alg_params_dict['experiment_str']}.npz",
            )

    solution = states_dict["x"]
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(solution, torch.Tensor):
        solution = solution.detach().cpu().numpy().astype(np.float64, copy=True)
    elif isinstance(solution, np.ndarray):
        solution = solution.copy()
    collectable_metrics_dict["solution"] = solution
    return collectable_metrics_dict


def run_algorithm(
    states_dict,
    data_dict,
    stop_criteruim_values_dict,
    stop_criteria_conditions_dict,
    alg_params_dict,
    collectable_metrics_dict,
    oracle_dict,
    update,
    init_collectable_metrics_dict,
    update_collectable_metrics_dict,
    init_states_dict,
    init_oracles_dict,
    fill_alg_params_dict,
):
    """Run the active no-Ray algorithm path and aggregate launches when needed."""
    if alg_params_dict["use_ray"]:
        raise NotImplementedError("Ray launches are not supported in the current synthetic algorithms path.")

    collectable_metrics_dicts_ar = []
    for seed in range(alg_params_dict["NUM_LAUNCHES"]):
        metrics_dict = run_single_launch(
            seed + alg_params_dict["seed"],
            clone_state_for_launch(states_dict),
            data_dict,
            stop_criteruim_values_dict,
            stop_criteria_conditions_dict,
            clone_alg_params_for_launch(alg_params_dict),
            clone_metrics_for_launch(collectable_metrics_dict),
            oracle_dict,
            update,
            init_collectable_metrics_dict,
            update_collectable_metrics_dict,
            init_states_dict,
            init_oracles_dict,
            fill_alg_params_dict,
        )
        collectable_metrics_dicts_ar.append(metrics_dict)

    if alg_params_dict["NUM_LAUNCHES"] == 1:
        collectable_metrics_dict = collectable_metrics_dicts_ar[0]
        save_data(collectable_metrics_dict, alg_params_dict["logs_path"], alg_params_dict["experiment_str"], alg_params_dict["print_status"])
        print("End-point:")
        print_last_point_metrics(collectable_metrics_dict)
        return collectable_metrics_dict

    avg_keys = set(ALLOWABLE_AVG_KEYS) & set(collectable_metrics_dict.keys())
    for key in avg_keys:
        values = [d[key] for d in collectable_metrics_dicts_ar]
        min_len = min(len(arr) for arr in values)
        truncated_values = [arr[:min_len] for arr in values]
        for i in range(alg_params_dict["NUM_LAUNCHES"]):
            collectable_metrics_dict[f"{key}_{i}"] = truncated_values[i]
        collectable_metrics_dict[f"{key}_mean"] = np.mean(truncated_values, axis=0)
        collectable_metrics_dict[f"{key}_median"] = np.median(truncated_values, axis=0)
        collectable_metrics_dict[f"{key}_std"] = np.std(truncated_values, axis=0)

    copy_keys = set(ALLOWABLE_NON_AVG_KEYS) & set(collectable_metrics_dict.keys())
    for key in copy_keys:
        values = [d[key] for d in collectable_metrics_dicts_ar]
        collectable_metrics_dict[key] = values[0]

    collectable_metrics_dict["solution"] = collectable_metrics_dicts_ar[0].get("solution")

    for key in avg_keys:
        collectable_metrics_dict.pop(key, None)

    save_data(collectable_metrics_dict, alg_params_dict["logs_path"], alg_params_dict["experiment_str"], alg_params_dict["print_status"])
    print("End-point:")
    print_last_point_metrics(collectable_metrics_dict)
    return collectable_metrics_dict


class Algorithm(Experiment):
    def __init__(self, args=None):
        self.arg_values = vars(self.parse_args(args))
        print("Input arguments:")
        for key, value in self.arg_values.items():
            print(f"{key}: {value}")

        self.general_arg_asserts()
        self.algorithm_launching_arg_asserts()
        self.project_specific_asserts()
        print_time(self.arg_values["print_status"])

        self.init_alg_launch_dicts()
        self.init_data_dict()
        self.init_load_params_dict()
        self.init_alg_params_dict()

        self.init_exp_param_extension()
        self.init_exp_data_extension()
        self.init_w_init_extension()
        self.init_exp_name_extension()
        self.init_paths_and_folders()

        self.load_prepared_datasets()
        self.load_w_init()
        self.load_parameters()
        # Active algorithms build their working Torch helpers inside the
        # subclass-specific init_oracles_dict(...) hook. We keep oracle_dict as an
        # interface placeholder here so the launch plumbing stays unchanged
        # without depending on the legacy Experiment oracle bootstrap.
        self.oracle_dict = {}
        self.fill_alg_params_dict_and_states()

    def script_directory(self):
        raise NotImplementedError("Algorithm subclasses must define script_directory().")

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser(description='Load prepared data and run stochastic algorithms.')

        parser.add_argument('--alg_name', action='store', dest='alg_name', type=str, default="", help='The name of the algorithm we run')
        parser.add_argument('--exp_name', action='store', dest='exp_name', type=str, default="", help='The name of the experiment')
        parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms', help='The name of the dataset')
        parser.add_argument('--loss_func', action='store', dest='loss_func', type=str, default='log-reg', help='Supported loss function')
        parser.add_argument('--regularizer_type', action='store', dest='regularizer_type', type=str, default='non-cvx', help='str-cvx, cvx or non-cvx')
        parser.add_argument('--lambda_reg', action='store', dest='lambda_reg', type=float, default=1e-3, help='Regularization coefficient')

        parser.add_argument('--is_grad_comp_init', action='store', dest='is_grad_comp_init', type=int, default=1, help='Whether to initialize with full local gradients')

        parser.add_argument('--max_epochs', action='store', dest='max_epochs', type=float, default=10.0, help='Maximum number of epochs')
        parser.add_argument('--max_bits', action='store', dest='max_bits', type=int, default=100, help='Maximum number of transmitted bits')
        parser.add_argument('--max_comms', action='store', dest='max_comms', type=int, default=100, help='Maximum number of communication rounds')
        parser.add_argument('--max_iters', action='store', dest='max_iters', type=int, default=100, help='Maximum number of iterations')
        parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-7, help='Tolerance for gradient-norm stopping')
        parser.add_argument('--stop_criteruim_params', action='store', dest='stop_criteruim_params', type=str, default="['iters']", help='List of stopping criteriums')

        parser.add_argument('--computable_params', action='store', dest='comp_params_str', type=str, default="[]", help='List of computable params')
        parser.add_argument('--collectable_metrics', action='store', dest='collectable_metrics', type=str, default="['iters','epochs','sqnorm']", help='Metrics collected during optimization')
        parser.add_argument('--loadable_params', action='store', dest='loadable_params', type=str, default="['lambda_reg','delta2_emp','L_global_emp']", help='List of params loaded from preprocessing outputs')
        parser.add_argument('--loadable_datasets', action='store', dest='loadable_datasets', type=str, default="['X_train','y_train','X_groups_train','y_groups_train']", help='Prepared tensors loaded from disk')

        parser.add_argument('--factor', action='store', dest='factor', type=float, default=1.0, help='Constant-step-size tuning factor')
        parser.add_argument('--PRINT_EVERY', action='store', dest='PRINT_EVERY', type=int, default=1000, help='How often to print metrics')
        parser.add_argument('--SAVE_EVERY', action='store', dest='SAVE_EVERY', type=int, default=1000, help='How often to checkpoint metrics')
        parser.add_argument('--COLLECT_EVERY', action='store', dest='COLLECT_EVERY', type=int, default=1000, help='How often to evaluate and store metrics')
        parser.add_argument('--NUM_LAUNCHES', action='store', dest='NUM_LAUNCHES', type=int, default=1, help='Number of sequential stochastic launches')
        parser.add_argument('--print_status', action='store', dest='print_status', type=int, default=1, help='Whether to print status updates')
        parser.add_argument('--use_ray', action='store', dest='use_ray', type=int, default=0, help='Legacy flag; current synthetic path uses sequential launches only')
        parser.add_argument('--seed', action='store', dest='seed', type=int, default=1, help='Global random seed')
        parser.add_argument('--seeds', action='store', dest='seeds', nargs='+', type=int, default=[1], help='Legacy list of seeds')

        parser.add_argument('--device', action='store', dest='device', type=str, default='cpu', help='cpu|cuda|cuda:N')
        parser.add_argument('--synthetic_setting', action='store', dest='synthetic_setting', type=str, default='m_gt_n', choices=sorted(SYNTHETIC_SETTING_TO_NM.keys()), help='Synthetic grouped setting')
        parser.add_argument('--synthetic_regime', action='store', dest='synthetic_regime', type=str, default='d1_small_d2_small', choices=sorted(SYNTHETIC_REGIMES), help='Synthetic beta_i regime')
        parser.add_argument('--d', action='store', dest='synthetic_d', type=int, default=1000, help='Synthetic feature dimension d')
        parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, default=1, help='Constant batch size used by active minibatch algorithms')
        parser.add_argument('--client_subset_size', action='store', dest='client_subset_size', type=int, default=1, help='Active client subset size used by D-ZeroSARAH')
        parser.add_argument('--n_groups', action='store', dest='n_groups', type=int, default=None, help='Optional explicit number of groups')
        parser.add_argument('--m_per_group', action='store', dest='m_per_group', type=int, default=None, help='Optional explicit group size')

        # Legacy-only parser args are intentionally excluded from the active
        # project path so they do not pollute self.arg_values / self.alg_params_dict.

        return parser.parse_args(args)

    def project_specific_asserts(self):
        if self.arg_values["loss_func"] == "log-reg":
            self.logreg_asserts()
            return
        raise NotImplementedError(
            "Only the synthetic log-reg algorithm path is active in the current project. "
            "Archived compatibility helpers live under src/."
        )

    def logreg_asserts(self):
        assert self.arg_values["loss_func"] == "log-reg"
        assert self.arg_values["dataset"] in LOGREG_DATASETS
        if self.arg_values["dataset"] == SYNTHETIC_DIRICHLET_LOGREG_DATASET:
            assert self.arg_values["synthetic_setting"] in SYNTHETIC_SETTING_TO_NM
            assert self.arg_values["synthetic_regime"] in SYNTHETIC_REGIMES
            assert self.arg_values["synthetic_d"] > 0

    def algorithm_launching_arg_asserts(self):
        assert self.arg_values["is_grad_comp_init"] in [0, 1]
        assert isinstance(self.arg_values['tol'], float) and self.arg_values['tol'] > 0
        assert isinstance(self.arg_values['PRINT_EVERY'], int) and self.arg_values['PRINT_EVERY'] > 0
        assert isinstance(self.arg_values['SAVE_EVERY'], int) and self.arg_values['SAVE_EVERY'] > 0
        assert isinstance(self.arg_values['COLLECT_EVERY'], int) and self.arg_values['COLLECT_EVERY'] > 0
        assert isinstance(self.arg_values['NUM_LAUNCHES'], int) and self.arg_values['NUM_LAUNCHES'] > 0
        assert isinstance(self.arg_values['seed'], int) and self.arg_values['seed'] > 0
        assert isinstance(self.arg_values['batch_size'], int) and self.arg_values['batch_size'] > 0
        assert isinstance(self.arg_values['client_subset_size'], int) and self.arg_values['client_subset_size'] > 0
        assert all(isinstance(seed, int) and seed > 0 for seed in self.arg_values['seeds'])
        assert self.arg_values["use_ray"] in [0, 1]
        if self.arg_values["use_ray"]:
            raise NotImplementedError("Ray is not supported in the active synthetic algorithms path.")

        stop_criteria_list = ast.literal_eval(self.arg_values['stop_criteruim_params'])
        if 'epochs' in stop_criteria_list:
            assert isinstance(self.arg_values['max_epochs'], (int, float)) and self.arg_values['max_epochs'] > 0
        if 'bits' in stop_criteria_list:
            assert isinstance(self.arg_values['max_bits'], int) and self.arg_values['max_bits'] > 0
        if 'comms' in stop_criteria_list:
            assert isinstance(self.arg_values['max_comms'], int) and self.arg_values['max_comms'] > 0
        if 'iters' in stop_criteria_list:
            assert isinstance(self.arg_values['max_iters'], int) and self.arg_values['max_iters'] > 0

    def general_arg_asserts(self):
        print("Checking input arguments...")
        assert isinstance(self.arg_values['exp_name'], str) and self.arg_values['exp_name'] != ""
        assert isinstance(self.arg_values['alg_name'], str) and self.arg_values['alg_name'] != ""
        assert self.arg_values['exp_name'] in ALLOWABLE_EXPERIMENTS
        assert self.arg_values['alg_name'] in ALLOWABLE_ALGORITHMS

        assert isinstance(self.arg_values['dataset'], str) and self.arg_values['dataset'] != ""
        assert self.arg_values['loss_func'] in SUPPORTED_LOSS_FUNCS
        assert self.arg_values['regularizer_type'] in SUPPORTED_REGULARIZERS
        assert isinstance(self.arg_values['factor'], float) and self.arg_values['factor'] > 0
        assert self.arg_values["print_status"] in [0, 1]

        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['stop_criteruim_params']))
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['loadable_params']))
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['collectable_metrics']))
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['loadable_datasets']))
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['comp_params_str']))

    def init_data_dict(self):
        self.data_dict = parse_params_to_dict(self.arg_values['loadable_datasets'], ALLOWABLE_DATASETS)
        assert set(self.data_dict.keys()).issubset(ALLOWABLE_DATASETS)

    def init_alg_launch_dicts(self):
        self.stop_criteria_values_fed = dict(
            zip(
                ALLOWABLE_STOP_CRITERIA,
                [
                    self.arg_values['max_epochs'],
                    self.arg_values['max_bits'],
                    self.arg_values['max_comms'],
                    self.arg_values['max_iters'],
                    self.arg_values['tol'],
                    self.arg_values['tol'],
                    self.arg_values['tol'],
                ],
            )
        )
        self.stop_criteria_values_dict = parse_params_to_dict(self.arg_values['stop_criteruim_params'], ALLOWABLE_STOP_CRITERIA)
        self.stop_criteria_conditions_dict = {
            key: ALLOWABLE_STOP_CRITERIA_CONDITIONS[key] for key in self.stop_criteria_values_dict.keys()
        }
        self.collectable_metrics_dict = parse_params_to_dict(self.arg_values['collectable_metrics'], ALLOWABLE_COLLECTABLE_METRICS)
        assert set(self.stop_criteria_values_dict.keys()).issubset(ALLOWABLE_STOP_CRITERIA)
        assert set(self.collectable_metrics_dict.keys()).issubset(ALLOWABLE_COLLECTABLE_METRICS)
        for stop_crit in self.stop_criteria_values_dict.keys():
            self.stop_criteria_values_dict[stop_crit] = self.stop_criteria_values_fed[stop_crit]

    def init_paths_and_folders(self):
        script_directory = self.script_directory()
        self.project_path = script_directory + "/"
        self.data_path = self.project_path + "data_{0}/".format(self.arg_values['dataset'])
        self.init_dataset_path()

        factor_suffix = f"_f{myrepr(float(self.arg_values['factor']))}"
        self.experiment_str = self.arg_values['exp_name'] + self.exp_name_extension + factor_suffix
        self.logs_path = self.project_path + "logs/logs_{0}/".format(self.experiment_str)

        if not os.path.exists(self.project_path + "logs/"):
            os.makedirs(self.project_path + "logs/")
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

    def fill_alg_params_dict_and_states(self):
        # self.alg_params_dict["step_size_init"] = self.arg_values["step_size_init"]
        self.alg_params_dict["max_iters"] = int(self.arg_values["max_iters"])
        self.alg_params_dict["tol"] = self.arg_values["tol"]

        lambda_reg = self.alg_params_dict.get("lambda_reg", self.arg_values["lambda_reg"])
        if isinstance(lambda_reg, np.ndarray):
            lambda_reg = float(lambda_reg.reshape(-1)[0])
        self.alg_params_dict["lambda_reg"] = float(lambda_reg)
        self.alg_params_dict["oracle_params"] = {"lambda_reg": self.alg_params_dict["lambda_reg"]}
        self.alg_params_dict["dim"] = int(self.x_0.shape[0])

        self.alg_params_dict["exp_name"] = self.arg_values["exp_name"]
        self.alg_params_dict["alg_name"] = self.arg_values["alg_name"]
        self.alg_params_dict["factor"] = self.arg_values["factor"]
        self.alg_params_dict["print_status"] = self.arg_values['print_status']
        self.alg_params_dict["NUM_LAUNCHES"] = self.arg_values['NUM_LAUNCHES']
        self.alg_params_dict["PRINT_EVERY"] = self.arg_values['PRINT_EVERY']
        self.alg_params_dict["SAVE_EVERY"] = self.arg_values['SAVE_EVERY']
        self.alg_params_dict["COLLECT_EVERY"] = self.arg_values['COLLECT_EVERY']
        self.alg_params_dict["use_ray"] = self.arg_values['use_ray']
        self.alg_params_dict["seed"] = self.arg_values['seed']
        self.alg_params_dict["seeds"] = self.arg_values['seeds']
        self.alg_params_dict["logs_path"] = self.logs_path
        self.alg_params_dict["experiment_str"] = self.experiment_str
        self.alg_params_dict["data_path"] = self.data_path
        self.alg_params_dict["dataset_path"] = self.dataset_path
        self.alg_params_dict["project_path"] = self.project_path
        self.alg_params_dict["device"] = self.arg_values["device"]
        self.alg_params_dict["batch_size"] = int(self.arg_values["batch_size"])
        self.alg_params_dict["client_subset_size"] = int(self.arg_values["client_subset_size"])
        # Preserve legacy parser values in alg_params_dict so older algorithm
        # files remain loadable even though the active path no longer depends on them.
        for legacy_key in [
            "sampling",
            "is_sparse_dataset",
            "is_replacement",
            "cond_number",
            "beta1",
            "beta2",
            "dim",
            "prob",
            "num_samples",
            "rank",
            "noise_scale",
            "batchsize",
            "momentum",
            "compressor",
            "qC",
        ]:
            if legacy_key in self.arg_values:
                self.alg_params_dict[legacy_key] = self.arg_values[legacy_key]
                
        self.alg_params_dict["iter"] = 0
        self.alg_params_dict["training_epochs"] = 0.0
        self.alg_params_dict["latest_sqnorm"] = None
        self.alg_params_dict["last_sqnorm_iter"] = 0
        self.states_dict = {"x": self.x_0.copy(), "x_prev": None, "g": None}

    def run(self):
        print('------------------------------------------------------------------------------------------------')
        start_time = time.time()
        print("Running algorithm...")

        run_algorithm(
            self.states_dict,
            self.data_dict,
            self.stop_criteria_values_dict,
            self.stop_criteria_conditions_dict,
            self.alg_params_dict,
            self.collectable_metrics_dict,
            self.oracle_dict,
            self.update,
            self.init_collectable_metrics_dict,
            self.update_collectable_metrics_dict,
            self.init_states_dict,
            self.init_oracles_dict,
            self.fill_alg_params_dict,
        )
        print("Experiment finished.")
        time_diff = time.time() - start_time
        print(f"Computation time: {time_diff} sec")
        peak_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Peak Memory Usage: {peak_memory_usage / 1024} MB")
        print('------------------------------------------------------------------------------------------------')
