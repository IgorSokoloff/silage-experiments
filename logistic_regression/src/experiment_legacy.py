"""
Legacy experiment-level oracle bootstrap kept outside the active base class.

The current SILAGE algorithm path builds its working Torch oracles inside the
algorithm subclasses, so the NumPy oracle/regularizer bootstrap below is only
retained for older preprocessing branches that still use ``self.oracle_dict``.
"""

from src.utils import my_print
from src.oracle_functions_legacy import *


class LegacyExperimentOracleMixin:
    """
    Provide the archived regularizer/oracle bootstrap used by legacy
    preprocessing code paths.

    Active SILAGE runs should not depend on this mixin.
    """

    def init_regularizers(self):
        self.regularizer = {
            "str-cvx": regularizer_scvx,
            "cvx": regularizer_cvx,
            "non-cvx": regularizer_noncvx,
        }[self.arg_values["regularizer_type"]]
        self.regularizer_grad = {
            "str-cvx": regularizer_scvx_grad,
            "cvx": regularizer_cvx_grad,
            "non-cvx": regularizer_noncvx_grad,
        }[self.arg_values["regularizer_type"]]
        self.regularizer_hess = {
            "str-cvx": regularizer_scvx_hess,
            "cvx": regularizer_cvx_hess,
            "non-cvx": regularizer_noncvx_hess,
        }[self.arg_values["regularizer_type"]]
        self.regularizer_hess_bound = {
            "str-cvx": regularizer_scvx_hess_bound,
            "cvx": regularizer_cvx_hess_bound,
            "non-cvx": regularizer_noncvx_hess_bound,
        }[self.arg_values["regularizer_type"]]

    def init_oracles(self):
        my_print("Defining legacy NumPy oracles...", self.arg_values["print_status"])

        # The old bootstrap owns its own regularizer setup so callers only need
        # one legacy initialization entrypoint.
        self.init_regularizers()

        self.oracle_loss = {
            "log-reg": logreg_loss_ij,
            "quartic": quartic_loss_ij,
            #"quadratic": quad_loss_ij,
            #"lin-reg": linreg_loss_ij,
            #"l1_norm": l1_norm_loss_i_distributed
        }[self.arg_values["loss_func"]]

        self.oracle_grad = {
            "log-reg": logreg_grad_ij,
            "quartic": quartic_grad_ij,
            #"quadratic": quad_grad_ij,
            #"lin-reg": linreg_grad_ij,
            #"l1_norm": l1_norm_grad_i_distributed
        }[self.arg_values["loss_func"]]

        self.oracle_minibatch_grad = {
            "log-reg": logreg_grad_ij,
            "quartic": quartic_grad_ij,
            #"quadratic": quad_grad_ij,
            #"lin-reg": linreg_minibatch_grad_ij,
            #"l1_norm": l1_norm_grad_i_distributed
        }[self.arg_values["loss_func"]]

        self.oracle_hess = {
            "log-reg": logreg_hess_ij,
            "quartic": quartic_hess_ij,
            #"quadratic": quad_hess_ij,
            #"lin-reg": linreg_hess_ij,
            #"l1_norm": None
        }[self.arg_values["loss_func"]]

        self.oracle_hess_bound = {
            "log-reg": logreg_hess_ij_bound,
            "quartic": quartic_hess_ij_bound,
            #"quadratic": quad_hess_ij,
            #"lin-reg": linreg_hess_ij_bound,
            #"l1_norm": None
        }[self.arg_values["loss_func"]]

        # These archived worker-wise helpers are kept only so older code paths
        # can still resolve them if needed.
        self.local_losses = {
            "log-reg": None,
            "quartic": None,
            "quadratic": None,
            "lin-reg": None,
            "l1_norm": l1_norm_local_losses_i_distributed,
        }[self.arg_values["loss_func"]]

        self.local_grads = {
            "log-reg": None,
            "quartic": None,
            "quadratic": quad_local_grads,
            "lin-reg": None,
            "l1_norm": l1_norm_local_grads_i_distributed,
        }[self.arg_values["loss_func"]]

        self.non_local_grads = {
            "log-reg": None,
            "quartic": None,
            "quadratic": None,
            "lin-reg": None,
            "l1_norm": l1_norm_non_local_grads_i_distributed,
        }[self.arg_values["loss_func"]]

        self.oracle_dict = {
            "f": lambda w, X, y, params: self.oracle_loss(w, X, y, params, self.regularizer),
            "grad": lambda w, X, y, params: self.oracle_grad(w, X, y, params, self.regularizer_grad),
            "hess": lambda w, X, y, params: self.oracle_hess(w, X, y, params, self.regularizer_hess),
            "hess_bound": lambda w, X, params: self.oracle_hess_bound(w, X, params, self.regularizer_hess_bound),
            #"local_losses": lambda W, X, Y, params: self.local_losses(W, X, Y, params, self.regularizer),
            #"local_grads": lambda W, X, Y, params: self.local_grads(W, X, Y, params, self.regularizer_grad),
            #"non_local_grads": lambda w, X, Y, params: self.non_local_grads(w, X, Y, params, self.regularizer_grad),
        }
