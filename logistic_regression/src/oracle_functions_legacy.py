from src.utils import *

###########################################
## Legacy NumPy regularizer / logistic-regression API ##
###########################################

# These NumPy oracle implementations used to live in `src/oracle_functions.py`.
# They are archived here so the active oracle module can now host the Torch
# full-batch objective used by preprocessing-side AdaN+, while existing
# experiment and diagnostics code keeps working through re-exported names.

##################
## Regularizers ##
##################

def regularizer_noncvx(w: np.ndarray) -> float:
    """
    Compute the non-convex regularizer value for a given vector.
    """
    return np.sum(w**2/(1 + w**2))

def regularizer_noncvx_grad(w: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient of the non-convex regularizer function.
    """
    return 2*w /(1 + w**2)**2

def regularizer_noncvx_hess(w: np.ndarray) -> np.ndarray:
    diag_elements = 2 * (1 - 3 * w**2) / ((1 + w**2)**3)
    return np.diag(diag_elements)

def regularizer_noncvx_hess_bound(w: np.ndarray) -> np.ndarray:
    """
    Calculate a bound for the Hessian of the non-convex regularizer function.
    """
    d_0 = w.shape[0]
    return 2*np.eye(d_0)

def regularizer_cvx(w: np.ndarray) -> float:
    """
    Compute the convex zero regularizer used by the project.
    """
    return 0.0

def regularizer_cvx_grad(w: np.ndarray) -> np.ndarray:
    """
    Gradient of the convex zero regularizer.
    """
    return np.zeros(w.shape, dtype=w.dtype)

def regularizer_cvx_hess(w: np.ndarray) -> np.ndarray:
    d_0 = w.shape[0]
    return np.zeros((d_0, d_0), dtype=w.dtype)

def regularizer_cvx_hess_bound(w: np.ndarray) -> np.ndarray:
    """
    Bound for the Hessian of the convex zero regularizer.
    """
    d_0 = w.shape[0]
    return np.zeros((d_0, d_0), dtype=w.dtype)

def regularizer_scvx(w: np.ndarray) -> float:
    """
    Compute the strongly convex regularizer value for a given vector.
    """
    return sq_two_norm(w)

def regularizer_scvx_grad(w: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient of the strongly convex regularizer function.
    """
    return 2*w

def regularizer_scvx_hess(w: np.ndarray) -> np.ndarray:
    """
    Compute the Hessian of the strongly convex regularizer function.
    """
    d_0 = w.shape[0]
    return 2*np.eye(d_0)

def regularizer_scvx_hess_bound(w: np.ndarray) -> np.ndarray:
    """
    Calculate a bound for the Hessian of the strongly convex regularizer function.
    """
    d_0 = w.shape[0]
    return 2*np.eye(d_0)


##########################################
## Logistic Regression Oracle Functions ##
##########################################

def _get_lambda_reg(params: dict) -> float:
    if "lambda_reg" not in params:
        raise KeyError("oracle params must contain 'lambda_reg'")
    return float(params["lambda_reg"])

def logreg_loss_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, params: dict, regularizer: callable) -> float:
    """
    Compute the logistic regression loss for one local dataset.
    """
    lambda_reg = _get_lambda_reg(params)

    assert lambda_reg >= 0, "Regularization parameter lambda_reg must be non-negative"
    assert len(y_ij) == X_ij.shape[0], "Length of y must equal the number of rows in X"
    assert len(w) == X_ij.shape[1], "Length of w must equal the number of columns in X"

    w = w.flatten()
    y_ij = y_ij.flatten()

    X_y = np.multiply(X_ij, y_ij[:, np.newaxis])
    denominator = 1 + np.exp(-X_y @ w)
    denominator = np.squeeze(np.asarray(denominator))
    l = np.log(denominator)

    return np.mean(l) + lambda_reg * regularizer(w)


def logreg_grad_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, params: dict, regularizer_grad: callable) -> np.ndarray:
    """
    Compute the gradient of the logistic regression loss function.
    """
    lambda_reg = _get_lambda_reg(params)
    assert lambda_reg >= 0, "Regularization parameter lambda_reg must be non-negative"
    assert y_ij.shape[0] == X_ij.shape[0], "Length of y must equal the number of rows in X"
    assert w.shape[0] == X_ij.shape[1], "Length of w must equal the number of columns in X"

    X_y = np.multiply(X_ij, y_ij[:, np.newaxis])
    denominator = 1 + np.exp(X_y @ w)
    denominator = np.squeeze(np.asarray(denominator))
    loss_grad = - np.mean(X_y / denominator[:, np.newaxis], axis=0)
    loss_grad = np.squeeze(np.asarray(loss_grad))
    return loss_grad + lambda_reg * regularizer_grad(w)


def logreg_hess_ij_bound(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], params: dict, regularizer_hess_bound: callable) -> Union[sparse.csr_matrix, np.ndarray]:
    """
    Compute a bound for the Hessian of the logistic regression loss function.
    """
    n_0, d_0 = X_ij.shape
    lambda_reg = _get_lambda_reg(params)
    return (1 / (4*n_0)) * (X_ij.T @ X_ij) + lambda_reg * regularizer_hess_bound(w)


def logreg_hess_ij(w: np.ndarray, X: np.ndarray, y: np.ndarray, params: dict, regularizer_hess: callable) -> np.ndarray:
    """
    Compute the Hessian matrix of the logistic regression loss function,
    including regularization, for a given dataset.
    """
    lambda_reg = _get_lambda_reg(params)
    assert lambda_reg >= 0, "Regularization parameter lambda_reg must be non-negative"

    n = X.shape[0]
    Xw = np.array(X @ w).flatten()
    yXw = y * Xw
    coeff = 1.0 / (np.exp(-0.5 * yXw) + np.exp(0.5 * yXw))**2
    H = (np.multiply(X.T, coeff.reshape(1, -1))) @ X / n
    return H + lambda_reg * regularizer_hess(w)

# appeared in the project firstly
##############################
## Quartic Oracle Functions ##
##############################
def quartic_loss_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, params: dict, regularizer: callable) -> float:
    """
    Compute f(w) = ||Xw||^4 + λ·R(w).
    Here ||Xw||^2 = w^T (X^T X) w, so f = (||Xw||^2)^2.
    """
    la = params["la"]
    Xw = X_ij.dot(w)
    u = float(np.dot(Xw, Xw))
    return u**2 + la * regularizer(w)

def quartic_grad_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, params: dict, regularizer_grad: callable) -> np.ndarray:
    """
    Gradient of f(w) = (w^T M w)^2 with M = X^T X:
      ∇f = 4 (w^T M w) · (M w) + λ·R'(w).
    """
    la = params["la"]
    Xw = X_ij.dot(w)
    u = float(np.dot(Xw, Xw))
    Mw = X_ij.T.dot(Xw)
    grad = 4 * u * Mw
    return grad + la * regularizer_grad(w)

def quartic_hess_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, params: dict, regularizer_hess: callable) -> np.ndarray:
    """
    Hessian of f(w) = (w^T M w)^2, M = X^T X:
      H = 8 (M w)(M w)^T + 4 (w^T M w)·M  + λ·R''(w).
    """
    la = params["la"]
    Xw = X_ij.dot(w)
    u = float(np.dot(Xw, Xw))
    Mw = X_ij.T.dot(Xw)
    M  = X_ij.T.dot(X_ij)
    H = 8.0 * np.outer(Mw, Mw) + 4.0 * u * M
    return H + la * regularizer_hess(w)

def quartic_hess_ij_bound(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], params: dict, regularizer_hess_bound: callable) -> np.ndarray:
    """
    Exact bound on Hessian of f(w) = ||Xw||^4:
      H_bound = 8 (M w)(M w)^T + 4 (w^T M w)·M  + λ·R''_bound(w).
    """
    la = params["la"]
    Xw = X_ij.dot(w)
    u = float(np.dot(Xw, Xw))
    Mw = X_ij.T.dot(Xw)
    M  = X_ij.T.dot(X_ij)
    Hb = 8.0 * np.outer(Mw, Mw) + 4.0 * u * M
    return Hb + la * regularizer_hess_bound(w)



# Are not used in this project
################################
## Quadratic Oracle Functions ##
################################

def quad_loss_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, c: float, la: float, regularizer: callable) -> float:
    """
    Compute the loss of the quadratic function for a single worker's dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ij (sparse.csr.csr_matrix or np.ndarray): A 2D sparse or dense matrix representing a single worker's feature matrix.
    y_ij (np.ndarray): A 1D numpy array representing a single worker's label vector.
    la (float): A non-negative regularization parameter.
    regularizer_grad (callable): A function to compute the gradient of the regularization term.

    Returns:
    float: The gradient of the quadratic loss function for the given data.

    Asserts:
    y_ij and X_ij must be of dtype 'float64'.
    X_ij must be a sparse CSR matrix or a numpy ndarray.
    y_ij must be a numpy ndarray.
    X_ij must have 2 dimensions and y_ij must have 1 dimension.
    """
    
    assert y_ij.dtype == 'float64', "y_ij must be of dtype 'float64'"
    assert X_ij.dtype == 'float64', "X_ij must be of dtype 'float64'"
    assert isinstance(X_ij, sparse.csr.csr_matrix) or isinstance(X_ij, np.ndarray), "X_ij must be a sparse CSR matrix or a numpy ndarray"
    assert isinstance(y_ij, np.ndarray), "y_ij must be a numpy ndarray"
    assert len(X_ij.shape) == 2, "X_ij must have 2 dimensions"
    assert len(y_ij.shape) == 1, "y_ij must have 1 dimension"
    
    return 0.5 * np.dot(X_ij.dot(w), w) - np.dot(y_ij,w) + la * regularizer(w)

def quad_loss_distributed(w: np.ndarray, X_ar_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]], y_ar_ar: np.ndarray, c: float, la: float, regularizer: callable) -> float:
    """
    Compute the quadratic loss function with regularization across a nested list of datasets in a distributed manner.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ar_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): A nested list where each inner list contains 2D sparse or dense matrices, each representing a dataset's feature matrix.
    y_ar_ar (np.ndarray): A 3D numpy tensor representing the label tensor for all datasets.
    la (float): A non-negative regularization parameter.
    regularizer (callable): A function to compute the regularization term.

    Returns:
    float: The average quadratic loss with regularization across all datasets contained in the nested list.

    Asserts:
    Each first element of the inner lists of X must be either a sparse CSR matrix or a numpy ndarray.
    y must be a 3D numpy tensor.
    """
    #assert (all(isinstance(x, (sparse.csr.csr_matrix, np.ndarray)) for x in X_ar), "Elements in X_ar must be numpy arrays or sparse CSR matrices")
    assert isinstance(y_ar_ar, np.ndarray), "y_ar must be a numpy array"
    assert len(X_ar_ar) == y_ar_ar.shape[0], "Length of X_ar must equal the number of rows in y_ar"

    cum_meanX = 0
    cum_meanY = 0
    num_workers = len(X_ar_ar)
    for i in range(num_workers):
        meanX_i = sum(X_ar_ar[i])/len(X_ar_ar[i])
        meanY_i = np.mean(y_ar_ar[i], axis=0)
        cum_meanX += meanX_i
        cum_meanY += meanY_i
    meanX = cum_meanX/num_workers
    meanY = cum_meanY/num_workers
    
    return 0.5 * np.dot(meanX.dot(w), w) - np.dot(meanY,w) + la * regularizer(w)

######
# Gradients of Loss functions #
######
def quad_grad_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, la: float, regularizer_grad: callable) -> np.ndarray:
    """
    Compute the gradient of the quadratic loss function for a single worker's dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ij (sparse.csr.csr_matrix or np.ndarray): A 2D sparse or dense matrix representing a single worker's feature matrix.
    y_ij (np.ndarray): A 1D numpy array representing a single worker's label vector.
    la (float): A non-negative regularization parameter.
    regularizer_grad (callable): A function to compute the gradient of the regularization term.

    Returns:
    np.ndarray: The gradient of the quadratic loss function for the given data.

    Asserts:
    y_ij and X_ij must be of dtype 'float64'.
    X_ij must be a sparse CSR matrix or a numpy ndarray.
    y_ij must be a numpy ndarray.
    X_ij must have 2 dimensions and y_ij must have 1 dimension.
    """
    
    assert y_ij.dtype == 'float64', "y_ij must be of dtype 'float64'"
    assert X_ij.dtype == 'float64', "X_ij must be of dtype 'float64'"
    assert isinstance(X_ij, sparse.csr.csr_matrix) or isinstance(X_ij, np.ndarray), "X_ij must be a sparse CSR matrix or a numpy ndarray"
    assert isinstance(y_ij, np.ndarray), "y_ij must be a numpy ndarray"
    assert len(X_ij.shape) == 2, "X_ij must have 2 dimensions"
    assert len(y_ij.shape) == 1, "y_ij must have 1 dimension"
    
    return X_ij.dot(w) - y_ij + la*regularizer_grad(w)

def quad_grad_distributed(w: np.ndarray, X_ar_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]], y_ar_ar: np.ndarray, la: float, regularizer_grad: callable) -> np.ndarray:
    """
    Compute the average gradient of the quadratic loss function across a nested list of datasets.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X (List[List[Union[sparse.csr_matrix, np.ndarray]]]): A nested list where each inner list contains 2D sparse or dense matrices, each representing a dataset's feature matrix.
    y (np.ndarray): A 3D numpy tensor representing the label tensor for all datasets.
    la (float): A non-negative regularization parameter.
    regularizer_grad (callable): A function to compute the gradient of the regularization term.

    Returns:
    np.ndarray: The average gradient of the quadratic loss function across all datasets contained in the nested list.

    Asserts:
    Each first element of the inner lists of X must be either a sparse CSR matrix or a numpy ndarray.
    y must be a 3D numpy tensor.
    """
    #assert (all(isinstance(X_ar[0], (sparse.csr_matrix, np.ndarray)) for X_ar in X_ar_ar), "Each first element of the inner lists of X must be either a sparse CSR matrix or a numpy ndarray")
    assert y_ar_ar.ndim == 3, "y must be a 3D numpy tensor"

    cum_meanX = 0
    cum_meanY = 0
    num_workers = len(X_ar_ar)
    for i in range(num_workers):
        meanX_i = sum(X_ar_ar[i])/len(X_ar_ar[i])
        meanY_i = np.mean(y_ar_ar[i], axis=0)
        cum_meanX += meanX_i
        cum_meanY += meanY_i
    meanX = cum_meanX/num_workers
    meanY = cum_meanY/num_workers       
    return meanX.dot(w) - meanY + la*regularizer_grad(w)

# Single node version
def quad_local_grads(W: np.ndarray, X_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]], y_ar: np.ndarray, la: float, regularizer_grad: callable) -> np.ndarray:
    """
    Calculate the gradient of the quadratic loss function for each local model.

    Args:
        W (np.ndarray): The weight matrix of shape (n_local_models, n_features).
        X_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): The input data for each local model.
        y_ar (np.ndarray): The target values for each local model.
        la (float): The regularization parameter.
        regularizer_grad (callable): The gradient of the regularization function.

    Returns:
        np.ndarray: The gradient matrix of shape (n_local_models, n_features).
    """
    G = np.zeros(W.shape)
    for i in range(len(X_ar)):
        G[i] = quad_grad_ij(W[i], X_ar[i], y_ar[i], la, regularizer_grad)
    return G
    
######
# Hessians  #
######
def quad_hess_non_reg_ij(X_ij: Union[sparse.csr_matrix, np.ndarray]) -> Union[sparse.csr_matrix, np.ndarray]:
    """
    Compute the Hessian matrix of the non-regularized quadratic loss function for a single dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ij (Union[sparse.csr_matrix, np.ndarray]): A 2D sparse or dense matrix representing the feature matrix.

    Returns:
    Union[sparse.csr_matrix, np.ndarray]: The Hessian matrix of the non-regularized quadratic loss function.
    
    Asserts:
    X_ij must be either a sparse CSR matrix or a numpy ndarray.
    """
    assert isinstance(X_ij, sparse.csr_matrix) or isinstance(X_ij, np.ndarray), "X_ij must be either a sparse CSR matrix or a numpy ndarray"
    return X_ij

def quad_hess_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], la: float, regularizer_hess: callable) -> Union[sparse.csr_matrix, np.ndarray]:
    """
    Compute the Hessian matrix of the quadratic loss function, including regularization, for a single dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ij (Union[sparse.csr_matrix, np.ndarray]): A 2D sparse or dense matrix representing the feature matrix.
    la (float): A non-negative regularization parameter.
    regularizer_hess (callable): A function to compute the Hessian of the regularization term.

    Returns:
    Union[sparse.csr_matrix, np.ndarray]: The Hessian matrix of the quadratic loss function.

    Asserts:
    X_ij must be either a sparse CSR matrix or a numpy ndarray.
    la must be non-negative.
    """
    assert isinstance(X_ij, sparse.csr_matrix) or isinstance(X_ij, np.ndarray), "X_ij must be either a sparse CSR matrix or a numpy ndarray"
    assert la >= 0, "Regularization parameter la must be non-negative"
    
    return quad_hess_non_reg_ij(X_ij) + la*regularizer_hess(w)

def quad_hess_non_reg_i(X_ar_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]]) -> List[Union[sparse.csr_matrix, np.ndarray]]:
    """
    Compute the average Hessian matrix of the non-regularized quadratic loss function for each group of datasets.

    Parameters:
    X_ar_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): A nested list where each inner list contains 2D sparse or dense matrices, each representing a group of dataset's feature matrices.

    Returns:
    List[Union[sparse.csr_matrix, np.ndarray]]: A list containing the average Hessian matrix for each group of datasets.
    """
    return [sum(X_ar)/len(X_ar) for X_ar in X_ar_ar]

def quad_hess_i(w: np.ndarray, X_ar_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]],la: float, regularizer_hess: callable ) -> List[Union[sparse.csr_matrix, np.ndarray]]:
    """
    Compute the Hessian matrix of the quadratic loss function, including regularization, for each group of datasets.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ar_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): A nested list where each inner list contains 2D sparse or dense matrices, each representing a group of dataset's feature matrices.
    la (float): A non-negative regularization parameter.
    regularizer_hess (callable): A function to compute the Hessian of the regularization term.

    Returns:
    List[Union[sparse.csr_matrix, np.ndarray]]: A list containing the Hessian matrix for each group of datasets, including the regularization term.
    """
    return [sum(X_ar)/len(X_ar) + la*regularizer_hess(w) for X_ar in X_ar_ar]

def quad_hess_non_reg_distributed(X_ar_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]]) -> Union[sparse.csr_matrix, np.ndarray]:
    """
    Compute the average Hessian matrix of the non-regularized quadratic loss function across a nested list of datasets.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ar_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): A nested list where each inner list contains 2D sparse or dense matrices, each representing a dataset's feature matrix.

    Returns:
    Union[sparse.csr_matrix, np.ndarray]: The average Hessian matrix across all datasets contained in the nested list.
    """
    # Removed assert since it would be redundant with the loop check

    cum_sumX = 0
    num_workers = len(X_ar_ar)
    for X_ar in X_ar_ar:
        sumX_ar = sum(X_ar)/len(X_ar)
        cum_sumX += sumX_ar
    meanX = cum_sumX/num_workers
    return meanX


def quad_hess_distributed(w: np.ndarray, X_ar_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]], y_ar_ar: np.ndarray, la: float, regularizer_hess: callable) -> Union[sparse.csr_matrix, np.ndarray]:
    """
    Compute the average Hessian matrix of the quadratic loss function across a nested list of datasets.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ar_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): A nested list where each inner list contains 2D sparse or dense matrices, each representing a dataset's feature matrix.
    y_ar_ar (np.ndarray): A 3D numpy tensor representing the label tensor for all datasets.
    la (float): A non-negative regularization parameter.
    regularizer_hess (callable): A function to compute the Hessian of the regularization term.

    Returns:
    Union[sparse.csr_matrix, np.ndarray]: The average Hessian matrix across all datasets contained in the nested list.
    """
    # Removed assert since it would be redundant with the loop check

    return quad_hess_non_reg_distributed(X_ar_ar) + la*regularizer_hess(w)

def quad_hess_bound_distributed(w: np.ndarray, X_ar_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]], la: float, regularizer_hess: callable) -> Union[sparse.csr_matrix, np.ndarray]:
    """
    Compute the average Hessian matrix bound of the quadratic loss function across a nested list of datasets.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ar_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): A nested list where each inner list contains 2D sparse or dense matrices, each representing a dataset's feature matrix.
    la (float): A non-negative regularization parameter.
    regularizer_hess (callable): A function to compute the Hessian of the regularization term.

    Returns:
    Union[sparse.csr_matrix, np.ndarray]: The average Hessian matrix across all datasets contained in the nested list.
    """
    # Removed assert since it would be redundant with the loop check
    return quad_hess_non_reg_distributed(X_ar_ar) + la*regularizer_hess(w)

#############################
## linreg Oracle Functions ##
#############################

#Currently, I need single node functions only
# Comments within functions may be incorrect

def linreg_loss_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, c_ij: float, params: dict, regularizer: callable) -> float:
    """
    Compute the loss of the quadratic function for a single worker's dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ij (sparse.csr.csr_matrix or np.ndarray): A 2D sparse or dense matrix representing a single worker's feature matrix.
    y_ij (np.ndarray): A 1D numpy array representing a single worker's label vector.
    c_ij (float): A non-negative constant parameter.
    params (dict): A dictionary containing "la" (regularization parameter) and "n_samples".
    regularizer (callable): A function to compute the regularization term.

    Returns:
    float: The gradient of the quadratic loss function for the given data.
    """
    la = params["la"]
    n_samples = params["n_samples"]

    assert y_ij.dtype == 'float64', "y_ij must be of dtype 'float64'"
    assert X_ij.dtype == 'float64', "X_ij must be of dtype 'float64'"
    assert isinstance(X_ij, sparse.csr.csr_matrix) or isinstance(X_ij, np.ndarray), "X_ij must be a sparse CSR matrix or a numpy ndarray"
    assert isinstance(y_ij, np.ndarray), "y_ij must be a numpy ndarray"
    assert len(X_ij.shape) == 2, "X_ij must have 2 dimensions"
    assert len(y_ij.shape) == 1, "y_ij must have 1 dimension"
    assert c_ij >= 0, "c_ij must be non-negative"

    return (0.5 / n_samples) * (np.dot(X_ij.dot(w), w) - np.dot(y_ij, w) + c_ij) + la * regularizer(w)

def linreg_grad_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, params: dict, regularizer_grad: callable) -> np.ndarray:
    """
    Compute the gradient of the quadratic loss function for a single worker's dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ij (sparse.csr.csr_matrix or np.ndarray): A 2D sparse or dense matrix representing a single worker's feature matrix.
    y_ij (np.ndarray): A 1D numpy array representing a single worker's label vector.
    params (dict): A dictionary containing "la" (regularization parameter) and "n_samples".
    regularizer_grad (callable): A function to compute the gradient of the regularization term.

    Returns:
    np.ndarray: The gradient of the quadratic loss function for the given data.
    """
    la = params["la"]
    n_samples = params["n_samples"]

    assert y_ij.dtype == 'float64', "y_ij must be of dtype 'float64'"
    assert X_ij.dtype == 'float64', "X_ij must be of dtype 'float64'"
    assert isinstance(X_ij, sparse.csr.csr_matrix) or isinstance(X_ij, np.ndarray), "X_ij must be a sparse CSR matrix or a numpy ndarray"
    assert isinstance(y_ij, np.ndarray), "y_ij must be a numpy ndarray"
    assert len(X_ij.shape) == 2, "X_ij must have 2 dimensions"
    assert len(y_ij.shape) == 1, "y_ij must have 1 dimension"

    return (1 / n_samples) * (X_ij.dot(w) - y_ij) + la * regularizer_grad(w)

def linreg_minibatch_grad_ij(w: np.ndarray, A_ij: Union[sparse.csr_matrix, np.ndarray], b_ij: np.ndarray, params: dict, regularizer_grad: callable) -> np.ndarray:
    """
    Compute the gradient of the quadratic loss function for a single worker's dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    A_ij (sparse.csr.csr_matrix or np.ndarray): A 2D sparse or dense matrix representing a single worker's feature matrix.
    y_ij (np.ndarray): A 1D numpy array representing a single worker's label vector.
    params (dict): A dictionary containing "la" (regularization parameter) and "n_samples".
    regularizer_grad (callable): A function to compute the gradient of the regularization term.

    Returns:
    np.ndarray: The gradient of the quadratic loss function for the given data.
    """
    assert b_ij.dtype == 'float64', "y_ij must be of dtype 'float64'"
    assert A_ij.dtype == 'float64', "X_ij must be of dtype 'float64'"
    assert isinstance(A_ij, sparse.csr.csr_matrix) or isinstance(A_ij, np.ndarray), "X_ij must be a sparse CSR matrix or a numpy ndarray"
    assert isinstance(b_ij, np.ndarray), "y_ij must be a numpy ndarray"
    assert len(A_ij.shape) == 2, "X_ij must have 2 dimensions"
    assert len(b_ij.shape) == 1, "y_ij must have 1 dimension"

    la = params["la"]
    b_size = params["batchsize"]
    inds = params["inds"]
    b_inds = b_ij[inds].copy()
    A_inds = A_ij[inds].copy()
    A_inds_T = A_inds.T.copy()
    # TODO check if .T is correct; check dim
    return (1 / b_size)*A_inds_T.dot(A_inds.dot(w) - b_inds) + la * regularizer_grad(w)

def linreg_hess_non_reg_ij(X_ij: Union[sparse.csr_matrix, np.ndarray], params: dict) -> Union[sparse.csr_matrix, np.ndarray]:
    """
    Compute the Hessian matrix of the non-regularized quadratic loss function for a single dataset.

    Parameters:
    X_ij (Union[sparse.csr_matrix, np.ndarray]): A 2D sparse or dense matrix representing the feature matrix.
    params (dict): A dictionary containing "n_samples".

    Returns:
    Union[sparse.csr_matrix, np.ndarray]: The Hessian matrix of the non-regularized quadratic loss function.
    """
    n_samples = params["n_samples"]

    assert isinstance(X_ij, sparse.csr.csr_matrix) or isinstance(X_ij, np.ndarray), "X_ij must be either a sparse CSR matrix or a numpy ndarray"
    return (1 / n_samples) * X_ij

def linreg_hess_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, params: dict, regularizer_hess: callable) -> Union[sparse.csr_matrix, np.ndarray]:
    """
    Compute the Hessian matrix of the quadratic loss function, including regularization, for a single dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ij (Union[sparse.csr_matrix, np.ndarray]): A 2D sparse or dense matrix representing the feature matrix.
    params (dict): A dictionary containing "la" (regularization parameter) and "n_samples".
    regularizer_hess (callable): A function to compute the Hessian of the regularization term.

    Returns:
    Union[sparse.csr_matrix, np.ndarray]: The Hessian matrix of the quadratic loss function.
    """
    la = params["la"]
    return linreg_hess_non_reg_ij(X_ij, params) + la * regularizer_hess(w)

def linreg_hess_ij_bound(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, params: dict, regularizer_hess_bound: callable) -> Union[sparse.csr_matrix, np.ndarray]:
    """
    Compute the bound of the Hessian matrix of the quadratic loss function, including regularization, for a single dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ij (Union[sparse.csr_matrix, np.ndarray]): A 2D sparse or dense matrix representing the feature matrix.
    params (dict): A dictionary containing "la" (regularization parameter) and "n_samples".
    regularizer_hess_bound (callable): A function to compute the Hessian bound of the regularization term.

    Returns:
    Union[sparse.csr_matrix, np.ndarray]: The Hessian matrix of the quadratic loss function.
    """
    la = params["la"]
    return linreg_hess_non_reg_ij(X_ij, params) + la * regularizer_hess_bound(w)



##############################
## l1_norm Oracle Functions ##
##############################
def l1_norm_loss_ij(w: np.ndarray, X_ij: Union[sparse.csr_matrix, np.ndarray], y_ij: np.ndarray, la: float, regularizer: callable) -> float:
    #in progress
    raise NotImplementedError("This function is not yet implemented")
    """
    Compute the loss of the quadratic function for a single worker's dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ij (sparse.csr.csr_matrix or np.ndarray): A 2D sparse or dense matrix representing a single worker's feature matrix.
    y_ij (np.ndarray): A 1D numpy array representing a single worker's label vector.
    la (float): A non-negative regularization parameter.
    regularizer_grad (callable): A function to compute the gradient of the regularization term.

    Returns:
    float: The gradient of the quadratic loss function for the given data.

    Asserts:
    y_ij and X_ij must be of dtype 'float64'.
    X_ij must be a sparse CSR matrix or a numpy ndarray.
    y_ij must be a numpy ndarray.
    X_ij must have 2 dimensions and y_ij must have 1 dimension.
    """
    
    assert y_ij.dtype == 'float64', "y_ij must be of dtype 'float64'"
    assert X_ij.dtype == 'float64', "X_ij must be of dtype 'float64'"
    assert isinstance(X_ij, sparse.csr.csr_matrix) or isinstance(X_ij, np.ndarray), "X_ij must be a sparse CSR matrix or a numpy ndarray"
    assert isinstance(y_ij, np.ndarray), "y_ij must be a numpy ndarray"
    assert len(X_ij.shape) == 2, "X_ij must have 2 dimensions"
    assert len(y_ij.shape) == 1, "y_ij must have 1 dimension"
    return 0.5 * np.dot(X_ij.dot(w), w) - np.dot(y_ij,w) + la * regularizer(w)

#Designed for deterministic setting; completed
def l1_norm_loss_i(w: np.ndarray, X_i: Union[sparse.csr_matrix, np.ndarray], y_i: np.ndarray, la: float, regularizer: callable) -> float:
    """
    Compute the loss of the quadratic function for a single worker's dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_i (sparse.csr.csr_matrix or np.ndarray): A 2D sparse or dense matrix representing a single worker's feature matrix.
    y_i (np.ndarray): A 1D numpy array representing a single worker's label vector.
    la (float): A non-negative regularization parameter.
    regularizer_grad (callable): A function to compute the gradient of the regularization term.

    Returns:
    float: The gradient of the quadratic loss function for the given data.

    Asserts:
    y_i and X_i must be of dtype 'float64'.
    X_i must be a sparse CSR matrix or a numpy ndarray.
    y_i must be a numpy ndarray.
    X_i must have 2 dimensions and y_i must have 1 dimension.
    """
    
    assert y_i.dtype == 'float64', "y_i must be of dtype 'float64'"
    assert X_i.dtype == 'float64', "X_i must be of dtype 'float64'"
    assert isinstance(X_i, sparse.csr.csr_matrix) or isinstance(X_i, np.ndarray), "X_i must be a sparse CSR matrix or a numpy ndarray"
    assert isinstance(y_i, np.ndarray), "y_i must be a numpy ndarray"
    assert len(X_i.shape) == 2, "X_i must have 2 dimensions"
    assert len(y_i.shape) == 1, "y_i must have 1 dimension"
    
    return one_norm(X_i.dot(w) - y_i) + la * regularizer(w)

#Designed for deterministic setting; completed
def l1_norm_loss_i_distributed(w: np.ndarray, X_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]], y_ar: np.ndarray, la: float, regularizer: callable) -> float:
    """
    Compute the quadratic loss function with regularization across a nested list of datasets in a distributed manner.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): A nested list where each inner list contains 2D sparse or dense matrices, each representing a dataset's feature matrix.
    y_ar (np.ndarray): A 3D numpy tensor representing the label tensor for all datasets.
    la (float): A non-negative regularization parameter.
    regularizer (callable): A function to compute the regularization term.

    Returns:
    float: The average quadratic loss with regularization across all datasets contained in the nested list.

    Asserts:
    Each first element of the inner lists of X must be either a sparse CSR matrix or a numpy ndarray.
    y must be a 3D numpy tensor.
    """
    #assert (all(isinstance(x, (sparse.csr.csr_matrix, np.ndarray)) for x in X_ar), "Elements in X_ar must be numpy arrays or sparse CSR matrices")
    assert isinstance(y_ar, np.ndarray), "y_ar must be a numpy array"
    assert len(X_ar) == y_ar.shape[0], "Length of X_ar must equal the number of rows in y_ar"

    cum_loss = 0
    num_workers = len(X_ar)
    for i in range(num_workers):
        cum_loss = l1_norm_loss_i(w, X_ar[i], y_ar[i], la, regularizer)
    return cum_loss/num_workers

#Designed for deterministic setting; completed
def l1_norm_local_losses_i_distributed(W: np.ndarray, X_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]], y_ar: np.ndarray, la: float, regularizer: callable) -> float:
    """
    Compute the quadratic loss function with regularization across a nested list of datasets in a distributed manner.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): A nested list where each inner list contains 2D sparse or dense matrices, each representing a dataset's feature matrix.
    y_ar (np.ndarray): A 3D numpy tensor representing the label tensor for all datasets.
    la (float): A non-negative regularization parameter.
    regularizer (callable): A function to compute the regularization term.

    Returns:
    float: The average quadratic loss with regularization across all datasets contained in the nested list.

    Asserts:
    Each first element of the inner lists of X must be either a sparse CSR matrix or a numpy ndarray.
    y must be a 3D numpy tensor.
    """
    #assert (all(isinstance(x, (sparse.csr.csr_matrix, np.ndarray)) for x in X_ar), "Elements in X_ar must be numpy arrays or sparse CSR matrices")
    assert isinstance(y_ar, np.ndarray), "y_ar must be a numpy array"
    assert len(X_ar) == y_ar.shape[0], "Length of X_ar must equal the number of rows in y_ar"

    num_workers = len(X_ar)
    L = np.zeros(num_workers)
    for i in range(num_workers):
        L[i] = l1_norm_loss_i(W[i], X_ar[i], y_ar[i], la, regularizer)
    return L

######
# Subgradients of Loss functions #
######
#Designed for deterministic setting; completed
def l1_norm_grad_i(w: np.ndarray, X_i: Union[sparse.csr_matrix, np.ndarray], y_i: np.ndarray, la: float, regularizer_grad: callable) -> np.ndarray:
    """
    Compute the gradient of the quadratic loss function for a single worker's dataset.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X_i (sparse.csr.csr_matrix or np.ndarray): A 2D sparse or dense matrix representing a single worker's feature matrix.
    y_i (np.ndarray): A 1D numpy array representing a single worker's label vector.
    la (float): A non-negative regularization parameter.
    regularizer_grad (callable): A function to compute the gradient of the regularization term.

    Returns:
    np.ndarray: The gradient of the quadratic loss function for the given data.

    Asserts:
    y_i and X_i must be of dtype 'float64'.
    X_i must be a sparse CSR matrix or a numpy ndarray.
    y_i must be a numpy ndarray.
    X_i must have 2 dimensions and y_i must have 1 dimension.
    """
    
    assert y_i.dtype == 'float64', "y_i must be of dtype 'float64'"
    assert X_i.dtype == 'float64', "X_i must be of dtype 'float64'"
    assert isinstance(X_i, sparse.csr.csr_matrix) or isinstance(X_i, np.ndarray), "X_i must be a sparse CSR matrix or a numpy ndarray"
    assert isinstance(y_i, np.ndarray), "y_i must be a numpy ndarray"
    assert len(X_i.shape) == 2, "X_i must have 2 dimensions"
    assert len(y_i.shape) == 1, "y_i must have 1 dimension"
    
    X_i_T = X_i.T
    return X_i_T.dot(np.sign(X_i.dot(w) - y_i)) + la*regularizer_grad(w)

#Designed for deterministic setting; completed
def l1_norm_grad_i_distributed(w: np.ndarray, X_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]], y_ar: np.ndarray, la: float, regularizer_grad: callable) -> np.ndarray:
    """
    Compute the average gradient of the quadratic loss function across a nested list of datasets.

    Parameters:
    w (np.ndarray): A 1D numpy array representing the weight vector.
    X (List[List[Union[sparse.csr_matrix, np.ndarray]]]): A nested list where each inner list contains 2D sparse or dense matrices, each representing a dataset's feature matrix.
    y (np.ndarray): A 3D numpy tensor representing the label tensor for all datasets.
    la (float): A non-negative regularization parameter.
    regularizer_grad (callable): A function to compute the gradient of the regularization term.

    Returns:
    np.ndarray: The average gradient of the quadratic loss function across all datasets contained in the nested list.

    Asserts:
    Each first element of the inner lists of X must be either a sparse CSR matrix or a numpy ndarray.
    y must be a 3D numpy tensor.
    """
    assert isinstance(y_ar, np.ndarray), "y_ar must be a numpy array"
    assert len(X_ar) == y_ar.shape[0], "Length of X_ar must equal the number of rows in y_ar"

    cum_grad = 0
    num_workers = len(X_ar)
    for i in range(num_workers):
        cum_grad = l1_norm_grad_i(w, X_ar[i], y_ar[i], la, regularizer_grad)
    return cum_grad/num_workers

#Designed for deterministic setting; completed
def l1_norm_local_grads_i_distributed(W: np.ndarray, X_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]], y_ar: np.ndarray, la: float, regularizer_grad: callable) -> np.ndarray:
    """
    Calculate the gradient of the quadratic loss function for each local model.

    Args:
        W (np.ndarray): The weight matrix of shape (n_local_models, n_features).
        X_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): The input data for each local model.
        y_ar (np.ndarray): The target values for each local model.
        la (float): The regularization parameter.
        regularizer_grad (callable): The gradient of the regularization function.

    Returns:
        np.ndarray: The gradient matrix of shape (n_local_models, n_features).
    """
    
    assert isinstance(y_ar, np.ndarray), "y_ar must be a numpy array"
    assert len(X_ar) == y_ar.shape[0], "Length of X_ar must equal the number of rows in y_ar"

    num_workers = len(X_ar)
    G = np.zeros(W.shape)
    for i in range(num_workers):
        G[i] = l1_norm_grad_i(W[i], X_ar[i], y_ar[i], la, regularizer_grad)
    return G

#Designed for deterministic setting; completed
def l1_norm_non_local_grads_i_distributed(w: np.ndarray, X_ar: List[List[Union[sparse.csr_matrix, np.ndarray]]], y_ar: np.ndarray, la: float, regularizer_grad: callable) -> np.ndarray:
    """
    Calculate the gradient of the quadratic loss function for each local model.

    Args:
        W (np.ndarray): The weight matrix of shape (n_local_models, n_features).
        X_ar (List[List[Union[sparse.csr_matrix, np.ndarray]]]): The input data for each local model.
        y_ar (np.ndarray): The target values for each local model.
        la (float): The regularization parameter.
        regularizer_grad (callable): The gradient of the regularization function.

    Returns:
        np.ndarray: The gradient matrix of shape (n_local_models, n_features).
    """
    
    assert isinstance(y_ar, np.ndarray), "y_ar must be a numpy array"
    assert len(X_ar) == y_ar.shape[0], "Length of X_ar must equal the number of rows in y_ar"

    num_workers = len(X_ar)
    dim = w.shape[0]
    G = np.zeros((num_workers, dim), dtype = np.float64)
    for i in range(num_workers):
        G[i] = l1_norm_grad_i(w, X_ar[i], y_ar[i], la, regularizer_grad)
    return G

# The code below is not used in the experiments and it is not ready for use
#########################################
# Supports partial gradient computation #
#########################################

# def regularizer_noncvx_part_grad(w, ids):
#     reg_part_grad = np.zeros(w.shape)
#     reg_part_grad[ids] = 2*w[ids] /(1 + w[ids]**2)**2    
#     return reg_part_grad

# def logreg_part_grad(w, X, y, la, ids, regularizer_part_grad):
#     """
#     Returns full gradient
#     :param w:
#     :param X:
#     :param y:
#     :param la:
#     :return:
#     """
#     assert la >= 0
#     assert (y.shape[0] == X.shape[0])
#     assert (w.shape[0] == X.shape[1])
    
#     loss_part_grad = np.zeros(w.shape)
    
#     numerator = np.multiply(X[:,ids], y[:, np.newaxis])
#     denominator = 1 + np.exp(np.multiply(X@w, y))

#     matrix = numerator/denominator[:,np.newaxis]
    
#     loss_part_grad[ids] = - np.mean(matrix, axis = 0)
    
#     assert len(loss_part_grad) == len(w)
#     return loss_part_grad + la * regularizer_part_grad(w,ids)
