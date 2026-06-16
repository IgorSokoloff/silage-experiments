from IPython import display
from IPython import get_ipython
from contextlib import redirect_stdout

from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter, FuncFormatter, FixedLocator, FixedFormatter
import matplotlib.patheffects as path_effects
from sklearn.preprocessing import StandardScaler

from scipy.sparse.linalg import lobpcg
from numpy.linalg import norm
from numpy.random import RandomState
from scipy import sparse
from scipy.interpolate import interp1d
from scipy.linalg import hessenberg
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.datasets import load_svmlight_file, make_regression
from time import time
from tqdm import tqdm
from typing import List, Union, Callable
import argparse
import ast
import datetime
import h5py
import itertools
import json
import math
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
import matplotlib.ticker as tck
# Optional dependency used only in interactive plotting flows.
try:
    import mpld3  # type: ignore
except ImportError:
    mpld3 = None
import numpy as np
import os
import pandas as pd
import ray
import resource
import scipy
import shutil
import subprocess
import sympy as smp
import sys
import time
import psutil
import bz2file
import copy


int_repr_prec = lambda x, prec: int(x) if x.is_integer() else round(x,prec)
myrepr = lambda x: repr(round(x, 8)).replace('.',',') if isinstance(x, float) else repr(x)
intrepr = lambda x: int(x) if x.is_integer() else round(x,8)
sq_two_norm = lambda x: norm(x, ord=2) ** 2
sq_fro_norm = lambda x: norm(x, ord='fro') ** 2

one_norm = lambda x: norm(x, ord=1)
two_norm = lambda x: norm(x, ord=2)


d_copy = lambda my_dict: copy.deepcopy(my_dict)

#d_copy = lambda my_dict:{key: val.copy() if isinstance(val, np.ndarray) else val for key, val in my_dict.items()}
#d_copy = lambda my_dict: my_dict.copy()

NUM_BITS_PER_FLOAT = 64

NUM_ZERO = 1e-10
def sign (x):
    if x > NUM_ZERO:
        return 1
    elif np.abs(x) <= NUM_ZERO:
        return 0
    else:
        return -1

def left_sketch_B(B):
    """
    Compute the projection matrix H_B = B @ ((B^T @ B)^†) @ B^T.
    """
    B_T = B.T
    pseudo_inverse = np.linalg.pinv(B_T @ B)
    H_B = B @ pseudo_inverse @ B_T
    return H_B

def right_sketch_A(A):
    """
    Compute the projection matrix H_A = A^T @ ((A @ A^T)^†) @ A.
    """
    A_T = A.T
    pseudo_inverse = np.linalg.pinv(A @ A.T)
    H_A = A_T @ pseudo_inverse @ A
    return H_A

def random_rotation(n_features, seed=0):
    rng = np.random.RandomState(seed)
    # Generate a random n_features x n_features matrix
    M = rng.randn(n_features, n_features)
    # QR decomposition gives an orthonormal Q
    Q, _ = np.linalg.qr(M)
    return Q

def apply_random_rotation(A, seed=0):
    Q = random_rotation(A.shape[1], seed=seed)
    return A @ Q

def compare_mean_std(A_pre, A_ft):
    """
    Compares the mean and standard deviation of two datasets A_pre and A_ft.
    Returns the feature-wise means, feature-wise standard deviations,
    and two scalar measures of how different they are.
    
    Parameters
    ----------
    A_pre : np.ndarray of shape (n_samples_pre, n_features)
        The first dataset (e.g., pretraining).
    A_ft : np.ndarray of shape (n_samples_ft, n_features)
        The second dataset (e.g., finetuning).
    
    Returns
    -------
    mean_pre : np.ndarray, shape (n_features,)
        Feature-wise mean of A_pre
    mean_ft : np.ndarray, shape (n_features,)
        Feature-wise mean of A_ft
    std_pre : np.ndarray, shape (n_features,)
        Feature-wise standard deviation of A_pre
    std_ft : np.ndarray, shape (n_features,)
        Feature-wise standard deviation of A_ft
    mean_diff : float
        The Euclidean distance between mean_pre and mean_ft
    std_diff : float
        The Euclidean distance between std_pre and std_ft
    """
    # Feature-wise mean
    mean_pre = A_pre.mean(axis=0)
    mean_ft  = A_ft.mean(axis=0)
    
    # Feature-wise standard deviation
    std_pre = A_pre.std(axis=0, ddof=1)  # ddof=1 => unbiased estimator
    std_ft  = A_ft.std(axis=0, ddof=1)
    
    # Euclidean distances (L2 norm)
    mean_diff = np.linalg.norm(mean_pre - mean_ft)
    std_diff  = np.linalg.norm(std_pre - std_ft)
    
    return mean_pre, mean_ft, std_pre, std_ft, mean_diff, std_diff


def scale_arrays(*arrays):
    """
    Scales a collection of NumPy arrays (each can be 1D or 2D).
    For 2D arrays (feature matrices), it applies:
        scaler.fit_transform(X)
    For 1D arrays (targets), it applies:
        scaler.fit_transform(y.reshape(-1, 1)).ravel()

    Parameters
    ----------
    *arrays : np.ndarray
        Any number of NumPy arrays to be scaled.

    Returns
    -------
    scaled_arrays : tuple of np.ndarray
        The scaled version of each array, in the same order.

    Example
    -------
    A_pre_scaled, b_pre_scaled, A_ft_scaled, b_ft_scaled = scale_arrays(A_pre, b_pre, A_ft, b_ft)
    """
    scaled_arrays = []

    for arr in arrays:
        scaler = StandardScaler()
        if arr.ndim == 1:
            # For 1D array (target), reshape first
            arr_reshaped = arr.reshape(-1, 1)
            arr_scaled = scaler.fit_transform(arr_reshaped).ravel()
        else:
            # For 2D array (feature matrix)
            arr_scaled = scaler.fit_transform(arr)

        scaled_arrays.append(arr_scaled)

    # Return as a tuple, so you can unpack them directly
    return tuple(scaled_arrays)

def scale_arrays_custom(*arrays, target_mean=0.0, target_std=1.0):
    """
    Scales a collection of NumPy arrays (each can be 1D or 2D) 
    to have the specified mean and standard deviation.
    
    Parameters
    ----------
    *arrays : np.ndarray
        Any number of NumPy arrays to be scaled.
    target_mean : float, optional
        The desired mean of the scaled data (default is 0.0).
    target_std : float, optional
        The desired standard deviation of the scaled data (default is 1.0).

    Returns
    -------
    scaled_arrays : tuple of np.ndarray
        The scaled version of each array, in the same order.

    Example
    -------
    A_pre_scaled, b_pre_scaled = scale_arrays_custom(A_pre, b_pre, target_mean=10, target_std=2)
    """
    scaled_arrays = []

    for arr in arrays:
        arr_mean = arr.mean()
        arr_std = arr.std()
        if arr_std == 0:
            raise ValueError("Array has zero standard deviation; cannot scale.")
        arr_scaled = (arr - arr_mean) / arr_std  # Standardize
        arr_scaled = arr_scaled * target_std + target_mean  # Adjust to target mean and std
        scaled_arrays.append(arr_scaled)

    return tuple(scaled_arrays)

def transform_to_quadratic_form(A, b):
    # A: shape = (n_samples, dim)
    # b: shape = (n_samples,)
    X_mat = A.T @ A
    y_vec = 2 * (A.T @ b)
    c_val = np.dot(b, b)
    return X_mat.copy(), y_vec.copy(), c_val

def frobenius_diff(X_pre, X_ft):
    numerator = np.linalg.norm(X_pre - X_ft, 'fro')
    denom = np.linalg.norm(X_pre, 'fro') + np.linalg.norm(X_ft, 'fro')
    return numerator / (denom + 1e-12)  # add eps to avoid /0

def compute_mean_matrix(X, S):
    """
    Computes the mean matrix over the set of indices S.

    Parameters:
    - X: Input array of matrices. 
         It can be a list of scipy sparse csr_matrices or a numpy ndarray of shape (num_samples, dim, dim).
    - S: A list of indices over which to compute the mean matrix.

    Returns:
    - mean_matrix: The mean matrix computed over the specified indices.
    """

    if isinstance(X, np.ndarray):
        # Dense case
        mean_matrix = np.mean(X[S], axis=0).copy()
    elif isinstance(X, list) and all(isinstance(x, csr_matrix) for x in X):
        # Sparse case
        temp_mean = sum(X[i] for i in S) / len(S)
        mean_matrix = temp_mean.toarray()
    else:
        raise ValueError("Unsupported type for X. It must be either a numpy ndarray or a list of scipy csr_matrices.")
    
    return mean_matrix.copy()

def sign_matrix(x):
    x = np.asarray(x)  # Ensure the input is a NumPy array
    signs = np.zeros_like(x)  # Initialize an array of zeros with the same shape as x
    
    signs[x > NUM_ZERO] = 1
    signs[np.abs(x) <= NUM_ZERO] = 0
    signs[x < -NUM_ZERO] = -1
    return signs

def generate_pos_def_matrix(dim, rs, scale):
    M = scale * rs.randn(dim, dim).astype(np.float64)  # Ensure double precision
    return (M @ M.T + 0.1 * np.eye(dim, dtype=np.float64))  # Ensure the matrix is positive definite and double precision

def create_str_id(input_list):
    # Convert all elements to strings
    str_list = [str(element) for element in input_list]
    # Join elements with "_" as separator
    result = "_".join(str_list)
    return result

def str_filter(string_list, exclude_substring):
    """
    Filters out strings from a list that contain a specific substring.

    Args:
    string_list (list): The list of strings to filter.
    exclude_substring (str): The substring to exclude.

    Returns:
    list: A list of strings that do not contain the exclude_substring.
    """
    # Use list comprehension to filter out strings containing the exclude substring
    filtered_list = [s for s in string_list if exclude_substring not in s]
    return filtered_list

def extract_str_multiple(param_set, substrings):
    """
    Extracts all strings from a set that contain all specified substrings.

    Args:
    param_set (set): The set of strings to search.
    substrings (list): A list of substrings to check against each string in the param_set.

    Returns:
    list: A list of strings containing all substrings.

    Raises:
    ValueError: If no strings are found containing all substrings.
    """
    results = []
    # Check each string in the set
    for param in param_set:
        # Check if all substrings are in the current string
        if all(sub in param for sub in substrings):
            results.append(param)
    
    if not results:
        raise ValueError("No strings found containing all specified substrings")
    
    return results

def to_exponential(number, precision=1):
    return f"{number:.{precision}e}"

def convert_array_to_str(array):
    # Convert each element in the array to a string with the desired format
    str_elements = [myrepr(float(x)) if isinstance(x, str) else myrepr(x) for x in array]
    # Join the string elements with dashes
    return '-'.join(str_elements)

def fix_shape(np_array):
    if len (np_array.shape)==2:
        if np_array.shape[0]==1:
            np_array = np_array.flatten()
        else: 
            raise ValueError("wrong shape")
    return np_array

def shapes_alignment(ar1, ar2):
    x_shape = ar1.shape[0]
    y_shape = ar2.shape[0]
    if x_shape != y_shape:
        min_shape =  min(x_shape, y_shape)
        ar1 = ar1[:min_shape]
        ar2 = ar2[:min_shape]
    return ar1.copy(), ar2.copy()

def load_np_array (path_to_file, pickle=True):
    try:
        np_array = np.load(path_to_file, allow_pickle=pickle)
    except IOError:
        print (path_to_file+": its failed to be loaded: IOError")
        np_array = np.array([-1])
    return np_array
    
def zero_small_values(matrix, threshold=1e-14):
    zeroed_matrix = np.where(np.abs(matrix) < threshold, 0, matrix)
    return zeroed_matrix

def extract_tridiagonal(matrix):
    tridiagonal = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(max(0, i - 1), min(matrix.shape[1], i + 2)):
            tridiagonal[i, j] = matrix[i, j]
    return tridiagonal

def exp_round(num, decimal_places):
    num_str = "{:.{precision}e}".format(num, precision=decimal_places+1)
    num_str_parts = num_str.split("e")
    
    # Round the number part
    num_part = round(float(num_str_parts[0]), decimal_places)
    
    # Combine the rounded number part and the exponent part
    return "{:.{precision}f}e{}".format(num_part, num_str_parts[1], precision=decimal_places)

def my_print(str, is_print):
    if is_print:
        print(str)

def is_float(string):
    try:
        float(string) or int(string)
        return True
    except ValueError:
        return False

def load_svmlight_dataset(file_path, is_sparse=True):
    
    if os.path.isfile(file_path):
        if file_path.endswith('.bz2'):
            with bz2file.open(file_path, 'rb') as f:
                data, labels = load_svmlight_file(f)
        else:
            data, labels = load_svmlight_file(file_path)
        
    else:
        raise ValueError("dataset not found ")
    
    enc_labels = labels.copy() 
    if not np.array_equal(np.unique(labels), np.array([-1, 1], dtype='float')):
        min_label = min(np.unique(enc_labels))
        max_label = max(np.unique(enc_labels))
        enc_labels[enc_labels == min_label] = -1
        enc_labels[enc_labels == max_label] = 1
    if is_sparse:
        return csr_matrix(data), enc_labels
    else:
        return data.todense(), enc_labels.copy()
    

def load_param(path, param_name, is_print=0):
    #new version of load_param that supports loading sparse matrices
    """
    Loads a parameter from either a .npy or .h5 file depending on which file exists.

    Parameters:
    - path: The path to the file without extension.
    - param_name: The name of the parameter being loaded.
    - is_print: Boolean flag to indicate if messages should be printed.

    Returns:
    - loaded_param: The loaded parameter.
    """
    npy_file = path + '.npy'
    h5_file = path + '.h5'
    npy_exists = os.path.exists(npy_file)
    h5_exists = os.path.exists(h5_file)

    if npy_exists and h5_exists:
        raise ValueError("Both .npy and .h5 files exist for the given path. Only one should be present.")
    elif npy_exists:
        loaded_param = np.load(npy_file, allow_pickle=True)
        # Unwrap scalar/object arrays to preserve original Python scalars/objects.
        if isinstance(loaded_param, np.ndarray) and loaded_param.shape == ():
            loaded_param = loaded_param.item()
        my_print(f"Loaded {param_name}", is_print)
    elif h5_exists:
        loaded_param = []
        with h5py.File(h5_file, 'r') as f:
            for i in tqdm(range(len(f.keys()))):
                key = f'matrix_{i}'
                matrix_grp = f[key]
                data = matrix_grp['data'][:]
                indices = matrix_grp['indices'][:]
                indptr = matrix_grp['indptr'][:]
                shape = tuple(matrix_grp['shape'][:])
                matrix = csr_matrix((data, indices, indptr), shape=shape)
                loaded_param.append(matrix)
                
        my_print(f"Loaded {param_name}", is_print)
    else:
        raise FileNotFoundError("Neither .npy nor .h5 file exists for the given path.")
    return loaded_param

def load_selected_sparse_matrices(path, param_name, inds, is_print=0):
    """
    Loads selected sparse matrices from a .h5 file based on provided indices.

    Parameters:
    - path: The path to the .h5 file without extension.
    - param_name: The name of the parameter being loaded.
    - inds: A numpy array of indices corresponding to the matrices to be loaded.
    - is_print: Boolean flag to indicate if messages should be printed.

    Returns:
    - loaded_param: A list containing the selected loaded matrices.
    """
    h5_file = path + '.h5'
    h5_exists = os.path.exists(h5_file)

    if not h5_exists:
        raise FileNotFoundError(f".h5 file does not exist for the given path: {path}")

    loaded_param = []
    with h5py.File(h5_file, 'r') as f:
        for i in inds:
            key = f'matrix_{i}'
            if key in f.keys():
                matrix_grp = f[key]
                data = matrix_grp['data'][:]
                indices = matrix_grp['indices'][:]
                indptr = matrix_grp['indptr'][:]
                shape = tuple(matrix_grp['shape'][:])
                matrix = csr_matrix((data, indices, indptr), shape=shape)
                loaded_param.append(matrix)
            else:
                raise KeyError(f"Matrix with key '{key}' not found in the .h5 file.")
    
    #my_print(f"Loaded {param_name} with selected indices", is_print)
    return loaded_param


def save_param(path, param_name, param_value, is_print):
    if param_value is None:
        raise ValueError(f"Cannot save parameter '{param_name}' because its value is None.")
    my_print(f"Saving {param_name}", is_print)
    np.save(path, param_value, allow_pickle=True)


def save_comp_params_bundle(folder_path, params_dict, is_print=0):
    """
    Save computable parameters as a single NPZ bundle.
    Supports scalars and arrays without forcing dtype conversion.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    invalid_keys = [k for k, v in params_dict.items() if v is None]
    if invalid_keys:
        raise ValueError(f"Cannot save comp params with unset values: {invalid_keys}")

    bundle_path = os.path.join(folder_path, "comp_params_bundle.npz")
    meta_path = os.path.join(folder_path, "comp_params_meta.json")

    serializable = {key: np.asarray(value) for key, value in params_dict.items()}
    np.savez(bundle_path, **serializable)

    meta = {
        "format": "comp_params_bundle_v1",
        "keys": {
            key: {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            }
            for key, arr in serializable.items()
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    my_print(f"Saved comp params bundle: {bundle_path}", is_print)
    return bundle_path


def load_comp_params_bundle(folder_path, requested_keys=None, is_print=0):
    """
    Load computable parameters from NPZ bundle if it exists.
    Returns a dict with found keys only.
    """
    bundle_path = os.path.join(folder_path, "comp_params_bundle.npz")
    if not os.path.exists(bundle_path):
        return {}

    loaded = {}
    with np.load(bundle_path, allow_pickle=True) as bundle:
        keys = bundle.files if requested_keys is None else requested_keys
        for key in keys:
            if key not in bundle.files:
                continue
            value = bundle[key]
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            loaded[key] = value

    if loaded:
        my_print(f"Loaded {len(loaded)} comp params from bundle.", is_print)
    return loaded


def save_metrics_bundle(folder_path, metrics_dict, experiment_str, is_print=0, bundle_name=None):
    """
    Save algorithm metrics as a single NPZ bundle for efficient periodic checkpoints.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if bundle_name is None:
        bundle_name = f"metrics_bundle_{experiment_str}.npz"
    bundle_path = os.path.join(folder_path, bundle_name)

    serializable = {}
    for key, value in metrics_dict.items():
        if value is None:
            continue
        serializable[key] = np.asarray(value)

    np.savez(bundle_path, **serializable)
    my_print(f"Saved metrics bundle: {bundle_path}", is_print)
    return bundle_path


def save_metrics_arrays(folder_path, metrics_dict, experiment_str, is_print=0):
    """
    Save algorithm metrics as individual NPY files for plotting compatibility.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for metric, value in metrics_dict.items():
        if value is None:
            continue
        metric_path = os.path.join(folder_path, f"{metric}_{experiment_str}")
        np.save(metric_path, np.asarray(value), allow_pickle=True)
        my_print(f"Saved metric array: {metric_path}.npy", is_print)


def resolve_torch_device(device_str, is_print=0):
    """
    Resolve a requested Torch device string while gracefully falling back to CPU
    when CUDA is unavailable.
    """
    import torch

    if device_str == "cpu":
        return torch.device("cpu")

    normalized = "cuda:0" if device_str == "cuda" else device_str
    if not normalized.startswith("cuda"):
        raise ValueError(f"Unsupported device string: {device_str}")

    if not torch.cuda.is_available():
        my_print("CUDA requested but unavailable. Falling back to CPU.", is_print)
        return torch.device("cpu")

    requested = torch.device(normalized)
    if requested.index is not None and requested.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested CUDA device index {requested.index}, but only {torch.cuda.device_count()} devices are available."
        )
    return requested


def get_silage_m_gt_n_stepsize(L_global, delta2, n_groups, p, factor=1.0):
    """
    Compute the SILAGE_m>n constant step size

        factor / (L + delta2 * sqrt((n - p) / (n p))).
    """
    L_global = float(L_global)
    delta2 = float(delta2)
    n_groups = float(n_groups)
    p = float(p)
    factor = float(factor)

    if L_global < 0 or delta2 < 0:
        raise ValueError(f"L_global and delta2 must be nonnegative, got {L_global}, {delta2}.")
    if n_groups <= 0:
        raise ValueError(f"n_groups must be positive, got {n_groups}.")
    if not (0.0 < p <= 1.0):
        raise ValueError(f"Expected probability p in (0, 1], got {p}.")
    if factor <= 0:
        raise ValueError(f"Expected positive factor, got {factor}.")

    curvature_term = L_global + delta2 * math.sqrt((n_groups - p) / (n_groups * p))
    if curvature_term <= 0:
        raise ValueError(f"Computed nonpositive curvature term {curvature_term}.")
    return factor / curvature_term


def get_silage_n_gt_m_stepsize(L_global, delta1, delta2, n_groups, batch_size, factor=1.0):
    """
    Compute the SILAGE_n>m constant step size

        factor / (L + sqrt(delta1^2 * A1(b) + delta2^2 * A2(b))).

    where

        A1(b) = ((n - b) * (5n + 4)) / (n - 1)
        A2(b) = (n^2 + 2n - 7b + 4b^2) / (2bn).
    """
    L_global = float(L_global)
    delta1 = float(delta1)
    delta2 = float(delta2)
    n_groups = int(n_groups)
    batch_size = int(batch_size)
    factor = float(factor)

    if L_global < 0 or delta1 < 0 or delta2 < 0:
        raise ValueError(
            f"L_global, delta1, and delta2 must be nonnegative, got {L_global}, {delta1}, {delta2}."
        )
    if n_groups <= 1:
        raise ValueError(f"Expected n_groups > 1 in the n>m regime, got {n_groups}.")
    if not (1 <= batch_size <= n_groups):
        raise ValueError(f"Expected batch_size in [1, n_groups], got b={batch_size}, n={n_groups}.")
    if factor <= 0:
        raise ValueError(f"Expected positive factor, got {factor}.")

    n_float = float(n_groups)
    b_float = float(batch_size)
    a1 = ((n_float - b_float) * (5.0 * n_float + 4.0)) / (n_float - 1.0)
    a2 = (n_float * n_float + 2.0 * n_float - 7.0 * b_float + 4.0 * b_float * b_float) / (2.0 * b_float * n_float)
    curvature_term = L_global + math.sqrt(delta1 * delta1 * a1 + delta2 * delta2 * a2)
    if curvature_term <= 0:
        raise ValueError(f"Computed nonpositive curvature term {curvature_term}.")
    return factor / curvature_term


def get_zerosarah_stepsize(L_ij_max, dataset_size, batch_size, factor=1.0):
    """
    Compute the ZeroSARAH constant step size

        factor / (L_ij_max * (1 + sqrt(8 * dataset_size) / batch_size)).
    """
    L_ij_max = float(L_ij_max)
    dataset_size = int(dataset_size)
    batch_size = int(batch_size)
    factor = float(factor)

    if L_ij_max < 0:
        raise ValueError(f"L_ij_max must be nonnegative, got {L_ij_max}.")
    if dataset_size <= 0:
        raise ValueError(f"Expected positive dataset_size, got {dataset_size}.")
    if not (1 <= batch_size <= dataset_size):
        raise ValueError(f"Expected batch_size in [1, dataset_size], got b={batch_size}, N={dataset_size}.")
    if factor <= 0:
        raise ValueError(f"Expected positive factor, got {factor}.")

    curvature_term = L_ij_max * (1.0 + math.sqrt(8.0 * float(dataset_size)) / float(batch_size))
    if curvature_term <= 0:
        raise ValueError(f"Computed nonpositive curvature term {curvature_term}.")
    return factor / curvature_term


def get_d_zerosarah_stepsize(L_ij_max, n_groups, m_per_group, client_subset_size, batch_size, factor=1.0):
    """
    Compute the D-ZeroSARAH constant step size

        factor / (L_ij_max * (1 + sqrt(8 * n * m) / (s * b))).
    """
    L_ij_max = float(L_ij_max)
    n_groups = int(n_groups)
    m_per_group = int(m_per_group)
    client_subset_size = int(client_subset_size)
    batch_size = int(batch_size)
    factor = float(factor)

    if L_ij_max < 0:
        raise ValueError(f"L_ij_max must be nonnegative, got {L_ij_max}.")
    if n_groups <= 0 or m_per_group <= 0:
        raise ValueError(f"Expected positive n_groups and m_per_group, got {n_groups}, {m_per_group}.")
    if not (1 <= client_subset_size <= n_groups):
        raise ValueError(
            f"Expected client_subset_size in [1, n_groups], got s={client_subset_size}, n={n_groups}."
        )
    if not (1 <= batch_size <= m_per_group):
        raise ValueError(f"Expected batch_size in [1, m_per_group], got b={batch_size}, m={m_per_group}.")
    if factor <= 0:
        raise ValueError(f"Expected positive factor, got {factor}.")

    dataset_size = float(n_groups * m_per_group)
    sampling_budget = float(client_subset_size * batch_size)
    curvature_term = L_ij_max * (1.0 + math.sqrt(8.0 * dataset_size) / sampling_budget)
    if curvature_term <= 0:
        raise ValueError(f"Computed nonpositive curvature term {curvature_term}.")
    return factor / curvature_term


def get_silver_stepsize(L, delta_flat, dataset_size, batch_size, factor=1.0):
    """
    Compute the SILVER Option I theory-inspired step size.

        factor * min(1 / L, b / (delta_flat * sqrt(N))).
    """
    L = float(L)
    delta_flat = float(delta_flat)
    dataset_size = int(dataset_size)
    batch_size = int(batch_size)
    factor = float(factor)

    if L <= 0:
        raise ValueError(f"L must be positive, got {L}.")
    if delta_flat < 0:
        raise ValueError(f"delta_flat must be nonnegative, got {delta_flat}.")
    if dataset_size <= 0:
        raise ValueError(f"Expected positive dataset_size, got {dataset_size}.")
    if not (1 <= batch_size <= dataset_size):
        raise ValueError(f"Expected batch_size in [1, dataset_size], got b={batch_size}, N={dataset_size}.")
    if factor <= 0:
        raise ValueError(f"Expected positive factor, got {factor}.")

    if delta_flat == 0.0:
        return factor / L

    dataset_size_float = float(dataset_size)
    batch_size_float = float(batch_size)
    return factor * min(1.0 / L, batch_size_float / (delta_flat * math.sqrt(dataset_size_float)))

def parse_params_to_dict(params_str, allowable_params):
    """
    Parses a string representation of a list into a dictionary with keys from the list and None values.
    
    Args:
    params_str (str): The string representation of a list of parameters.
    allowable_params (set): A set of allowable parameter names.

    Returns:
    dict: A dictionary with keys from the parsed list and None values if valid, else an empty dictionary.
    """
    try:
        params_list = ast.literal_eval(params_str)
    except ValueError:
        print("The string is not a valid list representation.")
        return {}

    if isinstance(params_list, list) and all(isinstance(item, str) for item in params_list):
        return {key: None for key in params_list}
    else:
        print("The list does not contain only string elements.")
        return {}
    
def nan_check (lst):
    """
    Check whether has any item of list np.nan elements
    :param lst: list of datafiles (eg. numpy.ndarray)
    :return:
    """
    for i, item in enumerate (lst):
        if np.sum(np.isnan(item)) > 0:
            raise ValueError("nan files in item {0}".format(i))

def print_time(is_print):
    currentDT = datetime.datetime.now()
    my_print(currentDT.strftime("%Y-%m-%d %H:%M:%S"), is_print)
    
################################
# Sparsificators
################################

def vec_sparsificator(vec, inds):
    output = np.zeros(vec.shape)
    output[inds] = vec[inds]
    return output

def permk_compressor(x, perm, k, i):
    dim = x.shape[0]
    output = np.zeros(x.shape, dtype=np.float64)
    output_perm = x[perm].copy()  ### permute the gradients
    output_small = output_perm[k*i:k*(i+1)] ### select only the part relevant to this node
    output[perm[k*i:k*(i+1)]] =  (dim/k) * output_small ### place the relevant part into the right place
    return output

# TopK and BottomK functions

def top_k_inds(x, k):
    output = np.arange(x.shape[0])
    x_abs = np.abs(x)
    idx = np.argpartition(x_abs, -k)[-k:]  # Indices not sorted
    inds = idx[np.argsort(x_abs[idx])][::-1] #topk inds
    return inds

def top_k_compressor(x, k):
    output = np.zeros(x.shape, dtype=np.float64)
    x_abs = np.abs(x)
    idx = np.argpartition(x_abs, -k)[-k:]  # Indices not sorted
    inds = idx[np.argsort(x_abs[idx])][::-1]
    output[inds] = x[inds]
    return output

def top_k_matrix(X, k):
    output = np.zeros(X.shape, dtype=np.float64)
    for i in range(X.shape[0]):
        output[i] = top_k_compressor(X[i], k)
    return output

def bottom_k_inds(x, k):
    output = np.arange(x.shape[0])
    x_abs = np.abs(x)
    idx = np.argpartition(x_abs, k)[:k]  # Indices of the smallest values
    inds = idx[np.argsort(x_abs[idx])]  # Indices sorted by smallest absolute values
    return inds

def bottom_k_compressor(x, k):
    output = np.zeros(x.shape)
    x_abs = np.abs(x)
    idx = np.argpartition(x_abs, k)[:k]  # Indices of the smallest values
    inds = idx[np.argsort(x_abs[idx])]
    output[inds] = x[inds]
    return output

#################################
# Data preproccessing functions #
#################################

def sort_dataset_by_label(X, y):
    sort_index = np.argsort(y)
    X_sorted = X[sort_index].copy()
    y_sorted = y[sort_index].copy()
    return X_sorted, y_sorted

def matrices_are_equal(matrices1, matrices2):
    if len(matrices1) != len(matrices2):
        return False

    for mat1, mat2 in zip(matrices1, matrices2):
        if not np.array_equal(mat1.shape, mat2.shape):
            return False
        if not np.allclose(mat1.data, mat2.data, atol=1e-10):
            differences = np.abs(mat1.data - mat2.data)
            print("Max difference:", np.max(differences))
            print("Mean difference:", np.mean(differences))
            return False
        if not np.array_equal(mat1.indices, mat2.indices):
            return False
        if not np.array_equal(mat1.indptr, mat2.indptr):
            return False

    return True

def max_eigval(A):
    # Check if A is a sparse matrix
    if scipy.sparse.issparse(A):
        # Compute the maximum eigenvalue of a sparse matrix
        return scipy.sparse.linalg.eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]
    else:
        # Assume A is a dense matrix (numpy.ndarray)
        n_0, d_0 = A.shape
        return np.float64(scipy.linalg.eigh(a=A, eigvals_only=True, turbo=True, type=1, eigvals=(d_0-1, d_0-1))[0])

def min_eigval(A):
    # Check if A is a sparse matrix
    if scipy.sparse.issparse(A):
        # Compute the minimum eigenvalue of a sparse matrix
        return scipy.sparse.linalg.eigsh(A, k=1, which='SA', return_eigenvectors=False)[0]
    else:
        # Assume A is a dense matrix (numpy.ndarray)
        n_0, d_0 = A.shape
        return np.float64(scipy.linalg.eigh(a=A, eigvals_only=True, turbo=True, type=1, eigvals=(0, 0))[0])

def second_matrix_norm(A, tol=1e-5, maxiter=100):
    if scipy.sparse.issparse(A):
        # n = A.shape[0]
        # X = np.random.rand(n, 1)  # Initial guess for the eigenvector
        # eigvals, _ = lobpcg(A, X, tol=tol, maxiter=maxiter)
        # return np.sqrt(eigvals[0])
        return scipy.sparse.linalg.norm(A, ord=2)
    else:
        return np.linalg.norm(A, ord=2)

def compute_L_0(any_vector, X, la, regularizer_hess_ubound, hess_ubound_func, is_print):
    my_print("Computing L_0...", is_print)
    return max_eigval(hess_ubound_func(any_vector, X, la, regularizer_hess_ubound))

def compute_mu_0(any_vector, X, la, regularizer_hess_lbound, hess_lbound_func, is_print):
    my_print("Computing mu_0...", is_print)
    return min_eigval(hess_lbound_func(any_vector, X, la, regularizer_hess_lbound))
    
def compute_Li(any_vector, X, la, regularizer_hess_bound, hess_bounds_func, is_print):
    my_print("Computing Li...", is_print)
    hess_bounds = hess_bounds_func(any_vector, X, la, regularizer_hess_bound)
    num_workers = len(hess_bounds)
    Li = np.zeros(len(hess_bounds), dtype=np.float64)
    for i in tqdm(range(num_workers), desc='Workers Progress'):
        Li[i] = max_eigval(hess_bounds[i])
    return Li

def compute_mui(any_vector, X, la, regularizer_hess_bound, hess_bounds_func, is_print):
    my_print("Computing mui...", is_print)
    hess_bounds = hess_bounds_func(any_vector, X, la, regularizer_hess_bound)
    num_workers = len(hess_bounds)
    mui = np.zeros(num_workers, dtype=np.float64)
    for i in tqdm(range(num_workers), desc='Workers Progress'):
        mui[i] = min_eigval(hess_bounds[i])
    return mui

def compute_muii(any_vector, X, la, regularizer_hess_bound, hess_bounds_func, is_print):
    my_print("Computing muii...", is_print)
    num_workers = len(X)
    num_samples = len(X[0])
    # we assume that each worker has the same number of samples
    assert all(len(X[i]) == num_samples for i in range(num_workers)) # additinal check that catches the case when the number of samples is not the same
    
    muii = np.zeros((num_workers, num_samples), dtype=np.float64)
    
    for i in tqdm(range(num_workers), desc='Workers Progress'):
        for j in range(num_samples):
            hess_ij_bound = hess_bounds_func(any_vector, X[i][j], la, regularizer_hess_bound)
            muii[i, j] = min_eigval(hess_ij_bound)
    return muii

def compute_Lii(any_vector, X, la, regularizer_hess_bound, hess_bounds_func, is_print):
    my_print("Computing Lii...", is_print)
    num_workers = len(X)
    num_samples = len(X[0])
    # we assume that each worker has the same number of samples
    assert all(len(X[i]) == num_samples for i in range(num_workers)) # additinal check that catches the case when the number of samples is not the same
    
    Lii = np.zeros((num_workers, num_samples), dtype=np.float64)
    
    for i in tqdm(range(num_workers), desc='Workers Progress'):
        for j in range(num_samples):
            hess_ij_bound = hess_bounds_func(any_vector, X[i][j], la, regularizer_hess_bound)
            Lii[i, j] = max_eigval(hess_ij_bound)
    return Lii

#calculating memmory ocupation for matrices

# Assuming 'matrices' is your list of numpy arrays
def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and their subclasses: tuple, list, deque, dict, set and frozenset.
    To find other objects, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)  # user-defined handlers
    seen = set()                  # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def size_of(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(size_of, handler(o)))
                break
        return s

    total = size_of(o)
    if verbose:
        print(f"Total memory: {total} bytes")
    return total

def bytes_to_gb(bytes, decimal_places=3):
    return round(bytes / (1024 ** 3), decimal_places)

# Example usage within your memory calculation context
# total_memory_bytes = total_size(matrices, verbose=True)
# total_memory_gb = bytes_to_gb(total_memory_bytes)

# print(f"Total memory usage of 'matrices': {total_memory_gb} GB")


##########################
# Functions for plotting #
##########################
def moving_average(y_metric, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(y_metric, window, 'valid')
    
def moving_average_with_padding(y_metric, window_size):
    # Padding the array
    pad_size = window_size // 2
    y_padded = np.pad(y_metric, (pad_size, pad_size), mode='edge')
    
    # Calculate the moving average
    window = np.ones(window_size) / window_size
    return np.convolve(y_padded, window, mode='same')[pad_size:-pad_size]
    
    
    
########################
# Archived probability-cost helpers #
########################

cost_function_biased = lambda c,k,p: (p + (1 - p)*c)*(1 + ((1 + np.sqrt(1 - p)) / p - 1) * k) # cost function in the biased case
def get_stepsize_biased(L_0, delta, p): 
    return 1/(L_0 + delta*np.sqrt((1-p)/((1-np.sqrt(1-p))**2)))
def get_optimal_params_biased(c, kappa):
    p = smp.symbols('p')
    expr = 0.5*(c*(-kappa*(-2*p**2 + smp.sqrt(1-p)*p + 2*smp.sqrt(1-p) + 2)/p**2 - 2) + kappa*(-1/smp.sqrt(1-p) - 2) + 2)
    sol = np.array(list(map(complex, smp.solve(expr, p, domain=smp.Interval.Lopen(0, 1)))))
    sol_real = np.real(sol)
    if sol_real.shape[0]==3:
        cost_1 = cost_function_biased(c, kappa, sol_real[1])
        cost_2 = cost_function_biased(c, kappa, sol_real[2])
        if cost_1 <= cost_2:
            p_opt = sol_real[1]
            cost_opt = cost_1
        else:
            p_opt = sol_real[2]
            cost_opt = cost_2
        if cost_opt>1:
            p_opt = 1
    elif sol_real.shape[0]==2:
        cost_opt = cost_function_biased(c, kappa, sol_real[1])
        p_opt = sol_real[1]
    else:
        print(c, kappa, sol_real)
        raise ValueError("")
    return p_opt, cost_opt

def cost_prime_unbiased (c,k,p):
    term1 = (-((1 - p) / p**2) - p**(-1)) * (c * (1 - p) + p) * k
    term2 = 2 * np.sqrt((1 - p) / p)
    term3 = (1 - c) * (1 + np.sqrt((1 - p) / p) * k)
    return sign(term1 / term2 + term3)

p_hat_unbiased = lambda c: (3*c)/(3*c + 1)

def cost_prime_grid_unbiased(c_vals,k_vals):
    P = np.array([p_hat_unbiased(c) for c in c_vals], dtype=np.float64)
    #dim = P.shape[0]
    z = np.zeros((k_vals.shape[0], c_vals.shape[0]))
    for i,k in enumerate(k_vals):
        for j,c in enumerate(c_vals):
            z[i,j] = cost_prime_unbiased(c,k,P[j])
    return z

cost_function_unbiased = lambda c,k,p: (p + (1 - p) * c) * (1 + k * np.sqrt((1 - p) / p))
def get_stepsize_unbiased(L_0, delta, p): 
    return 1/(L_0 + delta*np.sqrt((1-p)/(p)))

def get_optimal_params_unbiased(c, kappa):
    p = smp.symbols('p')
    expr = (-((1 - p) / p**2) - p**(-1)) * (c * (1 - p) + p) * kappa / (2 * smp.sqrt((1 - p) / p)) + (1 - c) * (1 + smp.sqrt((1 - p) / p) * kappa)
    
    sol = np.array(list(map(complex, smp.solve(expr, p, domain=smp.Interval.Lopen(0, 1)))))
    sol_real = np.real(sol)
    
    sol_attached = np.append(sol_real, 1.0)
    costs = np.array([cost_function_unbiased (c, kappa, sol) for sol in  sol_attached], dtype=np.float64)
    p_opt = sol_attached[np.argmin (costs)]
    return p_opt, cost_function_unbiased(c, kappa, p_opt)
####################################################################################
