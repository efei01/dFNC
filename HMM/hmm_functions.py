import numpy as np
from scipy.sparse.linalg import eigs
import random
import os
import re
import nibabel as nib
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import itertools
import json
import hashlib
import tempfile
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import time
import pickle
import pandas as pd
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from scipy.special import logsumexp
from osl_dynamics.utils import plotting

# Adapted get_eigenvectors() from github.com/PSYMARKER/leida-python/tree/master/pyleida/signal_tools/_signal_tools.py
def get_eigenvectors(dFC, n=1):
    """
    For a given subject, extract the leading
    eigenvector of each phase-coherence connectivity
    matrix at time t.
    
    Params:
    -------
    dFC : ndarray with shape (N_volumes,N_rois,N_rois). # my change
          Contains the phase-coherence matrices for each time point t.

    n   : int. 
          The number of desired eigenvalues and eigenvectors.
    
    Returns:
    --------
    LEi : ndarray with shape (N_time_points, N_ROIs)
          Extracted leading eigenvectors.
    """
    if not isinstance(dFC,np.ndarray) or (isinstance(dFC,np.ndarray) and dFC.ndim!=3):
        raise Exception("'dFC' must be a 3D array!")
    
    T, N = dFC.shape[0], dFC.shape[-1] # number of time points and number of regions (I also changed this line)
    
    LEi = np.empty((T,n*N))
    for t in range(T):
        avals, avects = eigs(dFC[t,:,:], n, which='LM') # (I also changed this line)
        ponderation = avals.real / np.sum(avals.real)
        for x in range(avects.shape[1]):
            # convention, negative orientation
            if np.mean(avects[:, x] > 0) > .5:
                avects[:, x] *= -1
            elif np.mean(avects[:, x] > 0) == .5 and np.sum(avects[avects[:, x] > 0, x]) > -1. * sum(avects[avects[:, x] < 0, x]):
                avects[:, x] *= -1

        LEi[t] = np.hstack([p * avects.real[:, x].real for x, p in enumerate(ponderation)])

    return LEi

def load_data(data_path: str, data_prep_method: str, standardize: bool, bad_ICs: list[int] | None = None):
    """
    Args:
        data_path        : str
                           Path to the data
        data_prep_method : str
                           'ICA' or 'LEiDA'
        standardize      : bool
                           Whether to standardize the data. If True, each channel (the third dimension of X) is standardized across time. Standardization is typical for ICA. For LEiDA, try without standardization first
        bad_ICs          : list[int]
                           1-based indices of independent components that need to be removed. None by default. Ignored if data_prep_method isn't 'ICA'
                           Note to self: [1, 6, 9, 18] for CIMT/rs, [6, 11, 12, 14] for CIMT/LLstim
    Returns:
        X                : ndarray with shape (n_subjects, n_timepoints, n_channels)
        full_data        : osl_dynamics.data.Data object
    """
    if data_prep_method == 'ICA':   
        with open(os.path.join(data_path, data_path.split('/')[-1] + 'SelectedDataFolders.txt'), 'r') as f:
            subj_order = f.readlines()
    
        X = []
        i = 0
        for tc in sorted(os.listdir(data_path)):
            if re.search(r"[0-9]_timecourses_ica_s1_.nii$", tc):
                nii = nib.load(os.path.join(data_path, tc))
                X.append(nii.get_fdata())
                i += 1
        
        X = np.array(X) # X.shape = (subjects, timepoints, ROIs)
        
        if bad_ICs:
            X = np.delete(X, obj=[IC - 1 for IC in bad_ICs], axis=2)
    elif data_prep_method == 'LEiDA':   
        phase_coherence = np.load(data_path)
        d1, d2, d3, _ = phase_coherence.shape
        phase_coherence_flat = phase_coherence.reshape(d1 * d2, d3, d3)
        X = get_eigenvectors(phase_coherence_flat) # returns ndarray with shape (N_time_points, N_ROIs)
        X = X.reshape(d1, d2, d3)
    else:
        raise ValueError("invalid data_prep_method. Only ICA and LEiDA are supported")

    full_data = Data(X)

    if standardize:
        full_data.prepare({
            "standardize": {},
        })
        
    return X, full_data

def create_folds(n_splits: int, X, y, inner_val_size: float, random_state: int, standardize: bool):
    """
    Args:
        n_splits       : int
                         Number of folds to create
        X              : ndarray with shape (n_subjects, n_timepoints, n_channels)
                         The first output of load_data()
        y              : ndarray with shape (n_subjects,)
                         The experimental cohort of each subject in the EXACT order that the subjects appear in X
        inner_val_size : float in (0, 1)
                         Within each fold, a training set and test set are created, and the training set is further split into an inner training set and an inner 
                         validation set. inner_val_size is the proportion of the training set reserved for the inner validaton set
        random_state   : int
                         The random_state passed into train_test_split() when splitting the training set into an inner training set and an inner validation set
        standardize    : bool
                         Whether to standardize the data. If True, each channel (the third dimension of the osl_dynamics.data.Data objects in split_plan) is standardized across time. Standardization is typical for ICA. For LEiDA, try without standardization first
    Returns:
        split_plan     : dict
                         Keys are 'outer_train', 'outer_val', 'inner_train', and 'inner_val'. Each of the values is a list of length n_splits containing
                         osl_dynamics.data.Data objects
    """
    skf = StratifiedKFold(n_splits=n_splits)

    split_plan = {
        'outer_train': [],
        'outer_val': [],
        'inner_train': [],
        'inner_val': [],
    }

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i + 1}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        
        outer_train, outer_val, y_train, y_test = X[train_index,], X[test_index,], y[train_index,], y[test_index,]
        inner_train, inner_val, y_inner_train, y_inner_val = train_test_split(
            outer_train, 
            y_train, 
            test_size=inner_val_size, 
            random_state=random_state, 
            stratify=y_train
        )
        
        outer_train = Data(outer_train)
        outer_val = Data(outer_val)
        inner_train = Data(inner_train)
        inner_val = Data(inner_val)

        if standardize:
            outer_train.prepare({"standardize": {}})
            outer_val.prepare({"standardize": {}})
            inner_train.prepare({"standardize": {}})
            inner_val.prepare({"standardize": {}})
    
        split_plan['outer_train'].append(outer_train)
        split_plan['outer_val'].append(outer_val)
        split_plan['inner_train'].append(inner_train)
        split_plan['inner_val'].append(inner_val)

    return split_plan

def print_n_param_updates_per_epoch(hyperparam_grid: dict[str, list[float]], full_data):
    """
    Args:
        hyperparam_grid : dict
                          Keys are hyperparameter names. Values should be lists containing values of the corresponding hyperparameter that need to be searched. For this function, values for 'sequence_length' and 'batch_size' must be provided
        full_data       : osl_dynamics.data.Data object
                          Second output of load_data()
    Returns:
        None
    """
    if hyperparam_grid['sequence_length'] and hyperparam_grid['batch_size']:
        for sequence_length in hyperparam_grid['sequence_length']:
            for batch_size in hyperparam_grid['batch_size']:  
                print(f"Number of parameter updates per epoch with sequence_length={sequence_length} and batch_size={batch_size}: {int(np.ceil(full_data.n_samples / (sequence_length * batch_size)))}")
    else:
        raise ValueError("Either 'sequence_length' or 'batch_size' (or both) is missing. Values for both 'sequence_length' and 'batch_size' must be provided in hyperparam_grid in order to print the number of parameter updates per epoch")
    
def get_hyperparam_combinations(hyperparam_grid: dict):
    """
    Args:
        hyperparam_grid         : dict
                                  Keys are hyperparameter names. Values should be lists containing values of the corresponding hyperparameter that need to be searched
    Returns:
        hyperparam_combinations : list[dict]
                                  Contains the hyperparameter combinations as dictionaries. Each combination dictionary has the hyperparameter names as keys and a
                                  single corresponding value for each hyperparameter as the values
    """
    combinations = itertools.product(*hyperparam_grid.values())
    hyperparam_combinations = [dict(zip(hyperparam_grid.keys(), combination)) for combination in combinations]
    return hyperparam_combinations

def make_seed(base_seed: int, *parts) -> int:
    payload = json.dumps([base_seed, *parts], sort_keys=True).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")

def set_all_seeds(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)

def as_serializable_history(history):
    """
    Convert osl-dynamics/Keras history into a pickle-friendly dict.
    """
    if hasattr(history, "history"):
        history = history.history

    return {
        key: [float(x) for x in values]
        for key, values in history.items()
    }

def save_log_atomic(obj, path):
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def run_grid_search(grid_search_log: dict, grid_search_log_save_path: str, hyperparam_grid: list[dict], seed: int, split_plan: dict):
    """
    Args:
        grid_search_log           : dict
                                    Can be empty or not. Will be modified by this function
        grid_search_log_save_path : str
                                    The path at which to save grid_search_log as a pickle. A save is done after the grid search for each k value
        hyperparam_grid           : dict
                                    Keys are hyperparameter names. Values should be lists containing values of the corresponding hyperparameter that need to be searched. Each combination dictionary must have keys 'k', 'sequence_length', 'learn_means', 'learn_covariances', 'set_regularizers', 'batch_size', 'learning_rate', 'lr_decay', 'n_epochs', and 'patience'
        seed                      : int
                                    For reproducibility
        split_plan                : dict
                                    Output of create_folds()
    Returns:
        None
    """
    for i, hyperparams in enumerate(get_hyperparam_combinations(hyperparam_grid)):
        print(f"\nHyperparam set {i + 1}: {hyperparams}")

        k = hyperparams.pop('k')
        if k not in grid_search_log:
            grid_search_log[k] = {}
        
        hp_key = tuple(sorted(hyperparams.items()))

        n_folds = len(split_plan['outer_train'])

        entry = grid_search_log[k].setdefault(hp_key, {
            'hyperparams': hyperparams,
            'inner_histories': [], # [history object from inner training for each fold]
            'best_epochs': [], # [the best inner training epoch for each fold]
            'inner_epochs': [], # [number of epochs actually run during inner training by early stopping]
            'outer_epochs': [], # [the number of epochs the outer model was fit for in each fold]
            'outer_free_energies': [], # [the free energy of the model fitted on outer_train and evaluated on outer_val for each fold]
            'status': "incomplete",
        })

        n_done = len(entry['outer_free_energies'])

        if entry.get('status') == "complete" and n_done == n_folds:
            print(f"Hyperparam set {i + 1} has already been evaluated, skipping...")
            continue

        for f in range(n_done, n_folds):
            start = time.perf_counter()
                
            print(f"\nFold {f + 1}/{n_folds}...")
            outer_train, outer_val, inner_train, inner_val = split_plan['outer_train'][f], split_plan['outer_val'][f], split_plan['inner_train'][f], split_plan['inner_val'][f]

            # Inner fit: mainly so we can plot validation loss curves later and for getting a rough estimate of what n_epochs should be
            inner_config = Config(
                n_states=k,
                n_channels=inner_train.n_channels,
                sequence_length=hyperparams['sequence_length'],
                learn_means=hyperparams['learn_means'],
                learn_covariances=hyperparams['learn_covariances'],
                batch_size=hyperparams['batch_size'], 
                learning_rate=hyperparams['learning_rate'], # Adam is the default optimizer
                lr_decay=hyperparams['lr_decay'], # exponential decay schedule by default
                n_epochs=hyperparams['n_epochs'],
            )
                
            set_all_seeds(make_seed(seed, "inner", int(k), int(f)))
            inner_model = Model(inner_config)

            if hyperparams['set_regularizers']:
                    inner_model.set_regularizers(inner_train)
                    
            try:
                inner_model.random_state_time_course_initialization(inner_train, verbose=0)
            except ValueError:
                print("random_state_time_course_initialization can't simulate a state time course where each state activates. Switching to using random_subset_initialization instead.")
                inner_model.random_subset_initialization(inner_train, verbose=0)
                    
            callback = EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=hyperparams['patience'],
                min_delta=0.0,
                restore_best_weights=False,
                verbose=0,
            )
            inner_history = inner_model.fit(
                inner_train,
                epochs=hyperparams['n_epochs'],
                validation_data=inner_val.dataset(
                    sequence_length=hyperparams['sequence_length'],
                    batch_size=hyperparams['batch_size'],
                ),
                verbose=0,
                callbacks=[callback],
            )

            inner_history = as_serializable_history(inner_history)
                        
            grid_search_log[k][hp_key]['inner_histories'].append(inner_history)
            best_epoch = np.argmin(inner_history['val_loss']) + 1
            grid_search_log[k][hp_key]['best_epochs'].append(best_epoch)
            grid_search_log[k][hp_key]['inner_epochs'].append(len(inner_history['val_loss']))

            # Outer fit: fit on all of outer_train for (approximately) the number of epochs determined by the inner fit, then evaluate on outer_val. The performance on outer_val is used to determine the best hyperparameters
            outer_config = Config(
                n_states=k,
                n_channels=outer_train.n_channels,
                sequence_length=hyperparams['sequence_length'],
                learn_means=hyperparams['learn_means'],
                learn_covariances=hyperparams['learn_covariances'],
                batch_size=hyperparams['batch_size'], 
                learning_rate=hyperparams['learning_rate'], # Adam is the default optimizer
                lr_decay=hyperparams['lr_decay'], # exponential decay schedule by default
                n_epochs=hyperparams['n_epochs'],
            )

            set_all_seeds(make_seed(seed, "outer", int(k), int(f)))
            outer_model = Model(outer_config)
                
            if hyperparams['set_regularizers']:
                    outer_model.set_regularizers(outer_train)
                    
            try:
                outer_model.random_state_time_course_initialization(outer_train, verbose=0)
            except ValueError:
                print("random_state_time_course_initialization can't simulate a state time course where each state activates. Switching to using random_subset_initialization instead.")
                outer_model.random_subset_initialization(outer_train, verbose=0)

            outer_epochs = min(
                    hyperparams['n_epochs'], 
                    int(np.ceil(best_epoch + 0.25 * hyperparams["patience"])) # add small buffer to best_epoch since we're fitting on a different amount of data now
            )
            outer_history = outer_model.fit(
                outer_train,
                epochs=outer_epochs,
                verbose=0,
            )
                
            grid_search_log[k][hp_key]['outer_epochs'].append(outer_epochs)
            outer_free_energy = outer_model.free_energy(outer_val) 
            grid_search_log[k][hp_key]['outer_free_energies'].append(outer_free_energy)

            save_log_atomic(grid_search_log, grid_search_log_save_path)

            end = time.perf_counter()
            time_elapsed = int(np.ceil(end - start))
            min_elapsed, sec_elapsed = divmod(time_elapsed, 60)
            print(f"Time elapsed for this fold: {min_elapsed} min. {sec_elapsed} sec.")
    
        entry['status'] = "complete"
        save_log_atomic(grid_search_log, grid_search_log_save_path)

def hyperparam_performance(grid_search_log: dict, k: int):
    """
    Args:
        grid_search_log : dict
                          grid_search_log as modified by run_grid_search()
        k               : int
                          k value for which to print searched hyperparameters and corresponding results
    Returns:
        performance_df  : pandas DataFrame
                          Contains the hyperparameter combinations that were searched and their corresponding average outer free energy across folds and average best inner training epoch across folds. Sorted by average outer free energy in ascending order (lowest average outer free energy at the top)
    """
    candidates = []

    for hp_key, entry in grid_search_log[k].items():
        outer_fe = np.asarray(entry.get('outer_free_energies', []), dtype=float)

        if entry.get('status') != "complete":
            continue

        if len(outer_fe) == 0 or np.all(np.isnan(outer_fe)):
            continue

        score = np.nanmean(outer_fe)
        candidates.append((score, hp_key))

    if len(candidates) == 0:
        raise ValueError(f"No complete grid-search entries found for k={k}.")

    candidates.sort(key=lambda res: res[0])  # lower free energy is better

    performance_df = pd.DataFrame({
        'k': [k] * len(candidates),
        'hyperparams': [grid_search_log[k][res[1]]['hyperparams'] for res in candidates],
        'average_outer_free_energy': [np.nanmean(grid_search_log[k][res[1]]['outer_free_energies']) for res in candidates],
        'best_inner_training_epochs': [grid_search_log[k][res[1]]['best_epochs'] for res in candidates]
    })

    return performance_df

def plot_cv_loss(grid_search_log: dict, k: int):
    """
    Args:
        grid_search_log : dict
                          grid_search_log as modified by run_grid_search()
        k               : int
                          k value for which to plot the training and validation loss curves of optimal hyperparameters
    Returns:
        None
    """
    performance_df = hyperparam_performance(grid_search_log, k)
    best_res = performance_df.iloc[0] 
    hyperparams, best_inner_training_epochs = best_res['hyperparams'], best_res['best_inner_training_epochs']
    hp_key = tuple(sorted(hyperparams.items()))
    
    for f in range(len(best_inner_training_epochs)):
        inner_history = grid_search_log[k][hp_key]['inner_histories'][f]

        fig, ax = plt.subplots(1, 1)
        x = range(1, len(inner_history['loss']) + 1)
        ax.plot(x, inner_history['loss'], label="Training Loss", color='blue', linestyle='-')
        ax.plot(x, inner_history['val_loss'], label="Validation Loss", color='orange', linestyle='--')
    
        ax.set_title(f"{k} States, {str(hyperparams)}\nLowest validation loss achieved: {np.min(inner_history['val_loss']):.3f} at epoch {best_inner_training_epochs[f]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Free Energy)")
        ax.legend()
        plt.show()

def choose_final_epochs(selected_entry: dict, quantile: float = 0.75) -> int:
    """
    Choose final full-data training duration from grid-search outer_epochs.
    """
    hyperparams = selected_entry['hyperparams']
    max_epochs = int(hyperparams['n_epochs'])

    outer_epochs = np.asarray(selected_entry['outer_epochs'], dtype=float)

    if len(outer_epochs) == 0 or np.all(np.isnan(outer_epochs)):
        raise ValueError("Selected hyperparameter entry has no usable outer_epochs.")

    final_epochs = int(np.ceil(np.nanquantile(outer_epochs, quantile)))
    final_epochs = max(1, min(max_epochs, final_epochs))

    return final_epochs

# Adapted osl_dynamics.models.inf_mod_base.MarkovStateInferenceModelBase.evidence() to return the full log-likelihood
def hmm_total_loglik(model, dataset):
    """
    Args:
        model       : osl_dynamics.models.hmm.Model object
        dataset     : osl_dynamics.data.Data object
    Returns:
        full_loglik : float
                      The full log-likelihood of the dataset given the model
    """
    ds = model.make_dataset(dataset, concatenate=True)

    eps = np.finfo(float).eps # to prevent taking np.log(0)
    log_pi = np.log(model.get_initial_state_probs() + eps) # log p(s_1)
    log_A = np.log(model.get_trans_prob() + eps) # log p(s_t | s_{t-1})

    total_loglik = 0.0

    for batch in ds:
        x = batch["data"]
        if hasattr(x, "numpy"):
            x = x.numpy()
        else:
            x = np.asarray(x)

        batch_size, sequence_length, _ = x.shape # x should have shape (batch_size, sequence_length, n_channels)

        log_filt = None 

        for t in range(sequence_length):
            if log_filt is None: # then initialize distribution for every sequence in the batch
                log_pred = np.broadcast_to(log_pi[None, :], (batch_size, model.config.n_states)) # log p(s_t | x_{1:t-1})
            else:
                log_pred = logsumexp(
                    log_filt[:, :, None] + log_A[None, :, :], # log_filt is expanded to have shape (batch_size, k, 1) and log_A is expanded to have shape (1, k, k). Their sum has shape (batch_size, k, k). logsumexp is taken across states from time t - 1, which are represented by axis 1
                    axis=1,
                ) # log p(s_t | x_{1:t-1})

            # log_pred has shape (batch_size, k)
            
            log_B = model.get_log_likelihood(x[:, t:t+1, :])[:, 0, :] # log p(x_t | s_t). log_B has shape (batch_size, k)
            log_filt = log_pred + log_B # log p(x_t, s_t∣x_{1:t-1}) (unnormalized). log_filt has shape (batch_size, k)

            log_c = logsumexp(log_filt, axis=1) # log p(x_t | x_{1:t-1}). # log_sum_exp is taken across states for each sequence in the batch. log_c has shape (batch_size,)
            total_loglik += log_c.sum() # sum is taken across sequences in the batch

            # normalize for next step
            log_filt -= log_c[:, None] # log p(s_t | x_{1:t}) (normalized). Expand log_c to have shape (batch_size, 1), then broadcast and subtract from log_filt

    return total_loglik

def count_hmm_params(k: int, d: int, learn_means: bool, learn_covariances: bool) -> int:
    """
    Approximate number of free parameters for BIC.
    """
    n_params = 0

    # Initial probabilities: k probabilities, constrained to sum to 1
    n_params += k - 1

    # Transition matrix: k rows, each constrained to sum to 1
    n_params += k * (k - 1)

    if learn_means:
        n_params += k * d # d x 1 mean vector per state

    if learn_covariances:
        n_params += k * d * (d + 1) / 2 # d x d covariance matrix per state. Since the covariance is symmetric, we just keep the upper triangle, including the main diagonal, which sums to (d * (d - 1)) / 2 + d = (d * (d - 1) + 2d) / 2 = (d * (d + 1)) / 2 covariance params per state

    return n_params

def off_diagonal_covariances(covs: np.ndarray) -> np.ndarray:
    """
    Return off-diagonal upper-triangle covariance entries as shape:
    n_edges x k
    """
    k = covs.shape[0]
    d = covs.shape[1]

    iu = np.triu_indices(d, k=1)
    return np.stack([covs[state][iu] for state in range(k)], axis=1)

def is_better_score(new_score: float, old_score: float | None, metric: str) -> bool:
    """
    Decide whether a new realization is better than the current best.
    """
    if old_score is None or np.isnan(old_score):
        return True

    if metric == "total_LL":
        return new_score > old_score

    if metric in {"free_energy", "BIC"}:
        return new_score < old_score

    raise ValueError(
        "model_save_metric must be one of: 'free_energy', 'total_LL', or 'BIC'."
    )

def run_full_model_eval(model_eval_log: dict, model_eval_log_save_path: str, grid_search_log: dict, k_values: list[int], n_realizations: int, model_save_metric: str, seed: int, full_data, results_path: str):
    """
    Fit n_realizations full-data HMMs for each k using hyperparameters selected
    from grid_search_log. Save all realization summaries and save the best model.

    Args:
        model_eval_log
        model_eval_log_save_path :
        grid_search_log
        k_values
        n_realizations
        model_save_metric
        seed
        full_data
        results_path
    Returns:
        None
    """
    valid_metrics = {"free_energy", "total_LL", "BIC"}
    if model_save_metric not in valid_metrics:
        raise ValueError(f"model_save_metric must be one of {valid_metrics}.")

    for k in k_values:
        print(f"\nFitting full-data model with {k} states...")

        # Select best hyperparameter setting from the grid_search_log
        performance_df = hyperparam_performance(grid_search_log, k)
        best_res = performance_df.iloc[0]
        best_hyperparams, avg_outer_fe = best_res['hyperparams'], best_res['average_outer_free_energy']
        hp_key = tuple(sorted(best_hyperparams.items()))
        best_entry = grid_search_log[k][hp_key]

        final_epochs = choose_final_epochs(best_entry, quantile=0.75)

        # Preserve the original schedule horizon
        schedule_n_epochs = int(best_hyperparams['n_epochs'])

        model_dir = f"{results_path}/{k}_states"
        os.makedirs(model_dir, exist_ok=True)

        best_realization_path = f"{model_dir}/{k}_states_model"

        # Initialize / validate model_eval_log entry
        if k not in model_eval_log:
            model_eval_log[k] = {}

        eval_entry = model_eval_log[k]

        # If this entry was created using a different selected hyperparameter setting, reset it. This prevents accidentally mixing results from different specs
        spec = {
            'hp_key': hp_key,
            'selected_grid_score': float(avg_outer_fe),
            'final_epochs': int(final_epochs),
            'schedule_n_epochs': int(schedule_n_epochs),
            'n_realizations': int(n_realizations),
            'model_save_metric': model_save_metric,
        }

        existing_spec = eval_entry.get('spec')

        if existing_spec is not None and existing_spec != spec:
            print("Existing evaluation spec differs from current grid-search selection. Resetting this k")
            model_eval_log[k] = {}
            eval_entry = model_eval_log[k]

        eval_entry.setdefault('spec', spec)
        eval_entry.setdefault('selected_hyperparams', best_hyperparams)
        eval_entry.setdefault('realizations', [])
        eval_entry.setdefault('best_realization_idx', None)
        eval_entry.setdefault('best_realization_score', None)
        eval_entry.setdefault('best_realization_path', best_realization_path)
        eval_entry.setdefault('status', "incomplete")

        n_done = len(eval_entry['realizations'])

        if eval_entry['status'] == "complete" and n_done >= n_realizations:
            print(f"The model with {k} states has already been evaluated, skipping...")
            continue

        for r in range(n_done, n_realizations):
            start = time.perf_counter()
            print(f"Realization {r + 1}/{n_realizations}:")

            realization_seed = make_seed(seed, "full", int(k), int(r))
            set_all_seeds(realization_seed)

            config = Config(
                n_states=k,
                n_channels=full_data.n_channels,
                sequence_length=best_hyperparams['sequence_length'],
                learn_means=best_hyperparams['learn_means'],
                learn_covariances=best_hyperparams['learn_covariances'],
                batch_size=best_hyperparams['batch_size'],
                learning_rate=best_hyperparams['learning_rate'],
                lr_decay=best_hyperparams['lr_decay'],
                n_epochs=schedule_n_epochs,
            )

            model = Model(config)

            if best_hyperparams['set_regularizers']:
                model.set_regularizers(full_data)

            init_method = "random_state_time_course"

            try:
                model.random_state_time_course_initialization(full_data, verbose=0)
            except ValueError:
                print("random_state_time_course_initialization can't simulate a state time course where each state activates. Switching to using random_subset_initialization instead.")
                init_method = "random_subset"
                model.random_subset_initialization(full_data, verbose=0)

            history = model.fit(
                full_data,
                epochs=final_epochs,
                verbose=0,
            )

            history = as_serializable_history(history)

            free_energy = model.free_energy(full_data)
            total_LL = hmm_total_loglik(model, full_data)

            n_estimated_params = count_hmm_params(
                k=k,
                d=full_data.n_channels,
                learn_means=best_hyperparams['learn_means'],
                learn_covariances=best_hyperparams['learn_covariances'],
            )

            BIC = n_estimated_params * np.log(full_data.n_samples) - 2 * total_LL

            realization = {
                'realization': r,
                'seed': int(realization_seed),
                'init_method': init_method,
                'history': history,
                'final_epochs': final_epochs,
                'schedule_n_epochs': schedule_n_epochs,
                'free_energy': free_energy,
                'total_LL': total_LL,
                'BIC': BIC,
            }

            means, covs = model.get_means_covariances()

            if best_hyperparams['learn_means']:
                realization['means'] = means

            if best_hyperparams['learn_covariances']:
                realization['covs'] = covs
                realization['off_diags'] = off_diagonal_covariances(covs)

            # Useful for later reproducibility checks
            realization['trans_matrix'] = model.get_trans_prob()
            realization['initial_state_probs'] = model.get_initial_state_probs()

            eval_entry['realizations'].append(realization)

            metric_value = realization[model_save_metric]

            if is_better_score(
                new_score=metric_value,
                old_score=eval_entry.get("best_realization_score"),
                metric=model_save_metric,
            ):
                if model_save_metric == "total_LL":
                    print("New best realization by highest total log-likelihood")
                else:
                    print(f"New best realization by lowest {model_save_metric}")

                model.save(best_realization_path)

                eval_entry["best_realization_idx"] = r
                eval_entry["best_realization_score"] = metric_value
                eval_entry["best_realization_path"] = best_realization_path

            save_log_atomic(model_eval_log, model_eval_log_save_path)

            end = time.perf_counter()
            time_elapsed = int(np.ceil(end - start))
            min_elapsed, sec_elapsed = divmod(time_elapsed, 60)
            print(f"Time elapsed for this realization: {min_elapsed} min. {sec_elapsed} sec")

            # Optional but helpful in long TensorFlow loops
            del model

        eval_entry['status'] = "complete"
        save_log_atomic(model_eval_log, model_eval_log_save_path)

def plot_state_time_course(subj_tc, n_samples: int, cmap: str, title: str, stim_times: list[int] | None = None):
    """
    Args:
        subj_tc    : ndarray with shape (n_timepoints, k)
                     The state time course of the subject of interest
        n_samples  : int
                     How many time points to plot
        cmap       : str
                     The Matplotlib colormap
        title      : str
                     The title of the plot
        stim_times : list[int]
                     The time points of stimulation/task, if relevant
    Returns:
        None
    """
    fig, ax = plotting.plot_alpha(subj_tc, n_samples=n_samples, cmap=cmap, title=title)
    
    if stim_times:
        for i in range(len(stim_times)):
            if stim_times[i]:
                ax[0].plot(i, 0, color='black', marker='o')

def get_mean_phase_coherences(phase_coherence, full_data, stc):
    """
    Args:
        phase_coherence       : ndarray with shape (n_subjects, n_timepoints, n_channels, n_channels)
                                The raw phase coherence data used to create X
        full_data             : osl_dynamics.data.Data object
                                Second output of load_data()
        stc                   : ndarray with shape (n_subjects, n_timepoints, k)
                                For each subject and timepoint, the ith state, 1 <= i <= k, is 1 if it is active (as determined by the HMM) and 0 otherwise
    Returns:
        mean_phase_coherences : ndarray with shape (k, n_channels, n_channels)
                                The mean phase coherence matrix of all the timepoints across all subjects assigned to each state
    """
    k = stc.shape[2]
    
    phase_coherence_flattened = np.reshape(
        phase_coherence, 
        (full_data.n_samples, phase_coherence.shape[2], phase_coherence.shape[3])
    )
    stc_flattened = np.reshape(
        stc,
        (full_data.n_samples, k)
    )
    
    mean_phase_coherences = []

    for state in range(k):
        assignment_idx = stc_flattened[:, state] == 1
        mean_phase_coherences.append(np.mean(phase_coherence_flattened[assignment_idx], axis=0))

    return np.array(mean_phase_coherences)

def leida_sanity_check(mean_phase_coherences, state_means, state_covs, figsize: tuple):
    """
    Args:
        mean_phase_coherences : ndarray with shape (k, n_channels, n_channels)
                                The output of get_mean_phase_coherences()
        state_means           : ndarray with shape (k, n_channels)
                                The Gaussian mean vectors of the states learned by the HMM
        state_covs            : ndarray with shape (k, n_channels, n_channels)
                                The Gaussian covariance matrices of the states learned by the HMM   
        figsize               : tuple with length 2
                                The fig_size of each Matplotlib subplot
    Returns:
        outer_products        : ndarray with shape (k, n_channels, n_channels)
                                The rank-1 approximation of the phase coherence of each state
    """
    outer_products = []

    for state in range(mean_phase_coherences.shape[0]):
        outer_products.append(np.outer(state_means[state], state_means[state]))
    
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        axs[0].imshow(mean_phase_coherences[state])
        axs[0].set_title("Mean Phase Coherence")
        axs[1].imshow(state_covs[state] + outer_products[state])
        axs[1].set_title("Expectation of Outer Product of \nLeading Eigenvector with Itself")
        plt.suptitle(f"State {state + 1}")
        plt.show()

    return np.array(outer_products)

def create_hypothesis_testing_dataframe(group_data: dict, group_labels: list, metrics: list, k: int):
    """
    Args:
        group_data   : dict
        group_labels : list
        metrics      : list
        k            : int
    Returns:
        ht_df        : pandas DataFrame
    """
    group_dfs = []

    for label in group_labels:
        group_df = pd.DataFrame({
            f'{metric}_state{state + 1}': list(group_data[label][metric][:, state]) for metric in metrics for state in range(k)
        })
        group_df['group'] = label
    
        group_dfs.append(group_df)
    
    ht_df = pd.concat(group_dfs, axis=0, ignore_index=True)
    return ht_df

def separate_trans_probs(group_data: dict, group_labels: list, k: int):
    """
    Args:
        group_data   : dict
        group_labels : list
        k            : int
    Returns:
        tp_df        : pandas DataFrame
    """
    group_dfs = []
    for label in list(group_labels):
        group_df = pd.DataFrame({
            f'tp_{i + 1}to{j + 1}': list(group_data[label]['trans_matrices'][:, i, j]) for i in range(k) for j in range(k)
        })
        group_df['group'] = label
        
        group_dfs.append(group_df)
        
    tp_df = pd.concat(group_dfs, axis=0, ignore_index=True)
    return tp_df