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
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import time
import pickle
from osl_dynamics.data import Data, processing
from osl_dynamics.models.hmm import Config, Model
from scipy.special import logsumexp
from osl_dynamics.models import load
from osl_dynamics.utils import plotting
from osl_dynamics.inference import modes
from osl_dynamics.analysis import post_hoc
from collections import defaultdict
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal, f_oneway, permutation_test

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

def run_grid_search(model_eval_log: dict, model_eval_log_save_path: str, hyperparam_grid: list[dict], seed: int, split_plan: dict):
    """
    Args:
        model_eval_log           : dict
                                   Can be empty or not. Will be modified by this function
        model_eval_log_save_path : str
                                   The path at which to save model_eval_log as a pickle. A save is done after the grid search for each k value
        hyperparam_grid          : dict
                                   Keys are hyperparameter names. Values should be lists containing values of the corresponding hyperparameter that need to be searched. Each combination dictionary must have keys 'k', 'sequence_length', 'learn_means', 'learn_covariances', 'set_regularizers', 'batch_size', 'learning_rate', 'lr_decay', 'n_epochs', and 'patience'
        seed                     : int
                                   For reproducibility
        split_plan               : dict
                                   Output of create_folds()
    Returns:
        None
    """
    for i, hyperparams in enumerate(get_hyperparam_combinations(hyperparam_grid)):
        print(f"\nHyperparam set {i + 1}: {hyperparams}")

        k = hyperparams.pop('k')
        if k not in model_eval_log:
            model_eval_log[k] = {}
        
        hp_key = sorted(tuple(hyperparams.items()))
        if hp_key not in model_eval_log[k]:
            model_eval_log[k][hp_key] = {}
            model_eval_log[k][hp_key]['hyperparams'] = hyperparams
            model_eval_log[k][hp_key]['inner_histories'] = [] # [history object from inner training for each fold]
            model_eval_log[k][hp_key]['best_epochs'] = [] # [the best inner training epoch for each fold]
            model_eval_log[k][hp_key]['inner_epochs'] = [] # [number of epochs actually run during inner training by early stopping]
            model_eval_log[k][hp_key]['outer_epochs'] = [] # [the number of epochs the outer model was fit for in each fold]
            model_eval_log[k][hp_key]['outer_free_energies'] = [] # [the free energy of the model fitted on outer_train and evaluated on outer_val for each fold]

            for f in range(len(split_plan['outer_train'])):
                start = time.perf_counter()
                
                print(f"\nFold {f + 1}...")
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
                        
                model_eval_log[k][hp_key]['inner_histories'].append(inner_history)
                best_epoch = np.argmin(inner_history['val_loss']) + 1
                model_eval_log[k][hp_key]['best_epochs'].append(best_epoch)
                model_eval_log[k][hp_key]['inner_epochs'].append(len(inner_history['val_loss']))

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
                
                model_eval_log[k][hp_key]['outer_epochs'].append(outer_epochs)
                outer_free_energy = outer_model.free_energy(outer_val) 
                model_eval_log[k][hp_key]['outer_free_energies'].append(outer_free_energy)

                end = time.perf_counter()
                time_elapsed = int(np.ceil(end - start))
                min_elapsed, sec_elapsed = divmod(time_elapsed, 60)
                print(f"Time elapsed for this realization: {min_elapsed} min. {sec_elapsed} sec.")
        else:
            print(f"Hyperparam set {i + 1} has already been evaluated, skipping...")
    
        with open(model_eval_log_save_path, 'wb') as f:
            pickle.dump(model_eval_log, f)

def plot_cv_loss(model_eval_log: dict, k: int, split_plan: dict):
    """
    Args:
        model_eval_log : dict
                         model_eval_log as modified by run_grid_search()
        k              : int
                         k value for which to plot the training and validation loss curves of optimal hyperparameters
        split_plan     : dict
                         Output of create_folds()
    Returns:
        None
    """
    sorted_model_eval_log = sorted(model_eval_log[k].items(), key=lambda item: np.mean(item[1]['outer_free_energies'])) # sort hyperparameter combinations by mean outer free energy across folds
    example = sorted_model_eval_log[0] # get the inner training history of the best hyperparameter combination (the one with the lowest mean outer free energy across folds)
    
    for f in range(len(split_plan['outer_train'])):
        fig, ax = plt.subplots(1, 1)
        x = range(1, len(example[1]['inner_histories'][f]['loss']) + 1)
        ax.plot(x, example[1]['inner_histories'][f]['loss'], label="Training Loss", color='blue', linestyle='-')
        ax.plot(x, example[1]['inner_histories'][f]['val_loss'], label="Validation Loss", color='orange', linestyle='--')
    
        ax.set_title(f"{k} States, {str(example[0])}\nLowest validation loss achieved: {np.min(example[1]['inner_histories'][f]['val_loss']):.3f} at epoch {example[1]['best_epochs'][f]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Free Energy)")
        ax.legend()
        plt.show()

def hyperparam_performance(model_eval_log: dict, k: int):
    """
    Args:
        model_eval_log : dict
                         model_eval_log as modified by run_grid_search()
        k              : int
                         k value for which to print searched hyperparameters and corresponding results
    Returns:
        None
    """
    for h, t, e in sorted(
        zip(
            model_eval_log[k]['hyperparams'], 
            np.mean(model_eval_log[k]['outer_free_energies'], axis=1), 
            np.mean(model_eval_log[k]['best_epochs'], axis=1)
        ), 
        key=lambda x: x[1]
    ):
        print(f"{h}: {t:.3f}, took {e:.1f} epochs on average")

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
    
def run_full_model_eval(model_eval_log: dict, model_eval_log_save_path: str, k_values: list[int], n_realizations: int, model_save_metric: str, seed: int, full_data, results_path: str):
    """
    Args:
        model_eval_log           : dict
                                   model_eval_log as modified by run_grid_search(). The actual values of the model evaluation metrics will be stored here
        model_eval_log_save_path : str
                                   The path at which to save model_eval_log as a pickle. A save is done after the grid search for each k value
        k_values                 : list[int]
                                   k values for which to run model evaluation for
        n_realizations           : int
                                   Number of models to fit for each k value to account for variance in model initialization
        model_save_metric        : str
                                   'free_energy', 'total_LL' (total log-likelihood), or 'BIC' (Bayesian information criterion).
                                   The realization that scores the best on model_save_metric is saved
        seed                     : int
                                   For reproducibility
        full_data                : osl_dynamics.data.Data object
                                   The full dataset
        results_path             : str
                                   Directory in which to save models
    Returns:
        None
    """
    for k in k_values: 
        print(f"Fitting model with {k} states...")
        if 'realizations' in model_eval_log[k] and len(model_eval_log[k]['realizations']) < n_realizations:
            del model_eval_log[k]['realizations']
            del model_eval_log[k]['free_energy']
            del model_eval_log[k]['total_LL'] 
            del model_eval_log[k]['BIC'] 
            # del model_eval_log[k]['MMDL'] 
        
        if 'realizations' not in model_eval_log[k]:
            model_eval_log[k]['realizations'] = []
            realizations = [] # contains model object from each realization. Models can't be pickled, so we will pick the best realization and save the corresponding model object separately
            model_eval_log[k]['free_energy'] = []
            model_eval_log[k]['total_LL'] = []
            model_eval_log[k]['BIC'] = []
            # model_eval_log[k]['MMDL'] = []
            
            random.seed(seed)
            np.random.seed(seed)
            
            for r in range(n_realizations): # account for variability in .random_state_time_course_initialization(), then average later
                start = time.perf_counter()
                
                print(f"Realization {r + 1}:")
                model_eval_log[k]['realizations'].append({})
                best_hyperparams_idx = np.nanargmin(np.mean(model_eval_log[k]['outer_free_energies'], axis=1))
                best_hyperparams = model_eval_log[k]['hyperparams'][best_hyperparams_idx]
                config = Config(
                    n_states=k,
                    n_channels=full_data.n_channels,
                    sequence_length=best_hyperparams['sequence_length'],
                    learn_means=best_hyperparams['learn_means'],
                    learn_covariances=best_hyperparams['learn_covariances'],
                    batch_size=best_hyperparams['batch_size'],
                    learning_rate=best_hyperparams['learning_rate'],
                    lr_decay=best_hyperparams['lr_decay'],
                    n_epochs=int(np.ceil(np.mean(model_eval_log[k]['best_epochs'][best_hyperparams_idx]))),
                )
        
                model = Model(config)

                if best_hyperparams['set_regularizers']:
                    model.set_regularizers(full_data)
                
                try:
                    model.random_state_time_course_initialization(full_data, verbose=0)
                except ValueError:
                    print("random_state_time_course_initialization can't simulate a state time course where each state activates. Switching to using random_subset_initialization instead.")
                    model.random_subset_initialization(full_data, verbose=0)
    
                history = model.fit(full_data, verbose=0)
                realizations.append(model)
                
                free_energy = model.free_energy(full_data) # note that .free_energy() returns the free energy averaged over batches
                model_eval_log[k]['free_energy'].append(free_energy)
        
                means, covs = model.get_means_covariances() 
                if best_hyperparams['learn_means']:
                    model_eval_log[k]['realizations'][r]['means'] = means
                if best_hyperparams['learn_covariances']:
                    model_eval_log[k]['realizations'][r]['covs'] = covs
            
                off_diags = []
                # dof = [] # degrees of freedom of inverse covariance matrices (precision matrices), used to compute MMDL
                for cov in covs:
                    ut = np.triu(cov, k=1) # upper triangle, excluding main diagonal
                    off_diag = ut[ut != 0] # np.triu() returns a full matrix, so filter out lower triangle entries and flatten to a 1-D array
                    off_diags.append(off_diag)
        
                    # precision_matrix = np.linalg.inv(cov)
                    # precision_matrix_ut = np.triu(precision_matrix) # upper triangle, incuding main diagonal
                    # dof.append(np.sum(precision_matrix_ut != 0)) # "Df(k) is the number of non-zeroes in the precision matrix"
            
                model_eval_log[k]['realizations'][r]['off_diags'] = np.array(off_diags).T # (number of upper triangle entries x k)
                
                total_LL = hmm_total_loglik(model, full_data)
                model_eval_log[k]['total_LL'].append(total_LL)
            
                # Compute Bayesian Information Criterion (BIC)
                d = full_data.n_channels
                n_estimated_params = (k - 1) + k * (k - 1) # k initial state probabilities (but must sum to 1, so only k - 1 need to be estimated), and k * (k - 1) transition probabilities
                if best_hyperparams['learn_means']:
                    n_estimated_params += k * d # d x 1 mean vector per state
                if best_hyperparams['learn_covariances']:
                    n_estimated_params += k * (d * (d + 1)) / 2 # d x d covariance matrix per state. Since the covariance is symmetric, we just keep the upper triangle, including the main diagonal, which sums to (d * (d - 1)) / 2 + d = (d * (d - 1) + 2d) / 2 = (d * (d + 1)) / 2 covariance params per state
                BIC = n_estimated_params * np.log(full_data.n_samples) - 2 * total_LL
                model_eval_log[k]['BIC'].append(BIC)
        
                # # Compute Mixture Minimum Description Length (MMDL) 
                # alpha = np.array(model.get_alpha(full_data)) # state probability time courses for each subject. Has shape (n_subjects, n_timepoints, k)
                # third_term = np.sum([np.log(full_data.n_samples * np.sum(alpha[:, :, state])) * dof[state] for state in range(k)]) / 2
                # MMDL = -total_LL + np.log(full_data.n_samples) * k * (k - 1) / 2 + third_term
                # model_eval_log[k]['MMDL'].append(MMDL)

                end = time.perf_counter()
                time_elapsed = int(np.ceil(end - start))
                min_elapsed, sec_elapsed = divmod(time_elapsed, 60)
                print(f"Time elapsed for this realization: {min_elapsed} min. {sec_elapsed} sec.")
            
            model_dir = f'{results_path}/{k}_states'
            os.makedirs(model_dir, exist_ok=True)
            best_realization_path = f'{model_dir}/{k}_states_model'

            if model_save_metric == 'total_LL':
                print("Saving model with highest total log-likelihood ...")
                realizations[np.nanargmax(model_eval_log[k][model_save_metric])].save(best_realization_path)
            else:
                print(f"Saving model with lowest {model_save_metric} ...")
                realizations[np.nanargmin(model_eval_log[k][model_save_metric])].save(best_realization_path) 
            
            model_eval_log[k]['best_realization_path'] = best_realization_path
            with open(model_eval_log_save_path, 'wb') as f:
                pickle.dump(model_eval_log, f)
        else:
            print(f"The model with {k} states has already been evaluated on the full dataset, skipping ...")

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
        axs[1].set_title("Rank-1 Approx. of Phase Coherence")
        plt.suptitle(f"State {state + 1}")
        plt.show()

    return np.array(outer_products)

