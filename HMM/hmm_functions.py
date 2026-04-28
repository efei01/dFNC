import numpy as np
from scipy.sparse.linalg import eigs
import random
import os
import re
import nibabel as nib
from sklearn.model_selection import StratifiedKFold, train_test_split
import math
import matplotlib.pyplot as plt
import itertools
import time
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
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
def get_eigenvectors(dFC,n=1):
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

def load_data(data_path, data_prep_method, standardize, bad_ICs=None):
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

def create_folds(n_splits, X, y, inner_val_size, random_state, standardize):
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
                         Keys are 'outer_train', 'outer_test', 'inner_train', and 'inner_val'. Each of the values is a list of length n_splits containing
                         osl_dynamics.data.Data objects
    """
    skf = StratifiedKFold(n_splits=n_splits)

    split_plan = {
        'outer_train': [],
        'outer_test': [],
        'inner_train': [],
        'inner_val': [],
    }

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Fold {i + 1}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        
        outer_train, outer_test, y_train, y_test = X[train_index,], X[test_index,], y[train_index,], y[test_index,]
        inner_train, inner_val, y_inner_train, y_inner_val = train_test_split(
            outer_train, 
            y_train, 
            test_size=inner_val_size, 
            random_state=random_state, 
            stratify=y_train
        )
        
        outer_train = Data(outer_train)
        outer_test = Data(outer_test)
        inner_train = Data(inner_train)
        inner_val = Data(inner_val)

        if standardize:
            outer_train.prepare({"standardize": {}})
            outer_test.prepare({"standardize": {}})
            inner_train.prepare({"standardize": {}})
            inner_val.prepare({"standardize": {}})
    
        split_plan['outer_train'].append(outer_train)
        split_plan['outer_test'].append(outer_test)
        split_plan['inner_train'].append(inner_train)
        split_plan['inner_val'].append(inner_val)

    return split_plan

def print_n_param_updates_per_epoch(hyperparam_grid, full_data):
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
                print(f"Number of parameter updates per epoch with sequence_length={sequence_length} and batch_size={batch_size}: {math.ceil(full_data.n_samples / (sequence_length * batch_size))}")
    else:
        raise ValueError("Either 'sequence_length' or 'batch_size' (or both) is missing. Values for both 'sequence_length' and 'batch_size' must be provided in hyperparam_grid in order to print the number of parameter updates per epoch")
    
def get_hyperparam_combinations(hyperparam_grid):
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

def run_grid_search(model_eval_log, model_eval_log_save_path, hyperparam_grid, seed, split_plan):
    """
    Args:
        model_eval_log           : dict
                                   Can be empty or not. Will be modified by this function
        model_eval_log_save_path : str
                                   The path at which to save model_eval_log as a pickle. A save is done after the grid search for each k value
        hyperparam_grid          : list[dict]
                                   Output of get_hyperparam_combinations(). Each combination dictionary must have keys 'k', 'sequence_length', 'learn_means', 'learn_covariances', 'set_regularizers', 'batch_size', 
                                   'learning_rate', 'lr_decay', 'n_epochs', and 'patience'
        seed                     : int
                                   For reproducibility
        split_plan               : dict
                                   Output of create_folds()
    Returns:
        None
    """
    i = 1
    for hyperparams in get_hyperparam_combinations(hyperparam_grid):
        print(f"\nHyperparam set {i}: {hyperparams}")

        random.seed(seed)
        np.random.seed(seed)

        k = hyperparams['k']
        del hyperparams['k']
        
        if k not in model_eval_log:
            model_eval_log[k] = {
                'hyperparams': [], # [{hyperparams}]
                'histories': [], # [[history object from best run for each fold]]
                'best_epochs': [], # [[the best training epoch for each fold]]
                'test_free_energies': [], # [[float]]
            }
        
        if hyperparams not in model_eval_log[k]['hyperparams']:
            histories = []
            best_epochs = []
            test_free_energies = []
            for f in range(len(split_plan['outer_test'])):
                start = time.perf_counter()
                
                print(f"\nFold {f + 1}...")
                outer_test, inner_train, inner_val = split_plan['outer_test'][f], split_plan['inner_train'][f], split_plan['inner_val'][f]
                config = Config(
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
                    
                model = Model(config)

                if hyperparams['set_regularizers']:
                    model.set_regularizers(inner_train)
                    
                try:
                    model.random_state_time_course_initialization(inner_train, verbose=0)
                except ValueError:
                    print("random_state_time_course_initialization can't simulate a state time course where each state activates. Switching to using random_subset_initialization instead.")
                    model.random_subset_initialization(inner_train, verbose=0)
                    
                callback = EarlyStopping(monitor='val_loss', patience=hyperparams['patience'], verbose=0) # we don't need restore_best_weights=True because we aren't saving the models
                history = model.fit(
                    inner_train,
                    validation_data=inner_val.dataset(
                        sequence_length=hyperparams['sequence_length'],
                        batch_size=hyperparams['batch_size'],
                    ),
                    verbose=0,
                    callbacks=[callback],
                )
                        
                histories.append(history)
                best_epochs.append(np.argmin(history['val_loss']) + 1)
                            
                test_free_energy = model.free_energy(outer_test)
                test_free_energies.append(test_free_energy)

                end = time.perf_counter()
                print(f"Time elapsed for this fold: {end - start:.3f} seconds")
        
            model_eval_log[k]['hyperparams'].append(hyperparams)
            model_eval_log[k]['histories'].append(histories)
            model_eval_log[k]['best_epochs'].append(best_epochs) # can take the mean across folds later (make sure to take the ceiling of the mean)
            model_eval_log[k]['test_free_energies'].append(test_free_energies) # can take the mean across folds later
        else:
            print(f"Hyperparam set {i} has already been evaluated, skipping...")
            
        i += 1
    
        with open(model_eval_log_save_path, 'wb') as f:
            pickle.dump(model_eval_log, f)

def plot_cv_loss(model_eval_log, k, split_plan):
    """
    Args:
        model_eval_log : dict
                         model_eval_log as modified by run_grid_search()
        k              : int
                         k value for which to plot the training and validation loss curves of optimal hyperparameters
        split_plan     : int
                         Output of create_folds()
    Returns:
        None
    """
    best_hyperparams_idx = np.nanargmin(np.mean(model_eval_log[k]['test_free_energies'], axis=1))
    
    for f in range(len(split_plan['outer_test'])):
        example = model_eval_log[k]['histories'][best_hyperparams_idx]
        fig, ax = plt.subplots(1, 1)
        x = range(1, len(example[f]['loss']) + 1)
        ax.plot(x, example[f]['loss'], label="Training Loss", color='blue', linestyle='-')
        ax.plot(x, example[f]['val_loss'], label="Validation Loss", color='orange', linestyle='--')
    
        ax.set_title(f"{k} States, {str(model_eval_log[k]['hyperparams'][best_hyperparams_idx])}\nLowest validation loss achieved: {np.min(example[f]['val_loss']):.3f} at epoch {model_eval_log[k]['best_epochs'][best_hyperparams_idx][f]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Free Energy)")
        ax.legend()
        plt.show()

def hyperparam_performance(model_eval_log, k):
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
            np.mean(model_eval_log[k]['test_free_energies'], axis=1), 
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
    
def run_full_model_eval(model_eval_log, model_eval_log_save_path, k_values, n_realizations, model_save_metric, seed, full_data, results_path):
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
                best_hyperparams_idx = np.nanargmin(np.mean(model_eval_log[k]['test_free_energies'], axis=1))
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
                    n_epochs=math.ceil(np.mean(model_eval_log[k]['best_epochs'][best_hyperparams_idx])),
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
                # third_term = np.sum([np.log(full_data.n_samples * np.sum(alpha[:, :, state_num])) * dof[state_num] for state_num in range(k)]) / 2
                # MMDL = -total_LL + np.log(full_data.n_samples) * k * (k - 1) / 2 + third_term
                # model_eval_log[k]['MMDL'].append(MMDL)

                end = time.perf_counter()
                print(f"Time elapsed for this realization: {end - start:.3f} seconds")
            
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
    