import numpy as np
from scipy.sparse.linalg import eigs
import random
import os
import re
import nibabel as nib
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import copy
from tqdm.auto import tqdm
import itertools
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from osl_dynamics.data import Data, processing
from osl_dynamics.models.hmm import Config, Model
import math
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

def load_data(data_path, data_prep_method, bad_ICs=None, standardize=True):
    """
    Args:
        data_path        : str
                           Path to the data
        data_prep_method : str
                           'ICA' or 'LEiDA'
        bad_ICs          : list
                           1-based indices of independent components that need to be removed. None by default. Ignored if data_prep_method isn't 'ICA'
                           Note to self: [1, 6, 9, 18] for CIMT/rs, [6, 11, 12, 14] for CIMT/LLstim
        standardize      : bool
                           Whether to standardize the data. True by default
    Returns:
        X                : ndarray with shape (n_subjects, n_timepoints, n_channels)
        full_data        : osl_dynamics Data object
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
    full_data.prepare({
        "standardize": {},
    })
    return X, full_data

def create_folds(n_splits, X, y, inner_val_size, random_state):
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
    Returns:
        split_plan     : dict
                         Keys are 'outer_train', 'outer_test', 'inner_train', and 'inner_val'. Each of the values is a list of length n_splits containing
                         osl_dynamics Data objects
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

        outer_train.prepare({"standardize": {}})
        outer_test.prepare({"standardize": {}})
        inner_train.prepare({"standardize": {}})
        inner_val.prepare({"standardize": {}})
    
        split_plan['outer_train'].append(outer_train)
        split_plan['outer_test'].append(outer_test)
        split_plan['inner_train'].append(inner_train)
        split_plan['inner_val'].append(inner_val)

    return split_plan

def get_hyperparam_combinations(hyperparam_grid):
    """
    Args:
        hyperparam_grid         : dict
                                  Keys are hyperparameter names. Values should be lists containing values of the corresponding hyperparameter that need to be searched
    Returns:
        hyperparam_combinations : list
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
        hyperparam_grid          : list
                                   Output of get_hyperparam_combinations(). Must have keys 'k', 'sequence_length', 'learn_means', 'learn_covariances', 'batch_size', 
                                   'learning_rate', 'lr_decay', 'n_epochs', and 'patience'
        seed                     : int
                                   For reproducibility
        split_plan               : dict
                                   Output of create_folds
    Returns:
        None
    """
    i = 1
    for hyperparams in get_hyperparam_combinations(hyperparam_grid):
        print(f"\nHyperparam set {i}: {hyperparams}")

        random.seed(42)
        np.random.seed(42)

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
                         Output of create_folds
    Returns:
        None
    """
    for f in range(len(split_plan)):
        example = model_eval_log[k]['histories'][np.nanargmin(np.mean(model_eval_log[k]['test_free_energies'], axis=1))]
        fig, ax = plt.subplots(1, 1)
        x = range(1, len(example[f]['loss']) + 1)
        ax.plot(x, example[f]['loss'], label="Training Loss", color='blue', linestyle='-')
        ax.plot(x, example[f]['val_loss'], label="Validation Loss", color='orange', linestyle='--')
    
        ax.set_title(f"{k} States, {str(model_eval_log[k]['hyperparams'][np.nanargmin(np.mean(model_eval_log[k]['test_free_energies'], axis=1))])}\nLowest validation loss achieved: {np.min(example[f]['val_loss']):.3f}")
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