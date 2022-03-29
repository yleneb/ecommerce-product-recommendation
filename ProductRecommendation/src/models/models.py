from src.visualisation.visualisation import plot_confusion_matrix, show_results
import time
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from IPython.display import display, clear_output
import pprint
import joblib
import os
import random


def normalise_fold(X_train, X_valid):
    """Normalise training and validation data using means and std from training"""
    # get means and std - sum is too big for float16
    means = X_train.astype(np.float32).mean(axis=0)
    stdev = X_train.astype(np.float32).std(axis=0)
    # normalise datasets
    X_train = (X_train - means) / stdev
    X_valid = (X_valid - means) / stdev
    return X_train, X_valid

def fit_simple_model(
    clf, X_train, y_train, X_valid, y_valid,
    resampler=False, confusion_matrix=False,
    n_partitions=10, normalise=False):
    
    """For training the initial models.
    Optionally resample data, train model, and report results on a range of metrics."""

    X_train = X_train.copy()
    X_valid = X_valid.copy()
    
    # some models need normalised inputs
    if normalise:
        X_train, X_valid = normalise_fold(X_train, X_valid)

    # optionally resample training data
    if resampler:
        X_train, y_train = resampler.fit_resample(X_train, y_train)
        
    # store customerIds to calculate mean AP
    customerId_train = X_train.pop('customerId')
    customerId_valid = X_valid.pop('customerId')
    
    # fit model
    clf.fit(X_train, y_train)
    
    # make predictions
    y_train_pred = clf.predict_proba(X_train)[:,1]
    y_valid_pred = clf.predict_proba(X_valid)[:,1]
    
    # log performance on a range of metrics
    results_df = show_results(
        y_train, y_train_pred, customerId_train,
        y_valid, y_valid_pred, customerId_valid, n_partitions)
    
    # optionally print a confusion matrix
    if confusion_matrix:
        plot_confusion_matrix(y_train, y_train_pred > 0.5, title='Training Confusion Matrix')
        plot_confusion_matrix(y_valid, y_valid_pred > 0.5, title='Validation Confusion Matrix')
        
    return clf, results_df

def fit_simple_model_cv(
    data, base_model, model_params=False, to_drop=None,
    n_folds=3, resampler=False, n_partitions=10,
    confusion_matrix=False, sample=False, normalise=False):
    
    # save trained models
    models = []
    
    # if model parameters not supplied then set as empty dict
    if not model_params:
        model_params = {}
    
    results = {}
    for fold_id in tqdm(range(n_folds), leave=True, desc='fold'):
        # get dataset fold
        X_train = data.folds_data[fold_id].train.copy().drop(columns=to_drop)
        y_train = X_train.pop('purchased')

        X_valid = data.folds_data[fold_id].valid.copy().drop(columns=to_drop)
        y_valid = X_valid.pop('purchased')
        
        # for quick testing
        if sample:
            X_train = X_train[:100000]
            y_train = y_train[:100000]
            X_valid = X_valid[:100000]
            y_valid = y_valid[:100000]
            
        t_start = time.time()
        # train model on fold
        model, results_df = fit_simple_model(
            base_model(**model_params),
            X_train, y_train, X_valid, y_valid,
            resampler=resampler,
            n_partitions=n_partitions,
            confusion_matrix=confusion_matrix,
            normalise=normalise)
        
        # save trained model
        models.append(model)
        
        # log results & training time
        results_df['time'] = time.time() - t_start
        results[f'Fold {fold_id}'] = results_df
        display(results_df)
        
    return models, pd.concat(results).unstack(1)


def grid_search(data, base_model, param_grid, to_drop, ckpt_files,
                n_folds=3, resampler=False, n_partitions=10,
                continue_training=False, n_runs=100,
                sample=False, normalise=False):
    """n_partitions controls how many partitions to use when
    claculating mean average precision score in parallel."""
    
    # optionally continue from a checkpoint
    if continue_training:
        # if file exists load previous state
        assert all([os.path.exists(fpath) for fpath in ckpt_files])
        results = joblib.load(ckpt_files[0])
        run_params = joblib.load(ckpt_files[1])
        names_list = joblib.load(ckpt_files[2])
        print('continuing training')
    # otherwise start from scratch
    else:
        results = []
        run_params = []
        names_list = []

    # do many runs
    pbar = tqdm(range(n_runs), desc='run')
    for run_id in pbar:
        
        # randomly select hyperparameters
        params = {key: random.choice(values) for (key, values) in param_grid.items()}
        
        # skip if this setup has already be tried
        run_name = ' '.join([str(x) for x in params.values()])
        if run_name in names_list:
            continue
        
        # print current params
        pprint.pprint(params)
        
        # record this new config
        names_list.append(run_name)
        run_params.append(params)
        
        # fit the model on each fold and return results
        _, results_df = fit_simple_model_cv(
            data, base_model, model_params=params, to_drop=to_drop,
            n_folds=n_folds, resampler=resampler, n_partitions=n_partitions,
            confusion_matrix=False, sample=sample, normalise=normalise)
        
        # record results
        for parameter, value in params.items():
            results_df[parameter] = value
        
        # add results to list
        results.append(results_df)
        
        # save a checkpoint of results
        joblib.dump(results, ckpt_files[0])
        joblib.dump(run_params, ckpt_files[1])
        joblib.dump(names_list, ckpt_files[2])
        
        # display results so far
        clear_output()
        display(pbar.container) # reprint progress bar
        # display results so far
        display(pd.concat(results, keys=[f'run {x:02}' for x in range(len(results))])
                .drop(columns=['precision','recall', 'f1','accuracy',
                                'roc_auc',('time','valid')])
                .groupby(level=0)
                .agg(lambda g: g.mean() if g.name[1]!='' else g[0])
                .assign(time=lambda df: df.time.round().astype(int),
                        mAP=lambda df: df.mAP.round(4),
                        avg_precision=lambda df: df.avg_precision.round(4)))
        
    return results