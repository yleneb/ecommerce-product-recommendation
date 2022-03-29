from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import (confusion_matrix,
                             ConfusionMatrixDisplay,
                             precision_recall_curve,
                             PrecisionRecallDisplay,
                             precision_score,
                             recall_score,
                             f1_score,
                             accuracy_score,
                             roc_auc_score,
                             average_precision_score)
import matplotlib.pyplot as plt
from src.tools.tools import mean_average_precision_score

import pandas as pd

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot a confusion matrix of results"""
    cm = confusion_matrix(y_true, y_pred)
    cm = ConfusionMatrixDisplay(cm)
    
    cm = cm.plot(values_format='.1e')
    plt.title(title)
    plt.show()
    plt.close()
    
def show_results(y_train, y_train_pred, customerId_train,
                 y_valid, y_valid_pred, customerId_valid,
                 n_partitions=10):
    
    """Print performance metrics on valid and train sets
    plot a confusion matrix and precision recall curve"""
    
    results_df = pd.DataFrame()
    
    ## mAP needs customerIds
    results_df.at['train','mAP'] = mean_average_precision_score(customerId_train, y_train, y_train_pred, n_partitions)
    results_df.at['valid','mAP'] = mean_average_precision_score(customerId_valid, y_valid, y_valid_pred, n_partitions)

    ## average precision (not mean) needs predicted probabilities
    results_df.at['train','avg_precision'] = average_precision_score(y_train, y_train_pred)
    results_df.at['valid','avg_precision'] = average_precision_score(y_valid, y_valid_pred)
    
    # other metrics want as binary classification
    metrics = ['precision','recall','f1','accuracy','roc_auc']
    metric_fns = [precision_score, recall_score, f1_score,
                  accuracy_score, roc_auc_score]
    
    for metric_name, metric_fn in zip(metrics, metric_fns):
        results_df.at['train',metric_name] = metric_fn(y_train, y_train_pred > 0.5)
        results_df.at['valid',metric_name] = metric_fn(y_valid, y_valid_pred > 0.5)
    
    return results_df