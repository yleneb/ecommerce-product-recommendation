from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

def _mean_average_precision(group):
    """Calculate AP for each customer individually.
    Skip those with no purchases"""
    
    ap_func = lambda x: \
        average_precision_score(x.purchased, x.purchase_probability) \
        if x.purchased.any() \
        else None
        
    return group.groupby('customerId', observed=True).apply(ap_func)

def applyParallel(dfGrouped, func):
    """Apply a function to each group in parallel then concat the results"""
    retLst = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

def mean_average_precision_score(customerId, purchased, purchase_probability, n_partitions=10):
    """Calculate AP for each customer then take the average.
    Excluding customers who make no purchases."""
                
    mAP = pd.DataFrame({'customerId':customerId,
                        'purchased':purchased,
                        'purchase_probability':purchase_probability})
    
    # assign each customer to a partition 
    mAP['partitionId'] = pd.qcut(mAP.customerId, n_partitions, labels=[x for x in range(n_partitions)])
    
    # then apply to each partition in parallel
    mAP = applyParallel(mAP.groupby('partitionId'), _mean_average_precision)    
    mAP = mAP.mean()
    
    return mAP