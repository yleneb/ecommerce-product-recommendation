import numpy as np
import pandas as pd
from src.config import DATA_DIR

def reduce_memory_usage(df, verbose=False):
    """Check if each col is numeric, then reduce it's dtype to the lowest memory
    
    pandas only supports datetime64[ns]"""

    if verbose:
        starting_memory_usage = df.memory_usage(deep=True).sum() / 1024**2

    # try each column
    for col in df.columns:
        col_type = df[col].dtypes

        # Downsample numeric dtypes
        if col_type in [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
            c_min, c_max = df[col].agg(['min','max'])

            # rare case of boolean column
            if c_min in [0,1] and c_max in [0,1]:
                if all(df[col].isin([0,1])):
                    df[col] = df[col].astype('bool')
                    continue

            # if col is float and cannot be converted to int then only try float dtypes
            if str(col_type).startswith('float') and not np.array_equal(df[col], df[col].astype(int)):
                dtypes = [np.float16, np.float32, np.float64]
            # if all values are positive we can use unsigned integer
            elif all(df[col]>=0):
                dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
            else:
                dtypes = [np.int8, np.int16, np.int32, np.int64]

            # get the info about each dtype
            dtype_info = [np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype) for dtype in dtypes]

            # try all the smaller dtypes in order of size
            for dtype, info in zip(dtypes, dtype_info):
                # if this dtype is suitable,. use it and break, else try the next dtype
                if c_min >= info.min and c_max <= info.max:
                    df[col] = df[col].astype(dtype)
                    break

    if verbose:
        ending_memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        reduction = 100*(starting_memory_usage-ending_memory_usage)/starting_memory_usage
        print(f"Memory usage decreased from {np.round(starting_memory_usage,3)} Mb "
              f"to {np.round(ending_memory_usage,3)} Mb ({np.round(reduction,1)}% reduction)")

    return df

def load_customer_dataset(filepath=DATA_DIR/'raw'/'customers.txt', verbose=False):
    """Load the customers csv, fill missing values and reduce memory usage"""
    customers_df = \
    (pd.read_csv(filepath)
    .fillna('Unknown')
    .astype({'country':'category'})
    .pipe(reduce_memory_usage, verbose=verbose))
    
    return customers_df

def load_product_dataset(filepath=DATA_DIR/'raw'/'products.txt', verbose=False):
    """Load the products csv, fill missing values and reduce memory usage"""
    
    def fillna_dateOnSite_with_brand_mean(df):
        """Replace pd.NaT in dateOnSite to mean of that brand"""
        fixed_dates = \
            (df.groupby('brand', as_index=True)
            .dateOnSite
            .transform(lambda g: g.fillna(g.mean(numeric_only=False))))
            
        df.dateOnSite = fixed_dates
        return df
        
    products_df = \
    (pd.read_csv(filepath)
    ## fix missing dates
    # replace 'NaN' with pd.NaT object
    .replace({r'\N': pd.NaT})
    .astype({'dateOnSite':'datetime64[ns]'})
    # first replace with mean of that brand
    .pipe(fillna_dateOnSite_with_brand_mean)
    # then fill any remaining with overall mean
    .transform(lambda df: df.fillna(df.mean(numeric_only=False)))
    # round to seconds precision
    .assign(dateOnSite=lambda df: df.dateOnSite.round('S'))
    ##  reduce memory usage
    # convert price to whole pence
    .eval('price = price * 100')
    .astype({'brand':'category', 'productType':'category', 'price': int})
    .pipe(reduce_memory_usage, verbose=verbose))
    
    return products_df

def load_purchase_dataset(filepath=DATA_DIR/'raw'/'purchases.txt', verbose=False):
    """Load the products csv, fill missing values and reduce memory usage"""
    
    purchase_df = \
    (pd.read_csv(filepath, parse_dates=['date'])
     # convert price to whole pence
     # fill missing discounts as No discount
    .assign(purchasePrice=lambda df: df.purchasePrice*100)
    .fillna({'discountType':'No Discount'})
    .astype({'purchasePrice':int, 'discountType':'category'})
    .pipe(reduce_memory_usage, verbose=verbose))
    
    return purchase_df

def load_views_dataset(filepath=DATA_DIR/'raw'/'views.txt', verbose=False):
    """Load the views csv and reduce memory usage"""
    views_df = \
    (pd.read_csv(filepath, parse_dates=['date'])
     .pipe(reduce_memory_usage, verbose=verbose))
    return views_df

def load_labels_dataset(filepath=DATA_DIR/'raw'/'labels_training.txt', verbose=False):
    """Load either labels csv and reduce memory usage
    for validation use labels_predict.txt"""
    
    labels_df = \
    (pd.read_csv(filepath)
     # NaN in labels_predict cannot save to feather
     .fillna({'purchase_probability':False})
     .pipe(reduce_memory_usage, verbose=verbose))
    return labels_df

def save_processed_datasets_to_feather(SAVE_DIR=DATA_DIR / 'interim'):
    """Load all dataset csv's, process, and save to feather for faster loading."""
    
    load_customer_dataset().to_feather(SAVE_DIR/'customers.feather')
    print('customers dataset saved (1/6)')
    
    load_product_dataset().to_feather(SAVE_DIR/'products.feather')
    print('products dataset saved (2/6)')
    
    load_purchase_dataset().to_feather(SAVE_DIR/'purchases.feather')
    print('purchases dataset saved (3/6)')
    
    load_views_dataset().to_feather(SAVE_DIR/'views.feather')
    print('views dataset saved (4/6)')
    
    load_labels_dataset().to_feather(SAVE_DIR/'labels_training.feather')
    print('labels training dataset saved (5/6)')
    
    load_labels_dataset(DATA_DIR/'raw'/'labels_predict.txt').to_feather(SAVE_DIR/'labels_predict.feather')
    print('labels testing dataset saved (6/6) - FINISHED')
    
def load_processed_datasets_from_feather(SAVE_DIR=DATA_DIR/'interim'):
    """load all the dataset from feather choosing either training or testing labels"""
    
    customer_df = pd.read_feather(SAVE_DIR/'customers.feather')
    print('customers dataset retrieved (1/6)')
    
    product_df = pd.read_feather(SAVE_DIR/'products.feather')
    print('products dataset retrieved (2/6)')
    
    purchase_df = pd.read_feather(SAVE_DIR/'purchases.feather')
    print('purchases dataset retrieved (3/6)')
    
    views_df = pd.read_feather(SAVE_DIR/'views.feather')
    print('views dataset retrieved (4/6)')
    
    labels_training_df = pd.read_feather(SAVE_DIR/'labels_training.feather')
    print("""labels training dataset retrieved (5/6)""")
    
    labels_testing_df = pd.read_feather(SAVE_DIR/'labels_predict.feather')
    print("""labels testing dataset retrieved (6/6) - FINISHED""")
    
    return customer_df, product_df, purchase_df, views_df, labels_training_df, labels_testing_df