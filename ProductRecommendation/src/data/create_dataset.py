import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.data.process_data import reduce_memory_usage, load_processed_datasets_from_feather
from src.features.feature_engineering import *

class Dataset:
    """An Dataset object contains training validation/testing data, usually for a single fold.
    The individual datasets are combined and new features are engineered.
    Using _fit and _transform ensures that the validation set is transformed using
    values calculated from the training data - eg we impute using the median customer age.
    These datasets can be saved to feather format for quick reloading."""
    # a pair of train and valid/test datasets
    # so that values from train can be used to impute in valid
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        # total sales by all customers
        self.overallCustomerPurchaseRate = {}
        # sales made by each customer
        self.customerPurchaseRate = {}
        
    def _fit(self, labels_df, customer_df, purchase_df, product_df, use_new_features):
        """To impute on the validation dataset we need to gather
        values from the training dataset.
        This function saves everything we will need to transform a validation dataset."""
        
        #########################################################
        ############## Median Customer yearOfBirth ##############
        #########################################################
        
        # use for imputing missing values in both train and validation
        self.median_customer_age = get_median_customer_age(customer_df, labels_df)
        
                        
        #########################################################
        ##### Overall product/brand/productType Popularity ######
        #########################################################
        
        # what % of our customer base have bought this 
        # product/brand/productType (pre-January) in the training data?
        # fast, so not concerned about repeating
        purchase_product_df = \
        (purchase_df
        .query("""date < '2017-01-01 00:00:00.000'""")
        .query("""customerId in @labels_df.customerId.unique()""")
        .merge(product_df, how='left', on='productId'))
        
        # get number of unique customers in training set
        self.nunique_customers = len(customer_df)
        
        # how many of the customers in training have bought x
        # as a rate and log10, as numbers are small        
        for feature in ['productId','productType','brand']:
            # save results to merge later
            self.overallCustomerPurchaseRate[feature] = \
                get_overall_purchase_rate(purchase_product_df, feature, self.nunique_customers)
            
        #########################################################
        ############ This Customer's Purchase History ###########
        #########################################################
            
        # same in each fold but so quick, not concerned about repeating
        # purchases pre January & product info
        purchase_product_df = \
        (purchase_df
        .query("""date < '2017-01-01 00:00:00.000'""")
        .merge(product_df, how='left', on='productId'))
        
        # Total purchases made by each customer & % bought during sale
        totalPurchases = \
        (purchase_product_df
        .groupby('customerId')
        .agg(total_customerPurchases=('purchasePrice','count'),
             onSale_customerPurchases=('onSale','mean')))
        
        # how many of this customer's purchases are of this type
        for feature in ['productId','productType','brand']:
             # save results to merge later
            self.customerPurchaseRate[feature] = get_customer_purchase_rate(purchase_product_df, feature, totalPurchases)
            
        # save results to merge later
        self.customerPurchaseRate['total'] = totalPurchases.reset_index()
        
        #########################################################
        ############## Create Similarity Features ###############
        #########################################################
        
        if use_new_features:
            self.brand_df = create_brand_features(product_df, purchase_df, customer_df)
            self.productId_df = create_productId_features(product_df)
            self.productType_df = create_productType_features(product_df, purchase_df, customer_df)
            self.customer_spend_df = create_customer_spend_features(purchase_df, product_df, self.productType_df)
        
        
    def _customer_merge(self, labels_df, customer_df):
        """Add customer info to dataset, imputing for new customers"""
        
        return \
        (labels_df # indicator=='left_only' occurs for customers not present in our customers table
        .merge(customer_df, how='left', on='customerId', indicator=True)
        # Handle new customers
        .rename(columns={'_merge':'newCustomer'})
        .eval("""newCustomer = newCustomer=='left_only'""")
        # impute yearOfBirth to median
        .fillna({'yearOfBirth': self.median_customer_age,
                 'country':'Unknown',
                 'isPremier':False})
        # isFemale is [-1,0,1] - 0 for unknown
        .assign(isFemale=lambda df: df.isFemale*2 - 1)
        .fillna({'isFemale':0})
        # fix broken datatypes
        .astype({'isFemale':np.int8, 'yearOfBirth':np.uint16}))
        

    def _merge_views_dataset(self, df, views_df, product_df):
        """Total views for the whole month.
        Effectively we are predicting at the time of their final view
        does the customer go on to buy the product?
        We can improve this by making an example for each view.
        eg: a customer with 3 views then buys the product
        hence, that customer with 2 views also went on to buy the product
        so, we get three examples from one customer-product pair.
        BUT this will increase the size of the dataset by ~50%"""
        
        # get the views info for these customers & join with product table
        viewCount_product_df = \
        (views_df
        .query('customerId in @df.customerId.unique()')
        .merge(product_df, how='left', on='productId'))

        # total times this customer has viewed this product/brand/productType in January
        for feature in ['productId','productType','brand']:
            featureViewCount = \
            (viewCount_product_df
            .groupby(['customerId',feature], observed=True)
            .agg(viewCount=('viewOnly','count'))
            .rename(columns=lambda x: f'{feature}_{x}')
            .reset_index())
            
            # merge into dataset
            df = df.merge(featureViewCount, how='left', on=['customerId',feature])
            print(f'views {feature} merged')
        return df
    

    def _merge_purchases_dataset(self, df):
        """Merge in purchase information prior to January.
        These trables were prepared in self._fit() but we still need to fill
        missing values, where eg: the customer has never bought a brand."""
        
        features = ['productId','productType','brand']
        
        # add in features computed in self._fit()
        for feature in features:
            # item information
            df = df.merge(self.overallCustomerPurchaseRate[feature], how='left', on=[feature])
            # customer information (relation to the product)
            df = df.merge(self.customerPurchaseRate[feature], how='left', on=['customerId', feature])
        
        # how many items total has each customer bought, and how often onSale
        df = df.merge(self.customerPurchaseRate['total'], how='left', on=['customerId'])

        # items the customer has never bought get filled to 0
        df = df.fillna({x+'_customerPurchaseRate': 0 for x in features})
        df = df.fillna({'total_customerPurchases': 0, 'onSale_customerPurchases':0})
        
        # overall purchase rate is different to avoid log(0)
        log_fill = np.log10(0+1/self.nunique_customers)
        df = df.fillna({x+'_logOverallCustomerPurchaseRate': log_fill for x in features})

        return df
    
    def _merge_similarity_features(self, df):
        """Join in all the computed similarity features and fill missing to 0"""
        
        df = \
        (df
         .merge(self.customer_spend_df, on=['customerId','productType'], how='left')
         .merge(self.productType_df, on='productType', how='left')
         .merge(self.productId_df, on='productId', how='left')
         .merge(self.brand_df, on='brand', how='left')
         .fillna({'customer_productTypeSpend':0,
                  'productTypePrice':0,'productType_purchasedByFemale':0,
                  'productId_priceToType':0,'productId_priceToBrand':0,
                  'brand_purchasedByFemale':0,'brand_pca_0':0,'brand_pca_1':0,
                  'brand_pca_2':0,'brand_pca_3':0,'brand_pca_4':0,}))
        
        return df
        
    
    def _transform(self, labels_df, customer_df, purchase_df, product_df, views_df, use_new_features, countries_to_keep):
        """Merge all the datasets into a single table to input to an ML model.
        Make use of values imputed in self._fit()"""
        
        # merge with customer info
        df = self._customer_merge(labels_df, customer_df)
        
        # merge with product info - no indicator needed
        df = df.merge(product_df, how='left', on='productId')
        
        # merge with views dataset
        df = self._merge_views_dataset(df, views_df, product_df)
        
        # merge with purchases (from pre-January)
        df = self._merge_purchases_dataset(df)
        
        # merge with similarities
        if use_new_features:
            df = self._merge_similarity_features(df)
        
        # merge =soume countries and remove unwanted columns.
        # convert timstamp to int days on site
        df = finalise_data(df, countries_to_keep)
        
        # finally reduce memory usage
        df = df.pipe(reduce_memory_usage)
        
        return df
    
    def _fit_transform(
        self, labels_df, customer_df, purchase_df,  product_df,
        views_df, use_new_features, countries_to_keep):
        self._fit(labels_df, customer_df, purchase_df, product_df, use_new_features)
        return self._transform(labels_df, customer_df, purchase_df, product_df, views_df, use_new_features, countries_to_keep)
    
    def create_train_valid_datasets(
        self, labels_train_df, labels_valid_df, customer_df,
        purchase_df, product_df, views_df, use_new_features, countries_to_keep):
        """fit_transform the training dataset then transform the validation set"""
        print('fitting training dataset')
        self._fit(labels_train_df, customer_df, purchase_df, product_df, use_new_features)
        print('training dataset fit')
        print('transforming training dataset')
        self.train = self._transform(labels_train_df, customer_df, purchase_df, product_df, views_df, use_new_features, countries_to_keep)
        print('transforming validation dataset')
        self.valid = self._transform(labels_valid_df, customer_df, purchase_df, product_df, views_df, use_new_features, countries_to_keep)
            
    def save_datasets(self, train_filepath, valid_filepath):
        self.train.to_feather(train_filepath)
        self.valid.to_feather(valid_filepath)
    
    def load_datasets_from_file(self, train_filepath, valid_filepath):
        self.train = pd.read_feather(train_filepath)
        self.valid = pd.read_feather(valid_filepath)


class DatasetComplete(Dataset):
    """The entire dataset include all folds.
    Containes additional functions for handling multiple folds
    and splitting data into folds.
    Saved as a class as we will impute missing values in
    validation and testing datasets using values in the training datasets."""
    def __init__(self, random_seed=42):
        super(DatasetComplete, self).__init__()
        self.folds_data = {}
    
    def load_datasets(self):
        """load all the datasets from files"""
        (self.customer_df,
         self.product_df,
         self.purchase_df, 
         self.views_df,
         self.labels_training_df,
         self.labels_testing_df) = load_processed_datasets_from_feather()
    
    def assign_folds(self, n_folds=3, load_from_path=False, save_path=False):
        """Get assigned folds from file, or generate with StratifiedGroupKFold.
        Using Stratified Group K Fold splitting, assign customers to n folds.
        Stratifying ensures that folds have similar proportions of True purchases.
        Optionally save the array of fold indexes to a file for reuse.
        eg: fold=2 is part of the validation set for fold 2"""
        
        if load_from_path:
            self.labels_training_df['fold'] = np.load(load_from_path)
            self.n_folds = self.labels_training_df['fold'].max() + 1 # starts at 0
            self.fold_path = load_from_path
            print('fold ids loaded from file')
            
        else:
            self.n_folds = n_folds
            self.fold_path = save_path
                        
            # create the cross validation object and split
            cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
            cv = cv.split(X=self.labels_training_df,
                          y=self.labels_training_df.purchased,
                          groups=self.labels_training_df.customerId)
            print('Cross validation split object created')
            
            # initialise the new column
            self.labels_training_df['fold'] = -1
            # assign fold ids by iterating through cv object
            for i, (train_idxs, valid_idxs) in enumerate(cv):
                # assign the fold number to the dataset
                self.labels_training_df.loc[valid_idxs, 'fold'] = i
                print(f'fold {i} assigned')
                
            # convert to lower memory use
            self.labels_training_df['fold'] = self.labels_training_df['fold'].astype(np.uint8)
                
            # save the folds for later if needed
            if save_path:
                np.save(save_path, self.labels_training_df['fold'].values)
                      
    def show_fold_sizes(self):
        """print number of samples in each validation fold """
        self.fold_sizes = \
        (self.labels_training_df
            .groupby('fold')
            .agg(foldPCT=('customerId','count'))
            .apply(lambda x: 100*(x/x.sum()).round(3)))
        
        print(self.fold_sizes)
    
    def fit_transform_fold(
        self, fold_id, use_new_features, countries_to_keep, save_filepath=False):
        """Given a fold ID create a training and validation dataset
        and optionally save to feather"""
        
        if save_filepath:
            assert self.n_folds
        
        # instantiate the fold
        fold = Dataset()
        print(f'Preparing fold {fold_id}')
        
        # fit and transform the datasets
        fold.create_train_valid_datasets(
            labels_train_df=self.labels_training_df.query("""fold!=@fold_id"""),
            labels_valid_df=self.labels_training_df.query("""fold==@fold_id"""),
            customer_df=self.customer_df,
            purchase_df=self.purchase_df,
            product_df=self.product_df,
            views_df=self.views_df,
            use_new_features=use_new_features,
            countries_to_keep=countries_to_keep)
        print(f'Fold {fold_id} complete')
        
        # save optionally
        if save_filepath:
            fold.save_datasets(save_filepath/f'train_{fold_id}_{self.n_folds-1}.feather',
                               save_filepath/f'valid_{fold_id}_{self.n_folds-1}.feather')
            
        # store in this object
        self.folds_data[fold_id] = fold
        
    def create_and_save_cross_validated_datasets(
        self, n_folds, fold_id_save_path, fold_data_save_path,
        use_new_features, countries_to_keep):
        """Create and save the complete dataset. K folds stratified and grouped,
        each dataset individually processed and saved to feather"""
        
        self.n_folds = n_folds
        self.assign_folds(n_folds=n_folds, save_path=fold_id_save_path)
        
        for fold_id in range(n_folds):
            self.fit_transform_fold(
                fold_id=fold_id,
                use_new_features=use_new_features,
                countries_to_keep=countries_to_keep,
                save_filepath=fold_data_save_path)
            
    def load_fold_from_file(self, fold_id, save_filepath):
        """Load a single fold dataset from feather"""
        assert self.n_folds
        # instantiate the fold
        fold = Dataset()
        # load contents
        fold.train = pd.read_feather(save_filepath/f'train_{fold_id}_{self.n_folds-1}.feather')
        fold.valid = pd.read_feather(save_filepath/f'valid_{fold_id}_{self.n_folds-1}.feather')
        # store in this object
        self.folds_data[fold_id] = fold
        
    def load_nfolds_from_files(self, n_folds, save_filepath):
        """Load all the folds from feather files"""
        self.n_folds = n_folds
        for fold_id in range(n_folds):
            self.load_fold_from_file(fold_id, save_filepath)