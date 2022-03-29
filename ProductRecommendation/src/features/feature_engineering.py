import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def get_median_customer_age(customer_df, train_df):
    median_customer_age = \
    (customer_df
    .query('customerId in @train_df.customerId.unique()')
    .groupby('customerId')
    .yearOfBirth
    .first()
    .median(skipna=True)
    .astype(int))
    return median_customer_age

def get_overall_purchase_rate(purchase_product_df, feature, nunique_customers):
    # how many of the customers in training have bought x
    # as a rate and log10, as numbers are small 
    logOverallCustomerPurchaseRate = \
    (purchase_product_df
    .groupby(feature)
    .agg(logOverallCustomerPurchaseRate=('customerId','nunique'))
    .apply(lambda df: np.log10( (df+1)/nunique_customers ))
    .rename(columns=lambda x: f'{feature}_{x}')
    .reset_index())
    
    return logOverallCustomerPurchaseRate

def get_customer_purchase_rate(purchase_product_df, feature, totalPurchases):
    # how many of this customer's purchases are of this type
    customerPurchaseRate = \
    (purchase_product_df
    .groupby(['customerId',feature], observed=True)
    .agg(customerPurchaseRate=('purchasePrice','count'))
    .div(totalPurchases['total_customerPurchases'], axis=0, level='customerId')
    .rename(columns=lambda x: f'{feature}_{x}')
    .reset_index())
    return customerPurchaseRate

def brand_features_pca(brand_df, n_features=5):
    """Reduce dimensionality of brand_df with PCA"""
    
    # create the PCA object
    pca = PCA(n_components=n_features)
    
    # transform the columns (85 dims)
    X_transformed = pca.fit_transform(brand_df.drop(columns=['brand_purchasedByFemale']).copy())
    
    # update brand df
    brand_df = brand_df[['brand_purchasedByFemale']].copy()
    
    headers = [f'brand_pca_{x}' for x in range(n_features)]
    brand_df[headers] = X_transformed
    
    return brand_df

def create_brand_features(product_df, purchase_df, customer_df):
    """Calculate features for this brand.
    - Is it expensive
    - What types of products do they sell
    - Do they mostly have female customers?"""
    
    brand_df = product_df.copy()

     # price of product relative to opther products of that type
    brand_df.price = \
    (product_df
    .groupby('productType')
    .price
    .transform(lambda g: (g-g.mean())/g.std()))

    # how much of this brands catalog is each productType
    # + how expensive is this brand for this productType
    brand_df = \
    (brand_df
    .groupby(['brand','productType'], observed=True)
    .agg(pct_productType=('price','count'),
        median_price=('price','median'))
    .unstack('productType'))

    brand_df['pct_productType'] = brand_df['pct_productType'].apply(lambda row: row/row.sum(), axis=1)
    brand_df = brand_df.fillna(0)

    # is the brand mostly bought by women?
    brand_purchasedByFemale = \
    (purchase_df
    .query("""date < '2017-01-01 00:00:00.000'""")
    [['customerId','productId']]
    #  .query("""customerId in @data.labels_training_df.customerId.unique()""")
    .merge(customer_df[['customerId','isFemale']], how='left', on='customerId')
    .merge(product_df[['productId','brand']], how='left', on='productId')
    .groupby('brand')
    .agg(brand_purchasedByFemale=('isFemale','mean'))
    .fillna(0))

     # merge together
    brand_df = pd.concat([
        brand_df.pct_productType.rename(columns=lambda x: f'pct_productType_{x}'),
        brand_df.median_price.rename(columns=lambda x: f'median_price_{x}'),
        brand_purchasedByFemale], axis=1)

    # reduce dimensionality of computed features with pca
    brand_df = brand_features_pca(brand_df)

    return brand_df.reset_index()


def create_productId_features(product_df):
    """Calculate features for this product.
    Is this product expensive relative to its brand, relative to its type?"""
    # productId_df
    productId_df = product_df.copy()

    # is this product expensive relative to its brand, relative to its type
    # productId_df = pd.DataFrame()
    productId_df['productId_priceToType'] = \
    (product_df
    .groupby('productType')
    .price
    .transform(lambda g: (g-g.mean())/g.std()))

    productId_df['productId_priceToBrand'] = \
    (product_df
    .groupby('brand')
    .price
    .transform(lambda g: (g-g.mean())/g.std()))
    
    return productId_df[['productId','productId_priceToType','productId_priceToBrand']]


def create_productType_features(product_df, purchase_df, customer_df):
    """Calculate features for each productType.
    Is it expensive? eg coats vs. socks
    Is it mostly bought by women?"""

    # price of productType relative to other productTypes
    productType_median_price = \
    (product_df
    .groupby('productType')
    .agg(productTypePrice=('price','mean'))
    .transform(lambda g: (g-g.mean())/g.std()))

    # productType mostly bought by women?
    productType_purchasedByFemale = \
    (purchase_df
    .query("""date < '2017-01-01 00:00:00.000'""")
    [['customerId','productId']]
    #  .query("""customerId in @data.labels_training_df.customerId.unique()""")
    .merge(customer_df[['customerId','isFemale']], how='left', on='customerId')
    .merge(product_df[['productId','productType']], how='left', on='productId')
    .groupby('productType')
    .agg(productType_purchasedByFemale=('isFemale','mean'))
    .transform(lambda g: (g-g.mean())/g.std())
    .fillna(0))

    # merge together
    productType_df = pd.concat([
        productType_median_price,
        productType_purchasedByFemale], axis=1)

    return productType_df.reset_index()

def create_customer_spend_features(purchase_df, product_df, productType_df):
    """Create customer budget features.
    How much does the customer usually spend on this productType?"""
    # does this customer tend to spend more or less on this productType?
    customer_spend_df = \
    (purchase_df
    .query("""date < '2017-01-01 00:00:00.000'""")
    .merge(product_df, how='left', on=['productId'])
    [['customerId','purchasePrice','productType']]
    # first get standardised spend on each productType
    .merge(productType_df, on='productType')
    # then get how this customer compares to that
    .groupby(['customerId','productType'], observed=True)
    .agg(customer_productTypeSpend=('productTypePrice','mean'))
    .fillna(0)
    .reset_index())
    return customer_spend_df
 
def finalise_data(df, countries_to_keep):
    """reduce dims of countries. drop unwanted cols. convert timestamp to int
    Ready for input into a ml model"""

    # set of all countries to be mapped to 'Other'
    countries_to_merge = set(df.country.cat.categories) ^ countries_to_keep

    # update the countries
    df.country = df.country.map(lambda x: 'Other' if x in countries_to_merge else x).astype('category')

    # add new columns to dataframe
    df = df.join(pd.get_dummies(df.country, prefix='country_', dtype=bool))

    # drop unwanted columns - do not remove customerId
    df = df.drop(columns=['productId','brand','productType','country'])
    
    if 'fold' in df.columns:
        df = df.drop(columns=['fold'])
    
    # convert timestamps to time on site
    df.dateOnSite = df.dateOnSite - pd.Timestamp('02-01-2017 00:00:00', unit='s')
    df.dateOnSite = df.dateOnSite.dt.days
    df = df.rename(columns={'dateOnSite':'daysOnSite'})
    
    return df