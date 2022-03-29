# Report

An online only retailer gathers vast amounts of data on how customers interact with their website and mobile app. This project focuses on how this data can be leveraged to improve customer experience and ultimately drive further sales. The available data includes:
- Product information
- Purchase history from September 2016 to end January 2017
- Customer demographic and account information
- Product views during January 2017

Using this data we are able to predict which customer-product pairs lead to a sale, returning a purchase probability. By further understanding customer behaviours, we can provide better product recommendations and reminders for products which may go out of stock, we can increase the likelihood of a customer making a purchase and thus increase sales.

We should bear in mind that the purpose of this project is to identify which products a customer is most likely to buy; prediction of purchase probability is not the final goal. For example, we may predict one customer has a 20% probability of purchasing a product, while another customer is predicted 80%. If this product is the customer's most likely purchase then we should recommend it, regardless of how it compares to predicted probabilities for other customers - some people simply buy more products.

Furthermore, with limited screen space on devices it is increasingly important to narrow down recommendations to the top few. Estimating purchase probability will allow us to predict which products to prioritise to each customer.

Data imbalance is a key challenge to this problem, 98% of customer-product pairs in the labelled training data are not a purchase. It will be important to choose a suitable performance metric and potentially oversample the minority class to combat the imbalance. Secondly, the purchases dataset includes all the sales from January, including those in the test set. We will need to be careful to avoid data leakage which would result in overly optimistic performance on the test data and poor generalisation in production.

## Data Exploration

The first step was to explore the available data, here I will cover the key findings but more detail and plots can be found in the [data exploration notebook](/ProductRecommendation/notebooks/Data%20Exploration.ipynb).

### Customers

There are 398,841 unique customers in the database, 77% of whom identify as female. 31% of customers are based in the UK, the most of any country, and 40,282 customers had no country information at all - these were imputed with 'Unknown'.
![Customers by country](/ProductRecommendation/visualisations/Data%20Exploration/country.png "Customers by country")
78% of customers are millennials (born between 1980 and 2000), with 1.5% of customers having highly unlikely birth years, before 1910 and after 2010. There are 150 customers who are classed as Premier.

![Customers by year of birth](/ProductRecommendation/visualisations/Data%20Exploration/yearOfBirth.png "Customers by year of birth")

### Purchases

From September to January there are 2,063,893 purchases and daily sales tend to increase over the period, with spikes during the Halloween and Black Friday discount code events, with significant dips around the days of Christmas Day and New Years Eve. In general, daily sales are highest on Mondays then drop through the week before recovering on Sunday.
![Daily sales](/ProductRecommendation/visualisations/Data%20Exploration/daily%20sales.png "Daily sales")
![Sales per weekday](/ProductRecommendation/visualisations/Data%20Exploration/weekday%20sales.png "Sales per weekday")
The 1,063,803 purchases (58%) with missing discount type are assumed to have no discount applied. Free delivery was the most frequent discount (12%) and some discounts are applied well after their run - there are sales in January using the August 2016 discount.

41% of purchases are for a single item, larger orders are less common.

![Basket sizes](/ProductRecommendation/visualisations/Data%20Exploration/basket_size.png "Basket sizes")
### Products

There are 42 unique product types in the database and 1720 brands, 1097 brands with at least 10 listed products and 100 brands with at least a 100 listed products. A single brand (53) accounts for 25% of all products.

142 products have missing ("\N") dates on site, these were imputed to the mean for their brand if possible and otherwise the mean of all products. The two products with £0 price were left unchanged.
![Products added](/ProductRecommendation/visualisations/Data%20Exploration/products_added.png "Products added")
Products can be as expensive as £600 with mean value £28.5, although this is skewed towards less expensive items and the maximum of a Gaussian kernel density estimator is at £12.
![Product prices](/ProductRecommendation/visualisations/Data%20Exploration/product_prices.png "Product prices")

### Views

There were no missing values in the views table and all products also existed in the products table. There were 635 customers who do not exist in the customer table who were assumed to be new. Most products have fewer than 100 views by all customers in January (~75%) and customers only viewing a product more than once 15% of the time. 

### Labels

There are 13.5 million labelled examples to learn from. Within this dataset there are 116,543 customers with no recorded purchases and 40,468 products which have never been bought. This may be because the product has been taken off the store or was not available at the time, and was still included in the dataset as a listed product. In general, decisions regarding purchase are made on the first viewing of a product, with 65% of purchases taking place on the first viewing of a product, and a further 20% at the second viewing.

![View counts](/ProductRecommendation/visualisations/Data%20Exploration/view_count.png "View counts")

If a customer views a product more than once their likelihood of a purchase increases. 1.5% of products which are viewed once are purchased, but 20% of products which are viewed at least 10 times are purchased. This can be explained by customers for example, hesitating to purchase or taking time to decide amongst other products, before committing to a purchase. We can therefore expect view count to be a useful predictor of purchase probability.

![View counts](/ProductRecommendation/visualisations/Data%20Exploration/purchase_probability.png "View counts")
### Test Data

Normally we do not look at the test data until making predictions with the final model, but I think this case is an exception. In practice, the experimentalist has gathered the data and knows how they were split into training and testing sets, perhaps by choosing a random subset or splitting across time. When training machine learning models the test set should be as representative as possible of the real world data the model will encounter. Furthermore the validation data should be representative of the test data to be confident that good performance on the validation data will lead to good performance on the test set and therefore in production.

I discovered there were no overlapping customers in the train and test sets, so it would be sensible to separate my validation data by customer Id. I also found that purchases table includes sales for customer-product pairs in the test set, so I must be careful with how I use that data to avoid data leakage.

## Data wrangling

I framed the project as a binary classification problem, given a productId and customerId I would need to compute a set of features from the provided datasets to input into a classification model.

I imputed the few missing values in the given datasets and reduced their memory use by casting to smaller data types. I then saved these processed datasets to feather format for quick reloading.

For more reliable model evaluation I used k-fold cross validation, comparing the mean average performance across k folds. In particular I used stratified grouped k-fold, so customers are not found in both training and validation data (as with the test set) and also to ensure the ratio of classes is maintained across folds.

When imputing missing values in the validation data we should use values calculated from the training data - for example we use the mean age of customers in the training set to fill for new customers. I created the Dataset class to keep track of this information, it includes methods to merge all the datasets and generate new features. After being fit and transformed an Dataset contains the prepared train and validation data ready for input to a model.

The DatasetComplete class combines multiple Dataset objects and handles cross validation and fold assignment. Due to the complexity of some of the computed features it was necessary to implement my own classes rather than using scikit-learn pipelines.

To avoid data leakage I did not use any purchases from January to train models.

## Feature engineering

For the first round of feature engineering I tried to generate stats which describe the product, brand, and customer:

| Feature      | Description |
| :---        |    :----:   |
| XXX_logOverallCustomerPurchaseRate | log of the % of customers who have bought XXX |
| total_customerPurchases   | Number of purchases that customer has made |
| onSale_customerPurchases   | Rate of that customer's purchases which are discounted |
| XXX_customerPurchaseRate   | % of this customers purchases which are XXX |
| XXX_viewCount   | Times this customer has viewed XXX in January |
| country__   | One hot encoded countries, outside the top ten mapped to 'Other' |
| price   | Product price in whole pence |
| daysOnSite | Number of days the product has been on the site |

I also merged in all the features from all the other datasets, such as whether the product is on discount or if the customer is female. These features need to be calculated individually for each fold for cross validation.
## Initial Modelling

The target of our model is the probability of class membership, the goal is to identify the most probable purchases. A ranking metric such as ROC fits this description but since our data is very imbalanced I chose to evaluate using average precision, which estimates the area under a precision-recall curve and focuses on performance on the minority class. However, since we are interested in making recommendations to each customer independently I will use the mean average precision (mAP) over all the customers.

For initial models I used a single fold. As a baseline, a model which always predicts the majority class achieved an mAP score of 0.140 and a random classifier scores 0.161.

<table>
  <thead>
    <tr>
      <th rowspan="2" style="border-top:0px;border-left:0px"></th>
      <th colspan="2" halign="left">mAP</th>
    </tr>
    <tr>
      <th>train</th>
      <th>valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>XGBClassifier</th>
      <td>0.406153</td>
      <td>0.392299</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.998187</td>
      <td>0.339013</td>
    </tr>
    <tr>
      <th>ExtraTreesClassifier</th>
      <td>0.998192</td>
      <td>0.316818</td>
    </tr>
    <tr>
      <th>GradientBoostingClassifier</th>
      <td>0.390538</td>
      <td>0.386919</td>
    </tr>
  </tbody>
</table>

I first trained a selection of simple models. The random forest and extra trees classifiers struggled with overfitting. Meanwhile the boosting classifiers did not suffer from this but were slow to train except for XGBoost which can be accelerated by a GPU.

A common approach to handling imbalanced data is to over/undersample. I retrained the XGBoost classifier after resampling the training data. 

<table>
  <thead>
    <tr>
      <th rowspan="2" style="border-top:0px;border-left:0px"></th>
      <th colspan="2" halign="left">mAP</th>
    </tr>
    <tr>
      <th>train</th>
      <th>valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>RandomOverSampler</th>
      <td>0.898405</td>
      <td>0.390845</td>
    </tr>
    <tr>
      <th>RandomUnderSampler</th>
      <td>0.92691</td>
      <td>0.387436</td>
    </tr>
    <tr>
      <th>SMOTE</th>
      <td>0.995906</td>
      <td>0.341304</td>
    </tr>
    <tr>
      <th>TOMEKS</th>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

Undersampling performed similarly to no/over sampling while training much quicker. SMOTE was very slow (15mins), performed worse and appeared to be overffitting. TOMEKS was far too slow to use.

![XGB Feature Importance](/ProductRecommendation/visualisations/Initial%20Modelling/XGB_feature_importance.png "XGB Feature Importance")

The feature importance plot highlights that most of the one-hot encoded countries are not necessary, so I updated the data to use just the top 5 countries. Additionally I removed newCustomer, isPremier, and productId_customerPurchaseRate. Only 6.5% of purchases are repeats, 150 customers are flagged isPremier, and 515 customers are new - it is likely these features have low importance as they are very rare and have low variance in the dataset. 

## Further Feature Engineering

I decided to take another look at the data in search of more useful features. In particular I wanted to create features to describe brands, products, and customers hoping that the model would learn to treat similar brands similarly, up until this point the only feature describing the brand is the number of customers who have bought from them.

We can characterize a brand by the types of products it sells, its average price point for those products compared to other brands, and the kind of people who shop there. I grouped the products by product type and standardised the prices, this gave the price of the product relative to other products of that type. Using this I calculated the median standardised price of product types from each brand. I also calculated the proportion of product types in each brand's catalog. I finally calculated the percentage of customers who are female for each brand, resulting in a total of 85 (42+42+1) features for each brand. I used principal component analysis to reduce the dimensionality to 5 and merged these into the train and validation datasets.

For individual products I calculated the deviations from the mean price for that brand and product type - is this an expensive product for its type. I did the same for product types (coats tend to be pricier than socks), and found the rate of purchases made by women for each type. Finally I calculated the mean price each customer spends on each product type, giving a sense of a customer's budget for each type of product.

With these new features I retrained the models on a single fold of data with and without random oversampling.

<table>
    <thead>
      <tr>
        <th rowspan="2" halign="left">Dataset</th>
        <th rowspan="2" halign="left">Sampling</th>
        <th rowspan="2" halign="left">Model</th>
        <th colspan="2" halign="left">mAP</th>
        <th colspan="2" halign="left">avg_precision</th>
        <th colspan="2" halign="left">precision</th>
        <th colspan="2" halign="left">recall</th>
        <th colspan="2" halign="left">f1</th>
        <th colspan="2" halign="left">accuracy</th>
        <th colspan="2" halign="left">roc_auc</th>
      </tr>
      <tr>
        <th>train</th>
        <th>valid</th>
        <th>train</th>
        <th>valid</th>
        <th>train</th>
        <th>valid</th>
        <th>train</th>
        <th>valid</th>
        <th>train</th>
        <th>valid</th>
        <th>train</th>
        <th>valid</th>
        <th>train</th>
        <th>valid</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th rowspan="8" halign="left">Initial</th>
        <th rowspan="4" halign="left">None</th>
        <th>XGBClassifier</th>
        <td>0.406153</td>
        <td>0.392299</td>
        <td>0.120920</td>
        <td>0.099647</td>
        <td>0.655348</td>
        <td>0.381068</td>
        <td>0.002116</td>
        <td>0.001744</td>
        <td>0.004218</td>
        <td>0.003472</td>
        <td>0.979767</td>
        <td>0.979850</td>
        <td>0.501046</td>
        <td>0.500843</td>
      </tr>
      <tr>
        <th>RandomForestClassifier</th>
        <td>0.998187</td>
        <td>0.339013</td>
        <td>0.999893</td>
        <td>0.074310</td>
        <td>0.999315</td>
        <td>0.227848</td>
        <td>0.991823</td>
        <td>0.000400</td>
        <td>0.995555</td>
        <td>0.000798</td>
        <td>0.999821</td>
        <td>0.979853</td>
        <td>0.995904</td>
        <td>0.500186</td>
      </tr>
      <tr>
        <th>ExtraTreesClassifier</th>
        <td>0.998192</td>
        <td>0.316818</td>
        <td>0.999951</td>
        <td>0.063803</td>
        <td>0.999873</td>
        <td>0.171631</td>
        <td>0.992798</td>
        <td>0.002688</td>
        <td>0.996323</td>
        <td>0.005293</td>
        <td>0.999852</td>
        <td>0.979665</td>
        <td>0.996398</td>
        <td>0.501211</td>
      </tr>
      <tr>
        <th>GradientBoostingClassifier</th>
        <td>0.390538</td>
        <td>0.386919</td>
        <td>0.095929</td>
        <td>0.094008</td>
        <td>0.662281</td>
        <td>0.520362</td>
        <td>0.000828</td>
        <td>0.001277</td>
        <td>0.001653</td>
        <td>0.002548</td>
        <td>0.979754</td>
        <td>0.979874</td>
        <td>0.500409</td>
        <td>0.500627</td>
      </tr>
    <tr>
      <th>RandomOverSampler</th>
      <th rowspan="4" halign="left">XGBClassifier</th>
      <td>0.898405</td>
      <td>0.390845</td>
      <td>0.816548</td>
      <td>0.09828</td>
      <td>0.740219</td>
      <td>0.053309</td>
      <td>0.778433</td>
      <td>0.749628</td>
      <td>0.758846</td>
      <td>0.099539</td>
      <td>0.752621</td>
      <td>0.727009</td>
      <td>0.752621</td>
      <td>0.738086</td>
    </tr>
    <tr>
      <th>RandomUnderSampler</th>
      <td>0.92691</td>
      <td>0.387436</td>
      <td>0.827609</td>
      <td>0.096558</td>
      <td>0.744686</td>
      <td>0.052409</td>
      <td>0.779828</td>
      <td>0.754126</td>
      <td>0.761852</td>
      <td>0.098008</td>
      <td>0.756233</td>
      <td>0.720606</td>
      <td>0.756233</td>
      <td>0.737022</td>
    </tr>
    <tr>
      <th>SMOTE</th>
      <td>0.995906</td>
      <td>0.341304</td>
      <td>0.995497</td>
      <td>0.067802</td>
      <td>0.999001</td>
      <td>0.154283</td>
      <td>0.960018</td>
      <td>0.009663</td>
      <td>0.979122</td>
      <td>0.018188</td>
      <td>0.979529</td>
      <td>0.979</td>
      <td>0.979529</td>
      <td>0.504288</td>
    </tr>
    <tr>
      <th>TOMEKS</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th rowspan="6" halign="left">Improved</th>
      <th rowspan="3" halign="left">None</th>
      <th>XGBClassifier</th>
      <td>0.407936</td>
      <td>0.393897</td>
      <td>0.121084</td>
      <td>0.101452</td>
      <td>0.705776</td>
      <td>0.489489</td>
      <td>0.002143</td>
      <td>0.001811</td>
      <td>0.004273</td>
      <td>0.003608</td>
      <td>0.979772</td>
      <td>0.979870</td>
      <td>0.501062</td>
      <td>0.500886</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.998175</td>
      <td>0.333339</td>
      <td>0.999890</td>
      <td>0.069770</td>
      <td>0.999260</td>
      <td>0.143529</td>
      <td>0.991620</td>
      <td>0.000678</td>
      <td>0.995425</td>
      <td>0.001349</td>
      <td>0.999815</td>
      <td>0.979804</td>
      <td>0.995802</td>
      <td>0.500297</td>
    </tr>
    <tr>
      <th>ExtraTreesClassifier</th>
      <td>0.998179</td>
      <td>0.303447</td>
      <td>0.999948</td>
      <td>0.056117</td>
      <td>0.999873</td>
      <td>0.100178</td>
      <td>0.992601</td>
      <td>0.004387</td>
      <td>0.996224</td>
      <td>0.008407</td>
      <td>0.999848</td>
      <td>0.979167</td>
      <td>0.996299</td>
      <td>0.501789</td>
    </tr>
    <tr>
      <th>RandomOverSampler</th>
      <th rowspan="3" halign="left">XGBClassifier</th>
      <td>0.898677</td>
      <td>0.390980</td>
      <td>0.816960</td>
      <td>0.099421</td>
      <td>0.741932</td>
      <td>0.053700</td>
      <td>0.778031</td>
      <td>0.748884</td>
      <td>0.759553</td>
      <td>0.100214</td>
      <td>0.753703</td>
      <td>0.729320</td>
      <td>0.753703</td>
      <td>0.738901</td>
    </tr>
    <tr>
      <th>RandomUnderSampler</th>
      <td>0.927283</td>
      <td>0.387719</td>
      <td>0.829230</td>
      <td>0.096175</td>
      <td>0.746676</td>
      <td>0.052747</td>
      <td>0.779104</td>
      <td>0.753882</td>
      <td>0.762546</td>
      <td>0.098596</td>
      <td>0.757389</td>
      <td>0.722544</td>
      <td>0.757389</td>
      <td>0.737891</td>
    </tr>
    <tr>
      <th>SMOTE</th>
      <td>0.995926</td>
      <td>0.341553</td>
      <td>0.995609</td>
      <td>0.068484</td>
      <td>0.999056</td>
      <td>0.151809</td>
      <td>0.960077</td>
      <td>0.009041</td>
      <td>0.979179</td>
      <td>0.017066</td>
      <td>0.979585</td>
      <td>0.979037</td>
      <td>0.979585</td>
      <td>0.504002</td>
    </tr>
  </tbody>
</table>

These new features had minimal impact on the performance of the model. Given the low variance in the XGBoost classifiers we may be underfitting the data or approaching the limit of performance. Therefore we should try training some larger models for longer to reduce bias error.

I double checked by repeating the experiment using XGBoost, no resampling, and 3-fold cross validation. As before the improved dataset did not improve performance.

<table>
  <thead>
    <tr>
      <th rowspan="2" style="border-top:0px;border-left:0px"></th>
      <th colspan="2" halign="left">mAP</th>
    </tr>
    <tr>
      <th>train</th>
      <th>valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Improved</th>
      <td>0.406948</td>
      <td>0.394856</td>
    </tr>
    <tr>
      <th>Initial</th>
      <td>0.404730</td>
      <td>0.393791</td>
    </tr>
  </tbody>
</table>

## Modelling & Tuning

With the XGBoost model as a starting point I moved on to hyperparameter tuning. I did a random search of parameter space, sampling on a log scale from coarse to fine and compared the models using k-fold cross validation. The final model chosen was the best performing undersampling model whith a mAP score of 0.3959 on the validation dataset. This trained in less than half the time of the best model without resampling while sacrificing less than 1% performance.

I also trained some larger models (4000 deeper estimators) with heavy regularisation to combat overfitting, but these did not outperform smaller models which were much faster to train and quicker at inference.

### Probability Calibration

The final outputs of our model will be a predicted probability, so we should investigate whether our probabilities are well calibrated.

![Uncalibrated](/ProductRecommendation/visualisations/xgb_calibration_ROS_uncalibrated.png "Uncalibrated")


In the above figure we split our predicted probabilities into 100 quantiles. For each quantile we sum our predicted probabilities and divide by count, this represents the proportion of samples in each bin that our model expects will be in the true class. We then compare this to the real proportion.

So, consider the top 1% quantile, the customer-product pairs with the highest predicted probability of leading to a purchase according to our model. Our model would expect over 80% of these pairs to be the True class, but in reality only 20% are. Evidently our model is poorly calibrated and overpredicts purchase probabilities. 

We can solve this by adjusting the predictions according to the class distributions in the original dataset and the oversampled dataset. The adjustment formula can be derived using Bayes Theorem [(in the derivation notebook)](/ProductRecommendation/reports/Adjustment%20formula%20derivation.ipynb).

![Calibrated](/ProductRecommendation/visualisations/xgb_calibration_ROS_calibrated.png "Calibrated")

Our calibrated predicted probabilities match the class distribution far better, and the mean average precision is unchanged.

### Final Training

In summary, we have merged the datasets and engineered features to assist training. We used random oversampling to combat class imbalance and selected the optimal model and hyperparameters using k-fold cross validation. Finally we calibrated the predicted probabilities using the class distributions of the true and resampled datasets.

We can now train the model with these settings on the entire dataset and make predictions on the test set. These are saved in [SUBMISSION.csv](/ProductRecommendation/reports/SUBMISSION.csv)
## Discussion

Since the purchases table includes sales for the test set we can use this to evaluate performance on the test data. The model achieved a mean average precision score of 0.399 showing good generalisation and a significant improvement over the baseline score of 0.161.

It will be interesting to see how often the customer purchases one of the top-n products we predicted - if we only recommend 3 products how often do they buy one of those? If we consider customers in the test set who bought at least one product in January, 34.2% bought the product with the highest purchase probability. 59.5% bought one of their top 3 products and 84.2% one of the top 10. Evidently the model is identifying products which a customer is likely to purchase.

I compared performance of top-3 predictions for different groups of customers. The model performs better for men (64.2%, mAP 0.445) than women (57.8%, mAP 0.383), despite having more examples of women to learn from. This trend continues when comparing age groups, for people born 1990 to 1999 the model performs slightly below average, but is much better for those born before 1970 where we have far fewer customers. It may be that women and 18-30s buy a wider range of products making them less predictable.


![Performance YOB](/ProductRecommendation/visualisations/performance_YOB.png "Performance YOB")

On further examination of the data, I found that on average women buy from 4.35 different brands with a standard deviation of 5.09, compared to 3.34 and 3.53 for men. Similarly, customers born between 1985 and 1995 buy a greater variety of product types compared to those born before 1975. This supports the idea that groups with greater variation in habits are harder to predict.

![Age group variety](/ProductRecommendation/visualisations/Age_groups_product_variety.png "Age group variety")

For countries with more than 500 customers in the test set there is some variation in model predictive power. The UK matches the average, 59.9% of buyers bought one of their top 3 predicted products. Australia was the top scorer at 62.5%, and Russia was a relative outlier at 55.1%.

The figure below shows predictions on a sample of 40 customers. Products are ordered left to right by purchase probability, each row is a customer. Yellow squares are products which were bought, and if the customer viewed fewer than 20 products we pad the right with blue. A perfect model would have all yellow squares on the left. While not a precise metric we can see how the highest ranked products tend to be bought more.

![Top 20 sample](/ProductRecommendation/visualisations/top_20_predictions_bought.png "Top 20 sample")

### Production

Fortunately the model has very fast inference speed, 10 million predictions took 29s on GPU and 58s on CPU. Due to the engineered features we should look closely at how we could put this model into production.

To predict the purchase probability of a new customer-product pair we will need to generate all the features the model requires, but it is not feasible to repeat this process for every single prediction. Furthermore, as products are bought and viewed throughout the day many of the features will change, such as brand_viewCount and customerPurchaseRates. We could calculate these features daily and hope their values do not drift too much, but this is not ideal. A customer's session time may be tens of minutes or less and we want to make relevant recommendations based on the session they are currently in - a customer is looking at shoes right now and we need to recommend products they may like while they are still shopping. Therefore a mixed approach would be appropriate.

I would propose running two simultaneous processes. The first is run every 12 hours or daily and performs inference on the entire dataset, updating all engineered features with that day's views and purchases, and saving these features for later use. This process should not be too infrequent, else new products and brands will suffer from a slow start.

The second process makes frequent predictions for live customers only, using the saved engineered features as an approximation and incrementally updating those which can be. We can easily update features like customer view counts during a session, but updating overall purchase rates may not be possible. In particular this process may keep a buffer of new views, making predictions in batches, and updating some purchase probability table.

With this pair of processes a customer can receive recommendations based on products they are currently viewing, without the need to repeatedly process the entire dataset.

The model itself took 8 minutes to run on GPU with the dataset loaded in RAM, this would not be possible with the entire dataset, but we would never need to load the full dataset at once with an alternative implementation. To create the features we always group by customer or some other category, these groups can be processed individually and their much smaller aggregated tables can be saved. To train the model we can divide the customers in to groups and join those with the saved aggregated features, this way we could load a batch of customers at a time rather than the entire training dataset. If multiple GPUs are available we can use the dask package to train the XGBoost classifier across a cluster, speeding up training time.

The classifier should be regularly retrained, daily or weekly depending on training time. It may also be possible to train the model incrementally, adding additional estimators to the model using only the new data, but the efficacy of this approach would need to be separately evaluated. Less frequently we should repeat the hyperparameter search with cross validation in case the optimal parameters drift over time.


## Future Work

### Data Augmentation

There are several ways in which our dataset could be augmented, optionally we could create more augmented samples for the minority class to reduce some of the data imbalance.

#### Customer Forgetting

Randomly transform a samples as though they are new customers, setting their purchase data to zero and making their age and country unknown. This should be simple to implement and could help performance on new customers.

#### Purchase Forgetting

Randomly reduce the number of purchases a customer has made, as though they bought those products elsewhere 

#### Multiple Views

Currently we have one training example per customer-product pair, but we could have an example for each view. If a customer views a product three times before buying we could make three training examples - this person who has viewed this product once goes on to buy the product (and also view it a few more times). Doing so would increase the size of the dataset by about 50% and in particular more of the minority class, as products with more views are more likely to be bought.

However, this adds complexity to the data processing step as to avoid data leakage we should only consider events that have occurred before each view. For example the number of people who have bought from a brand will change daily so that feature will be different for views at different times. Since it is not feasible to recalculate these features for all of the tens of millions of views we may need to find some way to approximate their values or use alternative features. An approach of this sort would allow us to make use of the January purchases data which has so far been excluded.

### Additional Data

With feature engineering I looked for ways to describe similar brands and products. With more information we would be able to build better representations which could encode the brand's style or perception. This could be achieved using an autoencoder with many more brand features. Alternatively we could train an image classifier to predict brands from product images, the penultimate layer which represents the images as vectors could then be used as product embeddings. Perhaps brand embeddings could be the mean of their products.