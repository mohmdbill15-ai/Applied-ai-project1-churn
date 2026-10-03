# Customer Churn Prediction 
 
## Week 1: Exploratory Data Analysis 
 
### Dataset - Source: Telco Customer Churn (Kaggle) - Size: 7,043 customers, 21 features - Target: Predict customer churn (Yes/No) 
 
### Key Findings 
-Customers with shorter tenure appear more likely to churn than long-term customers.

-Monthly charges show a noticeable relationship with churn, with higher-paying customers appearing more likely to leave.

-Customers on month-to-month contracts show higher churn compared with customers on longer-term contracts.

-Internet service type and payment method show different churn patterns across customer groups.

-The analysis suggests that tenure, contract type, charges, services, and payment method may be useful features for predicting customer churn.
 
### Setup 
Open the Kaggle notebook or run locally: 
pip install pandas numpy matplotlib seaborn

## Week 2: Building ML Models
-Baseline (always "stay"): accuracy 0.735

-Best model: Logistic Regression, AUC 0.842, recall 0.567 at threshold 0.50

-Top churn drivers (permutation importance): tenure, TotalCharges, Contract_Two year

-Threshold chosen: 0.15, because missing a churner costs PKR 6000, compared with PKR 1000 for an unnecessary retention offer

-Engineered features: n_services, is_new, charge_per_mo, price_jump; effect on AUC: 0.8422 → 0.8420

-Biggest lesson: Lowering the threshold can increase recall when missing a churner is more costly, while the engineered features did not improve Random Forest AUC in this experiment.

## Week 3: Model Optimization and Unsupervised Learning

* Split-to-split accuracy range across 20 seeds: **0.780 to 0.828**
* 5-fold CV AUC: LR **0.8464 ± 0.0129**, RF **0.8464 ± 0.0114**, XGBoost **0.8502 ± 0.0117**
* Tuning: best RF grid parameters: **max_depth=8, max_features=sqrt, min_samples_leaf=20**; grid search took **123s** and random search took **137s**
* Test AUC of final model (used once): **0.8483**
* Customer segments (k = 4):

  * **New, high-charge customers:** 43% churn
  * **New, low-service customers:** 32% churn
  * **Long-term, high-service customers:** 14% churn
  * **Long-term, low-charge customers:** 5% churn
* PCA: **15 of 30 components** explain 90% of the variance
* Biggest lesson: **A single train/test split can give a misleading picture of model performance, so cross-validation gives a more reliable estimate of how models perform.**
