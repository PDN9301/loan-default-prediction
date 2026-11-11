# Loan Default Prediction with Deep Learning

**Which loan applicants are likely to default?** This project trains a feed-forward neural network on 307,511 real loan applications (120 features, only 8% defaults) and compares two ways of handling heavily imbalanced data.

**Tools:** Python, pandas, TensorFlow/Keras, scikit-learn, imbalanced-learn

## Results

Evaluated on 76,878 held-out applications (6,206 defaults) that were never used for training or tuning:

| Approach | ROC-AUC | Recall (Default) | Precision (Default) |
|---|---|---|---|
| SMOTE oversampling, 0.5 threshold | 0.694 | 0.10 | 0.19 |
| **Class weights, 0.5 threshold** | **0.738** | **0.66** | 0.16 |
| Class weights, F1-tuned threshold (0.636) | 0.738 | 0.38 | 0.22 |

## What I learned

- **Accuracy is misleading with imbalanced data.** A model that approves everyone scores 92% accuracy. This project reports ROC-AUC and recall on defaulters instead.
- **Evaluation leakage.** My first version applied SMOTE *before* the train/test split, so synthetic rows ended up in the test set and inflated the scores (about 91% accuracy). After splitting first and oversampling only the training data, honest performance was much lower. The fix taught me more than the original score did.
- **SMOTE vs class weights.** SMOTE's synthetic defaulters were easy for the network to learn (~0.95 training AUC) but didn't look like real ones. Training on real data with class weights raised ROC-AUC by 0.04 and caught 6.5× more defaulters.
- **The threshold is a business decision.** A 0.5 threshold catches two-thirds of defaulters but flags many good applicants. A tuned threshold trades recall for fewer false alarms. The right choice depends on what a default costs compared with a rejected good customer.

## Approach

1. Median/mode imputation for missing values; label encoding for categoricals
2. Stratified train / validation / test split (67.5% / 7.5% / 25%)
3. Model: Dense(256) → Dense(128) → Dense(64) → sigmoid, with 30% dropout, Adam, and binary cross-entropy
4. Early stopping on validation AUC
5. Decision threshold tuned on the validation set only

**Next steps:** gradient-boosted trees (LightGBM), one-hot encoding for nominal categories, and the bureau / previous-application tables from the full dataset.

## Run it

The dataset (166 MB) is too large for GitHub:

1. Download `application_train.csv` from the Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) competition.
2. Save it in this folder as `loan_data.csv`.
3. Run:

```bash
pip install -r requirements.txt
jupyter notebook loan_default_prediction.ipynb
```

`Data_Dictionary.csv` describes every column. `TARGET` is 1 for clients with payment difficulties (default) and 0 otherwise.
