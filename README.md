# Bank Term Deposit Prediction

Built this project to practice classification using a real-world dataset. I used the UCI Bank Marketing dataset, which contains 41k+ records from a Portuguese bank's telemarketing campaign, and tried to predict whether a customer would subscribe to a term deposit.

Data source: UCI Bank Marketing Dataset

## What I did

* Trained a Random Forest and XGBoost model to predict whether a customer subscribes (`y`).
* Removed `duration` because it causes data leakage. The call duration is only known after the call, so using it would make the model look better than it would be in a real prediction.
* Used RFE to select a smaller set of important features.
* Used PCA to get a 2D view of the data and see how well the two outcomes separate.
* Used class weighting because only around 11% of customers subscribed. Without this, a model could get high accuracy just by predicting "no" most of the time.

## Results

* Random Forest: ~87% accuracy, ROC-AUC ~0.82
* XGBoost: ~85% accuracy, ROC-AUC ~0.81

ROC-AUC was used alongside accuracy because the dataset is quite imbalanced.

## What stood out

The features with the highest importance were mostly related to economic conditions, such as interest rates and employment figures, rather than just individual customer characteristics.

This was interesting because it suggests that the wider economic situation may have a significant effect on whether someone decides to subscribe to a term deposit.

## Run it

Install the required packages:

```bash
pip install scikit-learn xgboost pandas numpy matplotlib
```

Then run:

```bash
python train.py
```

Make sure `bank-additional-full.csv` is in the same folder as `train.py`.

## Things I could improve

* Use one-hot encoding instead of label encoding for categorical variables.
* Try SMOTE or other methods for dealing with class imbalance.
* Use cross-validation instead of relying on a single train/test split.
* Tune the model parameters to see if the results can be improved.
