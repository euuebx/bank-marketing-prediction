# Bank Term Deposit Prediction

Built this to practice classification on a real dataset. Used the UCI Bank Marketing dataset (41k+ records from a Portuguese bank's telemarketing campaign) to predict whether a customer would sign up for a term deposit.

Data source: https://archive.ics.uci.edu/dataset/222/bank+marketing

## What I did

Trained a Random Forest and an XGBoost model to predict if a customer subscribes (the `y` column).

Dropped `duration` (call length) because it basically leaks the answer, a 0 second call means no every time. Keeping it would've made the accuracy look better than it actually is.

Used RFE to cut down to the features that actually matter, and PCA to see how separable the two outcomes are.

Only ~11% of people actually said yes, so I had to weight the classes or the model would just guess "no" every time and still score high.

## Results

- Random Forest: ~87% accuracy, ROC-AUC ~0.82
- XGBoost: ~85% accuracy, ROC-AUC ~0.81

## What stood out

The features that mattered most weren't really about the customer. It was mostly economic stuff (interest rates, employment numbers at the time). Made sense once I thought about it, people's decisions probably track the economy more than their personal profile.

## Run it

pip install scikit-learn xgboost pandas numpy matplotlib
python train.py

Needs bank-additional-full.csv in the same folder.

## Could improve

- One-hot encode instead of label encoding (label encoding kind of implies an order that isn't really there for stuff like job type)
- Try SMOTE instead of class weights
- More thorough cross-validation instead of one train/test split
