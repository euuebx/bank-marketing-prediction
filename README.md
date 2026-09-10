# Bank Term Deposit Prediction

Built this to practice classification on a real dataset. Used the UCI Bank Marketing dataset (41k+ records from a Portuguese bank's telemarketing campaign) to predict whether a customer would sign up for a term deposit.

Data source: https://archive.ics.uci.edu/dataset/222/bank+marketing

## What I did

- Trained a Random Forest and an XGBoost classifier to predict the `y` column (subscribed or not)
- Dropped the `duration` column on purpose. It's call length, and the dataset notes say it basically gives away the answer (if the call was 0 seconds, obviously no). Keeping it in would've inflated the accuracy but made the model useless in practice.
- Used RFE to find the most useful features instead of just throwing everything in
- Ran PCA to visualize how separable the two classes actually are
- Had to use class weighting since only ~11% of customers said yes. Otherwise the model could just predict "no" every time and still look accurate

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
