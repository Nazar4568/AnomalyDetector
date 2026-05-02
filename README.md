# Name: Semiconductor Anomaly Detection (SECOM)

**Description** : Project for searching defective chips using 590 sensors 

**Tech Stack**: Python , Pandas, Scikit-learn, Joblib

## What was done:
1) **Displaying and analysis of the raw data** 
2) **Data cleaning** The raw dataset had a lot of missing values(41951). Dropped completely broken sensors (>50% NaNs) and filled the remaining gaps with zeros.
3) **Preprocessing & PCA:** Scaled the data using `StandardScaler`. Applied PCA to scaled data since 590 sensors make too much noise, reduced to the 30 most important components. 
4) **Anomaly Detection:** I used Isolation Forest (Unsupervised learning) using sklearn.
5) **Evaluation:** Checked the results using Confusion Matrix and Classification Report (adjusted predictions by multiplying by -1 to match the SECOM label logic).
6) **Deployment Simulation:** Saved the scaler, PCA, and model via `joblib`. Created a `deploy.py` script that loads these files to simulate real-time inference on a single new chip.

**How to run** 
First execute main.py to train the model, then execute the deploy.py to get the result of our new_chip.csv
