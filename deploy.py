import joblib
import pandas as pd

scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')
model = joblib.load('model.joblib')
new_chip = pd.read_csv('new_chip.csv')

chip_scaled = scaler.transform(new_chip.values)
chip_pca = pca.transform(chip_scaled)
prediction = model.predict(chip_pca)
prediction = prediction[0] * -1
if prediction == 1:print("Attention the chip is defected")
else:print("Chip is OK")
