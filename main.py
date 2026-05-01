import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import joblib

df_data = pd.read_csv("secom.data", sep= " ",header=None)
df_labels = pd.read_csv("secom_labels.data",sep=" ",header=None)
print("Starting shape of data is:", df_data.shape)
print("Starting shape of labels is:",df_labels.shape)

df_labels.columns = ["Target", "Time"]
print("Number of Nans in raw data is:",df_data.isnull().sum().sum())

df = pd.concat([df_data, df_labels], axis=1)

df =df.dropna(axis = 1, thresh = len(df)//2)
df = df.fillna(0)
print("Final shape:", df.shape)
print("Number of Nans in new data is:",df.isnull().sum().sum())

y = df["Target"]
X = df.drop(columns = ["Target", "Time"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = IsolationForest(contamination=0.07, random_state=42)

model.fit(X_train_scaled)
y_pred = model.predict(X_test_scaled)
y_pred = y_pred * -1

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

pca = PCA(n_components=30, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model_1 = IsolationForest(contamination=0.07, random_state=42)
model_1.fit(X_train_pca)
y_pred = model_1.predict(X_test_pca)
y_pred = y_pred * -1
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

