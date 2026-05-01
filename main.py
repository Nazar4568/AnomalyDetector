import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

