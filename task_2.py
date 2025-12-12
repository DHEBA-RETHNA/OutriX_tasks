import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = sns.load_dataset('iris')
x = df.drop('species', axis=1) 
y = df['species']               

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, knn_pred)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train) 
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("\nMODEL ACCURACY COMPARISON:")
print("Logistic Regression:", lr_acc)
print("KNN:", knn_acc)
print("Decision Tree:", dt_acc)