# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 14:14:01 2025

@author: Debroop
"""

#import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


iris = load_iris(as_frame = True)
df = iris.frame
#print(df)
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns="target")
 
print("Sample data: \n", df.head())

X = df.drop("species", axis = 1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state = 42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))

import seaborn as sns
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",cmap="Blues", xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.title("KNN confusion matrix")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.tight_layout()
plt.show()
