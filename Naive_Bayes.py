# External imports
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Internal imports
from utils import load_preprocess_data, shuffle_split_dataset

print("Loading Data...")
clean_data = load_preprocess_data()
X = clean_data.drop('state', axis=1)
y = pd.DataFrame(clean_data['state'].values)
y.columns = ['state']


# X = X.iloc[:500000]
# y = y.iloc[:500000]
del clean_data

min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)

del X
y = y['state'].to_numpy(dtype=np.float32)

X_train, y_train, X_test, y_test = shuffle_split_dataset(X_scaled, y)

print("\nX_train dims: ", X_train.shape)
print("X_test dims: ", X_test.shape)
print("y_train dims:", y_train.shape)
print("y_test dims:", y_test.shape)
print('_'*100)


print("Starting Bernoulli Naive Bayes Classifier...")
BernNB = BernoulliNB()
BernNB.fit(X_train, y_train)
y_pred = BernNB.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy = ', acc)
del BernNB
print("_"*100)

print("Starting Gaussian Naive Bayes Classifier")
GausNB = GaussianNB()
GausNB.fit(X_train, y_train)
y_pred = GausNB.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy = ", acc)
del GausNB
print("_"*100)

print("Starting Gaussian Naive Bayes Classifier")
MultiNB = MultinomialNB()
MultiNB.fit(X_train, y_train)
y_pred = MultiNB.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy = ", acc)
