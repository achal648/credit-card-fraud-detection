# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 20:14:43 2018

@author: achal
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Read data
data = pd.read_csv('german_credit_data.csv')

# Delete index column
del data['Unnamed: 0']

# Fill empty values
data = data.fillna(data['Saving accounts'].value_counts().index[0])

# Replace categorical data
replace_map = {'Saving accounts': {'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4}, 'Checking account': {'little': 1, 'moderate': 2, 'rich': 4}, 'Housing': {'own': 1, 'rent': 2, 'free': 3}, 'Sex': {'male': 1, 'female': 2}, 'Purpose': {'car': 1, 'radio/TV': 2, 'furniture/equipment': 3, 'business': 4, 'education': 5, 'repairs': 6, 'vacation/others': 7, 'domestic appliances': 8}, 'Risk': {'good': 0, 'bad': 1}}
data.replace(replace_map, inplace=True)


X = np.array(data.drop(['Risk'], 1).astype(float))
y = np.array(data['Risk'])

# Data scaling
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state=0)

# Train random forest
print ("Training random forest...\n")

forest = RandomForestClassifier(n_estimators = 1000, random_state = 42)

forest.fit(X_train, y_train)

# Testing data
y_pred = forest.predict(X_test)
print ('Accuracy on test set: {:.2f}'.format(forest.score(X_test, y_test)))

# Print confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
