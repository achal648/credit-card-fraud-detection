# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 00:55:38 2018

@author: achal
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

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

# Train logistic regression
print ("Training logistic regression...\n")
logreg = LogisticRegression()
parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(estimator= logreg,param_grid= parameters, cv=5,  n_jobs= -1)
grid_search.fit(X_train, y_train)

# Testing data
y_pred = grid_search.predict(X_test)
print ('Accuracy on test set: {:.2f}'.format(grid_search.score(X_test, y_test)))

# Print confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Generate ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, grid_search.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, grid_search.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
