import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#File deleted after each runtime, upload data_banknote_authentication.txt again if necessarry
bankdata = pd.read_csv("datasets/data_banknote_authentication.txt")

print(bankdata.shape)
print(bankdata.head())

X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

print(y_test.shape)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
