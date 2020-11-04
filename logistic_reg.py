import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd
from utilities import visualize_classifier

# Define sample input data
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

#Create logistic regr classifier

classifier = linear_model.LogisticRegression(solver='liblinear', C = 100)

# Train the classifier
classifier.fit(X, y)

# Visualize the performance of the classifier 
visualize_classifier(classifier, X, y)

pred_class=classifier.predict(np.array([[5.0, 4.3]]))
print(pred_class)

pred_class_prob=classifier.predict_proba(np.array([[5.0, 4.3]]))
print(pred_class_prob*100)

df1 = pd.read_csv('datasets/data_decision_trees.txt', sep=",", header=None)

#UPLOAD THE FILE AGAIN !!
df2 = pd.read_csv('datasets/data_decision_trees.txt', sep=",", header=None)

import pickle
input_file = "data_singlevar_regr.txt"
input_file2 = "my_data1.xls"
data = np.loadtxt(input_file, delimiter=",")

data2 = pd.read_excel(input_file2, delimiter=",")
X = data2.drop('L', axis=1)
y = data2['L']
data2.shape

data.shape

X.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
#create model
regressor = linear_model.LinearRegression()

#Train model
regressor.fit(X_train, y_train)

#predict

y_test_pred = regressor.predict(X_test)

y_test_pred

#plot
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# Compute performance metrics
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

output_model_file = 'model.pkl'

#save model
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# to read model
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)
