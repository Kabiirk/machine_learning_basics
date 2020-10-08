import numpy as np
from sklearn import linear_model 
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

df1 = pd.read_csv('data_decision_trees.txt', sep=",", header=None)

#UPLOAD THE FILE AGAIN !!
df2 = pd.read_csv('data_decision_trees.txt', sep=",", header=None)
