import pandas as pd
import matplotlib.pyplot as pls

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
 
# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv(url, names=colnames)
irisdata.head()

X = irisdata.drop('Class', axis=1)
y = irisdata['Class']

print(y)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)

y = le.transform(y)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf', random_state=0,C=1, gamma=0.01) #accuracy increases with degree
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
