import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 

#upload Food_Service.tsv before running
dataset = pd.read_csv('datasets/Food_Service.tsv', delimiter='\t', quoting=3)
 

print(dataset)
 

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
 

print(stopwords.words('english'))
 

#Cleaning the text
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
 

corpus = []
 
for i in range(0,len(dataset)):
    review = re.sub('[^a-zA-Z^]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    print(review)
    corpus.append(review)
 
print(corpus)
 

from sklearn.feature_extraction.text import CountVectorizer
 
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
 

#Splitting Dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
 
#Training a Naive Bayes Model on training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
 
#Predicting the result
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
 
#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
