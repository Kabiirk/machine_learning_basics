import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

play_tennis = pd.read_csv("PlayTennis.csv")

play_tennis

number = LabelEncoder()
play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])
play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])

print(play_tennis)

features = ["Outlook","Temperature"]
target = "Play Tennis"

features_train, features_test, target_train, target_test = train_test_split(play_tennis[features], play_tennis[target], test_size = 0.33, random_state = 54)

features_train

features_test

model = GaussianNB()
model.fit(features_train, target_train)

pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)

print(accuracy)

print(model.predict([[2,1]]))

#ans 0 implies 0=> 'No'
