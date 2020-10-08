import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read Data
data = pd.read_csv("Mall_Customers.csv")

data.head()

"""## **Data Exploration**"""

#PLOT AGE FREQUENCY OF CUSTOMERS
data.hist('Age', bins=35)
plt.title('Distribution of Age')
plt.xlabel('Age')

#PLOT GENDER DISTRIBUTION OF CUSTOMERS
sns.countplot(x='Genre', data=data)

#PLOT RANGES OF SPENDING SCORE AND ANNUAL INCOME
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.title("Spending Score Ranges")
sns.boxplot(y=data["Spending Score (1-100)"], color="red")
plt.subplot(1,2,2)
plt.title("Annual Income Ranges")
sns.boxplot(y=data["Annual Income (k$)"])
plt.show()

#DISTRIBUTION OF CUSTOMERS ACCORDING TO THEIR SPENDING SCORES
ss1_20 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 1) & (data["Spending Score (1-100)"] <= 20)]
ss21_40 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 21) & (data["Spending Score (1-100)"] <= 40)]
ss41_60 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 41) & (data["Spending Score (1-100)"] <= 60)]
ss61_80 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 61) & (data["Spending Score (1-100)"] <= 80)]
ss81_100 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 81) & (data["Spending Score (1-100)"] <= 100)]

ssx = ["1-20", "21-40", "41-60", "61-80", "81-100"]
ssy = [len(ss1_20.values), len(ss21_40.values), len(ss41_60.values), len(ss61_80.values), len(ss81_100.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=ssx, y=ssy, palette="BuPu")
plt.title("Spending Scores")
plt.xlabel("Score")
plt.ylabel("Number of Customer Having the Score")
plt.show()

#DISTRIBUTION OF CUSTOMERS ACCORDING TO THEIR ANNUAL INCOME
ai0_30 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 0) & (data["Annual Income (k$)"] <= 30)]
ai31_60 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 31) & (data["Annual Income (k$)"] <= 60)]
ai61_90 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 61) & (data["Annual Income (k$)"] <= 90)]
ai91_120 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 91) & (data["Annual Income (k$)"] <= 120)]
ai121_150 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 121) & (data["Annual Income (k$)"] <= 150)]

aix = ["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]
aiy = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=aix, y=aiy, palette="BuPu")
plt.title("Annual Incomes")
plt.xlabel("Income")
plt.ylabel("Number of Customer")
plt.show()

#SEGMENTATION BASED ON AGE, SPENDING SCORE AND GENDER
sns.scatterplot('Age', 'Spending Score (1-100)', hue='Genre', data=data, palette="BuPu_r");
plt.title('Age to Spending Score, Colored by Gender');

sns.heatmap(.corr(), annot=True,)

from sklearn.cluster import KMeans
import seaborn as sns

#CREATING A KMEANS OBJECT
km = KMeans(n_clusters=3)
km

#PREDICTING CLUSTERS FOR EACH POINT
y_predicted = km.fit_predict(data[['Age', 'Spending Score (1-100)']])

#ADDING CLUSTERS OF EACH POINT TO DATAFRAME FOR EASE OF PLOTTING
data['cluster'] = y_predicted

data.head()

#CENTROID FOR EACH CLUSTER
km.cluster_centers_

#SEPARATING CLUSTERS INTO THEIR SEPARATE DATAFRAMES FOR BETTER PLOTTING
df0 = data[data.cluster == 0]
df1 = data[data.cluster == 1]
df2 = data[data.cluster == 2]

plt.scatter(df0['Age'], df0['Spending Score (1-100)'], color='red')
plt.scatter(df1['Age'], df1['Spending Score (1-100)'], color='green')
plt.scatter(df2['Age'], df2['Spending Score (1-100)'], color='blue')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='yellow', marker='o', label='centroid', s=200)
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.legend(['Cluster1', 'Cluster2', 'Cluster3', 'centroid'])
