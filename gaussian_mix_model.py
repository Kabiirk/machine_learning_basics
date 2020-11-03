import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn import datasets
from sklearn.mixture import GaussianMixture 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = datasets.load_iris()

X, y = datasets.load_iris(return_X_y=True)

# Split dataset into training and testing (80/20 split)
skf = StratifiedKFold(n_splits=5) # 
skf.get_n_splits(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
