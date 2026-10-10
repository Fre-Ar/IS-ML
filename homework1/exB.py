import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import const.py from the utils package
from utils.const import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

#---------------------------------------------------------------------------------------
#B.I.
#---------------------------------------------------------------------------------------
#Generating range of values for the features
#1. Dataset 1 creation.
RANGE= (0.0,1.0)



cov1 = np.array([[0.015, 0.003], [0.003, 0.015]])

def generate_dataset1(cov:np.ndarray):
   """
   This function generates a synthetic dataset for binary calssification.
   We assume 2 classes, 2 features, 80 samples split across 2 states. 
   The features are continuous real values.
   
   The approach is inspired by the sklearn.datasets.make_classification function,
   meaning we will use Gaussian clusters to generate the data points for each class.
   The clusters will be centered around the mean values of the features for each class,
   and the spread of the clusters will be controlled by the standard deviation of the Gaussian distribution.
   params: 
        cov: covariance matrix for the features
   return: 
        features (numpy array), labels (numpy array)
   """
   rng = np.random.default_rng(SEED)
   
   samples = 80
   samples_per_class = samples // 2
   
   mean_class_0 = np.array([0.3, 0.3], dtype=float)
   mean_class_1 = np.array([0.6, 0.6], dtype=float)
   
   X0=rng.multivariate_normal(mean_class_0, cov, samples_per_class)
   X1=rng.multivariate_normal(mean_class_1, cov, samples_per_class)
   
   X = np.vstack((X0, X1))
   y = np.array([0]*samples_per_class + [1]*samples_per_class)
   
   X = np.clip(X, RANGE[0], RANGE[1])
   
   idx = rng.permutation(samples)
   X, y = X[idx], y[idx]
   return X, y

x1,y1 = generate_dataset1(cov1)
#---------------------------------------------------------------------------------------
#Dataset 2 creation (Dataset 1 + outliers)
cov2 = np.array([[0.16, 0.0], [0.0, 0.16]])
min_dist = 0.3
def generate_dataset2(cov:np.ndarray,x:np.ndarray,y:np.ndarray,min_dist:float):
    """
    Genarates outliers and adds them to the original dataset.
    params:
    cov: covariance matrix for the outliers
        x: original features
        y: original labels
        min_dist: minimum distance of the outliers from their class means
    return: 
        features (numpy array), labels (numpy array)
    """
    rng = np.random.default_rng(SEED)
    samples =4
    samples_per_class = samples // 2
    X_out_list, y_out_list = [], []

    for label in np.unique(y):
        mean = x[y == label].mean(axis=0)

        # Rejection sampling: keep drawing until the point is far from its class mean
        pts = []
        while len(pts) < samples_per_class:
            p = rng.multivariate_normal(mean, cov)
            if np.linalg.norm(p - mean) > min_dist:
                pts.append(p)

        X_out_list.append(np.array(pts))
        y_out_list.append(np.full(samples_per_class, label))

    X_out = np.vstack(X_out_list)
    X_out = np.clip(X_out, RANGE[0], RANGE[1])
    y_out = np.hstack(y_out_list)

    X_new = np.vstack([x, X_out])
    y_new = np.hstack([y, y_out])

    return X_new, y_new

x2,y2 = generate_dataset2(cov2,x1,y1,min_dist)
#---------------------------------------------------------------------------------------
#3.Split Dataset 1 and Dataset 2

train_test_split=0.8
def split_dataset(x:np.ndarray,y:np.ndarray,train_test_split:float):
    """
    Splits the dataset into training and testing sets.
    params:
        x: features (numpy array)
        y: labels (numpy array)
        train_test_split: proportion of the dataset to include in the train split
    return:
        x_train, x_test, y_train, y_test (numpy arrays)
    """
    rng = np.random.default_rng(SEED)
    samples = x.shape[0]
    indices = rng.permutation(samples)
    train_size = int(samples * train_test_split)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]

x1_train, x1_test, y1_train, y1_test = split_dataset(x1,y1,train_test_split)
x2_train, x2_test, y2_train, y2_test = split_dataset(x2,y2,train_test_split)

#---------------------------------------------------------------------------------------
#4. Train on Dataset 1 and Dataset 2

def accuracy(y_true:np.ndarray, y_pred:np.ndarray):
    """
    Calculates the accuracy of the predictions.
    params:
        y_true: true labels (numpy array)
        y_pred: predicted labels (numpy array)
    return:
        accuracy (float)
    """
    return np.mean(y_true == y_pred)


n_neighbors = 3
c = 1.0

gnb1 = GaussianNB()
gnb1.fit(x1_train, y1_train)
gnb2 = GaussianNB()
gnb2.fit(x2_train, y2_train)

knn1 = KNeighborsClassifier(n_neighbors=n_neighbors)
knn1.fit(x1_train, y1_train)
knn2 = KNeighborsClassifier(n_neighbors=n_neighbors)
knn2.fit(x2_train, y2_train)

svm1 = svm.SVC(kernel='linear', C=c)
svm1.fit(x1_train, y1_train)
svm2 = svm.SVC(kernel='linear', C=c)
svm2.fit(x2_train, y2_train)

answers1 = {
    "GaussianNB": accuracy(y1_test, gnb1.predict(x1_test)),
    "KNN": accuracy(y1_test, knn1.predict(x1_test)),
    "SVM": accuracy(y1_test, svm1.predict(x1_test))
}
answers2 = {
    "GaussianNB": accuracy(y2_test, gnb2.predict(x2_test)),
    "KNN": accuracy(y2_test, knn2.predict(x2_test)),
    "SVM": accuracy(y2_test, svm2.predict(x2_test))
}

for model, acc in answers1.items():
    print(f"Dataset 1 - {model} Accuracy: {acc:.4f}")
for model, acc in answers2.items():
    print(f"Dataset 2 - {model} Accuracy: {acc:.4f}")   
