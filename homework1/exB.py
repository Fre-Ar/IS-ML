import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import const.py from the utils package
from utils.const import *
from sklearn import make_classification

#Generating range of values for the features
RANGE= (0.0,1.0)


cov1 = np.array([[1.5, 0.3], [0.3, 1.5]])

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
   
   mean_class_0 = np.ndarray([0.3, 0.3])
   mean_class_1 = np.ndarray([0.6, 0.6])
   
   X0=rng.multivariate_normal(mean_class_0, cov, samples_per_class)
   X1=rng.multivariate_normal(mean_class_1, cov, samples_per_class)
   
   X = np.vstack((X0, X1))
   y = np.array([0]*samples_per_class + [1]*samples_per_class)
   
   X = np.clip(X, RANGE[0], RANGE[1])
   
   idx = rng.permutation(samples)
   X, y = X[idx], y[idx]
   return X, y

cov2 = np.array([[16, 0.0], [0.0, 16]])

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
    y_out = np.hstack(y_out_list)

    X_new = np.vstack([x, X_out])
    y_new = np.hstack([y, y_out])

    return X_new, y_new
    