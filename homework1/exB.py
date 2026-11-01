import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import const.py from the utils package
from utils.const import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
import seaborn as sns

from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
#---------------------------------------------------------------------------------------
# B.I.
#---------------------------------------------------------------------------------------
# Random num generation
RNG = np.random.default_rng(SEED)
# Parameters for the features

RANGE= (0.0, 1.0)
VAR = 0.02
COVAR = 0.015

MEAN_0 = np.array([0.40, 0.45])
COV_0 = np.array([[VAR, COVAR], [COVAR, VAR]])
MEAN_1 = np.array([0.60, 0.55])
COV_1 = np.array([[VAR, -COVAR], [-COVAR, VAR]])

SAMPLES = 80

# Dataset 2 params
OUTLIERS_PER_CLASS = 4
MIN_DIST = 3 # 3 standard deviations away from the mean of the class

# class colors for plotting
COLORS = {0: "blue", 1: "red"}

TRAIN_TEST_SPLIT = 0.8

# SVM Hyperparameters
N_NEIGHBORS = 3
C = 1.0
KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']

# B.II. params
K_LIST = [1,2,5,20,50]

# Dataset 1 creation.
def generate_dataset1():
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
    
    samples_per_class = SAMPLES // 2
   
    X0 = RNG.multivariate_normal(MEAN_0, COV_0, samples_per_class)
    X1 = RNG.multivariate_normal(MEAN_1, COV_1, samples_per_class)
   
    X = np.vstack((X0, X1))
    y = np.array([0]*samples_per_class + [1]*samples_per_class)
   
    idx = RNG.permutation(SAMPLES)
    X, y = X[idx], y[idx]
    
    return X, y

#---------------------------------------------------------------------------------------
# Dataset 2 creation (Dataset 1 + outliers)

def generate_outliers(mean, cov, n):
    """
        Generates n outliers given a mean and covariance matrix for a class.
        params:
            mean: mean of the class
            cov: covariance matrix of the class
            n: number of outliers to generate
        return: 
            outliers (numpy array)
    """
    outliers = []
    # Invert the covariance matrices once for the Mahalanobis calculation
    inv_cov = inv(cov)
    # Rejection sampling: keep drawing until the point is far from its class mean
    while len(outliers) < n:
        p = RNG.multivariate_normal(mean, cov)
        # Check if it's far enough (w.r.t. the cluster's covariance shape)
        if mahalanobis(p, mean, inv_cov) >= MIN_DIST:
            outliers.append(p)
    return np.array(outliers)  

def generate_dataset2(x:np.ndarray, y:np.ndarray):
    """
    Generates outliers and adds them to the original dataset.
    params:
        x: original features
        y: original labels
    return: 
        features (numpy array), labels (numpy array)
    """
    # generate the outliers
    outliers_0 = generate_outliers(MEAN_0, COV_0, OUTLIERS_PER_CLASS)
    outliers_1 = generate_outliers(MEAN_1, COV_1, OUTLIERS_PER_CLASS)

    # Append the outliers to the original dataset
    X_new = np.vstack([x, outliers_0, outliers_1])
    y_new = np.hstack([y, 
                       np.zeros(OUTLIERS_PER_CLASS, int), 
                       np.ones(OUTLIERS_PER_CLASS, int)])
    is_outlier = np.arange(len(X_new)) >= len(x)
    
    return X_new, y_new, is_outlier


def scatter_dataset(ax, X, y, is_outlier=None, title=""):
    """Scatter plot of one dataset; outliers get a lime ring and their own legend entry."""
    if is_outlier is None:
        is_outlier = np.zeros(len(y), dtype=bool)
    for c in (0, 1):
        # regular points                                   
        m = (y == c) & ~is_outlier
        ax.scatter(X[m, 0], X[m, 1], color=COLORS[c], label=f"Class {c}")
    for c in (0, 1):
        # outliers                                   
        m = (y == c) & is_outlier
        if m.any():
            ax.scatter(X[m, 0], X[m, 1], color=COLORS[c], edgecolor="lime",
                       linewidth=2.5, s=120, label=f"Class {c} outlier")
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_aspect("equal")
    
def plot_datasets(x1, y1, x2, y2, is_outlier):
    """
        Plots the original dataset and the dataset with outliers.
        params:
            x1: original features (numpy array)
            y1: original labels (numpy array)
            x2: features with outliers (numpy array)
            y2: labels with outliers (numpy array)
            is_outlier: boolean array indicating which points are outliers in dataset 2
        """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True,
                             constrained_layout=True)
    scatter_dataset(axes[0], x1, y1, title="Dataset 1")
    scatter_dataset(axes[1], x2, y2, is_outlier, title="Dataset 2 (with outliers)")
    handles, labels = axes[1].get_legend_handles_labels()   # Dataset 2 has every entry
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    plt.show()


#---------------------------------------------------------------------------------------
#3.Split Dataset 1 and Dataset 2

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

def split_indices(y, train_frac=TRAIN_TEST_SPLIT, seed=SEED):
    """
    Performs a stratified split of the dataset into training and testing sets.
    params:
            y: labels (numpy array)
            train_frac: proportion of the dataset to include in the train split
            seed: seed for rng
    return:
        train, test (index arrays for train and test set)
    """
    return train_test_split(np.arange(len(y)), train_size=train_frac,
                            stratify=y, random_state=seed)

def kfold_indices(y, k=5, seed=SEED):
    """
    Performs a stratified k-fold 
    params:
            y: labels (numpy array)
            k: number of folds
            seed: seed for rng
    return:
        returns a list of (train_idx, val_idx) pairs.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))

#---------------------------------------------------------------------------------------
#4. Train on Dataset 1 and Dataset 2 + 5. exploring parameters for SVM

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

def train_and_eval_models(x1_train, y1_train, x1_test, y1_test, x2_train, y2_train, x2_test, y2_test):
    """
    Trains and evaluates GaussianNB, KNN, and SVM models on both datasets.
    """
    
    gnb1 = GaussianNB()
    gnb2 = GaussianNB()
    gnb1.fit(x1_train, y1_train)
    gnb2.fit(x2_train, y2_train)

    knn1 = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn2 = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn1.fit(x1_train, y1_train)
    knn2.fit(x2_train, y2_train)

    svmresults1 = {}
    svmresults2 = {}
    for kernel in KERNELS:
        svm1 = svm.SVC(kernel=kernel, C=C)
        svm1.fit(x1_train, y1_train)
        svm2 = svm.SVC(kernel=kernel, C=C)
        svm2.fit(x2_train, y2_train)
        svmresults1[kernel] = svm1
        svmresults2[kernel] = svm2

    answers1 = {
        "GaussianNB": accuracy(y1_test, gnb1.predict(x1_test)),
        "KNN": accuracy(y1_test, knn1.predict(x1_test)),
        "SVM_linear": accuracy(y1_test, svmresults1['linear'].predict(x1_test)),
        "SVM_poly": accuracy(y1_test, svmresults1['poly'].predict(x1_test)),
        "SVM_rbf": accuracy(y1_test, svmresults1['rbf'].predict(x1_test)),
        "SVM_sigmoid": accuracy(y1_test, svmresults1['sigmoid'].predict(x1_test))
    }
    answers2 = {
        "GaussianNB": accuracy(y2_test, gnb2.predict(x2_test)),
        "KNN": accuracy(y2_test, knn2.predict(x2_test)),
        "SVM_linear": accuracy(y2_test, svmresults2['linear'].predict(x2_test)),
        "SVM_poly": accuracy(y2_test, svmresults2['poly'].predict(x2_test)),
        "SVM_rbf": accuracy(y2_test, svmresults2['rbf'].predict(x2_test)),
        "SVM_sigmoid": accuracy(y2_test, svmresults2['sigmoid'].predict(x2_test))
    }

    for model, acc in answers1.items():
        print(f"Dataset 1 - {model} Accuracy: {acc:.4f}")
    for model, acc in answers2.items():
        print(f"Dataset 2 - {model} Accuracy: {acc:.4f}")   
        
    return answers1, answers2, gnb1, gnb2, knn1, knn2, svmresults1, svmresults2
    
def plot_confusion_matrices(x_test, y_test, title, answers, gnb, knn, svmresults):
    """
    Plots the confusion matrices for all models on one dataset.
    params:
        y_true: true labels (numpy array)
        y_pred: predicted labels (numpy array)
        title: title for the plot (string)
        answers: dictionary of model accuracies
        gnb: Gaussian Naive Bayes model
        knn: K-Nearest Neighbors model
        svmresults: dictionary of SVM models
        x_test: testing features (numpy array)
        y_test: testing labels (numpy array)
    """
    from sklearn.metrics import confusion_matrix
        
    for model, acc in answers.items():
            y_pred = None
            if model == "GaussianNB":
                y_pred = gnb.predict(x_test)
            elif model == "KNN":
                y_pred = knn.predict(x_test)
            else:
                kernel = model.split('_')[1]
                y_pred = svmresults[kernel].predict(x_test)
                 
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            plt.title(f"{title}" + f"\n{model} Accuracy: {acc:.4f}")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
        
#---------------------------------------------------------------------------------------
#6. visualization of the decision boundaries for each model on both datasets
def plot_decision_boundary(model, x_train, y_train, x_test, y_test, title):
    """
    Plots the decision boundary for a given model.
    params:
        model: trained model (sklearn classifier)
        x_train: training features (numpy array)
        y_train: training labels (numpy array)
        x_test: testing features (numpy array)
        y_test: testing labels (numpy array)
        title: title for the plot (string)
    """
    plt.figure(figsize=(8, 6))
    plt.title(title)
    
    # Create a mesh grid
    x_min, x_max = x_train[:, 0].min() - 0.1, x_train[:, 0].max() + 0.1
    y_min, y_max = x_train[:, 1].min() - 0.1, x_train[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # Predict on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and training points
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='k', marker='o', label='Train')
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolors='k', marker='x', label='Test')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    

#---------------------------------------------------------------------------------------
# B.II.
#---------------------------------------------------------------------------------------
def plot_knn_decision_boundaries(x_train, y_train, k_list,dataset="1"):
    """
    Plots the decision boundaries for KNN with different values of k.
    params:
        x_train: training features (numpy array)
        y_train: training labels (numpy array)
        k_list: list of k values to evaluate
    """
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        
        plt.figure(figsize=(8, 6))
        plt.title(f"KNN Decision Boundary with k={k} for Dataset {dataset}")
        x_min, x_max = x_train[:, 0].min() - 0.1, x_train[:, 0].max() + 0.1
        y_min, y_max = x_train[:, 1].min() - 0.1, x_train[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='k', marker='o')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

if __name__ == "__main__":
    # B.I.
    # Generate datasets
    print("Generating Datasets...")
    x1,y1 = generate_dataset1()
    x2,y2,outlier_mask = generate_dataset2(x1,y1)
    print("Datasets generated successfully.")
    
    plot_datasets(x1, y1, x2, y2, outlier_mask)

    # split datasets into training and testing sets
    train1, test1 = split_indices(y1)
    x1_train, y1_train = x1[train1], y1[train1]
    x1_test,  y1_test  = x1[test1],  y1[test1]
    train2 = np.r_[train1, np.arange(len(y1), len(y2))]  
    test2  = test1

    x1_train, x1_test, y1_train, y1_test = split_dataset(x1,y1,TRAIN_TEST_SPLIT)
    x2_train, x2_test, y2_train, y2_test = split_dataset(x2,y2,TRAIN_TEST_SPLIT)
    
    answers1, answers2, gnb1, gnb2, knn1, knn2, svmresults1, svmresults2 = train_and_eval_models(x1_train, y1_train, x1_test, y1_test, x2_train, y2_train, x2_test, y2_test)
    
    plot_confusion_matrices(x1_test, y1_test, f"Dataset 1 - Confusion Matrix", answers1, gnb1, knn1, svmresults1)
    plot_confusion_matrices(x2_test, y2_test, f"Dataset 2 - Confusion Matrix", answers2, gnb2, knn2, svmresults2)

    for model_name, model in [("GaussianNB", gnb1), ("KNN", knn1)] + [(f"SVM_{kernel}", svmresults1[kernel]) for kernel in KERNELS]:
        plot_decision_boundary(model, x1_train, y1_train, x1_test, y1_test, f"Dataset 1 - {model_name} Decision Boundary")
    for model_name, model in [("GaussianNB", gnb2), ("KNN", knn2)] + [(f"SVM_{kernel}", svmresults2[kernel]) for kernel in KERNELS]:
        plot_decision_boundary(model, x2_train, y2_train, x2_test, y2_test, f"Dataset 2 - {model_name} Decision Boundary")
        
    # B.II. 
    plot_knn_decision_boundaries(x1_train, y1_train, K_LIST, dataset="1")
    plot_knn_decision_boundaries(x2_train, y2_train, K_LIST, dataset="2")
