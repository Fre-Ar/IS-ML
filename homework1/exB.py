import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn import svm

from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

from utils.const import *
#---------------------------------------------------------------------------------------
# B.I.
#---------------------------------------------------------------------------------------
# Random num generation
RNG = np.random.default_rng(SEED)

# Dataset parameters
SAMPLES = 80

VAR = 0.02
COVAR = 0.015

MEAN_0 = np.array([0.40, 0.45])
COV_0 = np.array([[VAR, COVAR], [COVAR, VAR]])
MEAN_1 = np.array([0.60, 0.55])
COV_1 = np.array([[VAR, -COVAR], [-COVAR, VAR]])


# Dataset 2 params
OUTLIERS_PER_CLASS = 4
MIN_DIST = 3 # 3 standard deviations away from the mean of the class

# class colors for plotting
COLORS = {0: "blue", 1: "red"}
LIGHT_COLORS = {0: "deepskyblue", 1: "lightcoral"}

# Train-eval hyperparameters
TRAIN_TEST_SPLIT = 0.8

# Model Hyperparameters
N_NEIGHBORS = 5
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

def plot_split(ax, X, y, train_idx, test_idx, title=""):
    """
    Plots the training and testing sets of a dataset.
    params:
        ax: matplotlib axis to plot on
        X: features (numpy array)
        y: labels (numpy array)
        train_idx: indices of the training set
        test_idx: indices of the testing set
        title: title for the plot (string)
    """
    for idx, marker, name in [(train_idx, "o", "train"), (test_idx, "s", "test")]:
        for c in COLORS.keys():
            sel = idx[y[idx] == c]
            color = LIGHT_COLORS[c] if name == "test" else COLORS[c]
            ax.scatter(X[sel, 0], X[sel, 1], color=color, marker=marker,
                       edgecolor="black" if name == "test" else "none",
                       label=f"Class {c} ({name})")
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_aspect("equal")

#---------------------------------------------------------------------------------------
#4. Train on Dataset 1 and Dataset 2 + 5. exploring parameters for SVM

def get_models():
    return {
        "Naive Bayes": GaussianNB(),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=N_NEIGHBORS),
        "SVM (RBF)": svm.SVC(kernel="rbf", C=C),
    }

def train_and_eval(models, x_train, y_train, x_test, y_test):
    """
    Trains and evaluates GaussianNB, KNN, and SVM on a given dataset.
    params:
        x_train: training features (numpy array)
        y_train: training labels (numpy array)   
        x_test: testing features (numpy array)
        y_test: testing labels (numpy array)
    return:
        results: dictionary of model accuracies
        fitted: dictionary of fitted models
    """
    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted[name] = model
        results[name] = {"train": accuracy_score(y_train, model.predict(x_train)),
                         "test":  accuracy_score(y_test,  model.predict(x_test))}
    return results, fitted

def plot_confusion_matrices(fitted, x_test, y_test, title):
    """
    Plots confusion matrices for all models on one dataset.
    params:
        fitted: dictionary of fitted models
        x_test: testing features (numpy array)
        y_test: testing labels (numpy array)
        title: title for the plot (string)
    """
    fig, axes = plt.subplots(1, len(fitted), figsize=(4 * len(fitted), 4),
                             constrained_layout=True)
    for ax, (name, model) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\ntest acc = {model.score(x_test, y_test):.4f}")
    fig.suptitle(title)
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
    x2_train, y2_train = x2[train2], y2[train2]
    x2_test,  y2_test  = x2[test2],  y2[test2]
    
    # print dataset table
    table = pd.DataFrame({"x1": x2[:, 0], "x2": x2[:, 1], "class": y2,
                       "outlier": np.where(outlier_mask, "yes", "")})
    table["set"] = ""
    table.loc[train2, "set"] = "train"
    table.loc[test2, "set"] = "test"
    print(table.round(4).to_string())                      
    print(table.to_latex(index=False, float_format="%.4f", longtable=True,
                        caption="All data points (rows without an outlier flag form Dataset 1).",
                        label="tab:data"))
    
    # plot the split datasets
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
    plot_split(axes[0], x1, y1, train1, test1, "Dataset 1")
    plot_split(axes[1], x2, y2, train2, test2, "Dataset 2")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    plt.show()

    # train and evaluate models on both datasets
    res1, fitted1 = train_and_eval(get_models(), x1_train, y1_train, x1_test, y1_test)
    res2, fitted2 = train_and_eval(get_models(), x2_train, y2_train, x2_test, y2_test)
    print(pd.DataFrame(res1).T.round(4).to_latex())
    print(pd.DataFrame(res2).T.round(4).to_latex())    
        
    plot_confusion_matrices(fitted1, x1_test, y1_test, "Dataset 1 - Confusion Matrices")
    plot_confusion_matrices(fitted2, x2_test, y2_test, "Dataset 2 - Confusion Matrices")
    #answers1, gnb1, knn1, svmresults1 = train_and_eval_models(x1_train, y1_train, x1_test, y1_test, "Dataset 1")
    #answers2, gnb2, knn2, svmresults2 = train_and_eval_models(x2_train, y2_train, x2_test, y2_test, "Dataset 2")

    #plot_confusion_matrices(x1_test, y1_test, f"Dataset 1 - Confusion Matrix", answers1, gnb1, knn1, svmresults1)
    #plot_confusion_matrices(x2_test, y2_test, f"Dataset 2 - Confusion Matrix", answers2, gnb2, knn2, svmresults2)

    #for model_name, model in [("GaussianNB", gnb1), ("KNN", knn1)] + [(f"SVM_{kernel}", svmresults1[kernel]) for kernel in KERNELS]:
    #    plot_decision_boundary(model, x1_train, y1_train, x1_test, y1_test, f"Dataset 1 - {model_name} Decision Boundary")
    #for model_name, model in [("GaussianNB", gnb2), ("KNN", knn2)] + [(f"SVM_{kernel}", svmresults2[kernel]) for kernel in KERNELS]:
    #    plot_decision_boundary(model, x2_train, y2_train, x2_test, y2_test, f"Dataset 2 - {model_name} Decision Boundary")
        
    # B.II. 
    #plot_knn_decision_boundaries(x1_train, y1_train, K_LIST, dataset="1")
    #plot_knn_decision_boundaries(x2_train, y2_train, K_LIST, dataset="2")
