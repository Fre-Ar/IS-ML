from exB import generate_dataset1, split_indices, kfold_indices, train_and_eval 
from sklearn import svm
import pandas as pd
import numpy as np

Cs = [0.01,1,100]
kernel = 'rbf'
def get_models():
    return {
        f"SVM RBF (C={c})": svm.SVC(kernel=kernel, C=c) for c in Cs
    }

x,y = generate_dataset1()

train1, test1 = split_indices(y)
x_train, y_train = x[train1], y[train1]
x_test,  y_test  = x[test1],  y[test1]


res1, fitted1 = train_and_eval(get_models(), x_train, y_train, x_test, y_test)
print(pd.DataFrame(res1).T.round(4).to_latex())

models = get_models()
accuracies = {name: [] for name in models.keys()}

for train_idx, val_idx in kfold_indices(y_train):
    x_fold_train, y_fold_train = x_train[train_idx], y_train[train_idx]
    x_fold_val, y_fold_val = x_train[val_idx], y_train[val_idx]
    res, fitted = train_and_eval(models, x_fold_train, y_fold_train, x_fold_val, y_fold_val)
    for name, acc in res.items():
       accuracies[name].append(acc['test'])

acc_stats = {
    name: {"Mean": np.mean(acc_list), "Std": np.std(acc_list)} 
    for name, acc_list in accuracies.items()
}

print("K-Fold Accuracies:")
for name, stats in acc_stats.items():
    print(f"{name}: {stats['Mean']:.4f} ± {stats['Std']:.4f}")
    

df_acc = pd.DataFrame(acc_stats).T.round(4)
print(df_acc.to_latex())