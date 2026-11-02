from exB import generate_dataset1, split_indices,kfold_indices,train_and_eval, TRAIN_TEST_SPLIT
from sklearn import svm
from sklearn.metrics import accuracy_score as accuracy

x1,y1 = generate_dataset1()

x1_train, x1_test, y1_train, y1_test = split_indices(y1, TRAIN_TEST_SPLIT)
folded_idx = kfold_indices(y1_train, 5)

Cs = [0.01,1,100]
kernel = 'rbf'

def train_svm(x_train, y_train, kernel, Cs):
    svmresults = {}
    for c in Cs:
        model = svm.SVC(kernel=kernel, C=c)
        model.fit(x_train, y_train)
        svmresults[c] = model
    return svmresults

def evaluate_svm(svmresults, x_test, y_test):
    accuracies = {}
    for c, model in svmresults.items():
        predictions = model.predict(x_test)
        accuracies[c] = accuracy(y_test, predictions)
    return accuracies

svmresults = train_svm(x1_train, y1_train, kernel, Cs)
accuracies = evaluate_svm(svmresults, x1_test, y1_test)

for model, acc in accuracies.items():
    print(f"Dataset 1 - {model} Accuracy: {acc:.4f}")
    
def train_svm_kfold(x_train, y_train, kernel, Cs, folded_idx):
    kfold_results = {}
    for c in Cs:
        fold_accuracies = []
        for train_idx, val_idx in folded_idx:
            x_fold_train, y_fold_train = x_train[train_idx], y_train[train_idx]
            x_fold_val, y_fold_val = x_train[val_idx], y_train[val_idx]
            svmresults = train_svm(x_fold_train, y_fold_train, kernel, [c])
            accuracies = evaluate_svm(svmresults, x_fold_val, y_fold_val)
            fold_accuracies.append(accuracies[c])
        kfold_results[c] = sum(fold_accuracies) / len(fold_accuracies)
    return kfold_results

kfold_accuracies = train_svm_kfold(x1_train, y1_train, kernel, Cs, folded_idx)

for model, acc in kfold_accuracies.items():
    print(f"Dataset 1 - {model} K-Fold Accuracy: {acc:.4f}")