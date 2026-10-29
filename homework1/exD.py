from exB import generate_dataset1, split_dataset, COV, TRAIN_TEST_SPLIT, accuracy
from sklearn import svm

print("Generating Dataset 1...")
x1,y1 = generate_dataset1(COV)
print("Dataset 1 generated successfully.")
print(f"Features shape: {x1.shape}, Labels shape: {y1.shape}")
print(f"Features:\n{x1}\nLabels:\n{y1}")

x1_train, x1_test, y1_train, y1_test = split_dataset(x1,y1,TRAIN_TEST_SPLIT)

Cs = [0.01,1,100]
kernel = 'poly'
svmresults = {}
for c in Cs:
    svm1 = svm.SVC(kernel=kernel, C=c)
    svm1.fit(x1_train, y1_train)
    svmresults[c] = svm1
    
answers = {
    f"SVM_C={str(c)}": accuracy(y1_test, svmresults[c].predict(x1_test)) for c in Cs
}
for model, acc in answers.items():
    print(f"Dataset 1 - {model} Accuracy: {acc:.4f}")

import pandas as pd


results_df = pd.DataFrame([
    {
        "Model": "SVM",
        "Kernel": kernel,
        "C Parameter": c,
        "Accuracy": f"{acc:.4f}"
    }
    for c, acc in zip(Cs, answers.values())
])

print("\n=== Model Performance Table ===")
print(results_df.to_string(index=False))