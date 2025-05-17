from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def final_evaluation(gbest, X_train, y_train, X_test, y_test):
    # Train final classifier using gbest features and evaluate on test set.
    selected_features = np.where(gbest == 1)[0]
    
    if len(selected_features) == 0:
        print(" No features selected. Evaluation skipped.")
        return

    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]

    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    reduction_rate = 1 - (len(selected_features) / X_train.shape[1])

    print("\n Final Model Evaluation on Test Set:")
    print(f" Accuracy        : {acc:.4f}")
    print(f" Precision       : {prec:.4f}")
    print(f" Recall          : {rec:.4f}")
    print(f" F1 Score        : {f1:.4f}")
    print(f" Feature Reduction Rate: {reduction_rate*100:.2f}%")
