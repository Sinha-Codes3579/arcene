import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 6.1 Plot Accuracy vs Iteration 
def plot_accuracy_over_time(accuracy_hs, accuracy_pso):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_hs, label='Harmony Search', marker='o')
    plt.plot(accuracy_pso, label='Particle Swarm Optimization', marker='s')
    plt.title(" Accuracy Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot Features vs Iteration 
def plot_features_selected(feature_counts_hs, feature_counts_pso):
    plt.figure(figsize=(10, 5))
    plt.plot(feature_counts_hs, label='HS: Features', marker='o')
    plt.plot(feature_counts_pso, label='PSO: Features', marker='s')
    plt.title(" Features Selected Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Selected Features")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Confusion Matrix 
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(" Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Accuracy vs Feature Reduction 
def plot_accuracy_vs_reduction(acc_list, red_list, labels):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=acc_list)
    for i, red in enumerate(red_list):
        plt.text(i, acc_list[i] + 0.01, f"↓{red:.1f}% fewer features", ha='center', fontsize=10)
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.title(" Accuracy vs Feature Reduction %")
    plt.tight_layout()
    plt.show()
