import pandas as pd

results = {
    "Model": ["Simple RNN", "LSTM", "GRU"],
    "Accuracy": [0.7291, 0.7529, 0.7668],
    "Macro F1-Score": [0.6008, 0.6638, 0.6743]
}

comparison_df = pd.DataFrame(results)

print("\nModel Performance Comparison:")
print(comparison_df)

best_model = comparison_df.loc[
    comparison_df["Macro F1-Score"].idxmax()
]

print("\nBest Model:")
print(best_model)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.bar(
    comparison_df["Model"],
    comparison_df["Accuracy"]
)

plt.title("Accuracy Comparison of RNN, LSTM and GRU")
plt.xlabel("Deep Learning Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("artifacts/model_accuracy_comparison.png")
plt.show()

plt.figure(figsize=(8, 5))

plt.bar(
    comparison_df["Model"],
    comparison_df["Macro F1-Score"]
)

plt.title("Macro F1-Score Comparison of RNN, LSTM and GRU")
plt.xlabel("Deep Learning Models")
plt.ylabel("Macro F1-Score")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("artifacts/model_f1_comparison.png")
plt.show()