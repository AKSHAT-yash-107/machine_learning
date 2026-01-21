import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================
# 1. LOAD DATASET
# =========================
# dataset.csv must be in the SAME folder as knn.py
df = pd.read_csv("dataset.csv")

# Assume last column is the target
TARGET_COLUMN = df.columns[-1]

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Encode labels if target is categorical
if y.dtype == "object":
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

# =========================
# 2. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 4. TRAIN KNN FOR MULTIPLE K
# =========================
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# =========================
# 5. SAVE RESULTS TO FILE
# =========================
best_k = k_values[accuracies.index(max(accuracies))]
best_accuracy = max(accuracies)

with open("results.txt", "w") as f:
    for k, acc in zip(k_values, accuracies):
        f.write(f"K={k} -> Accuracy: {acc:.4f}\n")
    f.write("\n")
    f.write(f"Best K: {best_k}\n")
    f.write(f"Best Accuracy: {best_accuracy:.4f}\n")

# =========================
# 6. PLOT ACCURACY VS K
# =========================
plt.figure()
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.grid(True)
plt.savefig("accuracy_vs_k.png", dpi=300)
plt.close()

# =========================
# 7. FINAL OUTPUT
# =========================
print("KNN analysis completed successfully.")
print(f"Best K: {best_k}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print("Saved files:")
print("- accuracy_vs_k.png")
print("- results.txt")
