import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv("Iris.csv")

# 2. Preprocess the data
# Drop unnecessary Id column
data = df.drop("Id", axis=1)

# Features and target
X = data.drop("Species", axis=1)
y = data["Species"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 4. Feature scaling (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train KNN for multiple K values
k_values = range(1, 41)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# 6. Plot Accuracy vs K
plt.figure()
plt.plot(k_values, accuracies)
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K (Iris Dataset)")
plt.show()

# 7. Train final model with best K
best_k = k_values[accuracies.index(max(accuracies))]
best_accuracy = max(accuracies)

print(f"Best K: {best_k}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
