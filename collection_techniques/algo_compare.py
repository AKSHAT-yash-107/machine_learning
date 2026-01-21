import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Load & clean dataset
clean_lines = []
with open("pokemon_data.csv", "r") as f:
    for line in f:
        if not line.startswith(("<<<", "===", ">>>")) and line.strip():
            clean_lines.append(line)

df = pd.read_csv(StringIO("".join(clean_lines)))
df = df[df["name"] != "name"]

for col in ["height", "weight", "base_experience"]:
    df[col] = pd.to_numeric(df[col])

# Binary target
df["target"] = (df["base_experience"] >= df["base_experience"].median()).astype(int)

X = df[["height", "weight"]]
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = []

# Multiple Linear Regression
lr = LinearRegression()
lr.fit(X_train_s, y_train)
results.append(("Multiple Linear Regression",
                accuracy_score(y_test, lr.predict(X_test_s).round())))

# Logistic Regression
logr = LogisticRegression()
logr.fit(X_train_s, y_train)
results.append(("Logistic Regression",
                accuracy_score(y_test, logr.predict(X_test_s))))

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_s, y_train)
results.append(("KNN",
                accuracy_score(y_test, knn.predict(X_test_s))))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_s, y_train)
results.append(("Naive Bayes",
                accuracy_score(y_test, nb.predict(X_test_s))))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
results.append(("Decision Tree",
                accuracy_score(y_test, dt.predict(X_test))))

print(pd.DataFrame(results, columns=["Algorithm", "Accuracy"]))
