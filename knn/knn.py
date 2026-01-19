import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv('Iris.csv')

# 2. Preprocess the data
# The 'Id' column is just an index and doesn't help with classification
data = df.drop('Id', axis=1)

# Split features (X) and target variable (y)
X = data.drop('Species', axis=1)
y = data['Species']

# Encode the target labels (Species names) into numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Split the data into Training and Testing sets
# We use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Feature Scaling
# KNN relies on distance, so it's important to scale features to the same range
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train the KNN Classifier
# We'll use K=5 as a starting point
k = 100
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# 6. Make Predictions and Calculate Accuracy
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy with K={k}: {accuracy * 100:.2f}%")