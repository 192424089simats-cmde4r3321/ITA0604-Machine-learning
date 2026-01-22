from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

# Prediction
y_pred = lr.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
