import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample credit dataset
data = {
    'Income':[40,60,80,30,90],
    'Debt':[20,10,30,40,10],
    'CreditScore':[0,1,1,0,1]
}

df = pd.DataFrame(data)
X = df[['Income','Debt']]
y = df['CreditScore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
