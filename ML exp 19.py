import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample bank loan dataset
data = {
    'Income':[50,60,30,80,90],
    'CreditScore':[700,750,650,800,820],
    'LoanApproved':[1,1,0,1,1]
}

df = pd.DataFrame(data)
X = df[['Income','CreditScore']]
y = df['LoanApproved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
