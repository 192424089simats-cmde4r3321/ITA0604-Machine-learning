import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample mobile dataset
data = {
    'RAM':[4, 6, 8, 12],
    'Storage':[64, 128, 128, 256],
    'Price':[15000, 20000, 25000, 35000]
}

df = pd.DataFrame(data)
X = df[['RAM','Storage']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

print("Predicted Price:", model.predict([[8, 128]]))
