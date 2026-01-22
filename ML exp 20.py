import numpy as np
from sklearn.linear_model import LinearRegression

# Sales data (Year vs Sales)
X = np.array([[2018],[2019],[2020],[2021],[2022]])
y = np.array([200, 220, 250, 280, 300])

model = LinearRegression()
model.fit(X, y)

future_sales = model.predict([[2023]])
print("Predicted Sales for 2023:", future_sales)
