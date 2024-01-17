import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Example model
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing(as_frame=True)
x = housing.data
y = housing.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()  # Example, you can choose other models as needed
model.fit(x_train, y_train)
predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
