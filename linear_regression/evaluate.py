import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import sys
sys.path.append('../thesmarthealthinitiative')
from data import *
import numpy as np

data = load_file('neighborhood_health_and_wellness.csv')
X = data[['Average_Income']]
y = data['Life_Expectancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = joblib.load('life_expectancy_model.pkl')

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r_squared = r2_score(y_test, y_pred)

epsilon = 1e-10
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100
accuracy_perc = 100 - mape

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R-squared: {r_squared:.2f}')
print(f'Accuracy: {accuracy_perc:.2f}%')

# output graph
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Life Expectancy')
plt.scatter(X_test, y_pred, color='red', label='Predicted Life Expectancy')
plt.plot(X_test, y_pred, color='red', linewidth=1)
plt.title('Actual vs. Predicted Life Expectancy')
plt.xlabel('Average Income')
plt.ylabel('Life Expectancy')
plt.legend()
plt.show()