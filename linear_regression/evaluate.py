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
features = ['Average_Income', 'Obesity_Rate', 'Smoking_Rate', 'Access_to_Gym', 'Number_of_Hospitals']
X = data[features]
y = data['Life_Expectancy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)
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
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs. Actual')
plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r--', linewidth=1)
plt.title('Actual vs. Predicted Life Expectancy')
plt.xlabel('Average Income')
plt.ylabel('Life Expectancy')
plt.legend()
plt.show()