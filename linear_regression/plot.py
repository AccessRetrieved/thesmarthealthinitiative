import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
import joblib
import statsmodels.api as sm
sys.path.append('../thesmarthealthinitiative')
from data import *
model = joblib.load('life_expectancy_model.pkl')


data = load_file('neighborhood_health_and_wellness.csv')

def scatter_plot(x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', alpha=0.6)
    plt.title('Average Income vs Life Expectancy')
    plt.xlabel('Average Income')
    plt.ylabel('Life Expectancy')
    plt.grid(True)

def box_plot(data, feature):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot(data[feature])
    plt.title(f'Box Plot of {feature}')
    plt.ylabel(feature)

features = ['Average_Income', 'Obesity_Rate', 'Smoking_Rate', 'Access_to_Gym', 'Number_of_Hospitals']
X = data[features]
y = data['Life_Expectancy']
predictions = model.predict(X)

plt.scatter(y, predictions)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)

box_plot(data, 'Average_Income')
box_plot(data, 'Life_Expectancy')
plt.show()
print(data.describe())

