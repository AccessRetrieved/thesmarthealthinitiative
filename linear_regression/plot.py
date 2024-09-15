import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
import joblib
sys.path.append('../thesmarthealthinitiative')
from data import *
model = joblib.load('life_expectancy_model.pkl')

data = load_file('neighborhood_health_and_wellness.csv')

def scatter_plot(data, x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', alpha=0.6)
    plt.title('Average Income vs Life Expectancy')
    plt.xlabel('Average Income')
    plt.ylabel('Life Expectancy')
    plt.grid(True)
    plt.show()

def box_plot(data, x, y):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot(data['Average Income'])
    plt.title('Box Plot of Average Income')
    plt.ylabel('Average Income')
    plt.boxplot(data['Life Expectancy'])
    plt.tight_layout()
    plt.show()

scatter_plot(data, data['Average Income'], data['Life Expectancy'])
