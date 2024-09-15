import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
import joblib
sys.path.append('../thesmarthealthinitiative')
model = joblib.load('life_expectancy_model.pkl')

def scatter_plot(data, x, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', alpha=0.6)
    plt.title('Average Income vs Life Expectancy')
    plt.xlabel('Average Income')
    plt.ylabel('Life Expectancy')
    plt.grid(1)
    plt.show()


