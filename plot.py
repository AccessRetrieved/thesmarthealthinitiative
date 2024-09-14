import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    features = ['Population', 'Median_Age', 'Average_Income', 'Obesity_Rate', 'Smoking_Rate']
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    
    data['High_Obesity_Rate'] = (data['Obesity_Rate'] > 0.25).astype(int)
    
    return data

def plot_correlation_matrix(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    
def plot_distribution(data, feature):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()
    
def plot_scatter(data, x, y):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x=x, y=y)
    plt.title(f'{x} vs. {y}')
    plt.show()
    
data = pd.read_csv('neighborhood_health_and_wellness.csv')
plot_correlation_matrix(data)
plot_distribution(data, 'Obesity_Rate')
plot_scatter(data, 'Average_Income', 'Obesity_Rate')