import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import sys

sys.path.append('../thesmarthealthinitiative')
from data import *

data = load_file('neighborhood_health_and_wellness.csv')
X = data[['Average_Income']]
y = data['Life_Expectancy']

model = LinearRegression()
model.fit(X, y)
joblib.dump(model, './life_expectancy_model.pkl')

print('Trained life exppectancy model.')