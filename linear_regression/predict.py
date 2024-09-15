import joblib
import sys
import pandas as pd
import numpy as np
sys.path.append('../thesmarthealthinitiative')
from data import *

model = joblib.load('life_expectancy_model.pkl')

try:
    average_income_input = float(input('Enter average income: '))
except ValueError:
    print('Invalid input.')
    sys.exit(0)
    
X_new = [[average_income_input]]

predicted_life_expectancy = model.predict(X_new)[0]

print(f'Predicted life exepctancy: {predicted_life_expectancy:.2f} year')