import joblib
import sys
import pandas as pd
import numpy as np
sys.path.append('../thesmarthealthinitiative')
from data import *

model = joblib.load('life_expectancy_model.pkl')

features = ['Average_Income', 'Obesity_Rate', 'Smoking_Rate', 'Access_to_Gym', 'Number_of_Hospitals']
input_data = {}

for feature in features:
    while True:
        try:
            value = float(input(f"Enter the {feature.replace('_', ' ')}: "))
            input_data[feature] = value
            break
        except ValueError:
            print(f"Invalid input for {feature}. Please enter a numeric value.")
    
X_new = [input_data[feature] for feature in features]
X_new = [X_new]

predicted_life_expectancy = model.predict(X_new)[0]

print(f'Predicted life exepctancy: {predicted_life_expectancy:.2f} year')