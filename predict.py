import numpy as np
import pandas as pd
import joblib
from model import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score


features = ['Population', 'Median_Age', 'Average_Income', 'Number_of_Hospitals', 'Obesity_Rate', 
            'Access_to_Gym', 'Smoking_Rate', 'Life_Expectancy']
target = 'Health_Score'
X = data[features]
y = data[target]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifer = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifer.fit(X_train, y_train)

# evaluate data
y_pred = rf_classifer.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')

# classification report
print(classification_report(y_test, y_pred))
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Precision: {precision}')
print(f'F1: {f1}')