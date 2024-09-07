import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_file(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    features = ['Population', 'Median_Age', 'Average_Income', 'Obseity_Rate', 'Smoking_Rate']
    
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    
    X = data[features]
    y = data['High_Obesity_Rate']
    
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'health_model.pkl')
    joblib.dump(X_test, 'X_test.pkl')
    joblib.dump(y_test, 'y_test.pkl')
    
    print('Models trained and saved.')