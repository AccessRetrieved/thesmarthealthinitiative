import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_model_data():
    model = joblib.load('health_model.pkl')
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')
    
    return model, X_test, y_test

def evaluate():
    model, X_test, y_test = load_model_data()
    
    predictions = model.predict(X_test)
    
    # evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    
    print(f'Accuracy: {accuracy}')
    
    # plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
evaluate()