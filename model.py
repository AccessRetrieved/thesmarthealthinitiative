from data import *

data = load_file('neighborhood_health_and_wellness.csv')
X, y = preprocess_data(data)

train_model(X, y)