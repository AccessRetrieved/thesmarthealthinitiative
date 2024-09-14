import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('neighborhood_health_and_wellness.csv')

# output data stats
print('Data stats:', end='\n')
print(data.head())
print(data.describe())
print(data.isnull().sum(), end='\n\n\n')

corr_matrix = data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
sns.pairplot(data)
plt.show()

# create mlr variables
predictors = ['Average_Income', 'Obesity_Rate', 'Smoking_Rate', 'Access_to_Gym', 'Number_of_Hospitals', 'Median_Age']
X_mlr = data[predictors]
y_mlr = data['Life_Expectancy']

# check for multicollinearity
X_mlr_with_const = sm.add_constant(X_mlr)
vif_data = pd.DataFrame()
vif_data['features'] = X_mlr_with_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_mlr_with_const.values, i) for i in range(X_mlr_with_const.shape[1])]
print(vif_data)

model_mlr = sm.OLS(y_mlr, X_mlr_with_const).fit()
print(model_mlr.summary())