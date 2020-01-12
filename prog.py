from sklearn.tree import DecisionTreeClassifier
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('ds/car_evaluation.csv')



enc=OneHotEncoder()

X = pd.get_dummies(df)
print(X)