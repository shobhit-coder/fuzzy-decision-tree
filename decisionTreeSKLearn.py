from sklearn.tree import DecisionTreeClassifier
import pandas as pd 
from sklearn.metrics import accuracy_score
import numpy as np  

attributes=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


df = pd.read_csv('ds/car_evaluation.csv')
X = pd.get_dummies(df[attributes[0:6]])
Y = pd.get_dummies(df[attributes[6]])
print('one hot encoding done')
msk = np.random.rand(len(df)) < 0.8 #80 percent split
print(msk)
X_train = X[msk]
Y_train = Y[msk]
Y_test = Y[~msk]
X_test = X[~msk]
print('split done')




clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_train)

print("train accuracy:")
print(accuracy_score(Y_train, Y_pred))

Y_pred = clf.predict(X_test)

print("test accuracy:")
print(accuracy_score(Y_test, Y_pred))