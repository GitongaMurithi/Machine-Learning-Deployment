import pickle

import pandas as pd

data = pd.read_csv("EPL1.csv")
print(data.head())
clas = pd.get_dummies(data["hclass"])
data = pd.concat([data, clas], axis=1)
print(data.head())
X = data.iloc[:, [0, 1, 5, 6, 7]].values
y = data.iloc[:, 4].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, prediction) * 100)
pickle.dump(lr, open('model.pkl', 'wb'))
