from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X=[
  [163,45,4]
  [172,60,5]
  [169,40,4]
  [165,52,5]
  [176,63,6]
]
Y=["Female","Male","Female","Female","Male"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
model=GaussianNB()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print("accuracy score:",accuracy_score(Y_test,y_pred))
print("predictions:",y_pred)
