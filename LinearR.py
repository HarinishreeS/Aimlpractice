from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
X=[[1],[5],[3],[9],[4]]
Y=[2,10,6,18,8]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
model=LinearRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print("Predictions:",y_pred)
print("Mean squared error:",mean_squared_error(y_test,y_pred)
print("r2 score:",r2_score(y_test,y_pred)


