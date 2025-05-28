from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score      #Linear Regression 

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline                    #Polynomial Regression

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
X=[[1],[5],[3],[9],[4]]
Y=[2,10,6,18,8]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
model=LinearRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
print("Predictions:",Y_pred)
print("Mean squared error:",mean_squared_error(Y_test,Y_pred)
print("r2 score:",r2_score(Y_test,Y_pred)

poly_model=make_pipeline(PolynomialFeatures(2),LinearRegression())
poly_model.fit(X_train,Y_train)
Y_pred=poly_model.predict(X_test)
print("Predictions:",Y_pred)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
print("Ridge Prediction:", ridge_model.predict(X_test))

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
print("Lasso Prediction:", lasso_model.predict(X_test))

