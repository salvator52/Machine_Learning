from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

california_housing = fetch_california_housing()

X  = california_housing.data
y = california_housing.target


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf_reg = RandomForestRegressor(random_state=42)

rf_reg.fit(X_train, y_train)


y_pred = rf_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(mse) #mean squared error
rsme = np.sqrt(mse) #root mean squared error
print(rsme)











