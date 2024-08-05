from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
diabets = load_diabetes()

X = diabets.data
y = diabets.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)


rmse = mean_squared_error(y_test,y_pred,squared =False) #mean square fonksiyonuna bu parametreye false verirrsek root mena square error'u bulabiliriz
print("rmse :",rmse)












