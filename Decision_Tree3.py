from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#karar ağacı regresyon modeli

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train,y_train)


y_pred = tree_reg.predict(X_test)
#Bir modelin tahmin ettiği değerler ile gerçek değerler arasındaki farkın karesinin ortalamasını hesaplar.
mse = mean_squared_error(y_test, y_pred)
print(mse)
rmse = np.sqrt(mse) #kare alır
print(rmse) #70 puanlık hataya kadar geldik



















