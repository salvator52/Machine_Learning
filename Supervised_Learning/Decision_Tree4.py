import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

X =np.sort( 5*np.random.rand(80,1),axis = 0)
y = np.sin(X).ravel() #array'imizi vektör haline getircek

y[::5] += 0.5 * (0.5 -np.random.rand(16))

#plt.scatter(X,y)


regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)

regr_1.fit(X, y)
regr_2.fit(X, y)


X_test = np.arange(0,5,0.05)[:,np.newaxis] #bu kısımda artık nd array'imiz 2 boyutlu olcak
y_pred1 = regr_1.predict(X_test)
y_pred2 = regr_2.predict(X_test)


plt.figure()
plt.scatter(X,y, c = "red" ,label ="data")
plt.plot(X_test,y_pred1,color="blue",label = "Max Depth :2",linewidth = 2)
plt.plot(X_test,y_pred2,color="green",label = "Max Depth :5",linewidth = 2)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()






