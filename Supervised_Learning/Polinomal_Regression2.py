import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
X = 4 * np.random.rand(100,1)
#y = 2 + 3* X**2   #y = 2 +3*x^2
y = 2 + 3* X**2 +np.random.rand(100,1) #gürültü ekledik
#plt.scatter(X,y)

poly_feat = PolynomialFeatures(degree = 2)
X_poly = poly_feat.fit_transform(X) # burada pythonun polinom fonksiyonundan yardım aldık 


lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Test verileri oluşturma ve tahmin yapma
X_test = np.linspace(0, 4, 100).reshape(-1, 1)
X_test_poly = poly_feat.transform(X_test)
y_pred = lin_reg.predict(X_test_poly)

plt.scatter(X,y,color="blue")
plt.plot(X_test,y_pred,color="red")

