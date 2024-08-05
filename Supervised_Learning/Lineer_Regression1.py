import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



#veri oluştur
X = np.random.rand(100,1) #100 satır tek sütundan oluşan bir nd array
y = 3 + 4 * X +np.random.rand(100,1) #y=3+4x olsun





lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.figure()
plt.scatter(X,y)
plt.plot(X,lin_reg.predict(X),color="red")
plt.xlabel("x değeri")
plt.ylabel("y değeri")
plt.title("lineer regresyon")

a1 = lin_reg.coef_[0][0] #x'in katsayısı
a0 = lin_reg.intercept_[0] #sabitin katsayısını alırız

for i in range(100):
    y_ =a0 + a1 * X
    plt.plot(X,y_,color="orange")