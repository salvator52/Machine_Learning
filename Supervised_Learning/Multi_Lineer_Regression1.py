import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Veri oluşturma
X = np.random.rand(100, 2)  # 100x2 boyutunda rastgele veri
coef = np.array([3, 5])
y = 0 + np.dot(X, coef)  # Doğrusal regresyon denklemi: y = a0 + a1 * x1 + a2 * x2

# Model oluşturma ve eğitme
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Meshgrid oluşturma
x1, x2 = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))  # 50x50'lik meshgrid

# Tahmin yapmak için veriyi düzleştirme ve tahmin
X_grid = np.c_[x1.ravel(), x2.ravel()]  # Meshgrid'i düzleştir
y_pred = lin_reg.predict(X_grid).reshape(x1.shape)  # Tahminleri yeniden şekillendir

# 3D grafik çizme
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')  # Veri noktalarını çizme
ax.plot_surface(x1, x2, y_pred, color='b', alpha=0.5)  # Tahmin yüzeyi çizme

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.title('3D Linear Regression Surface')
plt.show()

#katsayılara bakmak içinse
print("katsayılar :",lin_reg.coef_)
#sabitin değerine bakmak içinse
print("intercept(sabit) :",lin_reg.intercept_)


#gürültü artarsa katsayılar ve sabit ler değişir


