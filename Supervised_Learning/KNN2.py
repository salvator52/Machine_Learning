import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

#buradaki axis=0 değeri sütun değerlerini küçükten büyüğe sıralamak için kullanılır
#burası ayrıca feature kısmımız
X = np.sort(5 * np.random.rand(40,1),axis = 0)#uniform türünde 40*1 lik bir vektör oluşturduk 0-5 arası
y = np.sin(X).ravel() #target   düzleştirme işlemi yapar ravel()

#problemi zorlaştırmak için birazcık gürültü ekleyelim
#add noise

y[::5] +=1 *(0.5 - np.random.rand(8))


#test veri seti oluşturma
T = np.linspace(0,5,500)[:,np.newaxis] #diziyi bir sütun vektörüne dönüştürür linsapce ise 0 ile 500 arasında hepsi eşit uzaklıkta 500 değer üretir



weight = "uniform"
knn = KNeighborsRegressor(n_neighbors= 5 ,weights=weight)
knn.fit(X, y).predict(T)


for i, weight in enumerate(["uniform", "distance"]):
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X, y).predict(T)
    
    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color="green", label="data")
    plt.plot(T, y_pred, color="blue", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weights = {}".format(weight))

plt.tight_layout()
plt.show()

#plt.plot(X,y)
#scatter bize nokta nokta şeklinde aralığı boşluk olacak şekilde verir
#plt.scatter(X,y)

"""
uniform değeri her bir veri noktasının değerini eşit sayar ve işlemleri ona göre yapar : homojen dağılmış veriler için uygundur
distance değeri ise noktalar arası uzaklık istediğimizden fazlaysa kullanırıız
"""








