"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

#bu sefer feature'larımızı paid'ler hallinde kullanarak sonuçlara ulaşacağız

#(1) veri setini ekle
iris = load_iris()

for pairid,pair in enumerate([0,1],[0,2],[0,3],[1,2],[1,3],[2,3]):
    X = iris.data[:,pair]
    y = iris.target
    
    clf = DecisionTreeClassifier().fit(X,y)
    
    ax =plt.subplot(2,3,pairidx + 1) #burada ki pair id 0 olamaz o yüzden
    plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=2.5)
    
plt.legend()   

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Iris veri setini yükleyin
iris = load_iris()
X = iris.data
y = iris.target

# Özelliklerin tüm ikili kombinasyonlarını belirleyin
feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Grafikler için bir figür oluşturun
plt.figure(figsize=(15, 10))

for pairidx, pair in enumerate(feature_pairs):
    X_pair = X[:, pair]  # İkili özellik kombinasyonunu seçin
    
    # Karar ağacı sınıflandırıcısı oluşturup eğitin
    clf = DecisionTreeClassifier().fit(X_pair, y)
    
    # Alt grafik oluşturun
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    
    # Karar sınırlarını çizmek için meshgrid oluşturun
    x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
    y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Karar sınırlarını plot edin
    ax.contourf(xx, yy, Z, alpha=0.3)
    
    # Veri noktalarını plot edin
    for i, color in zip(range(3), "ryb"):
        idx = np.where(y == i)
        ax.scatter(X_pair[idx, 0], X_pair[idx, 1], c=color, label=iris.target_names[i],
                   edgecolor='black', s=20)
    
    ax.set_xlabel(iris.feature_names[pair[0]])
    ax.set_ylabel(iris.feature_names[pair[1]])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(f"Pair: {iris.feature_names[pair[0]]} & {iris.feature_names[pair[1]]}")

plt.legend()
plt.show()
