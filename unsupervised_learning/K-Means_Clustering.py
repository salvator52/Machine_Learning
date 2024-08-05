from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#kaç tane değer istiyoz ,kaç tane kümeye ayrılcak standart sapma yani gürültü ne olacak onu belirliyoruz
#standart sapma ne kadar yüksekse o kadar ayrık bir dağılım vardır
X , _ = make_blobs(n_samples=300, centers=4 ,cluster_std=0.6,random_state=42) #bize 2 tane değer return eder ikinic değeri kullanmayız


plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Original Data")
plt.show()
#kaç tane küme olacağını seç
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

labels = kmeans.labels_ #küme etiketlerine bakabiliriz

plt.figure()
#kümelerin farklı renkte kullanılması için cmap ="viridis" kullandık
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.title("KMeans Clustering")
plt.show()

#her bir kümenin merkezini görebiliriz
centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1],c="red",marker="X")


