from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Veri oluşturma: 4 merkezde, 300 örnekli ve standart sapması 0.6 olan bir veri kümesi oluşturuyoruz
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Oluşturulan veriyi görselleştirme
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Örnek Veri")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Kullanılacak bağlama (linkage) yöntemleri
linkage_methods = ["ward", "single", "average", "complete"]

# Çizim alanını oluşturma: 2 satır, 4 sütun şeklinde subplotlar için alan oluşturuyoruz
plt.figure(figsize=(20, 10))

# Her bir bağlama yöntemi için kümeleme ve dendrogram çizimi
for i, linkage_method in enumerate(linkage_methods, 1):
    # AgglomerativeClustering modelini oluşturma ve veriye uyarlama
    model = AgglomerativeClustering(n_clusters=4, linkage=linkage_method)
    cluster_labels = model.fit_predict(X)

    # Dendrogram çizimi için subplot oluşturma
    plt.subplot(2, 4, i)
    
    # Veri noktalarının uzaklık matrisini oluşturma
    linked = linkage(X, method=linkage_method)
    
    # Dendrogramı çizme
    dendrogram(linked, no_labels=True)
    plt.title(f"{linkage_method.capitalize()} Linkage Dendrogram")
    plt.xlabel("Veri Noktaları")
    plt.ylabel("Uzaklık")

    # Kümeleme sonuçlarını scatter plot ile gösterme için subplot oluşturma
    plt.subplot(2, 4, i + 4)
    
    # Veri noktalarını küme etiketlerine göre scatter plot ile gösterme
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis")
    plt.title(f"{linkage_method.capitalize()} Linkage Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")

# Tüm subplotların düzenli bir şekilde yerleşmesini sağlama
plt.tight_layout()

# Çizimleri gösterme
plt.show()







