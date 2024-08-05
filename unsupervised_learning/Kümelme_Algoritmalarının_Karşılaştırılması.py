from sklearn import datasets, cluster
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Numune sayısı
n_samples = 1500

# Farklı veri kümeleri oluşturma
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples)
no_structure = (np.random.rand(n_samples, 2), None)  # Diğerleri gibi iki değer döndürür

# Kullanılacak kümeleme algoritmaları isimleri
clustering_names = ["MiniBatchKMeans", "SpectralClustering", "Ward", "AgglomerativeClustering", "DBSCAN", "Birch"]

# Renkler
colors = np.array(["b", "g", "r", "c", "m", "y"])

# Veri kümeleri listesi
datasets_list = [noisy_circles, noisy_moons, blobs, no_structure]

# Her bir veri kümesi için döngü
for i_dataset, dataset in enumerate(datasets_list):
    X, y = dataset  # Veriyi ve etiketleri ayırma (etiketler bazı veri kümelerinde None olabilir)
    
    # Veriyi standardize etme (ölçekleme)
    X = StandardScaler().fit_transform(X)
    
    # Kümeleme algoritmalarını tanımlama
    two_means = cluster.MiniBatchKMeans(n_clusters=2)
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage="ward")
    spectral = cluster.SpectralClustering(n_clusters=2)
    dbscan = cluster.DBSCAN(eps=0.2)
    average_cluster = cluster.AgglomerativeClustering(n_clusters=2, linkage="average")
    birch = cluster.Birch(n_clusters=2)
    
    # Algoritmaların listesi
    clustering_algorithms = [two_means, ward, spectral, dbscan, average_cluster, birch]
    
    # Her bir algoritma için döngü
    for i_algo, (name, algo) in enumerate(zip(clustering_names, clustering_algorithms)):
        # Algoritmayı veriye uygula
        algo.fit(X)
        
        # Kümeleme sonuçlarını al
        if hasattr(algo, "labels_"):
            y_pred = algo.labels_.astype(int)
        else:
            y_pred = algo.predict(X)
        
        # Her bir subplot'u oluşturma
        plt.subplot(len(datasets_list), len(clustering_algorithms), i_dataset * len(clustering_algorithms) + i_algo + 1)
        
        # İlk satırda algoritma isimlerini başlık olarak ekleme
        if i_dataset == 0:
            plt.title(name)
        
        # Kümeleme sonuçlarını scatter plot ile gösterme
        plt.scatter(X[:, 0], X[:, 1], c=colors[y_pred].tolist(), s=10)
        plt.xticks(())
        plt.yticks(())

# Tüm subplotları düzenli bir şekilde yerleştirme
plt.tight_layout()

# Çizimleri gösterme
plt.show()