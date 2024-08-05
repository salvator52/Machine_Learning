import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#kategorik değişkenler için
#Kutu grafiği (Box plot)

veri = {
    'Grup': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'Değer': [10, 20, 15, 25, 30, 22, 17, 30, 25]
}

df = pd.DataFrame(veri)


sns.boxplot(x='Grup', y='Değer', data=df,palette="viridis",width=0.2)


plt.xlabel('Grup')
plt.ylabel('Değer')
plt.title('Box Plot Örneği')

"""
palette kullanılacak renk paleti belirtilir seabornun kendi içinde olan viridis'i kullandım
width kutu genişliklerinin ayarlanmasını sağlar
orient parametresiyle yine yatay ya da dikey hazırlayabiliriz
"""
"""
ÖNEMLİ KAVRAMLAR (boxplot'u anlama)
Q3(kutu üst kenarı) : verilerin %75'i bu kısmın altındadır
Q1(kutu alt kenarı) : verilerin %25'i bu kısmın altındadır
Q2(medyan) : Kutu içinde yatay çizgi olarak gösterilir, verilerin ortanca değeridir.
Whisker : Q1 - 1.5 * IQR(alt sınır) ve Q3 + 1.5 * IQR(üst sınır) aralığında olan değerleri gösterir.Eğer bu aralıkta olmayan veri varsa o verilere outlier(aykırı değer) denir
Aykırı Değer : Tekrar olarak Whisker aralığında olmayan değerlere denir

"""




