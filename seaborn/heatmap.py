import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
# heatmap(ısı haritası) Verilerin yoğunluğunu veya dağılımını renkler aracılığıyla hızlıca görmek
veri = np.random.rand(10, 12)  # 10x12 boyutunda rastgele  0-1 arası değerler

#list comprehension  tek satırda liste oluşturmak için kullanılır
df = pd.DataFrame(veri, columns=[f'Monat_{i}' for i in range(1, 13)],
                  index=[f'Gün_{i}' for i in range(1, 11)])

korelasyon = df.corr()

sns.heatmap(data=korelasyon, annot=True, cmap='RdYlBu', linewidths=0.5, fmt='.2f')


plt.xlabel('Ay')
plt.ylabel('Gün')
plt.title('Heatmap Örneği')

"""

#Eğer veri setimizde kategorik bir değişken varsa bunu sayısal değere dönüştürmeliyiz

veri = {'Kategori': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'Özellik 1': [3, 4, 2, 1, 5, 2, 3, 2, 4],
        'Özellik 2': [1, 2, 3, 4, 5, 1, 2, 3, 4],
        'Özellik 3': [4, 3, 2, 1, 2, 4, 3, 2, 1]}


df = pd.DataFrame(veri)
df.describe()
#baktığımızda Kategori sütunu object olarak gözükmekte ilk başta bunu kategorik hale getirelim
df['Kategori'] = pd.Categorical(df['Kategori'])
df.describe()
#.cat.codes kullanarak farklı her kategorik değere farklı bir sayısal değer atadık
df['Kategori'] = df['Kategori'].cat.codes

# Korelasyon matrisini oluşturma
korelasyon = df.corr()


sns.heatmap(korelasyon, annot=True, cmap='RdYlBu')

plt.title('Özellik Korelasyonları')


"""
annot parametresi hücre yoğunlukların değerlerini hücre içinde göstermeyi sağlar
cmap renk paletini belirler çeşitli renk paletleri vardır ('YlGnBu', 'coolwarm', 'viridis', vb.)
fmt ondalıklı kısmın nasıl gösterileceğini belirler .2f   0.97 gibi
"""
