import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#kategorik değişkenler için
#Barplot (Kutu Grafiği) 
veri_genis = {
    'Kategori': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
    'AltKategori': ['X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y'],
    'Değer': [10, 20, 15, 25, 12, 18, 13, 22]
}

df = pd.DataFrame(veri_genis)

sns.barplot(x='Kategori', y='Değer', hue='AltKategori', data=df,orient='h')

plt.xlabel('Kategori')
plt.ylabel('Değer')
plt.title('Kategoriye ve Alt Kategoriye Göre Değerler')

"""
hue yeni bir boyut ekler kategorideki A tek bir sütunken A karşılığında altkategorilerde X ve Y değerleri için 2 tane grafik çizilmiş olur
orient parametresi ise sütun grafiğin yatay mı dikey mi olacağını belirleyeceğimiz kısım

"""