import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Çizgi Grafiği(lineplot)

veri = {'Yıl': [2015, 2016, 2017, 2018, 2019, 2020],
        'Gelir': [5000, 5500, 6000, 6500, 7000, 7500],
        'group': ['A', 'A', 'A', 'B', 'B', 'B']}
df = pd.DataFrame(veri)

sns.lineplot(x="Yıl",y="Gelir",style="group",markers=True,data=df)

plt.xlabel("Yıl")
plt.ylabel("Gelir")
plt.title("Şirket Geliri")

"""
style seçili sütunun farklı değerlerinde farklı bir görünüm olması için
markers x ve y de kesişim noktalarını işaretlemek için
data dataframe kısmını alıcak
"""
