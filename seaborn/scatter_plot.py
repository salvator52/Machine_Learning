import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#Nokta grafiği (scatter plot)
#x ve y ekseninde kesişen noktaları gösterir 

veri = {'Boy': [160, 165, 170, 175, 180,185],
        'Kilo': [60, 65, 70, 75, 80,85],
        'group': ['A', 'A', 'A', 'B', 'B', 'B']}

df = pd.DataFrame(veri)

sns.scatterplot(data=df, x='Boy', y='Kilo',style="group")
plt.xlabel('Boy')
plt.ylabel('Kilo')
plt.title('Boy ve Kilo İlişkisi')
plt.xticks(veri["Boy"]) #x eksenini istediğimiz değerleri koyabiliriz




