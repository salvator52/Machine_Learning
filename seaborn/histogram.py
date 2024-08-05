import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#Histogram  veri dağılımını görselleştirmek için kullanılır
veri = {
    'Değer': [32, 45, 67, 55, 78, 84, 91, 62, 75, 82, 69, 73, 66]}

df = pd.DataFrame(veri)


sns.histplot(data=df, x='Değer', bins=5,element="bars",kde=True)


plt.xlabel('Değer')
plt.ylabel('Frekans')
plt.title('Değerlerin Dağı')

"""
en önemli parametre bins parametresidir bu parametre kaç tane sütun yani kaç parçaya ayırmak istediğimizi belirtir
element parametresi ise histogramın nasıl görünmesini istediğimizi gireriz  örn; ploy,bars,step 
kde  yoğunluk eğrisini grafiğe eklemeyi sağlar
"""