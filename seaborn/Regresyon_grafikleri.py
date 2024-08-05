import seaborn as sns
import matplotlib.pyplot as plt

veri = sns.load_dataset("tips")
"""
# 1)lmplot()
# Veri kümesini oluşturma Değişkenler arasındaki doğrusal ilişkiyi gösteren regresyon grafiği çizme
sns.lmplot(data=veri, x="total_bill", y="tip")

#hue parametresi kullanarak ta yapılabilir
sns.lmplot(data=veri, x="total_bill", y="tip", hue="sex")

"""

# 2)pairplot()
#Her bir değişken çifti için scatter plotlar çizilir ve değişkenlerin dağılımları ve ilişkileri hakkında bilgi sağlar.
# Pair plot grafiğini çizme
sns.pairplot(data=veri)

#aynı şekilde hue parametresi kullanarakta yapılabilir
sns.pairplot(data=veri, hue="sex")