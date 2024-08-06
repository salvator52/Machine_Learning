import pandas as pd
import numpy as np
"""
tek boyutludur numpy array'lerinden farklı olarak birden fazla veri tipiyle çalışabilir
"""
"""
#seri oluşturma 
#seriler tuple,liste ve dictionary ile oluşabilir
seri1 = pd.Series(("muhammed",22,"arda",False))
seri2 = pd.Series([1,2,3,True])
seri3 = pd.Series({"istanbul" : 34, "ankara" : 6, "ordu" : 52}) 

#indexleri kullanarak değer değiştirme ya da değere ulaşabiliriz
#print(seri1[0]+" "+str(seri2[3]))
#fakat seri3'ün 2. elemanına erişmek istersek seri3[1] ile erişemeyiz
# print(seri3[1])
#print(seri3.index) #indexler istanbul ankara ve ordu olarak gözüküyor nedeni ise dictionary ile oluşturmuş olmamız key değerleri index value'lar yine value olur
# print(seri3["ankara"]) bu sefer 2. elemana ulaşabildik
"""
"""
seri4 = pd.DataFrame([12,23,32,52,17,98,87,65,46])

#Serilerde boolean işlemler nasıl yapılır ?
#aşağıdaki gibi yapılırsa sadece true false olarak döner
print(seri4 > 15)
#fakat bir kez daha seri4 içinde yazarsak istediğimiz değerleri görebiliriz false olan değerler NaN olarak çevrilir
print(seri4[seri4 > 35])
print(seri4[ (seri4 > 17) & (seri4 < 68) ]) #çift karşılaştırma için yapılabilir
"""

"""
#seri içinde nan değer var mı kontrolü  isna() ya da isnull() kullanılır farkı pek yoktur
seri5 = pd.Series([np.nan,4,23,5,np.nan,"mahmut"])
print(seri5.isnull())#bu şekilde olursa true false döner true olan kısımlar null değerlerdir eğer kaç tane var onu öğrenmek istersek eğer
print(seri5.isna().sum().sum()) #ile kaç tane null değer var öğrenebiliriz
#uniq değerleri öğrenmek için .unique() metodu yardımcı olur
print(seri5.unique()) 
"""

#Serilerde index mantığı
#serilerde default olarak index'ler 0 dan başlar fakat istersek bunu değişebiliriz
seri6 = pd.Series([5,3,1,7,93,2,342,131,"araba"],index=["a","b","c","d","e","f","g","h","i"])
print(seri6.items) #.index index lere .values değerlere .items her ikisinide verir

#.values ile alakalı olarak pandas serisini ya da dataframe'i nd array'e çevirmeye sağlar
seri7 = seri6.values
print(str(type(seri6))+" "+str(type(seri7)))

#loc ve iloc kullanımı
#loc direkt olarak indexlerle işlemler yapar
#iloc ise indexler farklıda olsa 0,1,2.... şeklinde işlemler yapar

print(seri6.loc["a":"e"]) #sondakinide dahil eder
print(seri6.iloc[0:4]) #burda e yani 4.index'i almadı


#Çok kullanılan Metodlar

#.max() -> en büyük değeri döner  .min() -> en küçük değeri döner 
#.sum() -> serinin içindeki elemanları toplar serideki değerler sayı türünden olmalıdır
#.mean() ->serinin ortalamasını alır hepsi sayı olmalı
#.std -> serinin standart sapmasını hesaplar


#Serilerde Birleştirme İşlemleri .concat metoduyla yapılır seriler tek sütundan oluştukları için alt alta birleşirler
seri8 = pd.Series([1,2,3,4,5,6])
seri9 = pd.Series([7,8,9,10,11,12])

#aşğıdaki gibi index birleştirme yapabiliriz index birleştirme yaparken ignore_index'i true yapmalıyız yoksa indexler karışabilir
#ignore index index'leri default haline getirmeyi sağlar
seri10 = pd.concat([seri8,seri9],ignore_index=True)


print(seri10.dtype) #serinin hangi tipte olduğunu verir  int64 çıktı
print(seri6.axes)  #serinin index'lerini verir
seri10.loc[10] = 16 #aynı veri türünü verirsek eleman eklemesi yapabiliriz













