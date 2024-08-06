import numpy as np
import pandas as pd
import seaborn as sns
"""
#Dataframe nasıl oluşturulur ?
#1)
df1 = pd.DataFrame([[1,2,3,4],
                    [1,2,3,4],
                    [7,3,9,0]
                    ])

#2)numpy array'ler kullanarak
n1 = np.random.randint(50,size=4)
n2 = np.random.randint(50,size=4)
n3 = np.random.randint(50,size=4)
df2 = pd.DataFrame([n1,n2,n3])

#ya da
n4 = np.random.randint(10,50,(3,4))
df3 = pd.DataFrame(n4)

#ya da 
n5 = np.arange(1,11).reshape(3,4)
df3 = pd.Dataframe(n4)

#3 dictionary kullanarak burada key'ler sütun adı olur 
df4 = pd.DataFrame({"s1":np.random.randint(20,size=4),
                     "s2":np.random.randint(20,size=4),
                     "s3":np.random.randint(20,size=4)})
"""
#.axes satır ve sütun bilgilerini verir
#.shape boyut bilgiisi verir
#.ndim boyut sayısını verir
#.size eleman sayısını verir
#.values dataframe'i bir nd array'e dönüştürür
#.ravel() çok boyutlu diziyi ya da df'yi tek boyuta indirger
"""


#sütun ve index adı ayarlama
#.index= liste  .columns = liste bunu siter oluştururken ister oluşturduktan sonra kullanabiliriz
df5 = pd.DataFrame([np.random.randint(50,size=4),
                    np.random.randint(50,size=4),
                    np.random.randint(50,size=4)],columns=["s1","s2","s3","s4"],index=["a","b","c"])

#sütun değerlerine ulaşmak için df5.columns
#index değerlerine ulaşmak için df5.index


#Eleman İşlemleri
#sütunlara erişmek için
#print(df5[sütun adı])
#birden fazla sütuna erişmek içinse fancy index  
print(df5[["s1","s2"]])

#hem sütun hem de satıra ulşamk için loc ,iloc kullanıcaz
# loc[ilk kısım satır,vrigülden sonrası sütun]
#loc kullanmazsak hata alırız
print(df5.loc["a":"c",:])


#Sütun eklemek için olmayan bir sütun adı yazmamız yeterli olacaktır
df5["s5"] = [10,12,2]

#satır eklemek için ise yine loc kullanıcaz yine olmayan bir isim yeterli
df5.loc["d"] = [12,16,14,1,9]

#Dataframeden sütun ya da satır silmek istersek .drop() kullanmalıyız ,inplace kullanarak yaptığımız işlemlerin df üzerinde de gerçekleşmesini sağlarız
#burada default olarak axis=0 dır yanis atır işlemi yapılacağını söyler sütun içinse 1 olmalıdır
df5.reset_index(drop=True,inplace=True) #index sıfırlaması yapabiliriz drop indexleri 0'lar
df5.drop(1,inplace=True) #1index'li sütun silindir axis=0
df5.drop(["s1","s2"],axis=1,inplace=True) #k1 ve k2 sütunları silinir

#satır veya sütun adı değiştirmek için .rename metodu kullanıcaz

#inplace parametresi kullanmazsak df5'in kopyası oluşturulcak ve işlemler orada gerçekleşcek inplace true yaparsak direkt dataframe'de değişiklikler olacak
#ya da pek tercih edilmeyen aşağıdaki gibide yapılabilir
#df5 = df5.rename(index={"a":"k","b":"l"},columns={"s1":"k1","s2":"k2"})
df5.rename(index={"a":"k","b":"l"},columns={"s3":"k1","s4":"k2"},inplace=True)

#Koşullu İfadeler
print(df5 > 30 )#true false döner
print(df5[df5 > 30 ]) #değer ya da NaN olarak görücez

print(df5.loc[(df5["k1"] > 15) & (df5["k2"] > 15), ["k1", "k2"]])
"""

"""
#Sık kullanılan metodlar    
#.apply(fonksiyon adı)  metodu bir series ya da df'in sütunlarındaki değerlerin hepsine bir fonksiyonu uygulamayı sağlar
data = np.random.randint(1, 10, size=(3, 4))
df6 = pd.DataFrame(data,columns=["s1","s2","s3","s4"])

#örneğin tüm elemanların karesini almak istiyoruz 
def kareAl(x):
    return x**2

#df6 = df6.apply(kareAl)

#bu fonksiyon genelde bir kere kullanılıp bırakılır o yüzden hafıza boş yere yer tutar bunun yerine lambda fonksiyonu kullanarak yaparsak daha efektif olacak
#df6 = df6.apply(lambda x : x**2)

#son 3.yol'a bakarsak en hızlısı vektörize yöntemlerdir 
#numpy'ın kendi kare alma .square metoduyla kullanalım bu genelde tamamı çin değil belli sütunlar için işlevler yapılırken daha mantıklıdır
df6["s2"] = np.square(df6["s2"])

#her satrın toplamını df'e ekleyelim
df6["toplam"] = df6.sum(axis=1) #satır toplamları
#sıradaki işlem toplam 40'ın altındaysa *  40 -60 arası ** 60 tan fazlaysa *** olacak şekilde df'e eleman eklicez
#df6[["toplam","s1"]].apply  şeklinde birden fazla sütunlada işlem yapabiliriz
df6["değerlendirme"] = df6["toplam"].apply(lambda x : '*' if x < 40  else('**' if 40 < x < 60 else '***'))


#.sort_values   sıralam işlemleri
#ascending true ise küçükten büyüğe , index'ler karışacağı için indexleri reset at
#örneğin toplamda iki satırda değerler aynı o zaman s2 sütununa bakarak sıralar
df6.sort_values(["toplam","s2"],ascending=False,ignore_index=True,inplace=True)

#value.counts  sütunlardak buluna değerler kaç kere geçiyo
# print(df6["s1"].value_counts())

#unique = kaç farklı değer var  nunique = kaç farklı değer var ama sayısını döndürür
# print(df6["s1"].unique())  
# print(df6["s1"].nunique())

#argmax max min argmin
#max en büyük değeri verir argmax ise en büyük değerin indexini 
print(df6["toplam"].max())
print(df6["toplam"].argmax())

#.describe  çok önemli bir metod  her sütun hakkında istatiksel bilgileri verir
# print(df6.describe())

#.info() genel bilgiler verir sütun hakkında nan/null değer sayısı dtype'ı vs. vs.
# print(df6.info())

#amaç isim değitirme ise örneğin male ve female değerleri M ya da F olsun
#1)  apply ile df["sütun adı"] = df["sütun adı"].apply(lambda x : "M" if x == "male" else : "F") şeklinde yapılabilir
#2) .replace ile  df["sütun adı"] = df["sütun adı"].replace("female","F",inplace = true)
#df["sütun adı"] = df["sütun adı"].replace("male","M",inplace = true)
#tek satırda yapmak istersek df["sütun adı"] = df["sütun adı"].replace(["female","male"],["F","M"],inplace = true)

#3) map işlemi df["sütun adı"] = df["sütun adı"].map({"female" :"F","male":"M"},inplace=True)
"""

#GROUPBY groupby gruplaştırma için kullanılır fakat tek başına kullanılmaz
#Buradaki ana mantık belli bir sütuna göre gruplama yapılır diğer sütunlarda işin içne girer
df7 = sns.load_dataset("planets")
grp = df7.groupby("method").mean() #method sütununda bulunan uniq değerler için diğer sütunların ortalması hesaplanır
print(grp)
#bir sürü değişken varsa onlarında seçilmesi gerekli
grp2 = df7.groupby("method")[["orbital_period","distance"]].mean() #yukarıdakinin aynısı sadece belli sütunların ortalamasını istedik
print(grp2)

#aggregate methodu birden fazla işlem yapmamızı sağlar

df8 = pd.DataFrame({"gruplar":["A","B","C","A","B","C"],
                    "degisken1":[10,23,33,22,11,99],
                    "degisken2":[100,253,333,262,111,969]
                    })

#gruplar da A B C  vardı A karşılığı olan sütunları alır min max median hesaplar aynı şekilde diğerleri içinde
print(df8.groupby("gruplar").aggregate([min,np.median,max]))

#aggregate metodunu agg ilede kullanabiliriz
#ya da belli başlı sütunlara istediğimiz işlemi yapmak istersek
print(df8.groupby("gruplar").agg({"degisken1":min,"degisken2":max}))
#var,std,max,min,first,last gibi metodlarıda kullanabiliriz


#fiter metodu adı üstüne sütunları belli bir ifade ile filtrelemek istersek kullanırız
def filter_func(x):
    return x["degisken1"].std() >9

#burada yaptığımız şey standart sapması 9 dan büyük olanı fonksiyondan döndürmek 
print(df8.groupby("gruplar").filter(filter_func))

#apply

df5 = pd.DataFrame({
                    "degisken1":[10,23,33,22,11,99],
                    "degisken2":[100,253,333,262,111,969]
                    })

#her bir değişkenin toplamını alcak
print(df5.apply(np.sum))

#gruplarlada çalışabilir
print(df8.groupby("gruplar").apply(np.sum))




