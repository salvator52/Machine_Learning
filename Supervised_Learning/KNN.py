from sklearn.datasets import load_breast_cancer #data set'i import ettik
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

#(1) veri setinin incelenmesi
cancer = load_breast_cancer()
#cancer verilerimizi bir dataframe koyalım
df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
#target yani label'ımızıda ekleyelim
df["target"] = cancer.target

#(2) modülün belirlenmesi
#(3) train işlemi

X = cancer.data #features
y = cancer.target #target

#train ve test diye ayırma kısmımız
#test_size 100 de kaç test kısmına ayrılcak ona bakıcaz
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#ölçeklendirme  belli bir aralığa değerleri sıkıştırmak
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#knn ile test işlemine geçelim
knn = KNeighborsClassifier() #hiperparametre gelicek
#train etmek için .fit() metodu kullanılır
knn.fit(X_train, y_train)

#(4) sonuçların değerlendirilmesi :test
#test kısmında ayrılan X değerlerine karşılık predict tahminler çıakrır
y_pred = knn.predict(X_test)
print(accuracy_score(y_test,y_pred)) #doğruluk oranını verir
print(confusion_matrix(y_test, y_pred)) #tablo şeklinde verir


#(5) Hiperparametre Ayarlarması
"""
    KNN: hiperparamtre = K
        K:1,2,3,...N
        Accuracy :%A,%B,%C, ...
"""
#n_neighbors parametresi yakında kaç tane komşu elemana bakılması gerektiğini ayarlar
accuracy_values = []
k_values = []

for k in range(1,21):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)

plt.figure()
plt.plot(k_values,accuracy_values,marker = "o",linestyle ="-")
plt.title("K değerine göre accuracy")
plt.xlabel("K değeri")
plt.ylabel("Doğruluk")
plt.xticks(k_values)
plt.grid(True)






