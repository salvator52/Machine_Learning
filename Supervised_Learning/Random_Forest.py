from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



#2D image var 4096 adet feature 1D   2D için(64*64)
oli = fetch_olivetti_faces()

#400 tane feature var  40 tane de target var yani her bir kişiye ait 10 tane feature var demek oluyor

"""
veri setimiz gray scale çünkü 
rgb olsaydı (400,64,64,3) olması gerekirdi

"""



"""
plt.figure()
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(oli.images[i],cmap="gray")
    plt.axis("off")
plt.show()

"""

X = oli.data #feature
y = oli.target #target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#estimator = içinde buluna her bir ağaç sayısı içinde kaç tane ağaç olcak yani
rf_clf = RandomForestClassifier(n_estimators=100,random_state=42)
rf_clf.fit(X_train, y_train)

y_pred=rf_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))


i_list = []
accuracy_list= []
for i in range(5,100,5):
        rf_clf = RandomForestClassifier(n_estimators=i,random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred=rf_clf.predict(X_test)
        i_list.append(i)
        accuracy_list.append(accuracy_score(y_test, y_pred))


plt.figure()
plt.scatter(i_list,accuracy_list)
plt.xticks(i_list)

plt.show()





