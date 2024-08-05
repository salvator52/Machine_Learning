from sklearn.datasets import load_iris #iris çiçeğinin veri setidir
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
#(1) veri seti elde edilmesi
iris = load_iris()

X = iris.data #feature
y = iris.target #target

#(2) ML modeli seç


#(3) Train işlemi yap
X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#kriter kısmı entropy ya da gini değeri max_Depth ise ağacın max derinliğini verir
tree_clf = DecisionTreeClassifier(criterion="gini",max_depth=5,random_state=42)
tree_clf.fit(X_train, y_train)

#prediction kısmı
y_pred = tree_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))


#tree göstermek için from sklearn.tree import plot_tree kullanırız
plt.figure(figsize=(15,10)) #figürü çoz uzakta çıkarmamak için 
#filled node'ları doldurmaya izin vermesi için neyle peki o da feature names parametresine verdiğimiz giridyle
#class names ise yapraklarda  yani en son ki node'larda gösterilecek şeyleri verir
#feature names ve class name liste şeklinde gönderilmelidir
plot_tree(tree_clf,filled =True,feature_names=iris.feature_names,class_names=list(iris.target_names))
plt.show()



#burada feature importance kavramıda devreye girmektedir hangi feature'ın önemli olduğunu öğrenmek istersek
feature_importance = tree_clf.feature_importances_
feature_names = iris.feature_names

feature_importance_sorted =sorted(zip(feature_importance,feature_names),reverse=True) #burası büyükten küçüğe önemliliği sıraladığımız yer
for importance ,f_name in feature_importance_sorted:
    print(f"{f_name} : {importance}")

















