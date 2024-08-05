from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas as pd

heart_disease = fetch_ucirepo(name = "heart_disease")

#burada kalp hastalığı verisetimizde missing value sıkıntımız var ilk olarak bunu düzeltmeliyiz
#bunun için pandas'dan yardım alcaaz

df = pd.DataFrame(data=heart_disease.data.features)
df["target"] = heart_disease.data.targets

#burada 2 tane any kullanmızın sebebi ilk any() her satır için yapar 2. any() ise her satırdan birinde bile true varsa true döncek

#drop missing value
if df.isna().any().any():
    df.dropna(inplace=True) #burası na/nan eksik sütun ya da satır varsa onları siler
    
    
X = df.drop(["target"],axis = 1).values  #target'ı çıkarıp nd array'e dönüştürmek için values kullandık
y = df.target.values #buda pandas serisi olmasın nd array olsun diye values ile nd array'e dönüştürdük

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

log_reg = LogisticRegression(penalty="l2",C=1,solver="lbfgs",max_iter=100)
log_reg.fit(X_train, y_train)

print(log_reg.score(X_test, y_test))
#üst kısım y_pred sonra accuracy_score hesaplamasını tek satırda yapmamızı sağlıyor















