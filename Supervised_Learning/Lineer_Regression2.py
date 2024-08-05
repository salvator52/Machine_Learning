from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt

#lineer regresyon olduğu için sadece tek bir feature üzerinden işlem yapıcaz
#bmi isimli feature'ı seçelim
diabets_X,diabets_y = load_diabetes(return_X_y=True) #bu şekilde target ve feature kısmını alabiliriz
# X = diabets.data ve y = diabets.target'a eşittir

diabets_X = diabets_X[:,np.newaxis,2] #sadece 2. sütunu alıyoruz

#son 20 hariç al
diabet_X_train = diabets_X[:-20]
#son 20 taneyi al
diabet_X_test = diabets_X[-20:]

#son 20 hariç al
diabet_y_train = diabets_y[:-20]
#son 20 taneyi al
diabet_y_test = diabets_y[-20:]

lin_reg =  LinearRegression()
lin_reg.fit(diabet_X_train,diabet_y_train)

diabet_y_pred = lin_reg.predict(diabet_X_test)

mse = mean_squared_error(diabet_y_test,diabet_y_pred)
r2 = r2_score(diabet_y_test,diabet_y_pred)




plt.scatter(diabet_X_test,diabet_y_test,color="black")
plt.plot(diabet_X_test,diabet_y_pred,color="blue")