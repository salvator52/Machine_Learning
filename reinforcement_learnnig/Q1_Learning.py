#ajan(model) environment içinde action'lar gerçekleştirerek hedefe ulaşmasını sağlamak
import gym 
import random
import numpy as np


#render mode görselleştirme için    slippery ise ajanın kaygan bir yüzeyde gitme işlevinin yapcak True olursa istediği state'e değil farklı state'e gitme olasılığı avrdır
environment = gym.make("FrozenLake-v1",is_slippery = False ,render_mode ="ansi")
environment.reset() #hem resetleme hem de hangi state'deyiz onu görürüz

#kaç tane state'miz var ona bakamak için
nb_states = environment.observation_space.n

#modelin yapabileceği action'lar burada sağ,sol,yuakrı,aşağı
nb_Actions =environment.action_space.n

qtable = np.zeros((nb_states ,nb_Actions)) #state'ler bizim satırlarımız action lar ise sütunlarımızdır

print("Q-table")
print(qtable)

#sol(left):0 aşağı(down):1 sağ(right):2 yuakrı(high):3

#environment içerisinde rastgele hareket etmeyei sağlar
action = environment.action_space.sample()
new_state , reward ,done ,info, _ = environment.step(action)


#burası şu ana kadar başlangıç environment içinde nasıl ahreket eder vs. içindi




