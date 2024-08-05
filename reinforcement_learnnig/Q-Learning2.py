import gym 
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#render mode görselleştirme için    slippery ise ajanın kaygan bir yüzeyde gitme işlevinin yapcak True olursa istediği state'e değil farklı state'e gitme olasılığı avrdır
environment = gym.make("FrozenLake-v1",is_slippery = False ,render_mode ="ansi")
environment.reset() #hem resetleme hem de hangi state'deyiz onu görürüz

#kaç tane state'miz var ona bakamak için
nb_states = environment.observation_space.n

#modelin yapabileceği action'lar burada sağ,sol,yuakrı,aşağı
nb_Actions =environment.action_space.n

qtable = np.zeros((nb_states ,nb_Actions)) #ajanın beyni

print("Q-table")
print(qtable)

#bölüm sayısı tanımla (kaç kere oynicak)
episodes = 1000
alpha = 0.5 #learning rate'i ifade edecek
gama = 0.9 #discount rate 

outcomes = [] #başarı ve başarısızlık durumlarını saklamak için kullanıcaz

#training
#tqdm sayaç grafiksel
for i in tqdm(range(episodes)):
    state , _ = environment.reset() #başlangıç state'ni aldık
    done = False #ajanın başarı durumu
    outcomes.append("Failure")
    
    while not done: #ajan başarılı olana akdar state içerisinde hareket et (Action seç ve uygula)
        #ACTION
        if np.max(qtable[state]) > 0: 
            action = np.argmax(qtable[state])
        else :
            action = environment.action_space.sample()
        
        new_state , reward ,done ,info, _ = environment.step(action)
        
        #UPDATE Q TABLE burası direkt forül
        qtable[state,action] = qtable[state,action] + alpha *(reward + gama *np.max(qtable[new_state])-qtable[state,action])
        
        state = new_state
        if reward:
            outcomes[-1] ="Success"


print("Q TABLE AFTER TRAINING")
print(qtable)


plt.bar(range(episodes),outcomes)

episodes = 100
nb_succes =0
#test
for i in tqdm(range(episodes)):
    state , _ = environment.reset() #başlangıç state'ni aldık
    done = False #ajanın başarı durumu
   
    while not done: #ajan başarılı olana akdar state içerisinde hareket et (Action seç ve uygula)
        #ACTION
        if np.max(qtable[state]) > 0: 
            action = np.argmax(qtable[state])
        else :
            action = environment.action_space.sample()
        
        new_state , reward ,done ,info, _ = environment.step(action)
        
        state = new_state
        
        nb_succes +=reward #episode'dan kaçta ne doğru değer bulursa reward artar



print("Success rate :",100*nb_succes /episodes)
