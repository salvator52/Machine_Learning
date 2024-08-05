import gym 
import random
import numpy as np
from tqdm import tqdm

# Taxi-v3 ortamını oluştur ve render modunu "ansi" olarak ayarla
env = gym.make("Taxi-v3", render_mode="ansi")
env.reset()
print(env.render())

"""
Eylemler:
0: Güney yönüne hareket et
1: Kuzey yönüne hareket et
2: Doğu yönüne hareket et
3: Batı yönüne hareket et
4: Yolcuyu al
5: Yolcuyu bırak
"""

# Eylem alanını (olası eylem sayısını) tanımla
action_space = env.action_space.n

# Durum alanını (olası durum sayısını) tanımla
state_space = env.observation_space.n

# Q-tablosunu sıfırlarla başlat
q_table = np.zeros((state_space, action_space))

# Hiperparametreleri tanımla
alpha = 0.1  # Öğrenme oranı
gamma = 0.6  # İndirim oranı
epsilon = 0.1  # Keşfetme oranı

# Ajanı eğitme
for i in tqdm(range(1, 100001)):
    state, _ = env.reset()  # Ortamı sıfırla ve başlangıç durumunu al
    done = False  # Görev tamamlandı mı?

    while not done:
        # Epsilon oranına göre rastgele bir eylem seç (keşfetme)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # Aksi takdirde, en yüksek Q-değerine sahip eylemi seç (istifade)
            action = np.argmax(q_table[state])

        # Eylemi gerçekleştir ve yeni durumu ve ödülü gözlemle
        next_state, reward, done, info, _ = env.step(action)

        # Q-tablosunu güncelle
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        # Bir sonraki duruma geç
        state = next_state

print("Eğitim tamamlandı")

# Eğitilmiş ajanın test edilmesi
total_epochs, total_penalties = 0, 0
episodes = 100  # Test edilecek bölüm sayısı

for i in tqdm(range(episodes)):
    state, _ = env.reset()  # Ortamı sıfırla ve başlangıç durumunu al
    
    epochs, penalties, reward = 0, 0, 0  # Bölüm içindeki adım sayısı ve ceza sayısını başlat
    done = False  # Görev tamamlandı mı?

    while not done:
        # En yüksek Q-değerine sahip eylemi seç
        action = np.argmax(q_table[state])
        
        # Eylemi gerçekleştir ve yeni durumu ve ödülü gözlemle
        next_state, reward, done, info, _ = env.step(action)
        
        # Bir sonraki duruma geç
        state = next_state
        
        # Ceza (negatif ödül) sayısını hesapla
        if reward == -10:
            penalties += 1
        
        epochs += 1

    total_epochs += epochs
    total_penalties += penalties
    
print(f"{episodes} bölüm sonrası sonuçlar")
print("Bölüm başına ortalama adım sayısı:", total_epochs / episodes)
print("Bölüm başına ortalama ceza sayısı:", total_penalties / episodes)
