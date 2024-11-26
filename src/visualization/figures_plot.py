# Ana betik
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Utility dosyasını dahil et
sys.path.append("..")
import utility.plot_settings  # plot_settings.py dosyasındaki tüm ayarları buraya dahil ediyoruz

# Örnek veri (gerçek veriyi burada kullanmalısınız)
data = pd.read_pickle("../../data/interim/housing.pkl")
data2 = pd.read_csv("../../data/raw/Housing.csv")

df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
df2_int = df2.select_dtypes(include=["int64"])
corr_matrix = df.corr()
corr_matrix2 = df2_int.corr()

fig, ax = plt.subplots(1, 2, figsize=(20, 16))

sns.heatmap(
    corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax[0]
)
ax[0].set_title("Korelasyon Haritası 1")

# İkinci korelasyon haritası
sns.heatmap(
    corr_matrix2, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax[1]
)
ax[1].set_title("Korelasyon Haritası 2")

# Grafiği gösterme
plt.tight_layout()  # Alt grafiklerin birbirine yakın olmaması için
plt.show()
