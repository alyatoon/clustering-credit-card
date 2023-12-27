import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('german_credit_data.csv')
df = df.drop(['Unnamed: 0'], axis=1)

st.header('Segmentation German Credit Data')
st.write('Data German Credit Data adalah data yang berisi informasi mengenai kredit yang diberikan kepada nasabah. Data ini terdiri dari 1000 baris dan 9 kolom. Kolom-kolom yang ada di dalam data ini adalah: Age, sex, job, housing, saving accounts, checking account, credit amount, duration, purpose. Pada project ini, akan dilakukan segmentasi data menggunakan metode K-Means Clustering.')
st.subheader('Data Asli')
st.write(df)

x = df.loc[:, ["Age", "Credit amount", "Duration"]]
cluster_log = np.log(x)
scaler = StandardScaler()
cluster_S = scaler.fit_transform(cluster_log)

st.subheader('Mencari Jumlah Kluster Terbaik')
k = []
for i in range(1,11):
    kmeans = KMeans(i)
    kmeans.fit(cluster_S)
    k.append(kmeans.inertia_)

plt.figure(figsize=(20,10))
plt.plot(range(1,11), k, marker='*')
plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False)
elbow = st.pyplot()
st.write('Dari grafik di atas, didapatkan bahwa jumlah kluster terbaik adalah 3.')

st.sidebar.subheader('Jumlah Kluster')
cluster = st.sidebar.slider('Masukkan jumlah kluster', 1, 10) 

def k_means(n):
    kmeans = KMeans(n_clusters=n, random_state=42).fit(cluster_S)
    df['cluster'] = kmeans.labels_

    plt.figure(figsize=(12,8))
    sns.scatterplot(x='Credit amount', y='Age', hue='cluster', data=df, palette='viridis', marker='o', s=100)
    
    for label in df['cluster']:
        plt.annotate(label, 
        (df[df['cluster']==label]['Credit amount'].mean(),
        df[df['cluster']==label]['Age'].mean()),
        horizontalalignment='center',
        verticalalignment='center',
        size=20, weight='bold',
        color='black')

    st.header('Hasil Klustering')
    st.subheader('Data Setelah Dilakukan Klustering')
    st.pyplot()
    st.write(df)

k_means(cluster)
#buatkan kesimpulan dari hasil klustering yang telah dilakukan
st.subheader('Kesimpulan')
st.write('Berdasarkan hasil klustering(jika jumlah cluster adalah 3) yang telah dilakukan, didapatkan bahwa:')
st.write('1. Kluster 0 adalah nasabah yang memiliki kredit amount yang kecil, umur yang muda, dan durasi kredit yang pendek.')
st.write('2. Kluster 1 adalah nasabah yang memiliki kredit amount yang besar, umur yang muda, dan durasi kredit yang pendek.')
st.write('3. Kluster 2 adalah nasabah yang memiliki kredit amount yang besar, umur yang tua, dan durasi kredit yang lama.')
