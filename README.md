# Laporan Proyek Machine Learning

### Nama : Alya Nur Oktapiani

### Nim : 211351015

### Kelas : Pagi B

## Domain Proyek
Web app ini merupakan mengelompokan data pelanggan bank. Dimana pengelompokan ini diciptakan menggunakan algorithma K-Means Clustering. 

## Business Understanding
Web app ini dapat digunakan untuk memahami pelanggan bank. Dengan mengetahui pelanggan mana yang mendapatkan cluster mana berdasarkan jumlah credit-score dan faktor lain, pihak bank bisa menentukan apakah cocok untuk diberikan credit atau tidak. 

### Problem Statement
Banyaknya pelanggan bank yang mendaftar memjadikannya sulit untuk melakukan pengelompokan secara manual.

### Goals
Berhasil membagikan pelanggan-pelanggan berdasarkan faktor-faktor yang berpengaruh untuk menentukan kecocokan dalam pemberian credit. 

### Solution Statements

- Membuatkan web app yang bisa membuat mengelompokan secara otomatis menggunakan K-Means Clustering.

## Data Understanding
Dataset originalnya mengandung 1000 baris data dengan 20 kategorial attribut yang disiapkan oleh Prof. Hofmann. Pada dataset ini setiap baris data merepresentasikan seorang pelanggan yang mengambil credit pada bank.
Dataset = [German Credit Risk](https://www.kaggle.com/datasets/uciml/german-credit)

### Variabel-variabel pada Diabetes Prediction adalah sebagai berikut:

- Age : Menunjukkan umur pelanggan. [int, 19-75]
- Sex : Menunjukkan jenis kelamin pelanggan. [string, male/female]
- Job : Menunjukkan kategori pekerjaan pelanggan. [int, 0-3, 0 : unskilled and non-resident, 1 : unskilled and resident, 2 : skilled, 3 : highly skilled]
- Housing : Menunjukkan kategori akomodasi pelanggan. [string, own, rent, or free]
- Saving accounts : Menunjukkan kategori akun simpanan pelanggan. [string, little, moderate, quite rich, rich]
- Checking account : Menunjukkan kategori akun checking pelanggan [string, little, moderate, quite rich, rich]
- Credit amount : Menunjukkan jumlah credit yang dimiliki pelanggan. [int, dalam Deutsch Mark, 250- 18,400]
- Duration : Menunjukkan berapa lama pelanggan memiliki credit tersebut. [int, dalam bulan, 4-72]
- Purpose : Menunjukkan alasan pelanggan mengambil credit pada bank. [string, car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others]

## Data Preparation
Pada tahap ini saya akan menunjukkan visualisasi data dan melakukan preprocessing seperti pembersihan data.

### Import Dataset
Langkah pertama adalah memasukkan token kaggle untuk mendapatkan akses datasetnya yang berada pada kaggle.
```python
from google.colab import files
files.upload()
```
Membuat folder untuk menyimpan token yang tadi telah diupload.
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
```python
!kaggle datasets download -d uciml/german-credit 
```

### Extract File
Melakukan extract file yang tadi diunduh lalu memasukkannya kedalam folder.
```python
!unzip german-credit.zip -d datas
!ls datas
```
Selesai!
### Import Library
```python
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```
### Inisialisasi DataFrame
```python
df = pd.read_csv('datas/german_credit_data.csv')
```
```python
df.head()
```
Duration diatas adalah durasi dalam jangka per bulan.
```python
df.isnull().sum()
```
Ada missing data, tpi sebelum itu kita akan meneruskan proses data discovering kita.
```python
df.info()
```
Terdapat 5 kolom yang memiliki Dtype integer dan 4 object, kita akan menentukan kolom apa saja yang akan kita gunakan pada bagian EDA setelah menganalisis grafik dan melakukan pengambilan datanya pada tahap preprocessing.
```python
df.describe()
```
Terdapat 1000 baris data per sebelum ianya diproses alias pembersihan data, karena sebelumnya kita menemukan terdapat data null pada beberapa baris data. ohiya, describe ini hanya menunjukkan kolom yang bertype data integer karena datatype itulah yang bisa dicari nilai mean, std, min, dan lain lainnya.<br>
<br>
Okeh, mari lanjut dengan meng-visualisasikan data data yang nantinya akan melewati tahap preprocessing. <br>
<br>
Pertama mari kita visualisasikan sebuah scatter plot yang menunjukkan durasi dan credit amount berdasarkan gender.
```python
sns.scatterplot(x="Credit amount",y="Duration", hue="Sex", data = df)
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/7c3b352b-25b8-4ea6-a63f-11249fb83568) <br>
Disini kita bisa lihat bahwa tidak ada seorangpun pria yang memiliki credit diatas 17500DM meskipun sudah memiliki credit cardnya selama 40 bulan.
```python
sns.scatterplot(x="Age",y="Credit amount", hue="Sex", data = df)
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/6d4af465-cff0-41ca-ab5b-64e636d76984)<br>
Dari scatterplot diatas bisa disimpulkan bahwa sangat sedikit sekali seorang lansia berumur diatas 70 tahun memiliki credit amount diatas 5000DM. <br>
<br>
Selanjutnya mari lihat data data ini menggunakan garis linear untuk melihat perbedaan antar 2 kolom.
```python
sns.lmplot(x="Credit amount",y="Duration", hue="Sex", data=df, palette="Set1", aspect=2)
plt.show()
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/709ea714-8072-48f5-aab8-891af48ef910)<br>
Perlu dicatat bahwa jarak antara kedua garis tersebut tidak terlalu jauh yang artinya perbedaannya tidak terlalu significant.<br>
<br>
Next, kita akan melihatnya by housing(apakah pelanggan memiliki rumah atau tidak).
```python
sns.lmplot(x="Credit amount",y="Duration", hue="Housing", data=df, palette="Set1", aspect=2)
plt.show()
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/b7ff583e-c817-4b7e-8ff8-752c6fbcad94)<br>
Sama seperti gender, tampaknya perbedaannya tidak terlalu signifikan, bisa dilihat dari jarak antar ketiga garis diatas. <br>
<br>
Kita akan melihat secara keseluruhan alasan orang-orang menggunakan creditnya dan mendapatkan credit tersebut.
```python
n_credits = df.groupby("Purpose")["Age"].count().rename("Count").reset_index()
n_credits.sort_values(by=["Count"], ascending=False, inplace=True)

plt.figure(figsize=(10,6))
bar = sns.barplot(x="Purpose",y="Count",data=n_credits)
bar.set_xticklabels(bar.get_xticklabels(), rotation=60)
plt.ylabel("Number of granted credits")
plt.tight_layout()
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/29d9cee1-8ac3-4580-960a-c1e2eebb748e)<br>
Kebanyakan credit yang diberikan itu bertujuan untuk membeli mobil, diikuti dengan entertainment serta furniture ruangan.
### Pre Processing
Data null pada 2 kolom di atas(saving accounts, checking accounts) merupakan sesuatu yang wajar karena tidak semua orang ingin membuka akun tabungan maka dari itu kita tidak akan menghilangkan baris null jika kolom tersebut memiliki nilai null. <br>
<br>
Disini kita hanya menggunakan kolom Age, Credit Amount dan Duration untuk menentukan clustering pelanggan.
```python
x = df.loc[:, ["Age", "Credit amount", "Duration"]]
```
Mari kita lihat data-data tersebut dalam sebuah grafik.
```python
sns.distplot(df["Age"])
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/a6878146-ccac-4e6b-b894-f8117f4a1910)<br>
```python
sns.distplot(df["Credit amount"])
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/d217c6b3-8bf0-4b0a-8d62-e0047dd481e6)<br>
```python
sns.distplot(df["Duration"])
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/a215f05b-4bf7-4453-800e-7a3b8e5fe9ec)<br>
Hasil dari grafik diatas menujukan sebuah skewness ke kiri, yang artinya data tersebut cenderung lebih besar di bagian kiri, mari kita perbaiki.
```python
cluster_log = np.log(x)
```
```python
sns.distplot(cluster_log["Age"])
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/665753ff-1547-40c3-854c-30e71aa0c126)<br>
```python
sns.distplot(cluster_log["Credit amount"])
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/b83c805e-b23e-4589-900f-479cd7309135)<br>
```python
sns.distplot(cluster_log["Duration"])
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/f09b22f1-b0f1-4847-8c14-a4b24c689104)<br>
Sudah tidak terlihat condong ke kirikan? bagus bagus, mari lanjut, kita akan melakukan scaling variable kita dengan menggunakan StandardScaler.
```python
scaler = StandardScaler()
cluster_S = scaler.fit_transform(cluster_log)
```
Ditahap selanjutnya kita akan mencari nilai K yang paling optimal untuk digunakan dengan cara looping semua hasilnya dari K 1 hingga K 11 dan melihat pada plot yang dihasilkan. Dimana ada kemiringan yang paling signifikan maka disekitar itulah nilai K kita.
```python
k = []
for i in range(1,11):
    kmeans = KMeans(i)
    kmeans.fit(cluster_S)
    k.append(kmeans.inertia_)

plt.figure(figsize=(20,10))
plt.plot(range(1,11), k, marker='*')
plt.show()
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/eae3047f-0474-4e71-b59f-e2878e65403c) <br>
Elbownya sangat terlihat diantara angka 2 dan 3, disini kita akan menggunakan angka 3 sebagai nilai k.
## Modeling
Bagian sulitnya sudah dilewati, mari kita mulai modeling yang seharusnya cukup mudah dilakukan
```python
kmeans = KMeans(n_clusters=3).fit(cluster_S)
print(kmeans.cluster_centers_)
labels = pd.DataFrame(kmeans.labels_)
res = x.assign(Cluster = labels)
```
Clustering sudah selesai dan label-nya sudah di masukkan pada variable 'x'. <br>
<br>
Data-datanya sudah kita cluster/kelompokkan, namun kita masih belum mengetahui titik tengah dari masing masing cluster, mari cari nilainya!
```python
km_th = res.groupby(['Cluster']).mean().round(1)
#Kita gunakan round(1) agar angka decimalnya tidak terlalu panjang saat kita mencari nilai meannya
km_th
```
Selesai!, diatas merupakan titik tengahnya.
### Visualisasi hasil algoritma
```python
sns.scatterplot(x="Credit amount",y="Duration", hue="Cluster", data = res)
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/44ebb641-237a-4345-bc66-76905e321f90)
```python
sns.scatterplot(x="Duration",y="Age", hue="Cluster", data = res)
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/4fe09b6d-667a-43e6-952c-00a6efd2b3cd)
```python
sns.scatterplot(x="Credit amount",y="Age", hue="Cluster", data = res)
```
![download](https://github.com/alyatoon/clustering-credit-card/assets/149295614/da64f865-93a8-48a7-b61c-167e4a4fa331)

## Deployment
[Cluster Credit Card](https://clustering-credit-card-germany-alya.streamlit.app/) <br>
![image](https://github.com/alyatoon/clustering-credit-card/assets/149295614/3ca70c41-b367-49c4-8447-c5b5b65caf62)

