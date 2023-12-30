# Laporan Proyek Machine Learning
### Nama : Refadli Dwi Ilham
### Nim : 211351121
### Kelas : Pagi B

## Domain Proyek
Evaluasi Mobil ini bisa digunakan sebagai patokan bagi semua yang ingin mengetahui penilaian kelas yang sesuai dengan kebutuhan konsumen maupun bisnis untuk menjamin kepuasanmya
## Business Understanding

Industri otomotif  berkembang seiring dengan perubahan kebutuhan konsumen akan kendaraan pribadi.
Evaluasi kendaraan merupakan aspek penting dalam proses  keputusan kelas yang baik bagi konsumen individu maupun bisnis.
Oleh karena itu, diperlukan suatu metode evaluasi yang komprehensif yang dapat memberikan informasi yang akurat kepada konsumen untuk menjamin kepuasan dan nilai investasi yang optimal.

Bagian laporan ini mencakup:

### Problem Statements

Namun, saat ini ada beberapa masalah pada penilaian mobil.
Konsumen seringkali  kesulitan memperoleh informasi yang lengkap dan terkini mengenai berbagai model kelas kendaraan yang ada di pasaran.
Selain itu, beragamnya kebutuhan dan preferensi konsumen seringkali tidak sesuai dengan rekomendasi yang diberikan oleh sumber informasi otomotif.
Oleh karena itu, Anda memerlukan solusi yang menyederhanakan proses evaluasi kendaraan dan memberikan rekomendasi kelas yang lebih disesuaikan dengan kebutuhan pribadi Anda.

### Goals

Tujuan utama dari proyek evaluasi kendaraan ini adalah untuk menyediakan platform yang dapat memberikan informasi yang akurat dan relevan tentang berbagai kelas model kendaraan yang tersedia di pasar.
Platform tersebut juga bertujuan untuk memberikan pemahaman lebih mendalam mengenai kelas kebutuhan dan preferensi konsumen, sehingga rekomendasi yang diberikan  lebih relevan dan memenuhi harapan konsumen.
Perusahaan juga bertujuan untuk meningkatkan pengalaman pemilihan kelas mobil bagi konsumen dan berkontribusi pada pertumbuhan industri otomotif secara keseluruhan.
    ### Solution statements
    - Pengembangan Platform Pencarian Harga yang cocok untuk membeli atau menjual ponsel Berbasis Web, Solusi pertama adalah mengembangkan platform pencarian Harga yang cocok untuk membeli atau menjual ponsel mengintegrasikan data dari Kaggle.com untuk memberikan pengguna akses cepat dan mudah ke informasi tentang estimasi Harga yang cocok untuk membeli atau menjual ponsel
    - Model yang dihasilkan dari datasets itu menggunakan metode Linear Regression.

## Data Understanding
Dataset yang saya gunakan berasal jadi Kaggle yang berisi Kelas Evaluasi Mobil.Dataset ini mengandung 1726 baris dan lebih dari 7 columns.

https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set

### Variabel-variabel sebagai berikut:
Buying     = Pembelian
maint      = pemeliharaan
doors      = pintu
persons    = orang
lug_boot   =
safety     = keamanan
class      = kelas
## Data Preparation
# IMPORT DATASET
```python
from google.colab import files
files.upload()
```
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
```python
!kaggle datasets download -d elikplim/car-evaluation-data-set
```
```python
!mkdir car-evaluation-data-set
!unzip car-evaluation-data-set.zip -d car-evaluation-data-set
!ls car-evaluation-data-set
```
# IMPORT LIBRARY
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
```
# DATA DISCOVERY
```python
df = pd.read_csv('car-evaluation-data-set/car_evaluation.csv')
df.sample()
```
```python
df.head()
```
```python
df.describe()
```
```python
df.info()
```
```python
df.isna().sum()
```
```python
df.nunique()
```
```python
df.duplicated().sum()
```
```python
df
```
# EDA
```python
sns.heatmap(df.isnull())
```
![image](ml1.png)
```python
sns.countplot(data=df,x='vhigh',hue='unacc')
plt.xticks(rotation=45, ha='right');
```
![image](ml2.png)
```python
sns.countplot(data=df,x='vhigh.1',hue='unacc')
plt.xticks(rotation=45, ha='right');
```
![image](ml3.png)
```python
sns.histplot(x="class",data=df ,color = 'rosybrown')
```
![image](ml4.png)
```python

```
```python

```
```python

```
```python

```
```python

```
```python

```
```python

```
