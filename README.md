# Laporan Proyek Machine Learning
### Nama : Refadli Dwi Ilham
### Nim : 211351121
### Kelas : Pagi B

## Domain Proyek

Estimasi harga ponsel ini boleh digunakan sebagai patokan bagi semua orang yang ingin membeli atau menjual ponsel
## Business Understanding

Lebih menghemat waktu agar tidak perlu menanyakan harga yang cocok untuk menjual atau membeli ponsel

Bagian laporan ini mencakup:

### Problem Statements

- Tidak mungkin seseorang yang ingin menjual atau membeli ponsel harus menanyakan kepada setiap orang yang memiliki ponsel agar tau harga yang pas

### Goals

- mencari solusi untuk memudahkan orang-orang yang mencari harga yang cocok untuk menjual atau membeli ponsel


    ### Solution statements
    - Pengembangan Platform Pencarian Harga yang cocok untuk membeli atau menjual ponsel Berbasis Web, Solusi pertama adalah mengembangkan platform pencarian Harga yang cocok untuk membeli atau menjual ponsel mengintegrasikan data dari Kaggle.com untuk memberikan pengguna akses cepat dan mudah ke informasi tentang estimasi Harga yang cocok untuk membeli atau menjual ponsel
    - Model yang dihasilkan dari datasets itu menggunakan metode Linear Regression.

## Data Understanding
Dataset yang saya gunakan berasal jadi Kaggle yang berisi Harga yang cocok untuk membeli atau menjual ponsel.Dataset ini mengandung 162 baris dan lebih dari 14 columns.

kaggle datasets download -d mohannapd/mobile-price-prediction 

### Variabel-variabel sebagai berikut:
- Sale  : Penjualan Ponsel(int64)
- weight    : Berat Ponsel(int64)
- ppi       : Ukuran Resolusi Pada Layar Ponsel(int64)
- cpu core  : Processor CPU(int64)
- ram       : memori jangka pendek(int64)
- Front_Cam : Kamera Depan(int64)
- battery   : Penyimpan Daya Listrik(int64)
- thickness : Ketebalan Ponsel(int64)
- price     : Harga Ponsel(int64)

## Data Preparation
