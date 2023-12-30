# Laporan Proyek Machine Learning
### Nama : Refadli Dwi Ilham
### Nim : 211351121
### Kelas : Pagi B

## Domain Proyek

Estimasi harga ponsel ini boleh digunakan sebagai patokan bagi semua orang yang ingin membeli atau menjual ponsel
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
