# Content-Based Movie Recommender System

Sistem rekomendasi film berbasis konten (**Content-Based Filtering**) yang menggunakan **TF-IDF** pada fitur genre dan **Cosine Similarity** untuk menghasilkan rekomendasi film yang dipersonalisasi.

Dibangun dengan **Python**, **Flask**, dan **HTML/Tailwind CSS**.


## DATASET
```
Harap mengunduh dataset https://www.kaggle.com/datasets/abdallahwagih/movies terlebih dahulu, simpan dataset movie.csv dan rating.csv pada folder project ini jangan semua
```
---
## Struktur Proyek

```
MovieRecomendation/
├── app.py              # Backend Flask + pipeline rekomendasi
├── movie.csv           # Dataset film (movieId, title, genres)
├── rating.csv          # Dataset rating (userId, movieId, rating, timestamp)
├── requirements.txt    # Dependensi Python
├── README.md           # Dokumentasi
├── templates/
│   └── index.html      # Halaman web utama
└── static/
    └── style.css       # Custom CSS (animasi, scrollbar, dsb.)
```

---

## Cara Menjalankan

### Install dependensi

```bash
pip install -r requirements.txt
```

### Jalankan aplikasi

```bash
python app.py
```

## Metode Sistem (Step-by-Step)

### Step 1: Load Dataset
- Membaca `movie.csv` dan `rating.csv` menggunakan **pandas**.
- Menampilkan informasi dasar (jumlah baris, kolom).
- Mendukung **sampling** rating untuk dataset besar (default: 500.000 baris).

### Step 2: Preprocessing Data
- Menghapus film dengan genre kosong atau `"(no genres listed)"`.
- Mengubah format genre: `"Action|Adventure|Sci-Fi"` → `"Action Adventure Sci-Fi"`.
- Memastikan tipe data `movieId` konsisten (`int`).
- Menghapus rating yang `movieId`-nya tidak ada di data movie.
- Menangani missing values.

### Step 3: Data Splitting
- Membagi rating menjadi **train (80%)** dan **test (20%)** menggunakan `train_test_split` dari scikit-learn.
- Train → membentuk user profile.
- Test → evaluasi (ground truth).

### Step 4: TF-IDF Embedding
- Menggunakan `TfidfVectorizer` pada kolom genre yang sudah dibersihkan.
- Menghasilkan matriks TF-IDF: setiap film direpresentasikan sebagai vektor numerik.

### Step 5: Pembentukan User Profile
- Input: **User ID** sebagai active user.
- Film dianggap **disukai** jika `rating >= 4.0` pada data train.
- Ambil vektor TF-IDF dari film yang disukai.
- Hitung **rata-rata vektor** sebagai **user profile vector**.

### Step 6: Perhitungan Similarity
- Hitung **cosine similarity** antara user profile vector dan semua film.
- Urutkan berdasarkan similarity score tertinggi.
- **Tidak** merekomendasikan film yang sudah pernah di-rating user.
- Tampilkan **Top-N** film.

### Step 7: Evaluasi Sistem
- **Precision@K** = jumlah rekomendasi relevan / K
- **Recall@K** = jumlah rekomendasi relevan / total item relevan di test
- Ground truth: film pada data test dengan `rating >= 4.0`.

### Step 8: User Interface
- Web interface menggunakan Flask + HTML + Tailwind CSS.
- Input: User ID dan Top-N.
- Output: Tabel rekomendasi + metrik evaluasi.

---

## Konfigurasi

Konfigurasi dapat diubah di bagian atas `app.py`:

| Parameter | Default | Keterangan |
|---|---|---|
| `RATING_SAMPLE_SIZE` | `....` | Jumlah sampel rating (None = semua) |
| `LIKE_THRESHOLD` | `4.0` | Rating minimum agar film dianggap disukai |
| `TEST_SIZE` | `0.2` | Proporsi data test |
| `RANDOM_STATE` | `42` | Seed untuk reproducibility |

---

## Library yang Digunakan

| Library | Fungsi |
|---|---|
| `pandas` | Membaca dan memanipulasi dataset |
| `numpy` | Operasi numerik pada vektor |
| `scikit-learn` | TF-IDF, cosine similarity, train/test split |
| `flask` | Web framework untuk UI |

---

## Catatan Teknis

- **Tidak** menghitung cosine similarity full matrix antar semua film (O(n²)).
- Menggunakan pendekatan **user profile vector** sehingga hanya perlu menghitung similarity 1 vektor vs seluruh film (O(n)).
- Mendukung **sampling rating** agar tetap responsif pada dataset besar (MovieLens 20M).
- Kode dilengkapi **komentar jelas** di setiap tahap untuk kemudahan pemahaman.
