# UTS Deep Learning: Klasifikasi Citra Makanan Khas Yogyakarta

Klasifikasi citra tiga makanan khas Yogyakarta, yaitu **Bakpia**, **Gudeg**, dan **Sate Klathak**, dengan membandingkan dua pendekatan:

1. **Custom CNN**, dilatih dari nol (4 blok konvolusi, ±249 ribu parameter).
2. **MobileNetV2**, *transfer learning* dari ImageNet (feature extraction lalu fine-tuning).

Proyek ini terdiri dari dua notebook yang dijalankan berurutan:

| Notebook | Isi |
|---|---|
| `notebook/01_dataCleaning.ipynb` | Validasi & pembersihan citra, deteksi duplikat, grup anti-leakage, pembagian train/val/test |
| `notebook/02_modelling.ipynb` | Augmentasi, training dua model, eksperimen, evaluasi di test set, penyimpanan model |

---

## 1. Struktur Folder

```text
UTS/
├── data/
│   ├── raw/                         # foto mentah, TIDAK pernah diubah
│   │   ├── bakpia/
│   │   ├── gudeg/
│   │   └── sate_klathak/
│   ├── interim/                     # hasil konversi RGB + resize (sementara, dibuat ulang tiap cleaning)
│   │   ├── bakpia/
│   │   ├── gudeg/
│   │   └── sate_klathak/
│   └── split/                       # dataset final yang dipakai training
│       ├── train/<kelas>/
│       ├── val/<kelas>/
│       └── test/<kelas>/
│
├── notebook/
│   ├── 01_dataCleaning.ipynb
│   └── 02_modelling.ipynb
│
├── outputs/
│   ├── cleaning/                    # keluaran notebook 01
│   │   ├── manifest.csv             # jejak setiap foto: sumber, perangkat, status, alasan dibuang, grup, split
│   │   ├── tabel_alur_pembersihan.csv
│   │   ├── tabel_split.csv
│   │   ├── tabel_syarat.csv
│   │   ├── tabel_parameter.csv
│   │   ├── grafik_split.png
│   │   ├── contoh_citra_per_kelas.png
│   │   └── grids/                   # contact sheet per kelas & split, contoh: bakpia_test.jpg
│   └── modelling/                   # keluaran notebook 02
│       ├── custom_cnn_best.keras    # Model terbaik dari Custom CNN 
│       ├── mobilenetv2_best.keras   # Model terbaik dari MobileNetV2
│       ├── model_info.json          # urutan kelas, format input, versi TensorFlow
│       ├── history_custom_cnn.csv
│       ├── history_mobilenetv2.csv
│       ├── history_dropout_{0.0,0.3,0.5}.csv
│       ├── history_lr_{A,B}.csv
│       ├── eksperimen_dropout.{csv,png}
│       ├── eksperimen_strategi_training.{csv,png}
│       ├── kurva_training.png
│       ├── contoh_augmentasi.png
│       ├── confusion_matrix.png
│       ├── classification_report_custom_cnn.csv
│       ├── classification_report_mobilenetv2.csv
│       ├── tabel_efisiensi.csv
│       ├── tabel_perbandingan.csv
│       ├── prediksi_test.csv        # semua prediksi test + keyakinan + asal foto
│       ├── contoh_prediksi_custom_cnn.png
│       └── contoh_prediksi_mobilenetv2.png
│
├── requirements.txt
├── Strategi_Modelling_UTS_DL.md
├── summary.md
└── .gitignore
```

Catatan:
- Pada `data/raw/<kelas>/`, **foto dari internet** diletakkan langsung di dalam folder kelas, sedangkan **foto sendiri** diletakkan di subfolder. Notebook memakai aturan ini untuk menentukan kolom `source` (`internet` / `sendiri`). Folder atau file yang berawalan `.` atau `_` (misalnya `_dibuang`) dilewati.
- `data/interim/`, `data/split/`, dan `outputs/` dibuat otomatis oleh notebook. Setiap kali notebook 01 dijalankan, `interim/` dan `split/` dihapus lalu dibuat ulang dari awal.

---

## 2. Ringkasan Hasil

**Dataset akhir:** 388 citra (268 train, 59 val, 61 test) untuk 3 kelas, dan seluruh syarat jumlah minimal dataset terpenuhi.

**Hasil di test set (61 citra, dievaluasi satu kali):**

| Metrik | Custom CNN | MobileNetV2 |
|---|---|---|
| Test accuracy | 0,8361 | **0,9836** |
| Precision (macro) | 0,8807 | **0,9841** |
| Recall (macro) | 0,8341 | **0,9841** |
| F1-score (macro) | 0,8261 | **0,9837** |
| Jumlah citra salah | 10 dari 61 | **1 dari 61** |
| Total parameter | **249.283** | 2.261.827 |
| Waktu inferensi (ms/citra) | **4,7** | 25,9 |
| Ukuran file model | **3,1 MB** | 24,6 MB |

Kesimpulan singkat:
- **MobileNetV2 jauh lebih akurat** dan stabil di semua kelas (hanya 1 kesalahan: satu bakpia terprediksi sebagai sate klathak).
- **Custom CNN** unggul dalam ukuran dan kecepatan inferensi, tetapi lemah pada kelas **gudeg** (recall hanya 0,55). Sembilan dari sebelas kesalahannya berupa gudeg yang diprediksi sebagai sate klathak.
- Test set sengaja dibuat **lebih sulit** dengan menahan foto dari perangkat tertentu yang tidak pernah dilihat saat training. Hasilnya menguji kemampuan generalisasi, bukan sekadar hafalan.

---

## 3. Teknik Pengumpulan Dataset

### 3.1 Sumber data

Dataset dikumpulkan dari dua sumber:

- **Foto sendiri** (204 foto) diambil langsung dengan 4 perangkat berbeda.
- **Foto dari internet** (199 foto) dikumpulkan sebagai tambahan variasi.

| Kelas | Internet | Foto sendiri | Total mentah |
|---|---:|---:|---:|
| Bakpia | 55 | 71 | 126 |
| Gudeg | 64 | 63 | 127 |
| Sate Klathak | 80 | 70 | 150 |
| **Total** | **199** | **204** | **403** |

### 3.2 Perangkat pengambilan foto sendiri

Perangkat dan waktu pengambilan dibaca dari metadata **EXIF** (tag merek, model, dan waktu jepret). Foto internet umumnya tidak punya metadata ini.

| Kelas | Perangkat | Jumlah foto | Waktu pengambilan |
|---|---|---:|---|
| Bakpia | FUJIFILM X-T30 III | 21 | 21 Okt 2026 |
| Bakpia | Xiaomi Redmi Note 13 Pro 5G | 42 | 20 Okt 2026 |
| Bakpia | Samsung Galaxy A35 5G | 8 | 20 Okt 2026 |
| Gudeg | Xiaomi Redmi Note 13 Pro 5G | 51 | 20 Okt 2026 |
| Gudeg | Samsung Galaxy A35 5G | 12 | 20 Okt 2026 |
| Sate Klathak | FUJIFILM X-T30 III | 26 | 22 Okt 2026 |
| Sate Klathak | Xiaomi POCO X7 Pro | 14 | 22 Okt 2026 |
| Sate Klathak | Xiaomi Redmi Note 13 Pro 5G | 13 | 22 Okt 2026 |
| Sate Klathak | Samsung Galaxy A35 5G | 17 | 22 Okt 2026 |

Pengambilan foto dengan beberapa perangkat dan sesi berbeda menghasilkan variasi kualitas kamera, warna, dan resolusi. Variasi ini dimanfaatkan sebagai dasar pembagian test set (lihat bagian 4.1, langkah 6).

### 3.3 Syarat dataset

| Syarat | Nilai minimum | Hasil |
|---|---|---|
| Train + val per kelas | ≥ 100 | Bakpia 104, Gudeg 104, Sate Klathak 119 |
| Test per kelas | ≥ 20 | Bakpia 21, Gudeg 20, Sate Klathak 20 |
| Total citra | ≥ 360 | 388 |

---

## 4. Metode Pengerjaan

### 4.1 Notebook 01: Data Cleaning

| # | Langkah | Detail |
|---|---|---|
| 1 | **Pengecekan metadata** | Membaca perangkat dan waktu jepret dari EXIF. Seluruh 403 file terbaca (`status = ok`). |
| 2 | **Validasi & standarisasi** | Cek file rusak (`verify`), koreksi **rotasi EXIF**, konversi ke **RGB** (PNG transparan ditempel di latar putih), buang sisi terpendek < 128 px, perkecil sisi terpanjang menjadi maks 800 px (rasio tetap). Hasil disimpan sebagai PNG di `data/interim/`. |
| 3 | **Seleksi manual** | Dua foto dibuang setelah pengecekan visual karena menampilkan tusuk bambu, bukan sate klathak (`sate_066.jpg`, `sate_072.jpg`). |
| 4 | **Deteksi duplikat** | **Perceptual hash (pHash)** 64 bit, jarak Hamming ≤ 8 dianggap duplikat. Foto yang dipertahankan adalah foto sendiri terlebih dahulu, lalu yang resolusinya terbesar. Duplikat lintas kelas akan dibuang semua (hasilnya 0). |
| 5 | **Grup anti-leakage** | Foto digabung dalam satu grup jika (a) pHash ≤ 14, atau (b) jepretan beruntun dari perangkat yang sama dengan selisih ≤ 10 detik. Grup tidak boleh lintas kelas. |
| 6 | **Split per grup** | **Test dipisahkan lebih dulu** dari perangkat yang ditahan (*held-out device*). Sisanya dibagi train/val 80:20 dengan proporsi sumber (internet / sendiri) yang sama. Pembagian selalu per grup utuh, tidak per foto. |
| 7 | **Cek kebocoran** | Assertion otomatis: tiap grup hanya berada di satu split, jarak pHash antar split > 14, dan foto dari perangkat test tidak masuk train/val. |
| 8 | **Simpan & dokumentasi** | Salin ke `data/split/`, simpan `manifest.csv`, tabel ringkasan, grafik, dan contact sheet visual untuk pengecekan akhir. |

**Perangkat yang ditahan untuk test (tidak pernah dilihat saat training):**

| Kelas | Perangkat test |
|---|---|
| Bakpia | FUJIFILM X-T30 III |
| Gudeg | Samsung Galaxy A35 5G (ditambah foto internet hingga 20) |
| Sate Klathak | Samsung Galaxy A35 5G |

**Parameter cleaning:** `MAX_SIDE=800`, `MIN_SIDE=128`, `PHASH_DUP=8`, `PHASH_GROUP=14`, `BURST_GAP_SEC=10`, `VAL_FRAC=0.2`, `TEST_PER_CLASS=20`, `SEED=42`.

### 4.2 Notebook 02: Modelling

**Konfigurasi umum:** TensorFlow 2.21.0 (CPU), seed 42, input 224×224, batch 16, loss `sparse_categorical_crossentropy`, optimizer Adam.

**Augmentasi (hanya aktif saat training, berupa layer di dalam model):**

| Layer | Nilai |
|---|---|
| RandomFlip | horizontal |
| RandomRotation | 0,1 (±36°) |
| RandomZoom | 0,15 |
| RandomContrast | 0,15 |

Layer ini otomatis mati saat `evaluate()` dan `predict()`, sehingga data val dan test tidak pernah diaugmentasi (dibuktikan di notebook: perubahan piksel maksimum = 0 saat `training=False`).

**Model 1: Custom CNN (dari nol)**

```text
Input 224×224×3 → Augmentasi → Rescaling(1/255)
→ [Conv 3×3 (32) → MaxPool] → [Conv (64) → MaxPool]
→ [Conv (128) → MaxPool] → [Conv (128) → MaxPool]     # 224 → 14
→ GlobalAveragePooling2D → Dense(64, ReLU) → Dropout(0.5) → Dense(3, softmax)
```

- Total **249.283** parameter. GAP menggantikan Flatten (sekitar 249 ribu parameter vs ±1,85 juta).
- Adam lr 1e-3, maks 120 epoch, dengan tiga callback: `EarlyStopping` (patience 15), `ReduceLROnPlateau` (faktor 0,5, patience 6), dan `ModelCheckpoint` (val_loss terbaik).
- Run 1 (maks 60 epoch) belum konvergen sehingga diperpanjang. Run 2 yang dipakai berhenti di epoch 85 dengan epoch terbaik 70 (val_loss 0,192, val acc 0,932).

**Model 2: MobileNetV2 (transfer learning)**

```text
Input → Augmentasi → Rescaling([-1, 1]) → MobileNetV2 (ImageNet, include_top=False)
→ GlobalAveragePooling2D → Dropout(0.2) → Dense(3, softmax)
```

- **Fase 1, feature extraction:** backbone dibekukan, hanya head (**3.843** parameter) yang dilatih. Adam lr 1e-3, maks 30 epoch.
- **Fase 2, fine-tuning:** layer backbone ke-100 sampai akhir (54 dari 154 layer) dibuka sehingga **1.865.283** parameter dilatih. Learning rate diturunkan 100× menjadi 1e-5 dan model di-compile ulang. Bobot fase 2 hanya dipakai jika val_loss lebih baik dari fase 1.
- Backbone dipanggil dengan `training=False` agar BatchNorm tetap memakai statistik ImageNet.

**Eksperimen tambahan pada Custom CNN (dibandingkan hanya dengan validation):**

1. **Pengaruh Dropout** (0,0 / 0,3 / 0,5), dengan seed dan pengaturan yang sama.
2. **Strategi training**, yaitu LR tetap tanpa callback (A), LR tetap + EarlyStopping (B), dan model utama (LR turun + EarlyStopping).

**Disiplin evaluasi:** pemilihan model dan hyperparameter hanya memakai **validation**. Model dikunci (dimuat ulang dari file `.keras` dan dicek ulang di val) sebelum test set dipakai, dan test set hanya dievaluasi **satu kali**.

---

## 5. Hasil

### 5.1 Dataset setelah cleaning

**Alur pembersihan:**

| Kelas | Foto mentah | Duplikat dibuang | Dibuang manual | Lolos |
|---|---:|---:|---:|---:|
| Bakpia | 126 | 1 | 0 | 125 |
| Gudeg | 127 | 3 | 0 | 124 |
| Sate Klathak | 150 | 9 | 2 | 139 |
| **Total** | **403** | **13** | **2** | **388** |

**Jumlah per kelas dan split:**

| Kelas | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Bakpia | 83 | 21 | 21 | 125 |
| Gudeg | 90 | 14 | 20 | 124 |
| Sate Klathak | 95 | 24 | 20 | 139 |
| **Total** | **268** | **59** | **61** | **388** |

**Komposisi sumber per split:**

| Kelas | Sumber | Train | Val | Test |
|---|---|---:|---:|---:|
| Bakpia | internet | 44 | 11 | 0 |
| Bakpia | sendiri | 39 | 10 | 21 |
| Gudeg | internet | 43 | 10 | 8 |
| Gudeg | sendiri | 47 | 4 | 12 |
| Sate Klathak | internet | 56 | 14 | 0 |
| Sate Klathak | sendiri | 39 | 10 | 20 |

**Grup anti-leakage:** Bakpia 82 grup (terbesar 35 foto), Gudeg 69 grup (terbesar 48 foto), Sate Klathak 109 grup (terbesar 14 foto). Jarak pHash terkecil antar split adalah 16 (train↔val), 18 (train↔test), dan 20 (val↔test), semuanya di atas ambang grup 14, jadi **semua pengecekan anti-leakage lolos**.

### 5.2 Eksperimen pada Custom CNN (validation)

**Pengaruh Dropout:**

| Dropout | Epoch terbaik | Val loss | Val acc | Train acc | Gap acc (train − val) |
|---:|---:|---:|---:|---:|---:|
| 0,0 | 65 | 0,2697 | 0,9153 | 0,9627 | 0,0474 |
| 0,3 | 37 | 0,2938 | 0,8644 | 0,9067 | 0,0423 |
| **0,5** | 50 | **0,2483** | **0,9322** | 0,9067 | −0,0255 |

Dropout 0,5 memberikan val_loss terendah sehingga dipakai pada model utama.

**Strategi training:**

| Konfigurasi | Epoch dijalankan | Epoch model dipakai | Val loss | Val acc |
|---|---:|---:|---:|---:|
| A. LR tetap, tanpa callback | 120 | 120 (epoch terakhir) | 0,1798 | 0,9661 |
| B. LR tetap + EarlyStopping | 100 | 85 | 0,1653 | 0,9322 |
| Model utama (LR turun + EarlyStopping) | 85 | 70 | 0,1917 | 0,9322 |

Pada percobaan ini, tambahan `ReduceLROnPlateau` tidak menghasilkan val_loss yang lebih baik daripada konfigurasi B. Model utama dipertahankan sesuai keputusan yang sudah dikunci sebelum test dibuka.

### 5.3 Efisiensi

| | Custom CNN | MobileNetV2 |
|---|---|---|
| Total parameter | 249.283 | 2.261.827 |
| Parameter dilatih | 249.283 (semua) | 3.843 (fase 1) / 1.865.283 (fase 2) |
| Epoch dijalankan / terbaik | 85 / 70 | 60 / 58 (fase 1: 30, fase 2: 30) |
| Waktu training | 251,3 dtk | 211,2 dtk |
| Waktu per epoch | 2,96 dtk | 3,52 dtk |
| Waktu inferensi | 4,7 ms/citra | 25,9 ms/citra |
| Val accuracy (epoch terbaik) | 0,9322 | 1,0000 |

### 5.4 Evaluasi di test set

**Custom CNN**

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Bakpia | 0,9524 | 0,9524 | 0,9524 | 21 |
| Gudeg | 1,0000 | 0,5500 | 0,7097 | 20 |
| Sate Klathak | 0,6897 | 1,0000 | 0,8163 | 20 |
| **Accuracy** | | | **0,8361** | 61 |

**MobileNetV2**

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Bakpia | 1,0000 | 0,9524 | 0,9756 | 21 |
| Gudeg | 1,0000 | 1,0000 | 1,0000 | 20 |
| Sate Klathak | 0,9524 | 1,0000 | 0,9756 | 20 |
| **Accuracy** | | | **0,9836** | 61 |

**Akurasi berdasarkan asal foto test (jumlah benar / total):**

| Kelas | Asal | Jumlah | Custom CNN | MobileNetV2 |
|---|---|---:|---:|---:|
| Bakpia | FUJIFILM X-T30 III | 21 | 20 | 20 |
| Gudeg | Internet | 8 | 5 | 8 |
| Gudeg | Samsung Galaxy A35 5G | 12 | 6 | 12 |
| Sate Klathak | FUJIFILM X-T30 III | 3 | 3 | 3 |
| Sate Klathak | Samsung Galaxy A35 5G | 17 | 17 | 17 |

### 5.5 Analisis kesalahan

- **Custom CNN** salah pada 10 citra: 9 gudeg (8 diprediksi sate klathak, 1 sebagai bakpia) dan 1 bakpia (diprediksi sate klathak). Karena banyak gudeg yang "jatuh" ke sate klathak, precision sate klathak turun menjadi 0,69 meskipun recall-nya 1,0. Kesalahan gudeg terkonsentrasi pada foto Samsung A35 (6 dari 12 salah) dan foto internet (3 dari 8 salah), sedangkan foto bakpia dari FUJIFILM hampir seluruhnya benar.
- **MobileNetV2** hanya salah pada 1 citra, yaitu `bakpia_012.png` (foto FUJIFILM) yang diprediksi sate klathak dengan keyakinan 0,99.
- Daftar lengkap prediksi, keyakinan, dan asal foto ada di `outputs/modelling/prediksi_test.csv`, dan visualisasi contoh benar/salah ada di `contoh_prediksi_*.png`.

### 5.6 Keterbatasan

- Test set kecil (61 citra), sehingga satu citra salah mengubah akurasi sekitar 1,6 poin. Selisih kecil antar-eksperimen sebaiknya tidak ditafsirkan berlebihan.
- Validation juga kecil (59 citra, dan hanya 14 untuk gudeg), sehingga val accuracy 100% pada MobileNetV2 belum menjamin performa yang sama di data baru.
- Seluruh foto sendiri diambil dalam satu sesi per kelas, jadi variasi latar, pencahayaan, dan penyajian masih terbatas.
- Training dilakukan di CPU dengan satu seed (42). Hasil dapat sedikit berbeda jika seed atau lingkungan diubah (misalnya hasil dropout 0,5 pada eksperimen sedikit berbeda dari model utama).

---

## 6. Cara Menjalankan

1. Siapkan environment Python dan instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
   Pustaka utama: `tensorflow` (2.21.0), `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pillow`, `pillow-heif`, dan `imagehash`.
2. Letakkan foto mentah di `data/raw/<bakpia | gudeg | sate_klathak>/`. Foto internet langsung di folder kelas, foto sendiri di subfolder.
3. Jalankan `notebook/01_dataCleaning.ipynb` dari awal sampai akhir. Cukup **sekali** setiap dataset mentah berubah, karena `data/split/` dibuat ulang.
4. Jalankan `notebook/02_modelling.ipynb`. Hasil dan model tersimpan di `outputs/modelling/`.

**Memakai model yang sudah dilatih:**

```python
import tensorflow as tf

CLASS_NAMES = ["bakpia", "gudeg", "sate_klathak"]   # urutan sama dengan model_info.json
model = tf.keras.models.load_model("outputs/modelling/mobilenetv2_best.keras")

img = tf.io.decode_image(tf.io.read_file("foto.jpg"), channels=3, expand_animations=False)
arr = tf.image.resize(img, (224, 224))[None, ...]   # piksel 0-255, normalisasi sudah di dalam model
probs = model.predict(arr)[0]
print(CLASS_NAMES[probs.argmax()], probs.round(3))
```
