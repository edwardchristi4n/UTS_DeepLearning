# Ringkasan Proyek UTS Deep Learning
**Klasifikasi Citra Makanan Khas Yogyakarta: Bakpia, Gudeg, Sate Klathak**
Model 1: Custom CNN (dari nol) · Model 2: MobileNetV2 (pretrained ImageNet, bekukan lalu fine-tuning)
Framework: TensorFlow 2.21 / Keras 3 · Python 3.12 · CPU laptop

---

## 0. Gambaran singkat

| Tahap | Notebook | Hasil utama |
|---|---|---|
| Cleaning dan split | `notebook/01_dataCleaning.ipynb` | 403 foto mentah → **388 foto bersih** (train 268 · val 59 · test 61), bebas kebocoran antar split |
| Modelling dan evaluasi | `notebook/02_modelling.ipynb` | **MobileNetV2 test accuracy 0,984**, Custom CNN 0,836 |

**Model terpilih: MobileNetV2 dengan transfer learning dua fase.**

Struktur folder:
```
UTS/
├── data/
│   ├── raw/<kelas>/               foto mentah (internet langsung di folder kelas, foto sendiri di subfolder)
│   ├── interim/<kelas>/           hasil konversi + resize (sementara)
│   └── split/{train,val,test}/<kelas>/   dataset final yang dipakai training
├── notebook/01_dataCleaning.ipynb, 02_modelling.ipynb
├── outputs/cleaning/              manifest.csv, tabel & grafik BAB III, grid visual
├── outputs/modelling/             model .keras, history, grafik, tabel, prediksi test
├── Strategi_Modelling_UTS_DL.md   rencana awal modelling
└── requirements.txt
```

---

## 1. Dataset

### 1.1 Sumber dan jumlah mentah

| Kelas | Internet | Foto sendiri | Total |
|---|---|---|---|
| bakpia | 55 | 71 | 126 |
| gudeg | 64 | 63 | 127 |
| sate_klathak | 80 | 70 | 150 |
| **Total** | **199** | **204** | **403** |

- **Foto internet** dikumpulkan manual satu per satu dari pencarian web, bukan dari dataset publik seperti Kaggle atau Roboflow (ketentuan B.2).
- **Foto sendiri** diambil dengan 4 perangkat. Data perangkat dan waktu dibaca dari EXIF:

| Kelas | Perangkat (jumlah foto) | Waktu |
|---|---|---|
| bakpia | Redmi Note 13 Pro (42), Samsung A35 (8) | 20 Okt, ±4 menit |
| | **FUJIFILM X-T30 III (21)** | 21 Okt pagi, latar berbeda (batu, meja besi, kaca) |
| gudeg | Redmi Note 13 Pro (51), **Samsung A35 (12)** | 20 Okt, ±12 menit, satu warung |
| sate_klathak | FUJIFILM (26), POCO X7 Pro (14), Redmi (13), **Samsung A35 (17)** | 22 Okt, ±8 menit, satu warung |

Foto sendiri gudeg dan sate hanya berasal dari **satu sesi pemotretan** di satu warung. Ini menjadi dasar strategi pemisahan test (bagian 2.5).

---

## 2. Cleaning dan pembagian data (`01_dataCleaning.ipynb`)

### 2.1 Langkah dan metode

| Langkah | Metode | Parameter | Hasil |
|---|---|---|---|
| 1. Inventaris dan metadata | Scan folder; sumber dari posisi file (langsung di folder kelas = internet, di subfolder = sendiri); baca EXIF perangkat (tag 271/272) dan waktu jepret (tag 36867) | — | 403 foto, semua bisa dibaca; 0 foto sendiri tanpa waktu EXIF |
| 2. Validasi dan standarisasi | `verify()` untuk file rusak, `exif_transpose` untuk memperbaiki rotasi, konversi ke RGB (PNG transparan → latar putih), filter ukuran, resize dengan rasio tetap, simpan PNG | `MIN_SIDE=128`, `MAX_SIDE=800` | 401 lolos; 2 dibuang manual (`sate_066`, `sate_072`: tusuk bambu, bukan klathak); foto terkecil 183 px |
| 3. Hapus duplikat | Perceptual hash (pHash 64 bit), jarak Hamming, union-find; simpan foto sendiri dulu, lalu resolusi terbesar; duplikat lintas kelas → dibuang semua | `PHASH_DUP=8` | 13 duplikat dibuang (bakpia 1, gudeg 3, sate 9); 0 lintas kelas |
| 4. Grup anti-leakage | Foto digabung satu grup jika (a) mirip: pHash ≤ 14, atau (b) jepretan beruntun: perangkat sama dan jeda ≤ 10 detik. Grup tidak boleh lintas kelas | `PHASH_GROUP=14`, `BURST_GAP_SEC=10` | 113 pasangan beruntun digabung; jumlah grup: bakpia 82, gudeg 69, sate 109; grup terbesar: gudeg 48, bakpia 35, sate 14 |
| 5. Split | Test = semua grup berisi foto dari **perangkat yang ditahan**, ditambah grup internet bila < 20. Train/val 80:20 **per grup utuh**, terpisah untuk grup internet dan grup foto sendiri | `TEST_DEVICE`, `TEST_PER_CLASS=20`, `VAL_FRAC=0.2`, `SEED=42` | lihat 2.2 |
| 6. Cek anti-leakage | `assert`: tidak ada grup terpecah; jarak pHash antar split > 14; perangkat test tidak ada di train/val; syarat jumlah | — | **semua lolos**; jarak pHash terkecil antar split 16–20 |
| 7. Simpan | Salin ke `data/split/<split>/<kelas>/kelas_NNN.png` (nomor unik per kelas lintas split); `manifest.csv` mencatat seluruh 403 foto | — | 388 foto tersimpan |
| 8–9. Cek visual dan ringkasan | Grid per kelas × split; tabel dan grafik untuk BAB III | — | `outputs/cleaning/` |

**Alasan aturan jepretan beruntun dihitung per perangkat:** foto sate diambil serentak oleh 4 perangkat dengan jeda rata-rata ±4 detik. Kalau dihitung per folder, foto dari HP berbeda ikut tersambung dan grupnya membengkak (gudeg sampai 56 dari 63 foto).

### 2.2 Hasil pembagian data

| Kelas | Train | Val | Test | Dev (train+val) |
|---|---|---|---|---|
| bakpia | 83 | 21 | 21 | 104 |
| gudeg | 90 | 14 | 20 | 104 |
| sate_klathak | 95 | 24 | 20 | 119 |
| **Total** | **268** | **59** | **61** | **327** |

Semua ketentuan B.3–B.4 terpenuhi: dev ≥ 100 per kelas, test ≥ 20 per kelas, total 388 ≥ 360.

### 2.3 Komposisi sumber

| Kelas | Sumber | Train | Val | Test |
|---|---|---|---|---|
| bakpia | internet / sendiri | 44 / 39 | 11 / 10 | 0 / 21 |
| gudeg | internet / sendiri | 43 / 47 | 10 / 4 | 8 / 12 |
| sate_klathak | internet / sendiri | 56 / 39 | 14 / 10 | 0 / 20 |

### 2.4 Asal test set

| Kelas | Isi test |
|---|---|
| bakpia | 21 foto FUJIFILM (sesi berbeda, latar baru) |
| gudeg | 12 foto Samsung + 8 foto internet |
| sate_klathak | 17 foto Samsung + 3 foto FUJIFILM (satu grup dengan foto Samsung) |

### 2.5 Kenapa test diambil dari perangkat yang ditahan

Tidak ada kesempatan memotret di warung lain. Karena itu, sumber yang paling terpisah dari data train adalah **perangkat dan sesi yang berbeda**: foto dari perangkat ini tidak pernah dilihat model saat training. Pendekatan ini serupa dengan pemisahan per subjek atau per lokasi untuk menghindari kebocoran data (Yagis dkk., 2021; Roberts dkk., 2017; Beery dkk., 2018).

---

## 3. Modelling (`02_modelling.ipynb`)

### 3.1 Pipeline input

- `image_dataset_from_directory` membaca `data/split`, ukuran 224×224, batch 16, label integer. Urutan kelas: bakpia = 0, gudeg = 1, sate_klathak = 2.
- Train diacak setiap epoch. Val dan test tidak diacak supaya urutan prediksi cocok dengan nama file.
- Piksel tetap 0–255 di pipeline. Normalisasi dilakukan **di dalam masing-masing model**.

### 3.2 Augmentasi

Augmentasi berupa layer Keras dan hanya aktif saat training (ketentuan B.6):

| Layer | Nilai |
|---|---|
| RandomFlip | horizontal |
| RandomRotation | 0,1 (±36°) |
| RandomZoom | 0,15 |
| RandomContrast | 0,15 |

Bukti bahwa augmentasi tidak aktif di luar training: perubahan piksel saat `training=False` = **0,0**. Augmentasi tidak menambah file, jadi tidak ikut dihitung dalam jumlah minimum.

### 3.3 Model 1: Custom CNN (dari nol, dikembangkan dari Practice Lab 3)

```
Input 224×224×3 → Augmentasi → Rescaling(1/255)
→ [Conv2D 3×3 ReLU same → MaxPool 2×2] × 4   (filter 32, 64, 128, 128; 224 → 14)
→ GlobalAveragePooling2D → Dense(64, ReLU) → Dropout(0,5) → Dense(3, softmax)
```

- **Parameter: 249.283**, semuanya dilatih dari nol, tanpa bobot pretrained (ketentuan D).
- **GlobalAveragePooling dipakai alih-alih Flatten.** Dengan Flatten, model akan berisi ±1,85 juta parameter (87% di satu layer Dense), sehingga rawan overfitting.
- **Training:**
  - Adam lr 1e-3, `sparse_categorical_crossentropy`, maksimal 120 epoch.
  - EarlyStopping (val_loss, patience 15, `restore_best_weights`).
  - ReduceLROnPlateau (×0,5, patience 6, min 1e-5).
  - ModelCheckpoint.

### 3.4 Model 2: MobileNetV2 (transfer learning)

```
Input → Augmentasi (sama) → Rescaling(1/127,5, offset −1) → MobileNetV2 (ImageNet, include_top=False, training=False)
→ GlobalAveragePooling2D → Dropout(0,2) → Dense(3, softmax)
```

| | Fase 1: backbone beku | Fase 2: fine-tuning |
|---|---|---|
| Yang dilatih | head saja (**3.843** parameter) | layer ke-100 sampai 154 (54 layer) + head (**1.865.283** parameter) |
| Optimizer | Adam 1e-3 | Adam **1e-5**, compile ulang |
| Epoch maks / EarlyStopping / ReduceLR | 30 / patience 8 / patience 4 | 30 / patience 8 / patience 4 |
| Pengaman | — | checkpoint hanya ditimpa jika val_loss lebih baik dari fase 1 |

- Total parameter: **2.261.827**.
- `training=False` membuat BatchNorm di backbone tetap memakai statistik ImageNet.

---

## 4. Proses tuning dan eksperimen (semua keputusan hanya berdasarkan validation)

### 4.1 Penyesuaian Custom CNN

| Run | Epoch maks | Patience ES / LR | Berhenti | Epoch terbaik | val_loss | val acc |
|---|---|---|---|---|---|---|
| 1 | 60 | 10 / 4 | 60 (batas) | 60 → **belum konvergen** | 0,302 | 0,881 |
| **2 (dipakai)** | 120 | 15 / 6 | 85 (EarlyStopping) | **70** | **0,192** | **0,932** |

Pada run 2, setelah epoch 70, val_loss naik (0,19 → 0,25) sementara train loss terus turun. Ini tanda awal overfitting, dan EarlyStopping mengembalikan bobot ke epoch 70.

### 4.2 Eksperimen dropout (meniru Practice Lab 3)

| Dropout | Epoch terbaik | val_loss | val acc | train acc | Selisih acc (train − val) |
|---|---|---|---|---|---|
| 0,0 | 65 | 0,270 | 0,915 | 0,963 | +0,047 |
| 0,3 | 37 | 0,294 | 0,864 | 0,907 | +0,042 |
| **0,5** | 50 | **0,248** | **0,932** | 0,907 | −0,026 |

- **Keputusan: dropout tetap 0,5.**
- Variasi antar-run dengan pengaturan yang sama sekitar ±0,05 val_loss (run utama 0,192 vs ulangan 0,248). Selisih antar-dropout (≤ 0,05) masih di dalam variasi ini.
- Bukti yang lebih kuat ada di kurva: dropout 0 paling cepat menunjukkan train loss < val loss (menghafal), sedangkan dropout 0,5 paling lama menunda overfitting.

### 4.3 Eksperimen strategi training (manfaat callback)

| Konfigurasi | Epoch | Model dipakai | val_loss | val acc | Waktu |
|---|---|---|---|---|---|
| A. LR tetap 1e-3, tanpa callback | 120 | epoch 120 (terakhir) | 0,180 | 0,966 | 381 detik |
| B. LR tetap + EarlyStopping | 100 | epoch 85 | 0,165 | 0,932 | 326 detik |
| **Model utama** (LR turun + ES) | 85 | epoch 70 | 0,192 | 0,932 | **251 detik** |

Temuan:
- **Ketiganya setara.** Selisihnya masih di dalam variasi antar-run.
- **Tanpa EarlyStopping pun belum terjadi overfitting jelas dalam 120 epoch.** Augmentasi dan dropout sudah menahan overfitting.
- **Learning rate tetap menghasilkan kurva val yang bergejolak** (B: lonjakan val_loss sampai 1,4).
- **Model utama paling stabil dan paling cepat.**

Keputusan: tetap memakai model utama.

### 4.4 MobileNetV2: fase 1 vs fase 2

| Fase | Epoch terbaik | val_loss | val acc | Waktu |
|---|---|---|---|---|
| 1 (beku) | 25 / 30 | 0,0302 | 1,000 | 82,7 detik |
| **2 (fine-tuning, dipakai)** | 58 / 60 | **0,0193** | 1,000 | 128,5 detik |

Fine-tuning menurunkan val_loss sekitar 36%: model menebak dengan lebih yakin, sedangkan accuracy val sudah 100% sejak fase 1. Di akhir fase 2, val_loss sudah datar (0,0193–0,0196) dengan learning rate 5e-6.

### 4.5 Penguncian model

Sebelum test dibuka, kedua model dimuat dari file `.keras` dan dicek ulang di val:
- Custom CNN: val_loss 0,1917, val acc 0,9322
- MobileNetV2: val_loss 0,0193, val acc 1,0000

Keduanya identik dengan history. Setelah itu tidak ada lagi perubahan model, dan test dievaluasi **satu kali**.

---

## 5. Hasil akhir pada test set

### 5.1 Tabel perbandingan

| | Custom CNN | **MobileNetV2** |
|---|---|---|
| Total parameter | 249.283 | 2.261.827 |
| Parameter dilatih | 249.283 | 3.843 (fase 1) / 1.865.283 (fase 2) |
| Epoch dijalankan / terbaik | 85 / 70 | 60 / 58 |
| Waktu training | 251,3 detik (2,96 detik/epoch) | **211,2 detik** (3,52 detik/epoch) |
| Waktu inferensi | **4,7 ms/citra** | 25,9 ms/citra |
| Val accuracy | 0,9322 | 1,0000 |
| **Test accuracy** | 0,8361 (51/61) | **0,9836 (60/61)** |
| Precision (macro) | 0,8807 | 0,9841 |
| Recall (macro) | 0,8341 | 0,9841 |
| F1-score (macro) | 0,8261 | 0,9837 |
| Recall bakpia / gudeg / sate | 0,95 / **0,55** / 1,00 | 0,95 / 1,00 / 1,00 |

### 5.2 Confusion matrix (baris = label, kolom = prediksi; urutan bakpia, gudeg, sate)

```
Custom CNN              MobileNetV2
[20   0   1]            [20   0   1]
[ 1  11   8]            [ 0  20   0]
[ 0   0  20]            [ 0   0  20]
```

### 5.3 Akurasi per asal foto test

| Kelas | Asal | Jumlah | Benar CNN | Benar MNV2 |
|---|---|---|---|---|
| bakpia | FUJIFILM (latar baru) | 21 | 20 | 20 |
| gudeg | internet | 8 | 5 | 8 |
| gudeg | Samsung (warung sama) | 12 | **6** | 12 |
| sate | FUJIFILM + Samsung | 20 | 20 | 20 |

### 5.4 Analisis kesalahan

- **Custom CNN, gudeg → sate (8 foto):**
  - Foto Samsung `gudeg_015`–`020` (piring biru-putih, krecek/kerupuk, meja kayu cokelat) dan foto internet bersantan cokelat.
  - Keyakinan model rendah (0,47–0,69), artinya model ragu.
  - Kemungkinan penyebab: warna dan tekstur cokelat gudeg mirip daging sate, meja kayu yang juga muncul di foto sate sendiri, dan val gudeg dari foto sendiri hanya 4 sehingga kelemahan ini tidak terlihat saat training.
- **MobileNetV2, `bakpia_012` → sate (keyakinan 0,99):**
  - Dua bakpia kecil di atas **meja besi gelap berlubang** yang mirip panggangan.
  - Pada `bakpia_011` (meja yang sama, bakpia lebih besar), model benar dengan keyakinan 1,00.
  - Bukti bahwa **ketika objeknya kecil, latar ikut mendominasi keputusan** (*shortcut learning*, Geirhos dkk., 2020).
- **Foto yang sama, hasil berbeda:** `gudeg_016` benar di MobileNetV2 (1,00) tetapi salah di Custom CNN (0,47). Ini bukti langsung perbedaan kualitas fitur kedua model.

---

## 6. Jawaban ringkas pertanyaan analisis wajib

1. **Performa terbaik:** MobileNetV2 (test accuracy 0,984, F1 0,984) dibanding Custom CNN (0,836, F1 0,826).
2. **Pengaruh arsitektur:** fitur ImageNet pada MobileNetV2 lebih kaya dan lebih tahan terhadap foto yang berbeda dari data latih. Selisih val → test: Custom CNN turun ±10 poin (0,93 → 0,84), MobileNetV2 hanya ±1,6 poin (1,00 → 0,98). Custom CNN cukup untuk kelas berbentuk khas (bakpia, sate), tapi lemah di gudeg yang visualnya kompleks.
3. **Model lebih kompleks selalu lebih akurat?** Di sini model yang lebih besar memang lebih akurat, tetapi keunggulannya terutama berasal dari **bobot pretrained**. Dengan hanya 3.843 parameter yang dilatih (fase 1), MobileNetV2 sudah mencapai val acc 1,0. Eksperimen dropout juga menunjukkan kapasitas berlebih tanpa regularisasi mempercepat overfitting. Pembuktian tegas memerlukan eksperimen MobileNetV2 dari nol (lihat bagian 9).
4. **Overfitting/underfitting:**
   - Custom CNN: tanda awal overfitting setelah epoch 70, yang ditangani EarlyStopping. Generalisasi ke test turun (0,93 → 0,84).
   - MobileNetV2: tidak overfit (train/val/test ±0,98–1,00), tetapi val jenuh di 100%.
   - Run 1 Custom CNN sempat kurang dilatih (epoch terbaik = epoch terakhir).
5. **Kelas tersulit:** gudeg (recall Custom CNN 0,55), dengan penyebab seperti di 5.4.
6. **Waktu dan parameter:** MobileNetV2 punya 9× lebih banyak parameter, tetapi selesai training lebih cepat (211 vs 251 detik) karena lebih sedikit epoch. Per epoch dan saat inferensi, MobileNetV2 lebih lambat (25,9 vs 4,7 ms/citra).
7. **Paling sesuai:** MobileNetV2 dengan transfer learning, karena paling akurat, paling stabil antara val dan test, training cepat, dan ukurannya ringan (bobot ±9 MB). Custom CNN unggul dalam ukuran dan kecepatan inferensi.

---

## 7. Metode final yang dipakai

| Aspek | Keputusan |
|---|---|
| Dataset | 3 kelas, 388 foto (199 internet manual + foto sendiri dari 4 perangkat) |
| Pembersihan | verify, EXIF transpose, RGB, MIN_SIDE 128, resize maks 800, pHash dedup (≤ 8), eksklusi manual 2 foto |
| Anti-leakage | grup pHash ≤ 14 + jepretan beruntun per perangkat ≤ 10 detik; split per grup |
| Test | perangkat yang ditahan (FUJIFILM untuk bakpia, Samsung untuk gudeg/sate) + top-up internet untuk gudeg |
| Train/val | 80:20 per grup, proporsional per sumber |
| Augmentasi | flip horizontal, rotasi 0,1, zoom 0,15, kontras 0,15 (hanya train) |
| Model 1 | Custom CNN 4 blok Conv + GAP + Dense(64) + Dropout 0,5; Adam 1e-3; ES 15; ReduceLR 6 |
| Model 2 | MobileNetV2 ImageNet; fase 1 beku (Adam 1e-3); fase 2 fine-tune layer 100+ (Adam 1e-5) |
| Pemilihan model | val_loss terendah; test dibuka sekali setelah model dikunci |
| **Model terpilih** | **MobileNetV2** |

---

## 8. Batasan (untuk BAB V)

1. Foto test gudeg dan sate berasal dari **warung yang sama** dengan train (beda perangkat dan waktu). Akurasi belum mencerminkan performa di warung lain.
2. Test kecil (61 foto; 1 foto = 1,6%), sehingga selisih kecil antar model perlu ditafsirkan hati-hati.
3. Variasi antar-run Custom CNN ±0,05 val_loss; hasil berasal dari satu run per konfigurasi.
4. Indikasi pengaruh latar (meja kayu, meja besi) belum dibuktikan langsung, misalnya dengan Grad-CAM.
5. Val MobileNetV2 jenuh di 100%, sehingga val tidak lagi bisa membedakan konfigurasi yang lebih baik.

---

## 9. Pemeriksaan notebook dan yang perlu dilakukan sebelum dikumpulkan

### Wajib diperbaiki

| No | Notebook | Masalah | Perbaikan |
|---|---|---|---|
| 1 | `01_dataCleaning` sel konfigurasi | Kode menulis `SPLIT = ROOT / "data" / "processed"`, padahal output tersimpan dan data sebenarnya ada di **`data/split`** (`data/processed` kosong), dan notebook modelling membaca `data/split`. Kalau notebook cleaning dijalankan ulang, hasilnya masuk ke folder yang salah | Ubah menjadi `SPLIT = ROOT / "data" / "split"`. Ganti juga teks "data/processed" di markdown pembuka dan judul "SIMPAN HASIL CLEANING KE PROCESSED". Hapus folder `data/processed` yang kosong |
| 2 | `02_modelling` | Hasil di laporan harus sama dengan output notebook (ketentuan I.5). Menjalankan ulang sel training akan menghasilkan angka sedikit berbeda | **Jangan "Run All"/restart lalu training ulang.** Simpan notebook apa adanya, lengkap dengan output |

### Perbaikan kecil (kerapian)

- `01_dataCleaning`:
  - Ada dua sel konfigurasi (sel 1 dan 2) dengan isi berulang. Cukup sisakan import di sel 1.
  - Penomoran judul belum urut ("Pengecekan Metadata" tanpa nomor, sel cek anti-leakage tanpa judul, judul "RINGKASAN UNTUK BAB III" kosong). Sesuaikan juga daftar "Alur" di markdown pembuka dengan judul langkah yang sebenarnya.
  - `import os` dan `random` tidak dipakai.
  - Hapus sel kosong terakhir.
- `02_modelling`:
  - Markdown pembuka poin 5 (MobileNetV2) masih berisi deskripsi Custom CNN (salah salin).
  - Penomoran judul belum urut (judul "1. Membaca data" belum ada; eksperimen belum bernomor).
  - Baris `head_weights = mnv2_model.get_weights()` beserta komentarnya tidak dipakai; hapus.
  - Typo "ada do ketentuan".
  - Hapus sel kosong terakhir.
  - Peringatan `UserWarning: shuffle=True ... ignored` tidak berbahaya (data sudah diacak oleh `image_dataset_from_directory`).

### Opsional untuk memperkuat analisis

- **Grad-CAM** pada `bakpia_011`/`012` dan `gudeg_015`–`020`, untuk membuktikan dugaan pengaruh latar (pertanyaan no. 5).
- **MobileNetV2 dari nol** (`weights=None`) sebagai model pembanding, untuk menjawab pertanyaan no. 3 secara tegas. Ini tidak mengganti model utama.
- Catat cara pengumpulan foto internet (kata kunci pencarian, situs), sebagai pertanggungjawaban dataset (ketentuan I.4).

### Status ketentuan UTS

| Ketentuan | Status |
|---|---|
| B. Dataset (3 kelas, ≥ 100 dev dan ≥ 20 test per kelas, ≥ 360, test terpisah, augmentasi hanya train, tanpa duplikat antar split) | ✅ |
| D. Custom CNN dari nol + arsitektur pilihan (MobileNetV2), split sama | ✅ |
| E.1–7 grafik, confusion matrix, metrik, waktu dan parameter, tabel perbandingan, contoh benar/salah | ✅ |
| E. 7 pertanyaan analisis | ✅ bahan tersedia (bagian 6) |
| G.2 source code | ✅ 2 notebook |
| G.3 dataset + pemisahan | ✅ `data/split` + `manifest.csv` |
| G.4 model | ✅ `custom_cnn_best.keras` (3,1 MB), `mobilenetv2_best.keras` (24,6 MB), `model_info.json` |
| G.1 laporan PDF, G.5 pembagian kontribusi | ⬜ belum |

---

## 10. Berkas keluaran penting

| File | Isi |
|---|---|
| `outputs/cleaning/manifest.csv` | jejak lengkap 403 foto: sumber, perangkat, waktu, alasan dibuang, grup, split, nama akhir |
| `outputs/cleaning/tabel_*.csv`, `grafik_split.png`, `contoh_citra_per_kelas.png`, `grids/` | bahan BAB III |
| `outputs/modelling/contoh_augmentasi.png` | contoh augmentasi (BAB III) |
| `outputs/modelling/kurva_training.png` | grafik accuracy dan loss (BAB IV) |
| `outputs/modelling/confusion_matrix.png`, `classification_report_*.csv` | evaluasi test |
| `outputs/modelling/tabel_perbandingan.csv`, `tabel_efisiensi.csv` | tabel perbandingan |
| `outputs/modelling/contoh_prediksi_*.png`, `prediksi_test.csv` | contoh benar/salah dan seluruh prediksi |
| `outputs/modelling/eksperimen_dropout.*`, `eksperimen_strategi_training.*` | eksperimen tambahan |
| `outputs/modelling/*.keras`, `model_info.json` | model final |

---

## 11. Referensi utama

- Practice Lab 3 Deep Learning UAJY (arsitektur CNN dasar, Adam 1e-3, eksperimen dropout).
- TensorFlow. *Transfer learning and fine-tuning*. https://www.tensorflow.org/tutorials/images/transfer_learning
- Sandler dkk. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR. https://arxiv.org/abs/1801.04381
- Kumar dkk. (2022). *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*. ICLR. https://iclr.cc/virtual/2022/poster/5945
- Yosinski dkk. (2014). *How transferable are features in deep neural networks?* https://arxiv.org/abs/1411.1792
- Lin dkk. (2013). *Network In Network* (Global Average Pooling). https://arxiv.org/abs/1312.4400
- Srivastava dkk. (2014). *Dropout*. JMLR 15. https://www.jmlr.org/papers/v15/srivastava14a.html
- Shorten & Khoshgoftaar (2019). *A survey on Image Data Augmentation for Deep Learning*. https://doi.org/10.1186/s40537-019-0197-0
- Zauner (2010). *Implementation and Benchmarking of Perceptual Image Hash Functions*. https://phash.org/docs/pubs/thesis_zauner.pdf
- Barz & Denzler (2020). *Do We Train on Test Data? Purging CIFAR of Near-Duplicates*. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321059/
- Kapoor & Narayanan (2023). *Leakage and the reproducibility crisis in ML-based science*. Patterns. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10499856/
- Yagis dkk. (2021). *Effect of data leakage in brain MRI classification using 2D CNNs*. Scientific Reports. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8604922/
- Roberts dkk. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. Ecography. https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf
- Beery dkk. (2018). *Recognition in Terra Incognita*. ECCV. https://arxiv.org/abs/1807.04975
- Geirhos dkk. (2020). *Shortcut learning in deep neural networks*. Nature Machine Intelligence. https://is.mpg.de/publications/geirhosetal20
- Bossard dkk. (2014). *Food-101*. ECCV. https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101
- Faturrahman dkk. (2023). *Klasifikasi Jajanan Tradisional Indonesia berbasis Deep Learning dan Metode Transfer Learning*. ELKOMIKA 11(4). https://ejurnal.itenas.ac.id/index.php/elkomika/article/view/9319
- Wibisono dkk. (2020). *Traditional food knowledge of Indonesia*. Journal of Big Data 7:69. https://doi.org/10.1186/s40537-020-00342-5
