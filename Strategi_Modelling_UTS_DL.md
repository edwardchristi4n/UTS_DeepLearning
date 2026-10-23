# Strategi Modelling UTS Deep Learning
**Klasifikasi Citra Makanan Khas Yogyakarta: Bakpia, Gudeg, Sate Klathak**
Model 1: Custom CNN (dari nol) | Model 2: MobileNetV2 (pretrained ImageNet, bekukan lalu fine-tuning)

---

## 1. Ringkasan

| Aspek | Keputusan |
|---|---|
| Framework | TensorFlow / Keras (sama dengan Practice Lab 3) |
| Data | `data/split/{train,val,test}/<kelas>/` hasil notebook cleaning (388 citra) |
| Input | 224 × 224 × 3, batch size 16, seed 42 |
| Model 1 | Custom CNN 4 blok konvolusi + Global Average Pooling + Dense(64) + Dropout(0.5), ±249 ribu parameter |
| Model 2 | MobileNetV2 (ImageNet) + head GAP-Dropout-Dense, dilatih 2 fase: backbone beku → fine-tuning bagian akhir |
| Evaluasi | Test set disentuh **sekali** setelah kedua model selesai; metrik sesuai ketentuan soal |

---

## 2. Kondisi data yang dimodelkan

### 2.1 Jumlah citra per split

| Kelas | Train | Val | Test | Dev (train+val) |
|---|---|---|---|---|
| bakpia | 83 | 21 | 21 | 104 |
| gudeg | 90 | 14 | 20 | 104 |
| sate_klathak | 95 | 24 | 20 | 119 |
| **Total** | **268** | **59** | **61** | **327** |

Semua syarat soal terpenuhi (dev ≥ 100 per kelas, test ≥ 20 per kelas, total ≥ 360 → 388).

### 2.2 Komposisi sumber

| Kelas | Sumber | Train | Val | Test |
|---|---|---|---|---|
| bakpia | internet | 44 | 11 | 0 |
| | foto sendiri | 39 | 10 | 21 |
| gudeg | internet | 43 | 10 | 8 |
| | foto sendiri | 47 | 4 | 12 |
| sate_klathak | internet | 56 | 14 | 0 |
| | foto sendiri | 39 | 10 | 20 |

### 2.3 Asal test set (perangkat yang ditahan)

| Kelas | Isi test | Sifat |
|---|---|---|
| bakpia | 21 foto FUJIFILM X-T30 III (sesi 21 Okt pagi) | **Latar berbeda total** dari train (batu, meja besi, kaca) → menguji generalisasi |
| gudeg | 12 foto Samsung Galaxy A35 + 8 foto internet | Foto Samsung: warung dan piring sama dengan train |
| sate_klathak | 17 foto Samsung + 3 Fujifilm (satu grup) | Meja dan piring sama dengan train |

**Implikasi untuk analisis:** tingkat kesulitan test tidak sama antar kelas. Bakpia paling sulit karena latarnya baru; gudeg dan sate lebih mudah karena latarnya sama dengan train. Perbedaan recall antar kelas harus ditafsirkan dengan mempertimbangkan hal ini, bukan semata-mata "makanan mana yang sulit dikenali".

### 2.4 Catatan data lain
- Kelas cukup seimbang (train 83/90/95), sehingga **tidak perlu class weight**.
- Val gudeg dari foto sendiri hanya 4 citra karena 48 foto gudeg berada dalam satu grup anti-leakage. Kurva val gudeg akan lebih berfluktuasi.
- Foto sendiri ketiga kelas sering berlatar meja kayu cokelat → risiko model belajar dari latar (*shortcut learning*).

---

## 3. Framework dan lingkungan

| Hal | Keputusan | Alasan |
|---|---|---|
| Library | TensorFlow / Keras | Konsisten dengan Practice Lab 3 (`Sequential`, `Adam`, `sparse_categorical_crossentropy`) |
| Tempat menjalankan | Server lab (GPU) atau laptop (CPU) | TensorFlow versi baru tidak mendukung GPU di Windows native; dengan 268 citra, CPU tetap memungkinkan tetapi MobileNetV2 lebih lambat |
| Reproducibility | `SEED = 42` untuk `random`, `numpy`, `tf.random` | Hasil dapat diulang |

---

## 4. Prinsip perbandingan yang adil

Kedua model **wajib** memakai:
1. Split yang sama (`data/split`).
2. Ukuran input yang sama (224 × 224).
3. Augmentasi yang sama, aktif hanya saat training.
4. Batch size dan seed yang sama.
5. Aturan pemilihan model yang sama: bobot terbaik dipilih dari **val_loss**, bukan dari test.
6. Test set dievaluasi **sekali saja**, setelah kedua model selesai dilatih.

Yang **boleh berbeda** hanyalah hal yang memang melekat pada modelnya: preprocessing (MobileNetV2 memerlukan skala [-1, 1]), learning rate, dan jumlah fase training.

---

## 5. Pipeline input

| Langkah | Pengaturan |
|---|---|
| Membaca data | `tf.keras.utils.image_dataset_from_directory(data/split/<split>, image_size=(224, 224), batch_size=16, label_mode="int", seed=42)` |
| Urutan kelas | Alfabetis dari nama folder: `bakpia = 0`, `gudeg = 1`, `sate_klathak = 2` |
| Shuffle | Train: `True`; val dan test: `False` (supaya urutan prediksi cocok dengan nama file untuk contoh benar/salah) |
| Performa | `.cache().prefetch(tf.data.AUTOTUNE)` |

### Augmentasi (sebagai layer di dalam model, hanya aktif saat training)

| Layer | Nilai | Alasan |
|---|---|---|
| `RandomFlip("horizontal")` | — | Makanan tetap makanan yang sama bila dicerminkan |
| `RandomRotation` | 0.1 (±36°) | Variasi sudut pengambilan |
| `RandomZoom` | 0.15 | Variasi jarak kamera |
| `RandomContrast` | 0.15 | Variasi pencahayaan warung/rumah |

Karena berupa layer Keras, augmentasi **otomatis nonaktif** saat `evaluate()` dan `predict()`, sehingga val dan test tidak pernah teraugmentasi (sesuai ketentuan soal).

---

## 6. Model 1 — Custom CNN (dilatih dari nol)

### 6.1 Asal-usul desain: dikembangkan dari Practice Lab 3

| Komponen | Lab 3 (model challenge, MNIST) | UTS | Alasan perubahan |
|---|---|---|---|
| Input | 28 × 28 × 1 | 224 × 224 × 3 | Foto berwarna; disamakan dengan MobileNetV2 |
| Normalisasi | dibagi 255 | `Rescaling(1/255)` di dalam model | Sama prinsipnya |
| Blok konvolusi | 3 Conv 3×3 (32-64-64) + MaxPool | **4 Conv 3×3 (32-64-128-128) + MaxPool 2×2**, `padding="same"` | Citra 224 px perlu diperkecil lebih banyak (224 → 14) |
| Peratakan | `Flatten` | **`GlobalAveragePooling2D`** | Mencegah parameter Dense membengkak (lihat 6.3) |
| Dense tersembunyi | Dense(64, ReLU) | Dense(64, ReLU) | Tetap sama |
| Dropout | 0.5 | 0.5 | Tetap sama; sudah dieksplorasi di Lab 3 |
| Output | Dense(10, softmax) | Dense(3, softmax) | 3 kelas |

### 6.2 Arsitektur lengkap

| No | Layer | Output shape | Parameter |
|---|---|---|---|
| 0 | Input | 224 × 224 × 3 | 0 |
| 1 | Augmentasi (flip, rotasi, zoom, kontras) | 224 × 224 × 3 | 0 |
| 2 | Rescaling(1/255) | 224 × 224 × 3 | 0 |
| 3 | Conv2D(32, 3×3, ReLU, same) | 224 × 224 × 32 | 896 |
| 4 | MaxPooling2D(2×2) | 112 × 112 × 32 | 0 |
| 5 | Conv2D(64, 3×3, ReLU, same) | 112 × 112 × 64 | 18.496 |
| 6 | MaxPooling2D(2×2) | 56 × 56 × 64 | 0 |
| 7 | Conv2D(128, 3×3, ReLU, same) | 56 × 56 × 128 | 73.856 |
| 8 | MaxPooling2D(2×2) | 28 × 28 × 128 | 0 |
| 9 | Conv2D(128, 3×3, ReLU, same) | 28 × 28 × 128 | 147.584 |
| 10 | MaxPooling2D(2×2) | 14 × 14 × 128 | 0 |
| 11 | GlobalAveragePooling2D | 128 | 0 |
| 12 | Dense(64, ReLU) | 64 | 8.256 |
| 13 | Dropout(0.5) | 64 | 0 |
| 14 | Dense(3, softmax) | 3 | 195 |
| | **Total** | | **249.283** (semua trainable) |

### 6.3 Mengapa Global Average Pooling, bukan Flatten

Setelah blok ke-4, fitur berukuran 14 × 14 × 128 = 25.088 nilai.

| Pilihan | Parameter Dense(64) | Total parameter model |
|---|---|---|
| Flatten | 25.088 × 64 + 64 ≈ 1,61 juta | ≈ 1,85 juta |
| **GlobalAveragePooling** | 128 × 64 + 64 = 8.256 | **≈ 249 ribu** |

Dengan Flatten, ±87% parameter berada di satu layer Dense. Untuk 268 citra latih, ini sangat rawan overfitting. GAP merangkum setiap feature map menjadi satu angka (Lin dkk., 2013) dan juga dipakai oleh MobileNetV2, sehingga kedua model setara di bagian ini.

### 6.4 Pengaturan training

| Parameter | Lab 3 | UTS | Alasan |
|---|---|---|---|
| Optimizer | Adam, lr 1e-3 | Adam, lr 1e-3 | Sama |
| Loss | sparse_categorical_crossentropy | sama | Label integer dari nama folder |
| Metrik | accuracy | accuracy | Sama |
| Batch size | 64 | 16 | 268 citra ÷ 64 = hanya 5 langkah per epoch |
| Epoch | 5 | maksimal 60 | Model dari nol dengan data kecil butuh lebih banyak epoch |
| EarlyStopping | — | monitor `val_loss`, patience 10, `restore_best_weights=True` | Berhenti saat mulai overfit, pakai bobot terbaik |
| ReduceLROnPlateau | — | monitor `val_loss`, factor 0.5, patience 4, min_lr 1e-6 | Menstabilkan akhir training |
| ModelCheckpoint | — | simpan bobot terbaik ke `outputs/modelling/custom_cnn_best.keras` | Model untuk lampiran deliverable |

### 6.5 Eksperimen tambahan (opsional, meniru Lab 3)
Ulangi eksperimen dropout Lab 3 pada Custom CNN: **dropout 0.0 / 0.3 / 0.5**, catat train acc, val acc, dan selisihnya. Pemilihan dropout terbaik memakai **val**, bukan test. Hasilnya menjadi bukti langsung untuk pembahasan overfitting.

### 6.6 Catatan BatchNormalization
BatchNormalization **tidak dipakai** agar tetap setia pada Lab 3. Pada uji pilot sebelumnya, BatchNorm dengan setelan bawaan membuat model sempat menebak satu kelas saja pada val. Jika Custom CNN sulit belajar, BatchNorm (momentum 0.9) dapat ditambahkan sebagai eksperimen terpisah dan dilaporkan.

---

## 7. Model 2 — MobileNetV2 (transfer learning)

### 7.1 Alasan memilih MobileNetV2
- Arsitektur populer dan ringan (±2,26 juta parameter backbone), cocok untuk data kecil dan komputasi terbatas.
- Terbukti baik pada klasifikasi makanan tradisional Indonesia (Faturrahman dkk., 2023: MobileNetV2 98% mengungguli ResNet50 dan VGG16).
- Sesuai dengan abstrak/proposal kelompok.

### 7.2 Arsitektur

| No | Komponen | Keterangan | Parameter |
|---|---|---|---|
| 0 | Input | 224 × 224 × 3 | 0 |
| 1 | Augmentasi | **sama persis** dengan Custom CNN | 0 |
| 2 | `mobilenet_v2.preprocess_input` | Skala piksel ke **[-1, 1]** (bukan /255) | 0 |
| 3 | `MobileNetV2(include_top=False, weights="imagenet")` | Dipanggil dengan `training=False` | 2.257.984 |
| 4 | GlobalAveragePooling2D | 7 × 7 × 1280 → 1280 | 0 |
| 5 | Dropout(0.2) | | 0 |
| 6 | Dense(3, softmax) | | 3.843 |
| | **Total** | | **2.261.827** |

Dua hal teknis yang wajib:
1. **Preprocessing [-1, 1].** MobileNetV2 dilatih di ImageNet dengan skala ini. Jika diberi skala [0, 1], fitur yang dihasilkan tidak sesuai dan performa turun.
2. **`training=False` saat memanggil backbone.** BatchNorm di dalam backbone tetap memakai statistik ImageNet, tidak diperbarui oleh batch kecil kita, termasuk pada fase fine-tuning.

Struktur ini mengikuti tutorial resmi TensorFlow *Transfer learning and fine-tuning*.

### 7.3 Training dua fase

| | Fase 1 — Feature extraction | Fase 2 — Fine-tuning |
|---|---|---|
| Yang dilatih | Hanya head (3.843 parameter); `base.trainable = False` | Head + bagian akhir backbone: layer ke-100 sampai akhir (±54 layer terakhir dari ±154 layer) |
| Optimizer | Adam, lr **1e-3** | Adam, lr **1e-5** |
| Epoch | maksimal 15 | maksimal 20, dilanjutkan dari epoch terakhir fase 1 (`initial_epoch`) |
| EarlyStopping | `val_loss`, patience 5, `restore_best_weights=True` | sama |
| Wajib | — | **Compile ulang** setelah layer dibuka; tanpa compile ulang perubahan `trainable` tidak berlaku |

Alasan urutan dua fase:
- Head baru masih acak. Jika seluruh jaringan langsung dilatih, gradien besar dari head acak dapat merusak fitur ImageNet di backbone.
- Kumar dkk. (2022) menunjukkan bahwa melatih head dulu (*linear probing*) lalu fine-tuning lebih baik daripada langsung fine-tuning, terutama pada data yang berbeda distribusi. Ini relevan dengan test bakpia yang berlatar baru.
- Learning rate fase 2 dibuat 100 kali lebih kecil agar penyesuaian backbone halus.

Jumlah parameter trainable di fase 2 dicetak dari `model.summary()` dan dicatat di tabel perbandingan.

---

## 8. Pencatatan selama training

| Yang dicatat | Cara |
|---|---|
| Waktu training | `time.time()` sebelum dan sesudah `fit()`; MobileNetV2 = waktu fase 1 + fase 2 |
| Riwayat | `history.history` disimpan ke CSV; MobileNetV2: riwayat fase 1 dan 2 digabung |
| Parameter | `count_params()` total dan trainable (MobileNetV2: per fase) |
| Bobot terbaik | file `.keras` di `outputs/modelling/` |

---

## 9. Evaluasi (sesuai ketentuan soal)

Dijalankan **sekali** pada test set setelah kedua model selesai:

1. Grafik accuracy dan loss (train vs val) untuk tiap model. MobileNetV2 diberi garis batas fase 1 / fase 2.
2. Confusion matrix tiap model.
3. Accuracy, precision, recall, F1 per kelas dan macro average (`classification_report`).
4. Waktu training dan jumlah parameter.
5. Tabel perbandingan kedua model:

| Model | Params total | Params trainable | Epoch | Waktu training | Val acc terbaik | Test acc | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Custom CNN | 249.283 | 249.283 | … | … | … | … | … | … | … |
| MobileNetV2 | 2.261.827 | fase 1: 3.843 / fase 2: … | … | … | … | … | … | … | … |

6. Contoh citra yang diklasifikasikan benar dan salah, lengkap dengan nama file, sumber (internet/sendiri), dan perangkat dari `manifest.csv`.

### Keluaran file

```
outputs/modelling/
├── custom_cnn_best.keras
├── mobilenetv2_best.keras
├── history_custom_cnn.csv
├── history_mobilenetv2.csv
├── kurva_custom_cnn.png
├── kurva_mobilenetv2.png
├── confusion_matrix.png
├── classification_report_custom_cnn.csv
├── classification_report_mobilenetv2.csv
├── tabel_perbandingan.csv
├── contoh_prediksi_custom_cnn.png
└── contoh_prediksi_mobilenetv2.png
```

---

## 10. Hipotesis dan bahan analisis

| Hipotesis | Cara menguji | Bahan pembahasan |
|---|---|---|
| MobileNetV2 lebih akurat daripada Custom CNN | Tabel perbandingan, confusion matrix | Manfaat fitur pretrained pada data kecil |
| Custom CNN lebih cepat overfit | Selisih train–val acc dan kurva loss | Pengaruh jumlah data dan regularisasi (dropout, augmentasi) |
| Recall bakpia paling rendah, terutama di Custom CNN | Recall per kelas | Test bakpia berlatar baru → menguji generalisasi; kemungkinan model belajar dari latar |
| Sate dan gudeg mudah dikenali | Recall per kelas | Test berasal dari meja/piring yang sama dengan train → akurasi cenderung optimistis (batasan) |
| Kesalahan berkaitan dengan sumber foto | Contoh salah + kolom sumber/perangkat | Perbedaan foto internet vs foto sendiri |
| MobileNetV2 mencapai hasil baik dengan parameter trainable jauh lebih sedikit (fase 1) | Params trainable vs akurasi | Efisiensi transfer learning |

### Eksperimen opsional bila waktu cukup
- **Grad-CAM** pada beberapa citra test untuk melihat apakah model fokus pada makanan atau pada meja (Selvaraju dkk., 2017).
- **Eksperimen dropout** pada Custom CNN (bagian 6.5).

---

## 11. Risiko dan mitigasi

| Risiko | Mitigasi |
|---|---|
| Overfitting karena data kecil | Augmentasi, Dropout, GAP, EarlyStopping dengan `restore_best_weights` |
| Model belajar dari latar (meja kayu) | Foto internet berlatar beragam ada di train dan val; analisis kesalahan per sumber; Grad-CAM |
| Preprocessing MobileNetV2 keliru | Gunakan `mobilenet_v2.preprocess_input`, jangan Rescaling 1/255 |
| Fine-tuning merusak fitur ImageNet | 2 fase, lr 1e-5, `training=False` pada backbone |
| Lupa compile ulang setelah membuka layer | Compile ulang wajib sebelum fase 2 |
| Test ikut dipakai memilih model | Semua keputusan (epoch, dropout, lr) dari val; test hanya sekali di akhir |
| Hasil berubah tiap dijalankan | Seed 42 untuk semua sumber acak |
| Training lambat di CPU | Jalankan di server lab; atau kurangi epoch maksimal fase 2 |

---

## 12. Urutan pengerjaan notebook training

| Langkah | Isi |
|---|---|
| 0 | Import, konfigurasi, seed, cek GPU |
| 1 | Membaca `data/split` dengan `image_dataset_from_directory`, cek jumlah dan urutan kelas |
| 2 | Layer augmentasi + contoh visual augmentasi (untuk BAB III) |
| 3 | Membangun Custom CNN, `summary()` |
| 4 | Training Custom CNN (callbacks, catat waktu) |
| 5 | Membangun MobileNetV2 + head, `summary()` |
| 6 | Fase 1: backbone beku |
| 7 | Fase 2: buka layer akhir, compile ulang, fine-tuning |
| 8 | Grafik accuracy/loss kedua model |
| 9 | Evaluasi test (sekali): classification report, confusion matrix |
| 10 | Tabel perbandingan |
| 11 | Contoh prediksi benar dan salah (dengan sumber dari manifest) |
| 12 | (Opsional) eksperimen dropout dan Grad-CAM |

---

## 13. Referensi

- Practice Lab 3 Deep Learning UAJY — arsitektur CNN dasar, Adam 1e-3, eksperimen dropout.
- TensorFlow. *Transfer learning and fine-tuning*. https://www.tensorflow.org/tutorials/images/transfer_learning
- Sandler, M. dkk. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR. https://arxiv.org/abs/1801.04381
- Kumar, A. dkk. (2022). *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*. ICLR. https://iclr.cc/virtual/2022/poster/5945
- Yosinski, J. dkk. (2014). *How transferable are features in deep neural networks?* NeurIPS. https://arxiv.org/abs/1411.1792
- Lin, M., Chen, Q., Yan, S. (2013). *Network In Network* (Global Average Pooling). https://arxiv.org/abs/1312.4400
- Srivastava, N. dkk. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR 15. https://www.jmlr.org/papers/v15/srivastava14a.html
- Shorten, C., Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data 6:60. https://doi.org/10.1186/s40537-019-0197-0
- Faturrahman, R., Hariyani, Y. S., Hadiyoso, S. (2023). *Klasifikasi Jajanan Tradisional Indonesia berbasis Deep Learning dan Metode Transfer Learning*. ELKOMIKA 11(4). https://ejurnal.itenas.ac.id/index.php/elkomika/article/view/9319
- Geirhos, R. dkk. (2020). *Shortcut learning in deep neural networks*. Nature Machine Intelligence 2. https://is.mpg.de/publications/geirhosetal20
- Selvaraju, R. R. dkk. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV. https://arxiv.org/abs/1610.02391
