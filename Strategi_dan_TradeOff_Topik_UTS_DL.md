# UTS Deep Learning: Pemilihan Topik, Sumber Data, dan Strategi Pemodelan

> Klasifikasi Citra Nusantara dengan CNN. Proyek kelompok, 3 kelas, minimal 100 citra pengembangan + 20 citra test per kelas.
> Dokumen ini berisi perbandingan trade-off semua pilihan topik, sumber data, dan rencana pemodelan.

---

## 1. Ringkasan Cepat

| Topik | Kemudahan kumpul data | Peluang foto sendiri | Tingkat kesulitan klasifikasi | Bahan analisis BAB IV | Risiko utama |
|---|---|---|---|---|---|
| **Makanan Yogyakarta** (Gudeg, Bakpia, Geplak) | ⭐⭐⭐⭐⭐ | Sangat tinggi | Mudah | Sedang (harus "dipancing") | Akurasi ~100%, analisis dangkal; shortcut dari kemasan/latar |
| **Wayang Yogyakarta** (Arjuna, Bima, Gatotkaca) | ⭐⭐⭐ | Rendah–sedang | Sedang–sulit | Kaya | Duplikat internet, objek fisik terbatas, label tokoh harus benar |
| **Motif Batik** (Parang, Kawung, Truntum) | ⭐⭐⭐⭐ | Tinggi | Sedang | Kaya | Crop dari kain yang sama bocor ke test; motif campuran |
| **Keris** (3 jenis/bentuk) | ⭐ | Rendah | Sulit (fine-grained) | Kaya | Data hampir mustahil cukup; label butuh ahli |
| **Pola Candlestick** (Doji, Hammer, Engulfing) | ⭐⭐⭐⭐⭐ | Bisa digenerate | Mudah–sedang | Sedang | Data sintetis dinilai kurang orisinal; aspek "Nusantara" lemah |

**Rekomendasi:**
- **Pilihan paling aman:** **Makanan**. Nilai dataset (15%) mudah maksimal, risiko gagal kumpul data paling kecil.
- **Pilihan paling "menjual" untuk analisis:** **Wayang**. Pilih ini hanya kalau kelompok punya akses ke wayang fisik (museum, pengrajin, kolektor) dan siap membersihkan duplikat.
- **Alternatif kuat yang sering terlewat:** **Batik**. Mudah difoto sendiri dan secara visual menantang di level tekstur.

---

## 2. Trade-off per Topik

### 2.1 Makanan Tradisional Yogyakarta (Gudeg, Bakpia, Geplak)

**Kelebihan**
- Bisa beli dan memotret sendiri dalam jumlah besar, dari banyak warung, toko oleh-oleh, dan merek. Mayoritas data bisa 100% orisinal, yang paling disukai kriteria "kualitas dan orisinalitas dataset".
- Variasi alami tinggi: penyajian (piring, besek, kendil, kotak), isi bakpia, warna geplak, gudeg kering vs basah.
- Foto ulasan Google Maps warung/toko adalah sumber tambahan yang realistis dan beragam.
- Biaya murah dan logistik gampang dibagi ke semua anggota.

**Kekurangan / Risiko**
- **Kelasnya terlalu mudah dibedakan** (warna dan bentuk sangat berbeda). Akurasi kedua model bisa mendekati 100%, sehingga:
  - Pertanyaan "apakah model kompleks selalu lebih baik?" sulit dijawab karena semua model sama-sama bagus.
  - Confusion matrix nyaris diagonal sempurna, sehingga analisis kelas tersulit jadi tipis.
- **Shortcut learning:** model bisa belajar kotak/kemasan bermerek, besek, atau latar warung, bukan makanannya.
- Gudeg biasanya disajikan bersama lauk (telur, krecek, ayam). Perlu aturan label yang jelas: apakah foto nasi gudeg komplit dihitung "Gudeg"?

**Cara memitigasi**
- Sengaja ambil kondisi sulit: bakpia dibelah, geplak dalam plastik, gudeg kering tanpa kuah, foto dari atas/samping/dekat, cahaya redup warung, latar seragam untuk ketiga kelas.
- Jadikan **robustness test** sebagai eksperimen tambahan: uji model pada citra dengan latar atau kemasan yang tidak pernah dilihat saat training.
- Grad-CAM untuk membuktikan model melihat makanannya, bukan kemasannya.

---

### 2.2 Wayang Yogyakarta (Arjuna, Bima, Gatotkaca)

**Kelebihan**
- **Lebih menantang**: Bima dan Gatotkaca sama-sama tokoh gagah/bertubuh besar, kemungkinan besar sering tertukar. Ini bahan bagus untuk pertanyaan analisis no. 5 (kelas tersulit) dan no. 3 (model kompleks vs sederhana).
- Arjuna (tokoh halus/ramping) vs Bima/Gatotkaca memberi struktur kesulitan yang menarik dibahas.
- Topik budaya yang kuat, sehingga latar belakang di BAB I mudah ditulis.

**Kekurangan / Risiko**
- **Objek fisik terbatas.** 20 foto dari satu wayang yang sama itu hampir identik, sehingga variasi rendah dan berisiko bocor antar split.
- **Gambar internet banyak duplikat** (satu foto di-repost di banyak situs, di-crop, diberi watermark). Wajib dedup ketat, dan prosesnya memakan waktu.
- **Gaya campur aduk:** wayang kulit Yogya vs Solo, wayang golek, wayang orang, ilustrasi, kartun, patung. Lingkup harus dibatasi sejak awal (mis. hanya wayang kulit, termasuk gaya Surakarta atau tidak).
- **Validitas label:** tokoh harus diidentifikasi dengan benar. Salah label akan menurunkan nilai dataset. Sebaiknya validasi dengan referensi (museum, dalang, buku wayang).
- Kecil kemungkinan kelompok bisa memenuhi ≥50% foto sendiri.

**Cara memitigasi**
- Kunjungi museum wayang, sentra pengrajin wayang kulit, toko suvenir, atau pertunjukan. Foto setiap wayang dari banyak sudut dan latar, lalu perlakukan **satu wayang fisik = satu grup** saat split.
- Wikimedia Commons dan situs museum memberi sumber yang lisensi dan labelnya lebih jelas.
- Tentukan definisi kelas tertulis di BAB III (mis. "wayang kulit purwa gaya Yogyakarta, tampak depan/samping").

---

### 2.3 Motif Batik Nusantara (Parang, Kawung, Truntum)

**Kelebihan**
- Mudah difoto sendiri: lemari baju sendiri/keluarga, pasar batik, toko, kain di rumah.
- CNN sangat cocok untuk klasifikasi **tekstur/pola berulang**, sehingga perbedaan arsitektur terlihat jelas dalam eksperimen.
- Variasi warna dan skala motif tinggi, sehingga tidak terlalu mudah dan tidak terlalu sulit.

**Kekurangan / Risiko**
- **Leakage dari crop:** satu kain bisa menghasilkan banyak crop. Kalau crop dari kain yang sama masuk ke train dan test, hasil palsu-tinggi. Wajib split per kain.
- Satu kain sering berisi **motif campuran** (mis. parang dengan isen-isen lain), sehingga label bisa ambigu.
- Batik print vs tulis vs cap dan variasi turunan (Parang Rusak, Parang Barong, dll.) perlu definisi kelas yang jelas.

---

### 2.4 Keris (3 jenis/bentuk)

**Kelebihan**
- Fine-grained classification, analisisnya paling dalam dan paling "riset".

**Kekurangan / Risiko**
- **Paling sulit memenuhi 120 citra per kelas** dengan data orisinal. Keris fisik jarang bisa diakses bebas.
- Membedakan jenis/dhapur keris butuh pengetahuan ahli, sehingga risiko salah label tinggi.
- Gambar internet sedikit dan banyak berulang.
- **Tidak disarankan** kecuali ada akses ke kolektor/museum keris.

---

### 2.5 Pola Candlestick (Doji, Hammer, Engulfing)

**Kelebihan**
- Data bisa digenerate tanpa batas dari data harga (mis. `mplfinance` + data saham IDX), dan label bisa otomatis dengan aturan.
- Tidak perlu foto lapangan.

**Kekurangan / Risiko**
- Citra sintetis, sehingga nilai "orisinalitas" dan unsur "Nusantara" bisa dipertanyakan dosen.
- Pola rule-based sangat mudah dipelajari, akurasi bisa sempurna dan analisis jadi dangkal.
- Label rule-based bisa ambigu (satu jendela bisa memenuhi dua pola sekaligus).

---

## 3. Sumber Data (bukan dataset publik)

Ketentuan: **dilarang** Kaggle, Roboflow, atau repositori dataset lain. Yang boleh adalah pengumpulan mandiri.

| Sumber | Cocok untuk | Catatan |
|---|---|---|
| **Foto sendiri** (HP beberapa anggota) | Semua, terutama Makanan & Batik | Jadikan porsi terbesar (idealnya ≥50%). Variasikan sudut, latar, cahaya, dan HP. |
| **Foto ulasan Google Maps** | Makanan | Sangat realistis dan beragam. Catat URL tempat. |
| **Google/Bing Images** via `icrawler` / `bing-image-downloader` | Semua | Banyak duplikat, wajib dedup dan cek manual. |
| **Wikimedia Commons** | Wayang, Batik, Keris | Lisensi jelas, bagus untuk daftar pustaka. |
| **Situs museum / blog budaya / blog kuliner / marketplace** | Wayang, Batik, Makanan | Cek relevansi dan kualitas. |
| **Frame video YouTube** | Wayang (pertunjukan) | Semua frame dari satu video harus masuk **split yang sama**. |

**Wajib dicatat** dalam `metadata.csv` (bukti "dapat dipertanggungjawabkan" sekaligus bahan BAB III):

```
filename, kelas, sumber, url_atau_lokasi, kolektor, tanggal, group_id, split
```

- `group_id` = satu sesi foto / satu objek fisik / satu warung / satu video / satu kain.
- Konfirmasi ke dosen apakah hasil crawling web dihitung "dikumpulkan sendiri". Kalau mayoritas foto sendiri, posisi aman.

**Target pengumpulan:** ±150 citra pengembangan + ±30 citra test per kelas (lebih dari minimum, untuk cadangan setelah dedup dan pembersihan).

---

## 4. Pipeline Data (anti-kebocoran)

1. **Kumpulkan** lalu **filter manual** (buang buram, salah kelas, gaya di luar lingkup).
2. **Dedup**: perceptual hash (`imagehash`, pHash, jarak Hamming ≤ ~8). Untuk near-duplicate bisa ditambah cosine similarity embedding (mis. dari ResNet pretrained).
3. **Kunci test set sejak awal**: idealnya dari sumber/sesi/warung/objek yang **berbeda total**. Simpan di folder terpisah, jangan dibuka sampai evaluasi akhir.
4. **Split train/val berbasis grup**: `StratifiedGroupKFold` / `GroupShuffleSplit` dengan `group_id`.
5. **Augmentasi hanya di train**: RandomResizedCrop, HorizontalFlip, rotasi kecil (±15°), brightness/contrast. Untuk makanan, hue jitter dibuat kecil karena warna adalah fitur penting.
6. **Struktur folder** yang dikumpulkan:
   ```
   dataset/
   ├── development/   (train + val, split via metadata)
   │   ├── kelas_1/ kelas_2/ kelas_3/
   └── test/
       ├── kelas_1/ kelas_2/ kelas_3/
   metadata.csv
   ```

---

## 5. Strategi Pemodelan

**Prinsip:** kedua (atau ketiga) model memakai **split, augmentasi, ukuran input (224×224), seed, dan aturan early stopping yang sama**. Jalankan 3 seed jika sempat, lalu laporkan rata-rata ± standar deviasi.

### Model 1: Custom CNN (from scratch, wajib)
```
Input 224×224×3
4 × [Conv3×3-BN-ReLU → Conv3×3-BN-ReLU → MaxPool]   filter: 32 → 64 → 128 → 256
GlobalAveragePooling → Dropout(0.5) → Dense(3, softmax)
```
- ±1 juta parameter. Adam lr 1e-3, ReduceLROnPlateau, EarlyStopping (patience ~10).
- Ablasi opsional (nilai tambah): dengan/tanpa BatchNorm, dengan/tanpa augmentasi, jumlah blok 3 vs 4.

### Model 2: Arsitektur dikenal (transfer learning)
- **MobileNetV2** atau **EfficientNet-B0** (pretrained ImageNet). Ringan dan cocok untuk data kecil.
- Dua tahap:
  1. Backbone dibekukan, hanya head dilatih (lr 1e-3).
  2. Unfreeze blok terakhir, fine-tune dengan lr 1e-4 sampai 1e-5.

### Model 3 (opsional, untuk menjawab pertanyaan analisis)
- **ResNet50 / VGG16**, model besar untuk menjawab "apakah model kompleks selalu lebih baik?".
- Atau **ResNet18 from scratch vs pretrained**, untuk memperlihatkan efek pretrained weights.

### Yang dicatat (sesuai bagian E soal)
- Kurva train/val accuracy dan loss.
- Confusion matrix, accuracy, precision, recall, F1 (`classification_report`).
- Waktu training (`torch.cuda.synchronize()` sebelum mencatat waktu), jumlah parameter, waktu inferensi per citra.
- Galeri citra benar dan salah klasifikasi, ditambah **Grad-CAM** untuk menjawab analisis no. 2 dan 5.

### Hardware
- RTX 4050 (6 GB VRAM) cukup. Gunakan **mixed precision (AMP)**, batch 32, input 224×224. Google Colab sebagai cadangan.

---

## 6. Peta ke Pertanyaan Analisis Wajib

| Pertanyaan | Bukti eksperimen yang disiapkan |
|---|---|
| 1. Model terbaik | Tabel perbandingan metrik pada **test set** |
| 2. Pengaruh arsitektur | Grad-CAM + perbandingan kurva belajar |
| 3. Kompleks selalu lebih baik? | Model 3 (besar) vs Model 2 (ringan) vs Custom |
| 4. Over/underfitting | Gap train–val pada kurva |
| 5. Kelas tersulit | Confusion matrix + galeri error |
| 6. Waktu & parameter | Tabel parameter, waktu training, waktu inferensi |
| 7. Model paling sesuai | Gabungan akurasi, efisiensi, dan kemudahan deployment |

---

## 7. Pembagian Kerja (contoh 4 orang)

| Anggota | Tugas utama |
|---|---|
| A | Pengumpulan data lapangan (foto sendiri), `metadata.csv` |
| B | Crawling web, dedup, group split, kunci test set |
| C | Custom CNN + ablasi |
| D | Model pretrained + Grad-CAM + model pembanding |
| Semua | Analisis, laporan, dan latihan demo (setiap anggota wajib paham seluruh alur) |

---

## 8. Timeline Saran

| Tahap | Kegiatan |
|---|---|
| Minggu 1 | Tetapkan topik dan definisi kelas, kumpulkan data, susun metadata |
| Minggu 1–2 | Dedup, split, baseline Custom CNN |
| Minggu 2 | Model pretrained, eksperimen tambahan, evaluasi test set (sekali di akhir) |
| Minggu 3 | Analisis, laporan BAB I–V, latihan demo |
