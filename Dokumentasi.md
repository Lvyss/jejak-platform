# Dokumentasi Panduan Proyek: Deteksi Asal Platform Media Sosial pada Foto Digital

Dokumen ini dirangkum dari seluruh diskusi kita. Angka bertanda "ilustrasi" belum terbukti, nanti dibuktikan datamu.

---

## 1. Ringkasan proyek

**Judul:** Deteksi Asal Platform Media Sosial pada Foto Digital untuk Verifikasi Riwayat Kompresi Menggunakan Analisis Spektrum DFT 2D dan Fitur Spasial

**Pertanyaan inti:** Apakah foto yang sama, setelah dikirim lewat platform berbeda (WA, TG, IG, X), meninggalkan jejak statistik yang cukup berbeda sehingga platformnya bisa ditebak? Fitur mana yang paling membedakan, dan kapan metodenya gagal?

**Nilai jual:**
- Data berlabel pasti (kamu sendiri yang mengirim).
- Desain berpasangan: satu objek, lima versi, jadi isi foto tidak jadi pembeda.
- Fitur bisa dijelaskan secara fisik (kompresi, resize, blok 8×8).
- Ada analisis kegagalan yang jujur.

---

## 2. Tujuan khusus

1. **Dataset:** foto dengan asal platform pasti.
2. **Fitur:** ubah jejak kompresi jadi angka (DFT 2D + fitur spasial).
3. **Klasifier:** LogReg (utama) dan SVM (pembanding).
4. **Analisis fitur terbaik:** bandingkan kelompok fitur.

---

## 3. Tech stack dan spesifikasi

| Komponen | Pilihan |
|---|---|
| Bahasa | Python 3.10+ |
| Library | numpy, scipy, pillow, pandas, scikit-learn, matplotlib |
| Lingkungan | Jupyter/VS Code atau Google Colab |
| Perangkat | 1 HP (kamera bawaan, tanpa filter/beauty/HDR bila bisa), akun/chat cadangan di WA, TG, IG, X |
| Skala data | 30-40 objek × 5 versi = 150-200 foto |
| Komputasi | Laptop biasa cukup (ekstraksi beberapa menit) |

Instalasi: `pip install numpy scipy pillow pandas scikit-learn matplotlib`

---

## 4. Struktur folder

```
project/
├── data/
│   ├── obj01/  O.jpg  WA.jpg  TG.jpg  IG.jpg  X.jpg
│   ├── obj02/  ...
├── ekstraksi.py      # foto -> fitur.csv
├── evaluasi.py       # fitur.csv -> hasil
├── fitur.csv
├── hasil/            # gambar spektrum, confusion matrix, tabel
└── laporan/
```

Nama folder = ID grup (untuk GroupKFold). Nama file = label.

---

## 5. Rencana kerja (sesuaikan deadline)

1. **Minggu 1:** pengambilan dan pengiriman foto, penyusunan folder, cek format dan ukuran.
2. **Minggu 2:** tulis dan uji ekstraksi fitur, hasilkan fitur.csv, visualisasi spektrum rata-rata per platform.
3. **Minggu 3:** klasifikasi, evaluasi, perbandingan fitur, analisis kegagalan.
4. **Minggu 4:** laporan, slide, (opsional) demo.

---

## 6. Alur kerja detail

### Step 1: Ambil dan kirim foto
- Foto 30-40 objek beragam (tekstur halus dan kasar, terang dan gelap, dalam dan luar ruangan) dengan satu HP, pengaturan sama.
- O = file asli dari kamera, disalin langsung.
- WA/TG/IG/X: kirim ke chat/akun sendiri sebagai **foto biasa** (bukan "dokumen/file"), lalu unduh kembali.
- Pakai satu mode pengiriman per platform, konsisten.
- Catat versi aplikasi dan tanggal.
- Jangan screenshot dan jangan simpan ulang lewat aplikasi lain.

### Step 2: Susun folder
Sesuai struktur di bagian 4.

### Step 3: Cek awal (wajib sebelum ekstraksi)
```python
from PIL import Image
import glob
for p in sorted(glob.glob("data/*/*.*")):
    im = Image.open(p)
    print(p, im.format, im.size)
```
Catat: format tiap platform (JPEG/WebP/PNG), apakah ukuran berubah (artinya ada resize), apakah ada versi yang identik dengan O (bandingkan hash file atau selisih piksel). Platform yang identik secara statistik tidak bisa dibedakan, itu temuan sah.

Opsional: `im.quantization` untuk membaca tabel kuantisasi JPEG. Itu fitur tingkat-file, pisahkan dari fitur citra bila dipakai.

### Step 4: Pra-proses
Crop tengah 512×512 **tanpa resize**, grayscale, offset crop kelipatan 8 (agar grid blok JPEG tidak bergeser).

```python
import numpy as np
from PIL import Image

def load(path, size=512):
    img = Image.open(path).convert("L")
    w, h = img.size
    left = ((w - size)//2)//8*8
    top  = ((h - size)//2)//8*8
    img = img.crop((left, top, left+size, top+size))
    return np.asarray(img, dtype=np.float64)
```

Catatan: kalau platform meresize, ukuran asli adalah informasi yang membedakan. Catat, dan putuskan apakah dijadikan fitur tambahan (tulis di laporan).

### Step 5: Ekstraksi fitur (57 angka per foto)

| Fitur | Jumlah | Apa yang diukur | Cara |
|---|---|---|---|
| DFT radial | 32 | Energi per tingkat kerapatan pola | fft2, fftshift, log1p(abs), rata-rata per 32 cincin dari pusat |
| Puncak blok JPEG | 1 | Seberapa menonjol puncak periodik di jarak 64 dari pusat spektrum | selisih puncak vs sekitarnya |
| Blockiness | 2 | Loncatan piksel di perbatasan blok 8×8 dibanding tempat lain | rasio rata-rata diff di kolom/baris 7|8, 15|16, dst. |
| Noise | 2 | Butiran halus | residual = img − gaussian blur; ambil std dan kurtosis |
| Tepi | 2 | Ketajaman tepi | Sobel, magnitudo; rata-rata dan proporsi di atas ambang 50 |
| Histogram | 18 | Sebaran kecerahan | 16 bin + mean + std |

```python
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import kurtosis

def fitur_dft(img):
    w = np.outer(np.hanning(512), np.hanning(512))
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * w))
    P = np.log1p(np.abs(F))
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy-256, xx-256).astype(int)
    bins = np.linspace(0, 256, 33).astype(int)
    radial = [P[(r>=bins[i])&(r<bins[i+1])].mean() for i in range(32)]
    puncak  = np.mean([P[256,256+64], P[256,256-64], P[256+64,256], P[256-64,256]])
    sekitar = np.mean([P[256,256+60], P[256,256-60], P[256+60,256], P[256-60,256]])
    return radial + [puncak - sekitar]

def fitur_blockiness(img):
    dx = np.abs(np.diff(img, axis=1)); dy = np.abs(np.diff(img, axis=0))
    return [dx[:, 7::8].mean()/dx.mean(), dy[7::8, :].mean()/dy.mean()]

def fitur_noise(img):
    res = img - gaussian_filter(img, 1.0)
    return [res.std(), kurtosis(res.ravel())]

def fitur_tepi(img):
    mag = np.hypot(sobel(img, 0), sobel(img, 1))
    return [mag.mean(), (mag > 50).mean()]

def fitur_hist(img):
    h, _ = np.histogram(img, bins=16, range=(0,255), density=True)
    return list(h) + [img.mean(), img.std()]
```

Catatan teknis: ambang 50 pada tepi adalah tebakan awal, cek distribusinya lalu sesuaikan. Posisi puncak 64 berlaku untuk blok 8 pada gambar 512; bila platform meresize, grid blok bergeser dan puncaknya bisa pindah.

### Step 6: Susun fitur.csv
```python
import glob, os, pandas as pd
rows = []
for path in sorted(glob.glob("data/*/*.*")):
    obj = os.path.basename(os.path.dirname(path))
    label = os.path.splitext(os.path.basename(path))[0]
    img = load(path)
    f = (fitur_dft(img) + fitur_blockiness(img) + fitur_noise(img)
         + fitur_tepi(img) + fitur_hist(img))
    rows.append([obj, label] + f)

kolom = (["obj","label"] + [f"dft_{i}" for i in range(33)]
         + ["blk_x","blk_y","noise_std","noise_kurt","edge_mean","edge_frac"]
         + [f"hist_{i}" for i in range(16)] + ["lum_mean","lum_std"])
pd.DataFrame(rows, columns=kolom).to_csv("fitur.csv", index=False)
```
Hasil: sekitar 175 baris × 59 kolom (2 keterangan + 57 fitur). Satu file, satu tabel. Data mentah (gambar, spektrum 2D) tidak dimasukkan; simpan spektrum rata-rata sebagai gambar untuk laporan.

### Step 7: Klasifikasi
```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

df = pd.read_csv("fitur.csv")
X = df.drop(columns=["obj","label"]).values
y = df["label"].values
groups = df["obj"].values

logreg = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
svm    = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
```

**Logika singkat:**
- **LogReg:** tiap kelas punya bobot per fitur; skor = jumlah bobot × fitur; softmax mengubah skor jadi peluang; kelas terbesar jadi tebakan. Bobot dicari dengan tebak, ukur salah, geser bobot, ulang. Bobotnya bisa dibaca, jadi dipakai untuk tujuan 4.
- **SVM (RBF):** mencari batas pemisah dengan margin terlebar dari titik terdekat (support vector), batas boleh melengkung. Tebakan = sisi tempat foto baru jatuh; 5 kelas lewat voting antar pasangan. Tidak mengubah data, hanya memilih letak batas.
- **Kenapa keduanya:** cek silang. Akurasi mirip berarti polanya sederhana; SVM jauh lebih tinggi berarti ada pola tak lurus; keduanya rendah berarti masalah di fitur/data.

### Step 8: Evaluasi
```python
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

gkf = GroupKFold(n_splits=5)
labels = ["O","WA","TG","IG","X"]
for nama, model in [("LogReg", logreg), ("SVM", svm)]:
    pred = cross_val_predict(model, X, y, groups=groups, cv=gkf)
    print(nama, accuracy_score(y, pred))
    cm = confusion_matrix(y, pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot()
    skor = cross_val_score(model, X, y, groups=groups, cv=gkf)
    print("per fold:", skor, "std:", skor.std())
```

**Aturan wajib:**
- `GroupKFold` dengan grup = ID objek, supaya kelima versi satu objek tidak terpecah antara latih dan uji.
- Standarisasi di dalam pipeline (tidak bocor).
- Laporkan akurasi per fold dan simpangan baku (data kecil, hasil berfluktuasi).
- Pembanding acak: 20% untuk 5 kelas seimbang.

### Step 9: Perbandingan fitur (tujuan 4)
Model dikunci (LogReg), kolom yang diganti.

```python
kelompok = {
    "DFT saja": [c for c in df.columns if c.startswith("dft_")],
    "Blockiness saja": ["blk_x","blk_y"],
    "Noise saja": ["noise_std","noise_kurt"],
    "Tepi saja": ["edge_mean","edge_frac"],
    "Histogram saja": [c for c in df.columns if c.startswith(("hist","lum"))],
    "Spasial saja": [c for c in df.columns if c.startswith(("blk","noise","edge","hist","lum"))],
    "Gabungan": [c for c in df.columns if c not in ("obj","label")],
}
for nama, cols in kelompok.items():
    pred = cross_val_predict(logreg, df[cols].values, y, groups=groups, cv=gkf)
    print(f"{nama:18s} {accuracy_score(y, pred):.3f}")
```
Fitur individual terpenting: latih LogReg di seluruh data, lihat `logreg[-1].coef_` (5 kelas × 57 kolom), atau gunakan `permutation_importance`. Kolom DFT saling berkorelasi, jadi bobot hanya petunjuk.

Dua perbandingan jangan dicampur: model vs model (fitur dikunci), fitur vs fitur (model dikunci).

### Step 10: Visualisasi untuk laporan
1. Spektrum 2D rata-rata per platform, berdampingan.
2. Kurva radial rata-rata per platform.
3. Tabel akurasi LogReg vs SVM.
4. Confusion matrix.
5. Tabel perbandingan kelompok fitur.
6. Peta Sobel contoh O vs WA.

---

## 7. Keterbatasan dan risiko (wajib ditulis di laporan)

- **Data kecil** (sekitar 35 objek): hasil fluktuatif, laporkan std antar fold.
- **Foto O sudah JPEG:** versi platform membawa kompresi ganda; yang terukur terutama kompresi terakhir.
- **Platform bisa identik atau tak terbedakan** (misalnya TG kirim tanpa kompresi): temuan sah.
- **Bergantung versi aplikasi dan HP:** pipeline platform bisa berubah.
- **Resize platform** menggeser grid blok; fitur blockiness dan puncak DFT bisa melemah. Format WebP/HEIC tidak memakai blok 8×8.
- **Fitur sensitif isi foto** (noise, tepi, histogram): beragamkan objek; opsi fitur rasio terhadap O per objek.
- **Ketimpangan jumlah kolom:** DFT 33 kolom vs blockiness 2 kolom, jadi "DFT terbaik" bisa karena kolomnya banyak. Tulis atau tambahkan percobaan DFT yang diringkas.
- **Cakupan:** hanya berlaku untuk kondisi pengiriman yang kamu pilih; bukan klaim umum.

---

## 8. Peta konsep ke materi kuliah

| Konsep | Dipakai di | Slide |
|---|---|---|
| DFT 2-D, spektrum | Fitur DFT | Ch4: 14 |
| Pemusatan spektrum (fftshift) | Fitur DFT | Ch4: 14, 19 |
| Periodisitas | Puncak blok 8×8 | Ch4: 19 |
| Aliasing, Nyquist | Dampak resize/downsampling | Ch4: 6-10 |
| Teorema konvolusi | Dasar pemfilteran | Ch4: 5, 23 |
| Unsharp mask (f − f_blur) | Residual noise | Ch3: 61-74, Ch4: 26 |
| Sobel, magnitudo gradien | Fitur tepi | Ch3: 61-74 |
| Histogram, mean, varians | Fitur histogram | Ch3: 20-49 |
| Filter Gaussian | Pembuat residual | Ch3: 55-60, Ch4: 24 |
| Sampling, resolusi, interpolasi | Dampak resize | Ch2 |

Nomor slide Ch2-3 diambil dari ringkasan, bukan dari PDF yang kubuka langsung; cocokkan lewat judul slide.

---

## 9. Struktur laporan

1. Pendahuluan (masalah, tujuan, kontribusi)
2. Landasan teori (JPEG dan blok 8×8, DFT 2D, aliasing, Sobel, LogReg, SVM, cross-validation)
3. Metodologi (dataset, protokol pengiriman, pra-proses, fitur, model, evaluasi)
4. Hasil (visualisasi spektrum, akurasi, confusion matrix, perbandingan fitur)
5. Pembahasan (platform mudah/sulit dibedakan dan alasannya, peran DFT vs spasial, kegagalan)
6. Keterbatasan dan penelitian lanjutan
7. Kesimpulan

---

## 10. Daftar periksa sebelum mulai

- [ ] HP pengaturan seragam, filter dan HDR dimatikan
- [ ] Format asli HP: JPG (sudah dikonfirmasi)
- [ ] Akun/chat cadangan di WA, TG, IG, X
- [ ] Satu mode pengiriman per platform dan dicatat
- [ ] Objek beragam, 30-40 buah
- [ ] Format dan ukuran tiap versi dicek dan dicatat
- [ ] Versi aplikasi dan tanggal dicatat
- [ ] Python dan library terpasang

## 11. Ide pengembangan (opsional)

- Fitur rasio terhadap O per objek.
- Fitur tabel kuantisasi JPEG (dipisah sebagai fitur tingkat-file).
- Uji ketahanan: kirim ulang foto (WA lalu IG) untuk melihat jejak berlapis.
- Demo Streamlit: unggah foto, tampilkan spektrum dan tebakan platform dengan catatan keterbatasan.

---

Kalau mau, aku bisa gabungkan kode jadi `ekstraksi.py` dan `evaluasi.py` utuh, atau bantu menyusun kerangka laporan per bab.