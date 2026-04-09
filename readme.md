# 🐾 WildLens — Klasifikasi Wajah Hewan

> **UTS Praktikum Kecerdasan Buatan 2026**  
> Klasifikasi wajah hewan berbasis Deep Learning menggunakan Transfer Learning MobileNetV2 dengan akurasi **99.7%** dan selisih train/val < 0.3%.

![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-gold)
![Accuracy](https://img.shields.io/badge/Akurasi-99.7%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Good%20Fit-success)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Flask](https://img.shields.io/badge/Backend-Flask-blue)

---

## 📁 Repository & Dataset

| Sumber | Link |
|--------|------|
| 💻 Kode Program | [github.com/TEPPN/Animal-Classifier](https://github.com/akuangell/Responsi-Ai.git) |
| 📦 Dataset | [kaggle.com · AFHQ Animal Faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces) |

---

## 📋 Soal 1 — Optimalisasi Model

Model awal menunjukkan overfitting yang signifikan. Berikut analisis masalah dan seluruh perbaikan yang dilakukan untuk mencapai kondisi **good fit**.

### Analisis Masalah

| Metrik | ⚠️ Model Awal (Overfitting) | ✅ Model Optimized (Good Fit) |
|--------|---------------------------|------------------------------|
| Training Loss | ~0.00 | ~0.01 (konvergen) |
| Validation Loss | ~0.05–0.10 | ~0.01 (sejajar) |
| Training Accuracy | ~99–100% | ~99.7% |
| Val Accuracy | ~95–97% | ~99.5–99.7% |
| Selisih Acc | ~3–5% ❌ | < 0.3% ✅ |

---

### 7 Perbaikan yang Dilakukan

#### 1. Perbaikan Data Split

**Masalah:** Kode asal menggunakan `sample(frac=0.7)` tanpa shuffle, menyebabkan distribusi tidak merata dan potensi data leakage antar split.

**Solusi:** Data dikocok dengan `random_state=42` lalu dibagi bersih: 70% train · 15% val · 15% test menggunakan `iloc`.

```python
# Shuffle seluruh data dulu
data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(data_df)
train = data_df.iloc[:int(0.70*n)]           # 70%
val   = data_df.iloc[int(0.70*n):int(0.85*n)]  # 15%
test  = data_df.iloc[int(0.85*n):]           # 15%
```

---

#### 2. Transfer Learning MobileNetV2 ⭐ Kunci Utama

**Masalah:** CNN dari scratch butuh data sangat banyak dan mudah overfitting karena belajar dari nol tanpa representasi fitur yang kuat.

**Solusi:** MobileNetV2 pretrained ImageNet digunakan sebagai backbone. Bobot yang sudah matang membuat model langsung akurat dari epoch pertama — training dan validasi konvergen bersama tanpa gap besar. Classifier head diganti untuk 3 kelas dengan Dropout + BatchNorm1d.

```python
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 3),  # cat / dog / wild
)
```

---

#### 3. Augmentasi Data Lebih Ringan

**Masalah:** Augmentasi agresif (rotasi 15°, ColorJitter besar) membuat training loss tinggi di awal sehingga kurva val terlihat lebih tinggi dari training — tampak tidak wajar.

**Solusi:** Dikurangi menjadi hanya RandomHorizontalFlip dan ColorJitter ringan (0.1). Input disesuaikan ke 224×224 sesuai MobileNetV2. Val/Test tidak diberi augmentasi.

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
```

---

#### 4. Differential Learning Rate

**Masalah:** Satu learning rate untuk semua layer — backbone pretrained bisa rusak jika lr terlalu besar, sementara head baru butuh lr yang cukup besar untuk belajar.

**Solusi:** Backbone diberi lr kecil (1e-5) agar bobot pretrained tidak rusak. Classifier head diberi lr lebih besar (1e-3) agar cepat menyesuaikan diri ke tugas 3 kelas.

```python
optimizer = Adam([
    {'params': model.features.parameters(),    'lr': 1e-5},  # backbone
    {'params': model.classifier.parameters(), 'lr': 1e-3},  # head
], weight_decay=1e-4)
```

---

#### 5. Weight Decay — L2 Regularization

**Masalah:** Tanpa regularisasi, bobot model bisa tumbuh sangat besar dan mendorong model untuk menghafal data training alih-alih belajar pola general.

**Solusi:** Ditambahkan `weight_decay=1e-4` pada Adam sebagai L2 regularization — menambahkan penalty pada bobot besar dan mendorong solusi yang lebih general.

---

#### 6. LR Scheduler — ReduceLROnPlateau

**Masalah:** Learning rate yang konstan menyebabkan training melompat-lompat dan tidak menemukan minimum yang baik di tahap akhir.

**Solusi:** ReduceLROnPlateau mengurangi lr sebesar 50% jika val loss tidak membaik selama 3 epoch, membantu model fine-tune lebih halus saat konvergensi.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
scheduler.step(avg_val_loss)  # dipanggil tiap epoch
```

---

#### 7. Early Stopping + Simpan Model Terbaik

**Masalah:** Training berlanjut meskipun model sudah mulai overfit, membuang komputasi dan berpotensi memperburuk generalisasi.

**Solusi:** Early stopping patience=5: training berhenti otomatis jika val loss tidak membaik. Model terbaik disimpan ke `best_model.pth` — file bobot model yang dihasilkan dari training — dan dimuat kembali di akhir.

```python
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    patience_counter = 0
    torch.save(model.state_dict(), 'best_model.pth')  # simpan bobot terbaik
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:
        break  # early stopping
model.load_state_dict(torch.load('best_model.pth'))  # load terbaik
```

---

### Perbandingan Hasil

| Komponen | ⚠️ Model Awal | ✅ Model Optimized |
|----------|--------------|-------------------|
| Arsitektur | CNN dari scratch | MobileNetV2 pretrained |
| Input Size | 128 × 128 | 224 × 224 |
| Data Split | Tanpa shuffle | Shuffle random_state=42 |
| Augmentasi | Rotation + ColorJitter agresif | Flip + ColorJitter ringan |
| Regularisasi | Tidak ada | Dropout + BatchNorm + WeightDecay |
| LR Strategy | Flat 1e-4 | Differential + ReduceLROnPlateau |
| Early Stopping | Tidak ada | Patience=5 + best_model.pth |
| Selisih Acc | ~3–5% (tidak stabil) | < 0.3% (stabil) |
| **Status** | 🔴 Overfitting | 🟢 Good Fit |

---

## 🌐 Soal 2 — Implementasi Website

Model MobileNetV2 hasil training diimplementasikan ke website fungsional menggunakan **Flask Backend + HTML/CSS/JS Frontend**. Sepenuhnya memakai bobot model sendiri (`best_model.pth`) — tanpa API pihak lain.

| Sumber | Link |
|--------|------|
| 🌐 Repository Website | [github.com/TEPPN/Animal-Classifier](https://github.com/akuangell/Responsi-Ai.git) |

---

### Arsitektur Sistem

| Komponen | File | Deskripsi |
|----------|------|-----------|
| 🖥️ Frontend | `animal_classifier_web.html` | Antarmuka berbasis HTML/CSS/JS murni. Drag-and-drop upload, preview gambar, animasi loading, confidence bar per kelas, dan dark theme responsif. |
| ⚙️ Backend | `app.py` (Flask) | Server Python yang memuat best_model.pth, menerima gambar via POST /predict, melakukan preprocessing + inference, dan mengembalikan JSON prediksi. |

---

### Alur Kerja

```
1. [Browser — Frontend]
   User drag-and-drop atau klik untuk memilih foto hewan dari perangkat.
        │
        ▼
2. [Browser → Flask — HTTP POST]
   Gambar dikonversi ke FormData dan dikirim ke endpoint POST /predict.
        │
        ▼
3. [Flask — Preprocessing]
   Gambar dibuka PIL, di-resize ke 224×224, dinormalisasi dengan mean/std ImageNet.
        │
        ▼
4. [best_model.pth — MobileNetV2]
   Model hasil training melakukan prediksi, menghasilkan logits untuk 3 kelas (kucing/anjing/liar).
        │
        ▼
5. [Flask → Browser — JSON]
   Softmax → probabilitas dikembalikan sebagai JSON → ditampilkan dengan confidence bar animasi.
```

---

### Cara Menjalankan

**Step 1 — Download model dari Google Colab (setelah training selesai)**

```python
# Jalankan di sel Colab
from google.colab import files
files.download('best_model.pth')  # file bobot hasil training
```

**Step 2 — Install dependencies**

```bash
pip install flask flask-cors torch torchvision pillow
```

**Step 3 — Susun semua file dalam satu folder**

```
📁 project/
├── app.py
├── best_model.pth          ← hasil download dari Colab
├── requirements.txt
└── animal_classifier_web.html
```

**Step 4 — Jalankan backend lalu buka website**

```bash
python app.py
# Server aktif di http://localhost:5000
# Buka animal_classifier_web.html di browser, lalu upload foto!
```

---

### Fitur Website

| Fitur | Deskripsi |
|-------|-----------|
| 🖱️ Drag & Drop | Upload gambar dengan drag-and-drop atau klik — JPG, PNG, WEBP, GIF |
| 👁️ Preview Gambar | Tampilan preview beserta nama file dan ukuran sebelum dianalisis |
| 📊 Confidence Bar | Bar persentase keyakinan animasi dari output softmax model |
| 🔄 Model Sendiri | 100% menggunakan best_model.pth hasil training — tanpa API lain |
| 🌙 Dark Theme | Desain dark theme profesional dengan animasi dan tampilan responsif |
| ⚠️ Error Handling | Pesan error informatif jika server tidak aktif atau input tidak valid |

---

## 🛠️ Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-gold)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)

---

## 📊 Dataset

**AFHQ Animal Faces** — [kaggle.com/datasets/andrewmvd/animal-faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

- 🐱 Kucing (Cat)
- 🐶 Anjing (Dog)
- 🦁 Hewan Liar (Wild)

---
