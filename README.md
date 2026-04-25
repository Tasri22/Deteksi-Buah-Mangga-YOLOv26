## 📺 Deteksi Buah Mangga Real-Time dengan YOLOv26 (Roboflow + Google Colab + Streamlit)

# Mango Detection System using YOLOv8 🥭

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![Framework](https://img.shields.io/badge/Framework-Roboflow-orange.svg)](https://roboflow.com/)

Proyek ini bertujuan untuk mendeteksi buah mangga secara otomatis dalam gambar menggunakan model **YOLOv26**. Sistem ini dikembangkan sebagai bagian dari tugas mata kuliah **Machine Learning (Informatika C)**.

## 👤 Identitas Pengembang
- **Nama:** Tasri Zulfitriyati
- **NIM:** 231001074
- **Kelas:** Informatika C

## Video Presentasi
[Klik Link](https://youtu.be/pITB-8j-Wgw?si=NFz5LWzffJMSuwPh)


## 🚀 Fitur Utama
* **Dataset Custom:** Menggunakan 26 gambar mangga dengan variasi sudut dan pencahayaan.
* **High Accuracy:** Mencapai mAP50 yang optimal untuk deteksi di lapangan.
* **Interactive Testing:** Notebook dilengkapi dengan slider confidence untuk pengujian real-time.
* **Model Export:** Mendukung format `.pt` dan `.onnx` untuk deployment.

## 🛠️ Langkah Instalasi
1. Clone repository ini:
   ```bash
   git clone [https://github.com/username/yolo-mango-detection.git](https://github.com/username/yolo-mango-detection.git)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Hasil Training
Berdasarkan eksperimen selama 50 epoch menggunakan YOLOv26:
| Metrik | Nilai (Validation Set) |
|---|---|
| **Precision** | 0.89 |
| **Recall** | 0.84 |
| **mAP50** | 0.91 |

## 💻 Cara Penggunaan
Buka file `training.ipynb` di Google Colab. Pastikan Anda sudah memiliki API Key dari Roboflow untuk mengunduh dataset secara otomatis.

## 📂 Struktur Folder
```text
├── models/             # Menyimpan model terbaik (best.pt)
├── notebook/           # File .ipynb untuk training di Colab
├── requirements.txt    # Daftar library yang diperlukan
└── README.md           # Dokumentasi proyek
```

## 📝 Kesimpulan & Kendala
Proyek ini membuktikan bahwa YOLOv26 sangat efisien untuk dataset skala kecil. Kendala utama yang dihadapi adalah anotasi manual yang memakan waktu dan variasi pencahayaan pada gambar, yang diatasi dengan teknik augmentasi data.
```
