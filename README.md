# Sistem Pengenalan Wajah dengan Eigenface

Implementasi sistem pengenalan wajah menggunakan metode eigenface dengan perhitungan eigenvalue/eigenvector dan euclidean distance secara manual.

## Demo Online

**Akses aplikasi:** [https://facerec-eigen.streamlit.app/](https://facerec2-eigenfrhbxpudmdhpcdxr6xx3ez.streamlit.app/)

## Fitur 

- Perhitungan Eigenvalue/Eigenvector custom dengan algoritma power iteration
- Euclidean Distance manual
- Pengenalan berbasis eigenface dengan pendekatan PCA klasik untuk face recognition
- Antarmuka grafis interaktif dengan web interface berbasis Streamlit
- Persistensi model dengan menyimpan dan memuat model yang sudah dilatih
- Pengujian fleksibel dengan upload gambar atau pilih dari folder test

## Implementasi Teknis

### Komponen Utama

1. **FaceRec.py** - Engine pengenalan wajah
   - Dekomposisi eigenvalue custom menggunakan power iteration
   - Komputasi dan proyeksi eigenface
   - Perhitungan euclidean distance manual
   - Fungsi training dan recognition

2. **FaceRec_Gui.py** - Interface GUI Streamlit
   - Loading dan management model
   - Upload dan testing gambar
   - Visualisasi hasil
   - Kontrol yang user-friendly

## Struktur Proyek

```
face-recognition/
│
├── FaceRec.py          # Algoritma pengenalan inti
├── FaceRec_Gui.py      # Aplikasi GUI Streamlit
├── dataset_subset/     # Dataset training
├── test_img/           # Gambar test
├── saved_models/       # File model terlatih
└── README.md           # File ini
```

## Cara Menjalankan

1. **Install Dependencies**
   ```bash
   pip install streamlit numpy matplotlib opencv-python pillow
   ```

2. **Persiapkan Dataset**
   - Letakkan gambar training di folder `dataset_subset/`
   - Organisir per orang: `dataset_subset/person1/`, `dataset_subset/person2/`, dll.

3. **Training Model** (jika diperlukan)
   ```python
   from FaceRec import train_eigenface_model, save_model
   
   # Train dan simpan model
   mean_face, eigenfaces, projected_train, eigenvalues, X, labels = train_eigenface_model('dataset_subset')
   save_model(mean_face, eigenfaces, projected_train, eigenvalues, X, labels)
   ```

4. **Jalankan Aplikasi**
   ```bash
   streamlit run Main.py
   ```

## Cara Penggunaan

1. **Load Model**: Klik "Muat Model Terlatih" di sidebar
2. **Test Gambar**: 
   - Upload gambar baru ATAU pilih dari folder test
   - Sesuaikan threshold jika diperlukan
   - Klik "Kenali Wajah" untuk mengenali
3. **Lihat Hasil**: Melihat hasil pencocokan dan tingkat kepercayaan

## Konfigurasi

- **Threshold**: Sesuaikan sensitivitas pengenalan (5.0 - 30.0)
- **Eigenfaces**: Default 25 komponen utama
- **Ukuran Gambar**: 100x100 piksel (grayscale)

## Performa

- **Waktu Training**: ~30-60 menit (tergantung ukuran dataset)
- **Waktu Recognition**: <1 detik per gambar
- **Akurasi**: Tergantung kualitas dataset dan pengaturan threshold


Proyek ini mendemonstrasikan:
- Konsep aljabar linear dalam computer vision
- Implementasi Principal Component Analysis (PCA)
- Pengembangan algoritma matematika custom
- Fundamental pengenalan wajah

## Lisensi

Proyek ini dikembangkan untuk tujuan edukasi.
