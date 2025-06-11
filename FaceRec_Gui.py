# -*- coding: utf-8 -*-
"""
FaceRec_GUI.py - Interface Grafis untuk Sistem Pengenalan Wajah
GUI berbasis Streamlit untuk implementasi eigenface recognition
Tugas Aljabar Linear - Universitas Sebelas Maret
"""

import streamlit as st
from PIL import Image

# ===== KONFIGURASI STREAMLIT =====
# PENTING: set_page_config HARUS menjadi perintah streamlit pertama!
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Import library lainnya SETELAH konfigurasi page
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# ===== IMPORT MODULE FACE RECOGNITION DENGAN ERROR HANDLING =====
face_rec_available = False
face_rec_error = ""

def safe_import_facerec():
    """
    Import modul FaceRec dengan penanganan error yang aman
    Mencegah aplikasi crash jika modul tidak ditemukan
    
    Returns:
        bool: True jika import berhasil, False jika gagal
    """
    global face_rec_available, face_rec_error, load_model, recognize_face, quick_recognize

    try:
        # Import fungsi-fungsi utama dari modul FaceRec
        from FaceRec import load_model, recognize_face, quick_recognize
        face_rec_available = True
        return True
    except Exception as e:
        face_rec_error = str(e)
        face_rec_available = False
        return False

def show_import_status():
    """
    Menampilkan status import modul FaceRec
    Memberikan feedback kepada user tentang ketersediaan sistem
    """
    # Coba import setelah streamlit diinisialisasi dengan baik
    if not face_rec_available:
        safe_import_facerec()

    if face_rec_available:
        st.success("✅ Modul FaceRec berhasil dimuat!")
    else:
        st.error(f"❌ Error mengimport FaceRec: {face_rec_error}")
        st.info("💡 Pastikan file FaceRec.py berada di direktori yang sama!")

# ===== FUNGSI UTAMA APLIKASI =====

def main():
    """
    Fungsi utama yang menjalankan seluruh interface GUI
    Mengatur layout, komponen, dan alur kerja aplikasi
    """
    
    # ===== HEADER APLIKASI =====
    st.title("🎭 Sistem Pengenalan Wajah")
    st.markdown("**Implementasi Eigenface dengan Custom Eigenvalue/Eigenvector**")
    st.markdown("---")

    # Tampilkan status import modul
    show_import_status()

    # Hentikan eksekusi jika modul FaceRec tidak tersedia
    if not face_rec_available:
        st.stop()

    # ===== SIDEBAR UNTUK PENGATURAN MODEL =====
    with st.sidebar:
        st.header("🔧 Pengaturan Model")
        st.markdown("*Muat model yang sudah ditraining*")

        # Tombol untuk load model
        if st.button("📥 Muat Model Terlatih"):
            with st.spinner("Memuat model..."):
                try:
                    # Load model yang sudah disimpan dari training
                    mean_face, eigenfaces, projected_train, eigenvalues, X, labels = load_model('saved_models')

                    if mean_face is not None:
                        # Simpan komponen model dalam session state untuk persistensi
                        st.session_state.model_loaded = True
                        st.session_state.mean_face = mean_face
                        st.session_state.eigenfaces = eigenfaces
                        st.session_state.projected_train = projected_train
                        st.session_state.eigenvalues = eigenvalues
                        st.session_state.X = X
                        st.session_state.labels = labels

                        st.success("✅ Model berhasil dimuat!")
                        
                        # Tampilkan statistik model
                        unique_people = len(set(labels))
                        total_images = len(labels)
                        eigenface_count = eigenfaces.shape[1]
                        
                        st.info(f"📊 **Statistik Model:**")
                        st.write(f"• Jumlah orang: {unique_people}")
                        st.write(f"• Total gambar training: {total_images}")
                        st.write(f"• Jumlah eigenface: {eigenface_count}")
                        
                    else:
                        st.error("❌ Gagal memuat model!")
                        st.warning("💡 Pastikan sudah menjalankan training dan menyimpan model")

                except Exception as e:
                    st.error(f"❌ Error memuat model: {e}")

    # ===== LAYOUT UTAMA: DUA KOLOM =====
    col1, col2 = st.columns([1, 1])

    # ===== KOLOM 1: INFORMASI DATASET =====
    with col1:
        st.header("📁 Informasi Dataset")

        # Dropdown untuk pemilihan dataset (untuk tampilan)
        dataset_path = st.selectbox(
            "Pilih Folder Dataset:",
            ["dataset_subset", "custom_dataset", "lfw_subset"],
            index=0,
            help="Dataset yang digunakan untuk training model"
        )

        # Validasi keberadaan dataset
        if os.path.exists(dataset_path):
            st.success(f"✅ Dataset ditemukan: {dataset_path}")

            # Analisis informasi dataset
            try:
                # Dapatkan daftar orang dalam dataset
                persons = os.listdir(dataset_path)
                persons = [p for p in persons if os.path.isdir(os.path.join(dataset_path, p))]
                
                if persons:
                    st.info(f"👥 Ditemukan {len(persons)} orang dalam dataset")

                    # Expandable section untuk detail dataset
                    with st.expander("👀 Lihat Detail Dataset"):
                        total_images = 0
                        for person in sorted(persons):
                            person_path = os.path.join(dataset_path, person)
                            img_files = [f for f in os.listdir(person_path)
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                            img_count = len(img_files)
                            total_images += img_count
                            st.write(f"• **{person}**: {img_count} gambar")
                        
                        st.markdown(f"**Total: {total_images} gambar**")
                else:
                    st.warning("⚠️ Dataset kosong atau tidak berisi folder orang")

            except Exception as e:
                st.warning(f"⚠️ Tidak dapat membaca dataset: {e}")
        else:
            st.error(f"❌ Dataset tidak ditemukan: {dataset_path}")
            st.info("💡 Pastikan folder dataset ada di direktori yang sama")

    # ===== KOLOM 2: PENGUJIAN GAMBAR =====
    with col2:
        st.header("🖼️ Pengujian Gambar")

        # ===== OPSI 1: UPLOAD GAMBAR =====
        st.subheader("📤 Upload Gambar Test")
        uploaded_file = st.file_uploader(
            "Pilih gambar untuk diuji...",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=False,
            help="Upload gambar wajah yang ingin dikenali"
        )

        # ===== OPSI 2: PILIH DARI FOLDER TEST =====
        st.subheader("📂 Atau Pilih dari Folder Test")
        test_folder = "test_img"

        # Cek keberadaan folder test
        if os.path.exists(test_folder):
            test_images = [f for f in os.listdir(test_folder)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if test_images:
                selected_test = st.selectbox(
                    "Pilih gambar test:",
                    ["None"] + sorted(test_images),
                    help="Gambar test yang tersedia di folder test_img"
                )

                # ===== PROSES GAMBAR YANG DIPILIH =====
                if selected_test != "None":
                    test_image_path = os.path.join(test_folder, selected_test)

                    # Tampilkan gambar test
                    try:
                        test_img = Image.open(test_image_path)
                        st.image(test_img, caption=f"Gambar Test: {selected_test}", width=300)

                        # ===== TOMBOL PENGENALAN WAJAH =====
                        if st.button("🔍 Kenali Wajah", type="primary"):
                            
                            # Validasi model sudah dimuat
                            if hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded:

                                with st.spinner("Mengenali wajah..."):
                                    try:
                                        # Pengaturan threshold dari sidebar
                                        threshold = st.sidebar.slider(
                                            "🎯 Threshold Pengenalan", 
                                            5.0, 30.0, 15.0, 0.5,
                                            help="Batas jarak maksimal untuk menganggap wajah cocok"
                                        )

                                        # Lakukan pengenalan wajah menggunakan implementasi custom
                                        result, distance, top_matches = quick_recognize(
                                            test_image_path,
                                            st.session_state.mean_face,
                                            st.session_state.eigenfaces,
                                            st.session_state.projected_train,
                                            st.session_state.X,
                                            st.session_state.labels,
                                            threshold=threshold
                                        )

                                        # ===== TAMPILKAN HASIL PENGENALAN =====
                                        st.markdown("---")
                                        st.header("🎯 Hasil Pengenalan")

                                        if result:
                                            # WAJAH BERHASIL DIKENALI
                                            st.success(f"✅ **KECOCOKAN DITEMUKAN!**")
                                            st.write(f"👤 **Teridentifikasi sebagai:** {result}")
                                            st.write(f"📏 **Jarak Euclidean:** {distance:.3f}")
                                            
                                            # Confidence level berdasarkan jarak
                                            if distance < 10:
                                                confidence = "Sangat Tinggi"
                                                conf_color = "🟢"
                                            elif distance < 15:
                                                confidence = "Tinggi"
                                                conf_color = "🟡"
                                            else:
                                                confidence = "Sedang"
                                                conf_color = "🟠"
                                            
                                            st.write(f"📊 **Tingkat Kepercayaan:** {conf_color} {confidence}")

                                            # Tampilkan gambar yang cocok dari dataset
                                            try:
                                                # Cari index gambar yang cocok
                                                matched_indices = [i for i, label in enumerate(st.session_state.labels) 
                                                                 if label == result]
                                                
                                                if matched_indices:
                                                    # Ambil gambar pertama yang cocok
                                                    matched_idx = matched_indices[0]
                                                    matched_img_vec = st.session_state.X[:, matched_idx]
                                                    matched_img = matched_img_vec.reshape(100, 100)

                                                    st.write("**Gambar yang cocok dari dataset:**")
                                                    fig, ax = plt.subplots(figsize=(4, 4))
                                                    ax.imshow(matched_img, cmap='gray')
                                                    ax.set_title(f"Match: {result}")
                                                    ax.axis('off')
                                                    st.pyplot(fig)
                                                    plt.close()  # Cleanup matplotlib figure

                                            except Exception as e:
                                                st.warning(f"Tidak dapat menampilkan gambar yang cocok: {e}")

                                        else:
                                            # WAJAH TIDAK DIKENALI
                                            st.error(f"❌ **TIDAK ADA KECOCOKAN**")
                                            st.write(f"📏 **Jarak minimum:** {distance:.3f}")
                                            st.write(f"🎯 **Threshold:** {threshold:.3f}")
                                            st.info("💡 Coba sesuaikan threshold atau gunakan gambar yang lebih jelas")

                                        # ===== TAMPILKAN TOP 3 KECOCOKAN =====
                                        if top_matches and len(top_matches) > 0:
                                            st.write("**🏆 3 Kecocokan Terdekat:**")
                                            for i, (dist, label, idx) in enumerate(top_matches[:3]):
                                                # Tambahkan indikator visual untuk ranking
                                                rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
                                                st.write(f"{rank_emoji} {label} (jarak: {dist:.3f})")

                                    except Exception as e:
                                        st.error(f"❌ Error dalam pengenalan: {e}")
                                        st.info("💡 Pastikan gambar dalam format yang didukung dan model sudah dimuat")

                            else:
                                st.warning("⚠️ Silakan muat model terlebih dahulu menggunakan tombol di sidebar!")

                    except Exception as e:
                        st.error(f"❌ Error memuat gambar test: {e}")

            else:
                st.warning(f"⚠️ Tidak ada gambar dalam folder {test_folder}")
                st.info("💡 Tambahkan gambar test dengan format .jpg, .jpeg, atau .png")

        else:
            st.warning(f"⚠️ Folder test tidak ditemukan: {test_folder}")
            st.info("💡 Buat folder 'test_img' dan tambahkan gambar test di dalamnya")

        # ===== PROSES GAMBAR YANG DIUPLOAD =====
        if uploaded_file is not None:
            st.write("**Gambar yang diupload:**")
            uploaded_img = Image.open(uploaded_file)
            st.image(uploaded_img, caption="Gambar Test yang Diupload", width=300)

            # Simpan file upload sementara untuk diproses
            temp_path = f"temp_{uploaded_file.name}"
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Tombol pengenalan untuk gambar upload
                if st.button("🔍 Kenali Wajah Upload", type="primary"):
                    
                    if hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded:

                        with st.spinner("Mengenali wajah yang diupload..."):
                            try:
                                threshold = st.sidebar.slider(
                                    "🎯 Threshold Pengenalan", 
                                    5.0, 30.0, 15.0, 0.5,
                                    key="upload_threshold"  # Key unik untuk slider kedua
                                )

                                # Proses pengenalan (sama seperti gambar dari folder)
                                result, distance, top_matches = quick_recognize(
                                    temp_path,
                                    st.session_state.mean_face,
                                    st.session_state.eigenfaces,
                                    st.session_state.projected_train,
                                    st.session_state.X,
                                    st.session_state.labels,
                                    threshold=threshold
                                )

                                # Tampilkan hasil (logika sama seperti di atas)
                                st.markdown("---")
                                st.header("🎯 Hasil Pengenalan Upload")

                                if result:
                                    st.success(f"✅ **KECOCOKAN DITEMUKAN!**")
                                    st.write(f"👤 **Teridentifikasi sebagai:** {result}")
                                    st.write(f"📏 **Jarak:** {distance:.3f}")
                                else:
                                    st.error(f"❌ **TIDAK ADA KECOCOKAN**")
                                    st.write(f"📏 **Jarak minimum:** {distance:.3f}")
                                    st.write(f"🎯 **Threshold:** {threshold:.3f}")

                                # Cleanup file temporary
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)

                            except Exception as e:
                                st.error(f"❌ Error pengenalan upload: {e}")
                                # Pastikan cleanup file temporary
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)

                    else:
                        st.warning("⚠️ Silakan muat model terlebih dahulu!")

            except Exception as e:
                st.error(f"❌ Error menyimpan file upload: {e}")

    # ===== FOOTER AKADEMIK =====
    st.markdown("---")
    st.markdown("### 🎓 Tentang Sistem")
    
    col_footer1, col_footer2 = st.columns([1, 1])
    
    with col_footer1:
        st.markdown("""
        **🔬 Implementasi Teknis:**
        - Custom eigenvalue/eigenvector calculation
        - Power iteration algorithm
        - Manual euclidean distance
        - Eigenface projection method
        """)
    
    with col_footer2:
        st.markdown("""
        **📚 Informasi Akademik:**
        - Mata Kuliah: Aljabar Linear
        - Universitas Sebelas Maret
        - Program Studi: Teknik Informatika
        - Topik: Face Recognition with Eigenfaces
        """)

    st.markdown("---")
    st.caption("💡 Sistem ini menggunakan implementasi custom tanpa library shortcuts untuk tujuan pembelajaran aljabar linear")

# ===== ENTRY POINT APLIKASI =====
if __name__ == "__main__":
    # Inisialisasi session state jika belum ada
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    # Jalankan aplikasi utama
    main()
