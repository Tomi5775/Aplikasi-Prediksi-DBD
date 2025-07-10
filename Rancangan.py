import streamlit as st

# Konfigurasi tampilan halaman
st.set_page_config(page_title="Wireframe Aplikasi Prediksi DBD", layout="wide")

# ===== Styling untuk kotak wireframe =====
st.markdown("""
<style>
    .placeholder-box {
        background-color: #f0f0f0;
        border: 2px dashed #aaa;
        padding: 30px;
        text-align: center;
        border-radius: 10px;
        font-size: 1.1rem;
        color: #555;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
        padding: 1rem;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===== Navigasi Sidebar =====
with st.sidebar:
    st.title("🧭 Navigasi")
    menu = st.radio("Pilih Halaman:", [
        "🏠 Beranda",
        "📊 Visualisasi",
        "🧪 Prediksi",
        "📈 Evaluasi"
    ])
    st.markdown("---")
    st.caption("👨‍💻 Tomi – Tugas Akhir")

# ===== Halaman Beranda =====
if menu == "🏠 Beranda":
    st.title("🏠 Beranda Aplikasi Prediksi DBD")
    with st.expander("📘 Penjelasan Aplikasi"):
        st.markdown('<div class="placeholder-box">Penjelasan umum aplikasi</div>', unsafe_allow_html=True)

    with st.expander("🧠 Alasan Penggunaan"):
        st.markdown('<div class="placeholder-box">Alasan dan manfaat penggunaan</div>', unsafe_allow_html=True)

    with st.expander("🧪 Fitur Darah yang Digunakan"):
        st.markdown('<div class="placeholder-box">Tabel fitur-fitur darah</div>', unsafe_allow_html=True)

# ===== Halaman Visualisasi =====
elif menu == "📊 Visualisasi":
    st.title("📊 Visualisasi Data Darah")
    st.markdown("### 📦 Boxplot Fitur")
    st.markdown('<div class="placeholder-box">Boxplot fitur darah</div>', unsafe_allow_html=True)

    st.markdown("### 📈 Distribusi Berdasarkan Kelas")
    st.markdown('<div class="placeholder-box">Grafik distribusi fitur berdasarkan status DBD</div>', unsafe_allow_html=True)

# ===== Halaman Prediksi =====
elif menu == "🧪 Prediksi":
    st.title("🧪 Form Prediksi Status DBD")

    st.markdown("### 📝 Input Data Pasien")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="placeholder-box">Input Platelet</div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder-box">Input Lymphocytes</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="placeholder-box">Input WBC</div>', unsafe_allow_html=True)
        st.markdown('<div class="placeholder-box">Input Neutrophils</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="placeholder-box">Input Hemoglobin</div>', unsafe_allow_html=True)

    st.markdown('<div class="placeholder-box">[ Tombol Prediksi ]</div>', unsafe_allow_html=True)
    st.markdown('<div class="placeholder-box">[ Hasil Prediksi & Probabilitas ]</div>', unsafe_allow_html=True)
    st.markdown('<div class="placeholder-box">[ Interpretasi Input Pasien ]</div>', unsafe_allow_html=True)

# ===== Halaman Evaluasi =====
elif menu == "📈 Evaluasi":
    st.title("📈 Evaluasi Model Prediksi")
    st.markdown("### 📊 Classification Report - Data Uji")
    st.markdown('<div class="placeholder-box">Tabel klasifikasi hasil prediksi pada data uji</div>', unsafe_allow_html=True)

    st.markdown("### 📊 Classification Report - Data Latih")
    st.markdown('<div class="placeholder-box">Tabel klasifikasi hasil prediksi pada data latih</div>', unsafe_allow_html=True)

# ===== Footer Umum =====
st.markdown("---")
st.caption("📅 2025 | Wireframe Aplikasi Prediksi DBD | Dibuat oleh Tomi | Tugas Akhir Teknik Informatika")
