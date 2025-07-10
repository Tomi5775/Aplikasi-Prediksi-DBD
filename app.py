import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# ====== Konfigurasi Tampilan Streamlit ======
st.set_page_config(page_title="Prediksi DBD", layout="wide")

st.markdown(
    """
    <style>
        .main { background-color: #F8FAFC; }
        .block-container { padding-top: 2rem; }
        .stApp { font-family: 'Segoe UI', sans-serif; }
        .st-cb { background-color: #E9F5FF; border-radius: 0.5rem; padding: 1rem; }
        .st-cb:hover { background-color: #DCEEFF; }
        h1, h2, h3 { color: #006699; }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ====== Load Model dan Tools ======
with open("model_random_forest.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("x_test.pkl", "rb") as f:
    x_test = pickle.load(f)

with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

with open("x_train.pkl", "rb") as f:
    x_train = pickle.load(f)

with open("y_train.pkl", "rb") as f:
    y_train = pickle.load(f)

with open("y_pred.pkl", "rb") as f:
    y_pred = pickle.load(f)


# ====== Sidebar Navigasi ======
# ====== Sidebar Navigasi (Enhanced) ======
with st.sidebar:
    st.markdown("## 🧭 Menu Navigasi")
    menu = st.radio(
        label="Pilih Halaman:",
        options=["🏠 Beranda", "📊 Visualisasi", "🧪 Prediksi", "📈 Evaluasi"],
        index=0
    )

    st.markdown("---")
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.markdown("📅 Tahun: **2025**")
    st.markdown("👨‍💻 Oleh: **Tomi**")
    st.markdown("🎓 Tugas Akhir Teknik Informatika")

    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                background-color: #F0F8FF;
                border-right: 1px solid #D1E8FF;
                padding: 1rem;
            }
            .st-radio > div {
                gap: 0.5rem;
            }
            .st-radio label {
                font-weight: 600;
                color: #004466;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


# ====== Beranda ======
if menu == "🏠 Beranda":
    st.title("🦠 Aplikasi Prediksi Infeksi DBD (Demam Berdarah Dengue)")
    st.subheader("🔍 Berbasis Machine Learning - Random Forest dengan GridSearchCV")
    st.divider()

    with st.expander("📘 Apa itu Aplikasi Prediksi DBD?"):
        st.markdown("""
        Aplikasi ini dirancang untuk membantu memprediksi status infeksi **Demam Berdarah Dengue (DBD)** berdasarkan hasil pemeriksaan darah pasien.
        
        Aplikasi ini memanfaatkan algoritma **Random Forest** yang telah dioptimasi menggunakan **GridSearchCV** untuk menghasilkan prediksi yang **akurat, cepat, dan stabil**.

        **Manfaat aplikasi ini antara lain:**
        - 🏥 Membantu **dokter dan tenaga medis** dalam pengambilan keputusan awal.
        - 🔬 Mendukung **penelitian** berbasis data laboratorium.
        - 👨‍👩‍👧‍👦 Meningkatkan **kesadaran masyarakat** terhadap indikator awal DBD.

        > ⚠️ **Catatan:** Hasil dari aplikasi ini bukan pengganti diagnosis medis langsung.
        """)

    with st.expander("🧠 Mengapa Menggunakan Aplikasi Ini?"):
        st.markdown("""
        ✅ **Cepat dan Efisien** — Proses prediksi berlangsung dalam hitungan detik.

        ✅ **Akurat** — Didukung oleh pemodelan *Machine Learning* menggunakan Random Forest + GridSearchCV.

        ✅ **Interaktif** — Visualisasi fitur, interpretasi nilai, dan evaluasi model tersedia.

        ✅ **Fleksibel** — Cocok untuk **penelitian**, **skrining awal pasien**, dan **pengambilan kebijakan medis**.

        ✅ **Berdasarkan Data Nyata** — Model dibangun menggunakan data hasil pemeriksaan darah pasien.
        """)

    with st.expander("🧪 Fitur Darah yang Digunakan"):
        fitur_data = {
            "Fitur": [
                "Platelet", 
                "Lymphocytes", 
                "WBC (Sel Darah Putih)", 
                "Neutrophils", 
                "Hemoglobin"
            ],
            "Keterangan": [
                "Jumlah keping darah. Umumnya rendah pada pasien DBD.",
                "Sel limfosit yang meningkat saat infeksi virus.",
                "Total sel darah putih. Cenderung menurun pada DBD.",
                "Sel neutrofil. Dapat menurun saat terjadi infeksi virus.",
                "Kadar hemoglobin. Dapat meningkat akibat dehidrasi atau menurun jika terjadi perdarahan."
            ],
            "Indikasi DBD": [
                "Rendah",
                "Tinggi",
                "Rendah",
                "Rendah",
                "Tinggi / Rendah"
            ]
        }
        df_fitur = pd.DataFrame(fitur_data)
        st.dataframe(df_fitur)

    st.markdown("---")
    st.caption("🧑‍💻 Jelajahi berbagai fitur di Menu Sidebar")


# ====== Visualisasi ======
elif menu == "📊 Visualisasi":
    st.title("📊 Visualisasi Fitur Darah")
    df_test = pd.DataFrame(
        scaler.inverse_transform(x_test),
        columns=["Platelet", "Lymphocytes", "WBC", "Neutrophils", "Hemoglobin"]
    )
    df_test["Label"] = label_encoder.inverse_transform(y_test)

    # ===== Boxplot Setiap Fitur =====
    st.subheader("📦 Boxplot Setiap Fitur")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_test.drop("Label", axis=1), ax=ax1, palette="pastel")
    ax1.set_ylabel("(Nilai Pemeriksaan)")
    ax1.set_xlabel("(Fitur Darah)")
    ax1.grid(True)
    st.pyplot(fig1)

    # ===== Distribusi Semua Fitur dalam 1 Frame =====
    st.subheader("📈 Distribusi Fitur Berdasarkan Status DBD")

    fitur_cols = df_test.columns[:-1]
    n_fitur = len(fitur_cols)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    axs = axs.flatten()

    for i, col in enumerate(fitur_cols):
        sns.kdeplot(data=df_test, x=col, hue="Label", fill=True, ax=axs[i], palette="Set2", alpha=0.5)
        axs[i].axvline(df_test[df_test["Label"] == "YA DBD"][col].mean(), color='red', linestyle='--', label='Rata-rata YA DBD')
        axs[i].axvline(df_test[df_test["Label"] == "TIDAK DBD"][col].mean(), color='green', linestyle='--', label='Rata-rata TIDAK DBD')
        axs[i].set_title(f"{col}")
        axs[i].set_xlabel("Nilai")
        axs[i].set_ylabel("Kepadatan")
        axs[i].grid(True)
        axs[i].legend()

    # Kosongkan subplot terakhir jika jumlahnya ganjil
    if n_fitur < len(axs):
        fig.delaxes(axs[-1])

    plt.tight_layout()
    st.pyplot(fig)



# ====== Prediksi ======
elif menu == "🧪 Prediksi":
    st.title("🧪 Prediksi Status Infeksi DBD")

    default_values = {
        "platelet": 90000.0,
        "lymph": 45.0,
        "wbc": 4000.0,
        "neutro": 35.0,
        "hemo": 15.2
    }
    for key, val in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = val

    with st.form("form_prediksi"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.platelet = st.number_input("Jumlah Platelet (/cumm)", min_value=0.0, value=st.session_state.platelet, format="%.0f")
            st.session_state.lymph = st.number_input("Persentase Limfosit (%)", min_value=0.0, max_value=100.0, value=st.session_state.lymph)
        with col2:
            st.session_state.wbc = st.number_input("Jumlah Sel Darah Putih (/cumm)", min_value=0.0, value=st.session_state.wbc, format="%.0f")
            st.session_state.neutro = st.number_input("Persentase Neutrofil (%)", min_value=0.0, max_value=100.0, value=st.session_state.neutro)
        with col3:
            st.session_state.hemo = st.number_input("Hemoglobin (g/dL)", min_value=0.0, value=st.session_state.hemo)

        submit = st.form_submit_button("🔍 Prediksi Sekarang")

    if submit:
        input_df = pd.DataFrame([[st.session_state.platelet, st.session_state.lymph,
                                  st.session_state.wbc, st.session_state.neutro,
                                  st.session_state.hemo]],
                                columns=["Platelet", "Lymphocytes", "WBC", "Neutrophils", "Hemoglobin"])

        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        label_pred = label_encoder.inverse_transform([pred])[0]

        st.session_state.pred_label = label_pred
        st.session_state.pred_proba = proba

    if "pred_label" in st.session_state:
        st.success(f"Hasil Prediksi: **{st.session_state.pred_label}**")
        st.progress(int(np.max(st.session_state.pred_proba) * 100))
        st.caption(f"Probabilitas Prediksi: {np.max(st.session_state.pred_proba) * 100:.2f}%")

        with st.expander("🩸 Interpretasi Nilai Fitur Anda"):
            st.markdown("**Analisis Berdasarkan Nilai Medis Umum:**")
            fitur_input = {
                "Platelet": st.session_state.platelet,
                "Lymphocytes": st.session_state.lymph,
                "WBC": st.session_state.wbc,
                "Neutrophils": st.session_state.neutro,
                "Hemoglobin": st.session_state.hemo
            }
            batas = {
                "Platelet": 150000,
                "Lymphocytes": 40,
                "WBC": 5000,
                "Neutrophils": 45,
                "Hemoglobin": 14
            }
            for fitur, nilai in fitur_input.items():
                if fitur == "Platelet" and nilai < batas[fitur]:
                    arah = "Rendah (indikasi DBD)"
                elif fitur in ["Lymphocytes", "Hemoglobin"] and nilai > batas[fitur]:
                    arah = "Tinggi (indikasi DBD)"
                elif fitur in ["WBC", "Neutrophils"] and nilai < batas[fitur]:
                    arah = "Rendah (indikasi DBD)"
                else:
                    arah = "Normal"
                st.write(f"🔹 **{fitur}**: {nilai} → {arah}")


# ====== Evaluasi Model ======
elif menu == "📈 Evaluasi":
    st.title("📈 Evaluasi Model Random Forest")

    # ===== Evaluasi Data Uji =====
    st.header("🧪 Evaluasi pada Data Uji")
    y_pred_test = model.predict(x_test)

    st.subheader("📋 Classification Report - Data Uji")
    report_test = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, output_dict=True)
    df_report_test = pd.DataFrame(report_test).transpose().drop(columns=['support'])
    st.dataframe(df_report_test.round(2).style.background_gradient(cmap='Blues'), use_container_width=True)

    st.divider()

    # ===== Evaluasi Data Latih =====
    st.header("🧠 Evaluasi pada Data Latih")
    y_pred_train = model.predict(x_train)

    st.subheader("📋 Classification Report - Data Latih")
    report_train = classification_report(y_train, y_pred_train, target_names=label_encoder.classes_, output_dict=True)
    df_report_train = pd.DataFrame(report_train).transpose().drop(columns=['support'])
    st.dataframe(df_report_train.round(2).style.background_gradient(cmap='Greens'), use_container_width=True)


# ====== Footer ======
st.markdown("---")
st.caption("📅 2025 | 🧬 Aplikasi Prediksi DBD | Dibuat oleh Tomi | Tugas Akhir Teknik Informatika")
