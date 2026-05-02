import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prediksi Kualitas Air Sungai",
    page_icon="💧",
    layout="wide"
)

# ================= STYLE =================
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #1E88E5;
}
.stButton>button {
    background-color: #1E88E5;
    color: white;
    border-radius: 8px;
}
.stDownloadButton>button {
    background-color: #43A047;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div style="padding:15px; border-radius:10px; background:linear-gradient(90deg,#1E88E5,#42A5F5); color:white;">
    <h2>💧 Sistem Prediksi Kualitas Air Sungai</h2>
    <p style="margin:0;">Berbasis Machine Learning (XGBoost)</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load("16_model_xgboost.pkl")
metadata = joblib.load("metadata_model.pkl")
fitur = metadata["fitur"]

# =====================================================
# NAVIGASI DROPDOWN
# =====================================================
st.sidebar.title("📂 Navigasi")

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "🏠 Input & Prediksi",
        "📚 Hasil Pelatihan Model",
        "ℹ️ Tentang Aplikasi"
    ]
)

# =====================================================
# PARAMETER IP
# =====================================================
BM = {
    "Temperatur": 3,
    "pH_min": 6,
    "pH_max": 9,
    "DO": 4,
    "BOD": 3,
    "COD": 25,
    "TSS": 50,
    "TDS": 1000
}
TEMP_ALAMI = 28

def hitung_ip(row):
    rasio = []

    dev = abs(row["Temperatur"] - TEMP_ALAMI)
    rasio.append(dev / BM["Temperatur"])

    L_min, L_max = BM["pH_min"], BM["pH_max"]
    L_mid = (L_min + L_max) / 2

    if row["pH"] < L_mid:
        r = (L_mid - row["pH"]) / (L_mid - L_min)
    else:
        r = (row["pH"] - L_mid) / (L_max - L_mid)
    rasio.append(abs(r))

    rasio.append(BM["DO"] / row["DO"])
    rasio.append(row["BOD"] / BM["BOD"])
    rasio.append(row["COD"] / BM["COD"])
    rasio.append(row["TSS"] / BM["TSS"])
    rasio.append(row["TDS"] / BM["TDS"])

    rasio = np.array(rasio, dtype=float)
    return np.sqrt((rasio.max()**2 + rasio.mean()**2) / 2)

# =====================================================
# SESSION
# =====================================================
KOL = ["Tanggal","Temperatur","pH","DO","BOD","COD","TSS","TDS","Nilai IP","Hasil Prediksi"]

if "data_all" not in st.session_state:
    st.session_state.data_all = pd.DataFrame(columns=KOL)

# =====================================================
# UTIL DOWNLOAD
# =====================================================
def download_excel(df, filename):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button("⬇️ Download Excel", buffer, filename)

# =====================================================
# MENU 1
# =====================================================
if menu == "🏠 Input & Prediksi":

    st.title("📊 Input & Prediksi Data Kualitas Air")

    # ================= INPUT =================
    st.subheader("📝 Input Manual")

    c1, c2 = st.columns(2)

    with c1:
        temperatur = st.number_input("Temperatur (°C)", 0.0, 50.0, 25.0)
        ph = st.number_input("pH (Derajat Keasaman)", 0.0, 14.0, 7.0)
        do = st.number_input("DO - Dissolved Oxygen (mg/L)", 0.0, 20.0, 4.0)
        bod = st.number_input("BOD - Biochemical Oxygen Demand (mg/L)", 0.0, 50.0, 3.0)

    with c2:
        cod = st.number_input("COD - Chemical Oxygen Demand (mg/L)", 0.0, 100.0, 25.0)
        tss = st.number_input("TSS - Total Suspended Solid (mg/L)", 0.0, 500.0, 50.0)
        tds = st.number_input("TDS - Total Dissolved Solid (mg/L)", 0.0, 5000.0, 1000.0)
        tanggal = st.date_input("Tanggal")

    if st.button("🔍 Prediksi & Simpan"):
        df_new = pd.DataFrame([{
            "Tanggal": tanggal,
            "Temperatur": temperatur,
            "pH": ph,
            "DO": do,
            "BOD": bod,
            "COD": cod,
            "TSS": tss,
            "TDS": tds
        }])

        df_new["Nilai IP"] = df_new.apply(hitung_ip, axis=1)

        pred = model.predict(df_new[fitur])[0] # MetaData model sudah menyimpan urutan fitur
        df_new["Hasil Prediksi"] = "Memenuhi Baku Mutu" if pred == 0 else "Tidak Memenuhi"

        st.session_state.data_all = pd.concat(
            [st.session_state.data_all, df_new[KOL]],
            ignore_index=True
        )

        st.success("Data berhasil ditambahkan")

    # ================= TEMPLATE =================
    st.markdown("---")
    st.subheader("📥 Download Template Excel")

    template = pd.DataFrame(columns=[
        "Tanggal","Temperatur","pH","DO","BOD","COD","TSS","TDS"
    ])
    download_excel(template, "template_kualitas_air.xlsx")

    # ================= UPLOAD =================
    st.markdown("---")
    st.subheader("📤 Upload Data")

    file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

    if file is not None:
        df_up = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

        df_up["Tanggal"] = pd.to_datetime(df_up["Tanggal"]).dt.date

        df_up["Nilai IP"] = df_up.apply(hitung_ip, axis=1)

        df_up["Hasil Prediksi"] = np.where(
            model.predict(df_up[fitur]) == 0,
            "Memenuhi Baku Mutu",
            "Tidak Memenuhi"
        )

        st.session_state.data_all = pd.concat(
            [st.session_state.data_all, df_up[KOL]],
            ignore_index=True
        )

        st.success("Data berhasil diupload")

    # ================= OUTPUT =================
    if not st.session_state.data_all.empty:

        df = st.session_state.data_all.copy()
        # ================= FILTER DATA =================
        st.markdown("---")
        st.subheader("🔎 Filter Data")

        df["Tanggal"] = pd.to_datetime(df["Tanggal"])

        colf1, colf2, colf3 = st.columns(3)

        with colf1:
            tahun_list = sorted(df["Tanggal"].dt.year.unique())
            tahun = st.multiselect(
                "Filter Tahun",
                tahun_list,
                default=tahun_list
            )

        with colf2:
            bulan = st.multiselect(
                "Filter Bulan",
                list(range(1, 13)),
                default=list(range(1, 13))
            )

        with colf3:
            tanggal_range = st.date_input(
                "Filter Rentang Tanggal",
                [df["Tanggal"].min(), df["Tanggal"].max()]
            )

        # ================= APPLY FILTER =================
        df_filter = df[
            df["Tanggal"].dt.year.isin(tahun) &
            df["Tanggal"].dt.month.isin(bulan)
        ]

        # filter tanggal range
        if len(tanggal_range) == 2:
            start, end = pd.to_datetime(tanggal_range[0]), pd.to_datetime(tanggal_range[1])
            df_filter = df_filter[
                (df_filter["Tanggal"] >= start) &
                (df_filter["Tanggal"] <= end)
            ]

        if df_filter.empty:
            st.warning("⚠️ Data tidak tersedia berdasarkan filter yang dipilih")

        df = df.sort_values("Tanggal")

        st.markdown("---")
        st.subheader("📋 Tabel Hasil Prediksi")

        df_tampil = df.copy()
        df_tampil["Tanggal"] = pd.to_datetime(df_tampil["Tanggal"]).dt.date

        st.dataframe(df_tampil, use_container_width=True)
        
        col1, col2 = st.columns(2)

        with col1:
            download_excel(df, "hasil_prediksi.xlsx")

        with col2:
            if st.button("🗑️ Hapus Semua Data"):
                st.session_state.data_all = pd.DataFrame(columns=KOL)
                st.success("Semua data berhasil dihapus")
                st.rerun()

        # ================= BAR CHART =================
        if not df_filter.empty:

            st.markdown("---")
            st.subheader("📊 Status Kualitas Air")

            df_filter["Periode"] = df_filter["Tanggal"].dt.to_period("M").astype(str)

            bar_df = (
                df_filter
                .groupby(["Periode", "Hasil Prediksi"])
                .size()
                .unstack(fill_value=0)
            )

            fig, ax = plt.subplots(figsize=(9,4))

            x = np.arange(len(bar_df))
            width = 0.35

            ax.bar(x - width/2, bar_df.get("Memenuhi Baku Mutu", 0),
                width, label="Memenuhi")

            ax.bar(x + width/2, bar_df.get("Tidak Memenuhi", 0),
                width, label="Tidak Memenuhi")

            ax.set_xticks(x)
            ax.set_xticklabels(bar_df.index, rotation=45)
            ax.set_ylabel("Jumlah Data")
            ax.set_title("Distribusi Status Kualitas Air")
            ax.legend()

            st.pyplot(fig)

        # ================= LINE CHART =================
        if not df_filter.empty:

            st.markdown("---")
            st.subheader("📈 Tren Parameter")

            param = st.selectbox("Pilih Parameter", fitur)

            df_line = (
                df_filter
                .groupby(df_filter["Tanggal"].dt.to_period("M"))[param]
                .mean()
                .reset_index()
            )

            df_line["Periode"] = df_line["Tanggal"].astype(str)

            fig2, ax2 = plt.subplots(figsize=(9,4))

            x = np.arange(len(df_line))
            y = df_line[param]

            ax2.plot(x, y, marker='o', linewidth=2)

            # label di titik
            for i in range(len(y)):
                ax2.text(x[i], y.iloc[i], round(y.iloc[i], 2),
                        ha='center', va='bottom')

            ax2.set_xticks(x)
            ax2.set_xticklabels(df_line["Periode"], rotation=45)
            ax2.set_ylabel(param)
            ax2.set_title(f"Tren {param}")

            ax2.grid(True)

            st.pyplot(fig2)

        # ================= PERBANDINGAN =================
        st.markdown("---")
        st.subheader("📑 Perbandingan IP dan Prediksi")

        df_compare = df.copy()

        df_compare["Status IP"] = np.where(
            df_compare["Nilai IP"] <= 1,
            "Memenuhi Baku Mutu",
            "Tidak Memenuhi"
        )

        df_compare_tampil = df_compare.copy()
        df_compare_tampil["Tanggal"] = pd.to_datetime(df_compare_tampil["Tanggal"]).dt.date

        st.dataframe(
            df_compare_tampil[[
                "Tanggal","Temperatur","pH","DO","BOD","COD","TSS","TDS",
                "Nilai IP","Status IP","Hasil Prediksi"
            ]],
            use_container_width=True
        )

        st.caption(
            "Perbandingan antara hasil perhitungan Indeks Pencemaran (IP) "
            "dan hasil prediksi model."
        )

# =====================================================
# MENU 2
# =====================================================
elif menu == "📚 Hasil Pelatihan Model":

    st.title("Hasil Pelatihan Model")

    st.subheader("Data Pelatihan")
    df_latih = pd.read_excel("HASIL_PENELITIAN_XGBOOST/01_data_mentah_5_baris.xlsx")

    df_latih_tampil = df_latih.copy()
    df_latih_tampil["Tanggal"] = pd.to_datetime(df_latih_tampil["Tanggal"]).dt.date

    st.dataframe(df_latih_tampil, use_container_width=True)

    # ================= EVALUASI MODEL =================
    st.markdown("---")
    st.subheader("Evaluasi Kinerja Model")

    # ================= CLASSIFICATION REPORT =================
    st.subheader("Classification Report")

    col1, col2 = st.columns(2)

    with col1:
        df_report = pd.read_excel("HASIL_PENELITIAN_XGBOOST/02_classification_report.xlsx")
        st.dataframe(df_report, use_container_width=True)

    with col2:
        st.markdown("""
        Classification report digunakan untuk mengevaluasi performa model klasifikasi 
        secara lebih rinci dibandingkan hanya menggunakan akurasi.

        - **Precision** menunjukkan tingkat ketepatan prediksi pada masing-masing kelas.
        - **Recall** menunjukkan kemampuan model dalam menemukan seluruh data yang benar.
        - **F1-score** merupakan rata-rata harmonis antara precision dan recall.
        - **Support** menunjukkan jumlah data pada masing-masing kelas.

        Berdasarkan hasil tersebut, model mampu membedakan kelas 
        *memenuhi baku mutu* dan *tidak memenuhi baku mutu* dengan baik.
        """)

    st.markdown("---")

    # ===== 1. CONFUSION MATRIX =====
    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.image(
            "HASIL_PENELITIAN_XGBOOST/03_confusion_matrix.png",
            caption="Confusion Matrix"
        )

    with col2:
        st.markdown("""
        **Confusion Matrix**

        Confusion matrix digunakan untuk mengevaluasi kinerja model klasifikasi 
        dengan membandingkan antara hasil prediksi dan data aktual. Matriks ini 
        terdiri dari empat komponen utama, yaitu True Positive (TP), True Negative (TN), 
        False Positive (FP), dan False Negative (FN).

        Berdasarkan hasil yang diperoleh, model mampu mengklasifikasikan sebagian besar 
        data dengan benar, baik untuk kelas *Memenuhi Baku Mutu* maupun 
        *Tidak Memenuhi Baku Mutu*. Hal ini menunjukkan bahwa model memiliki kemampuan 
        yang baik dalam membedakan kondisi kualitas air.

        Dengan demikian, confusion matrix memberikan gambaran bahwa model yang 
        dibangun memiliki tingkat akurasi yang cukup tinggi dan dapat digunakan 
        untuk proses prediksi kualitas air.
        """)

    # ===== 2. FEATURE IMPORTANCE =====
    st.markdown("---")

    col3, col4 = st.columns([1.1, 1])

    with col3:
        st.image(
            "HASIL_PENELITIAN_XGBOOST/04_feature_importance.png",
            caption="Feature Importance"
        )

    with col4:
        st.markdown("""
        **Feature Importance**

        Feature importance digunakan untuk mengetahui tingkat kontribusi masing-masing 
        parameter dalam mempengaruhi hasil prediksi model.

        Berdasarkan visualisasi, terlihat bahwa beberapa parameter seperti DO, BOD, 
        dan COD memiliki pengaruh yang lebih dominan dibandingkan parameter lainnya. 
        Hal ini menunjukkan bahwa parameter tersebut berperan penting dalam menentukan 
        kualitas air.

        Analisis ini membantu dalam memahami variabel mana yang paling berpengaruh, 
        sehingga dapat menjadi dasar dalam pengambilan keputusan terkait pengelolaan 
        kualitas air.
        """)

    # ===== SPLIT DATA =====
    st.markdown("---")

    col_split1, col_split2 = st.columns([1.1, 1])

    with col_split1:
        st.image(
            "HASIL_PENELITIAN_XGBOOST/11_split_data.png",
            caption="Pengaruh Split Data terhadap Akurasi"
        )

    with col_split2:
        st.markdown("""
        **Skenario Split Data**

        Pengujian ini dilakukan untuk mengetahui pengaruh pembagian data latih 
        dan data uji terhadap performa model.

        Beberapa skenario pembagian data digunakan, seperti: (Data Latih : Data Uji)
        - 20:80
        - 30:70
        - 40:60
        - 50:50
        - 60:40
        - 70:30
        - 80:20
        - 90:10

        Hasil menunjukkan bahwa komposisi data latih yang lebih besar cenderung 
        menghasilkan akurasi yang lebih baik, karena model memiliki lebih banyak 
        data untuk proses pembelajaran.

        Namun, data uji tetap diperlukan untuk memastikan bahwa model mampu 
        melakukan prediksi dengan baik pada data yang belum pernah dilihat sebelumnya.
        """)

    # ================= PARAMETER MODEL =================
    st.markdown("---")
    st.subheader("Pengaruh Parameter Model XGBoost")

    # ===== MAX DEPTH =====
    col5, col6 = st.columns([1.1, 1])

    with col5:
        st.image("HASIL_PENELITIAN_XGBOOST/12_max_depth.png")

    with col6:
        st.markdown("""
        **Parameter Max Depth**

        Parameter max depth mengatur kedalaman maksimum pohon keputusan yang dibangun 
        oleh model. Semakin besar nilai max depth, maka model akan semakin kompleks.

        Berdasarkan hasil pengujian, pemilihan nilai max depth yang tepat dapat 
        meningkatkan performa model tanpa menyebabkan overfitting.
        """)

    # ===== LEARNING RATE =====
    st.markdown("---")

    col7, col8 = st.columns([1.1, 1])

    with col7:
        st.image("HASIL_PENELITIAN_XGBOOST/13_learning_rate.png")

    with col8:
        st.markdown("""
        **Parameter Learning Rate**

        Learning rate menentukan besarnya kontribusi setiap pohon dalam proses boosting. 
        Nilai yang terlalu besar dapat menyebabkan model tidak stabil, sedangkan nilai 
        yang terlalu kecil membuat proses pembelajaran menjadi lambat.

        Oleh karena itu, diperlukan keseimbangan agar model dapat belajar secara optimal.
        """)

    # ===== N ESTIMATORS =====
    st.markdown("---")

    col9, col10 = st.columns([1.1, 1])

    with col9:
        st.image("HASIL_PENELITIAN_XGBOOST/14_n_estimators.png")

    with col10:
        st.markdown("""
        **Parameter n_estimators**

        Parameter ini menunjukkan jumlah pohon keputusan yang dibangun dalam model. 
        Semakin banyak jumlah pohon, maka model cenderung memiliki kemampuan prediksi 
        yang lebih baik, namun juga meningkatkan waktu komputasi.

        Pemilihan jumlah estimators yang tepat dapat menghasilkan model yang optimal.
        """)

    # ===== SAMPLING =====
    st.markdown("---")

    col11, col12 = st.columns([1.1, 1])

    with col11:
        st.image("HASIL_PENELITIAN_XGBOOST/15_sampling.png")

    with col12:
        st.markdown("""
        **Parameter Sampling**

        Sampling digunakan untuk mengontrol proporsi data yang digunakan dalam 
        setiap iterasi pelatihan. Teknik ini membantu dalam mengurangi risiko overfitting 
        dan meningkatkan kemampuan generalisasi model.

        Hasil pengujian menunjukkan bahwa penggunaan sampling yang tepat dapat 
        meningkatkan stabilitas performa model.
        """)

# =====================================================
# MENU 3
# =====================================================
elif menu == "ℹ️ Tentang Aplikasi":

    st.title("ℹ️ Tentang Aplikasi")

    st.markdown("""
    ### Gambaran Umum Aplikasi
    Aplikasi ini merupakan sistem prediksi kualitas air sungai berbasis machine learning 
    yang dikembangkan untuk mendukung proses analisis dan pemantauan kondisi kualitas air 
    secara kuantitatif dan berbasis data. Sistem ini dirancang sebagai alat bantu dalam 
    kegiatan penelitian, khususnya pada bidang data mining dan analisis kualitas lingkungan.

    Aplikasi memungkinkan pengguna melakukan input data baik secara manual maupun melalui 
    unggah file (CSV/Excel), sehingga dapat digunakan untuk analisis data individual maupun 
    data historis dalam jumlah besar.
    """)

    st.markdown("""
    ### Parameter Kualitas Air
    Parameter yang digunakan dalam penelitian ini meliputi:
    - Temperatur (°C)
    - pH
    - Dissolved Oxygen (DO)
    - Biochemical Oxygen Demand (BOD)
    - Chemical Oxygen Demand (COD)
    - Total Suspended Solid (TSS)
    - Total Dissolved Solid (TDS)

    Parameter-parameter tersebut dipilih karena merupakan indikator utama dalam 
    menentukan kondisi kualitas air serta memiliki keterkaitan langsung terhadap 
    tingkat pencemaran dan keseimbangan ekosistem perairan.
    """)

    st.markdown("""
    ### Dasar Regulasi
    Penentuan status kualitas air mengacu pada Peraturan Pemerintah Republik Indonesia 
    Nomor 22 Tahun 2021 tentang Penyelenggaraan Perlindungan dan Pengelolaan Lingkungan Hidup, 
    khususnya pada baku mutu air Kelas II.

    Berdasarkan ketentuan tersebut, data kualitas air diklasifikasikan ke dalam dua kategori:
    - Memenuhi Baku Mutu
    - Tidak Memenuhi Baku Mutu

    Klasifikasi ini dilakukan berdasarkan hasil perhitungan Indeks Pencemaran (IP) 
    yang kemudian digunakan sebagai label dalam proses pembelajaran model.
    """)

    st.markdown("""
    ### Metode dan Model Prediksi
    Model yang digunakan dalam aplikasi ini adalah algoritma Extreme Gradient Boosting 
    (XGBoost) yang termasuk dalam metode supervised learning untuk klasifikasi biner.

    Model dilatih menggunakan data historis kualitas air yang telah melalui tahapan:
    - preprocessing data (pembersihan dan imputasi nilai hilang)
    - transformasi data numerik
    - perhitungan Indeks Pencemaran (IP)
    - pelabelan data berdasarkan baku mutu

    Selain itu, pendekatan time-aware digunakan dengan menggeser label target (t+1), 
    sehingga model mampu memprediksi kondisi kualitas air pada waktu berikutnya.
    """)

    st.markdown("""
    ### Evaluasi Model
    Kinerja model dievaluasi menggunakan metrik evaluasi klasifikasi, yaitu:
    - Accuracy
    - Precision
    - Recall
    - F1-Score

    Selain itu, digunakan confusion matrix untuk menganalisis distribusi hasil prediksi 
    terhadap data aktual. Berdasarkan hasil evaluasi, model menunjukkan kemampuan yang 
    baik dalam membedakan antara kelas memenuhi baku mutu dan tidak memenuhi baku mutu.
    """)

    st.markdown("""
    ### Visualisasi Data
    Aplikasi dilengkapi dengan fitur visualisasi untuk mendukung interpretasi hasil, yaitu:
    - Diagram batang (bar chart) untuk menampilkan distribusi status kualitas air per periode
    - Diagram garis (line chart) untuk menampilkan tren perubahan parameter kualitas air

    Visualisasi ini membantu dalam memahami pola perubahan kualitas air secara temporal 
    serta mempermudah analisis secara deskriptif.
    """)

    st.markdown("""
    ### Tujuan Pengembangan
    Tujuan dari pengembangan aplikasi ini adalah:
    1. Mengimplementasikan metode XGBoost dalam prediksi kualitas air  
    2. Menyediakan sistem analisis kualitas air berbasis data  
    3. Membantu visualisasi tren kualitas air secara temporal  
    4. Mendukung pengambilan keputusan berbasis data dalam pengelolaan lingkungan  

    Dengan demikian, aplikasi ini diharapkan dapat menjadi alat bantu yang efektif 
    dalam analisis kualitas air sungai serta mendukung kegiatan penelitian di bidang 
    data mining dan lingkungan.
    """)

    st.markdown("""
    ### Dibuat Oleh Figo Firnanda | Sistem Informasi | Universitas Muhammadiyah Pontianak | Copyright © 2026
    """)