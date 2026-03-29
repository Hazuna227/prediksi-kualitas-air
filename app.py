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

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--"
})

LINE_COLOR = "#1E88E5"
FILL_COLOR = "#BBDEFB"
BAR_OK = "#4CAF50"
BAR_BAD = "#F44336"

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load("model_xgboost_kualitas_air.pkl")
metadata = joblib.load("metadata_model.pkl")
fitur = metadata["fitur"]

# =====================================================
# BAKU MUTU & IP
# =====================================================
BM_KELAS_II = {
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

    # Temperatur
    dev = abs(row["Temperatur"] - TEMP_ALAMI)
    rasio.append(dev / BM_KELAS_II["Temperatur"])

    # pH
    L_min = BM_KELAS_II["pH_min"]
    L_max = BM_KELAS_II["pH_max"]
    L_mid = (L_min + L_max) / 2

    if row["pH"] < L_mid:
        r = (L_mid - row["pH"]) / (L_mid - L_min)
    else:
        r = (row["pH"] - L_mid) / (L_max - L_mid)
    rasio.append(abs(r))

    # DO
    rasio.append(BM_KELAS_II["DO"] / row["DO"])

    # BOD, COD, TSS, TDS
    rasio.append(row["BOD"] / BM_KELAS_II["BOD"])
    rasio.append(row["COD"] / BM_KELAS_II["COD"])
    rasio.append(row["TSS"] / BM_KELAS_II["TSS"])
    rasio.append(row["TDS"] / BM_KELAS_II["TDS"])

    rasio = np.array(rasio, dtype=float)

    R = rasio.mean()
    M = rasio.max()

    return np.sqrt((M**2 + R**2) / 2)

# =====================================================
# STRUKTUR DATA INTERNAL
# =====================================================
KOL_INTERNAL = [
    "Tanggal",
    "Temperatur", "pH", "DO", "BOD", "COD", "TSS", "TDS",
    "Nilai IP", "Status IP", "Hasil Prediksi"
]

if "data_all" not in st.session_state:
    st.session_state.data_all = pd.DataFrame(columns=KOL_INTERNAL)

# =====================================================
# UTIL
# =====================================================
def download_excel(df, filename):
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        "⬇️ Download Excel",
        buffer,
        filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =====================================================
# SIDEBAR
# =====================================================
menu = st.sidebar.radio("Menu", ["Input & Prediksi", "Tentang Aplikasi", "Proses Pelatihan Model"])

# =====================================================
# INPUT & PREDIKSI
# =====================================================
if menu == "Input & Prediksi":

    st.title("💧 Prediksi Kualitas Air Sungai")

    # ---------------- INPUT MANUAL ----------------
    st.subheader("📝 Input Manual")
    c1, c2 = st.columns(2)

    with c1:
        temperatur = st.number_input("Temperatur (°C)", 0.0, 50.0, 25.0)
        ph = st.number_input("pH", 0.0, 14.0, 7.0)
        do = st.number_input("DO (mg/L)", 0.0, 20.0, 4.0)
        bod = st.number_input("BOD (mg/L)", 0.0, 50.0, 3.0)

    with c2:
        cod = st.number_input("COD (mg/L)", 0.0, 100.0, 25.0)
        tss = st.number_input("TSS (mg/L)", 0.0, 500.0, 50.0)
        tds = st.number_input("TDS (mg/L)", 0.0, 5000.0, 1000.0)
        tanggal = st.date_input("Tanggal Pengambilan")

    if st.button("🔍 Prediksi & Simpan"):
        df_new = pd.DataFrame([{
            "Tanggal": tanggal,   # DATE ONLY
            "Temperatur": temperatur,
            "pH": ph,
            "DO": do,
            "BOD": bod,
            "COD": cod,
            "TSS": tss,
            "TDS": tds
        }])

        df_new["Nilai IP"] = df_new.apply(hitung_ip, axis=1)

        pred = model.predict(df_new[fitur])[0]
        df_new["Hasil Prediksi"] = (
            "Memenuhi Baku Mutu" if pred == 0 else "Tidak Memenuhi"
        )

        df_new["Status IP"] = np.where(
            df_new["Nilai IP"] <= 1,
            "Memenuhi Baku Mutu",
            "Tidak Memenuhi Baku Mutu"
        )

        st.session_state.data_all = pd.concat(
            [st.session_state.data_all, df_new[KOL_INTERNAL]],
            ignore_index=True
        )

        st.success("Data berhasil ditambahkan")

    # ---------------- TEMPLATE ----------------
    st.markdown("---")
    st.subheader("📥 Download Template Excel")

    template_df = pd.DataFrame(columns=[
        "Tanggal", "Temperatur", "pH", "DO", "BOD", "COD", "TSS", "TDS"
    ])
    download_excel(template_df, "template_input_kualitas_air.xlsx")

    # ---------------- UPLOAD ----------------
    st.markdown("---")
    st.subheader("📤 Upload CSV / Excel")

    file = st.file_uploader("Unggah file", type=["csv", "xlsx"])

    if file is not None:
        df_up = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        df_up["Tanggal"] = pd.to_datetime(df_up["Tanggal"]).dt.date

        df_up["Nilai IP"] = df_up.apply(hitung_ip, axis=1)
        df_up["Status IP"] = np.where(
            df_up["Nilai IP"] <= 1,
            "Memenuhi Baku Mutu",
            "Tidak Memenuhi Baku Mutu"
        )

        df_up["Hasil Prediksi"] = np.where(
            model.predict(df_up[fitur]) == 0,
            "Memenuhi Baku Mutu",
            "Tidak Memenuhi"
        )

        st.session_state.data_all = pd.concat(
            [st.session_state.data_all, df_up[KOL_INTERNAL]],
            ignore_index=True
        )

        st.success("File berhasil diproses")

    # =====================================================
    # TABEL HASIL PREDIKSI
    # =====================================================
    if not st.session_state.data_all.empty:
        st.markdown("---")
        st.subheader("📋 Tabel Hasil Prediksi")

        df = st.session_state.data_all.copy().sort_values("Tanggal")

        tabel_hasil = df[[
            "Tanggal",
            "Temperatur", "pH", "DO", "BOD", "COD", "TSS", "TDS",
            "Nilai IP",
            "Hasil Prediksi"
        ]]

        st.dataframe(tabel_hasil, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            download_excel(tabel_hasil, "hasil_prediksi_kualitas_air.xlsx")
        with c2:
            if st.button("🗑️ Hapus Semua Data"):
                st.session_state.data_all = pd.DataFrame(columns=KOL_INTERNAL)
                st.rerun()

        # =====================================================
        # BAR CHART + FILTER
        # =====================================================
        st.markdown("---")
        st.subheader("📊 Status Kualitas Air")

        fb1, fb2 = st.columns(2)
        with fb1:
            tahun_bar = st.multiselect(
                "Tahun",
                sorted(pd.to_datetime(df["Tanggal"]).dt.year.unique()),
                default=sorted(pd.to_datetime(df["Tanggal"]).dt.year.unique())
            )
        with fb2:
            bulan_bar = st.multiselect(
                "Bulan",
                list(range(1, 13)),
                default=list(range(1, 13))
            )

        df_bar = df[
            pd.to_datetime(df["Tanggal"]).dt.year.isin(tahun_bar) &
            pd.to_datetime(df["Tanggal"]).dt.month.isin(bulan_bar)
        ]

        if df_bar.empty:
            st.warning("⚠️ Data tidak tersedia untuk Bar Chart berdasarkan filter yang dipilih.")
        else:
            df_bar["Periode"] = pd.to_datetime(df_bar["Tanggal"]).dt.to_period("M").astype(str)

            bar_df = (
                df_bar
                .groupby(["Periode", "Hasil Prediksi"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )

            if bar_df.empty:
                st.warning("⚠️ Tidak ada data yang dapat divisualisasikan pada Bar Chart.")
            else:
                x = np.arange(len(bar_df))
                width = 0.35

                fig, ax = plt.subplots(figsize=(9,4))
                ax.bar(x - width/2, bar_df.get("Memenuhi Baku Mutu", 0),
                    width, label="Memenuhi", color=BAR_OK)
                ax.bar(x + width/2, bar_df.get("Tidak Memenuhi", 0),
                    width, label="Tidak Memenuhi", color=BAR_BAD)

                ax.set_xticks(x)
                ax.set_xticklabels(bar_df["Periode"], rotation=45)
                ax.set_ylabel("Jumlah Data")
                ax.legend()
                st.pyplot(fig)

        # =====================================================
        # LINE CHART + FILTER
        # =====================================================
        st.markdown("---")
        st.subheader("📈 Tren Parameter Kualitas Air")

        # ===== FILTER =====
        fl1, fl2, fl3 = st.columns(3)

        with fl1:
            tahun_line = st.multiselect(
                "Tahun (Line Chart)",
                sorted(pd.to_datetime(df["Tanggal"]).dt.year.unique()),
                default=sorted(pd.to_datetime(df["Tanggal"]).dt.year.unique())
            )

        with fl2:
            bulan_line = st.multiselect(
                "Bulan (Line Chart)",
                list(range(1, 13)),
                default=list(range(1, 13))
            )

        with fl3:
            param = st.selectbox("Parameter", fitur)

        # ===== FILTER DATA =====
        df_line = df[
            pd.to_datetime(df["Tanggal"]).dt.year.isin(tahun_line) &
            pd.to_datetime(df["Tanggal"]).dt.month.isin(bulan_line)
        ]

        # ===== VALIDASI =====
        if df_line.empty:
            st.warning("⚠️ Data tidak tersedia untuk Line Chart berdasarkan filter yang dipilih.")
        else:
            df_line = df_line.copy()
            df_line["Periode"] = pd.to_datetime(df_line["Tanggal"]).dt.to_period("M").astype(str)

            line_df = (
                df_line
                .groupby("Periode")[param]
                .mean()
                .reset_index()
            )

            if line_df.empty:
                st.warning("⚠️ Tidak ada data yang dapat divisualisasikan pada Line Chart.")
            else:
                y = pd.to_numeric(line_df[param], errors="coerce")

                if y.isna().all():
                    st.warning("⚠️ Nilai parameter tidak valid untuk divisualisasikan.")
                else:
                    x = np.arange(len(line_df))

                    fig2, ax2 = plt.subplots(figsize=(9,4))
                    ax2.plot(x, y, color=LINE_COLOR, linewidth=2.5)
                    ax2.fill_between(
                        x,
                        y,
                        y.min(),
                        color=FILL_COLOR,
                        alpha=0.6
                    )

                    ax2.set_xticks(x)
                    ax2.set_xticklabels(line_df["Periode"], rotation=45)
                    ax2.set_ylabel(param)
                    ax2.set_title(f"Tren {param}")

                    st.pyplot(fig2)


        # =====================================================
        # TABEL PERBANDINGAN
        # =====================================================
        st.markdown("---")
        st.subheader("📑 Perbandingan Status Mutu (IP dan Prediksi Model)")

        st.dataframe(df[KOL_INTERNAL], use_container_width=True)

        st.caption(
            "Tabel ini membandingkan status mutu air berdasarkan "
            "Indeks Pencemaran (IP) dan hasil prediksi model machine learning."
        )

# =====================================================
# TENTANG APLIKASI
# =====================================================
elif menu == "Tentang Aplikasi":

    st.title("ℹ️ Tentang Aplikasi")

    st.markdown("""
    ### Gambaran Umum Aplikasi
    Aplikasi ini merupakan **sistem prediksi kualitas air sungai** yang dikembangkan untuk
    membantu analisis dan pemantauan kondisi kualitas air secara **cepat, terstruktur,
    dan berbasis data**. Sistem ini ditujukan untuk mendukung kebutuhan akademik
    (penelitian dan tugas akhir) serta sebagai alat bantu analisis kualitas lingkungan.

    Aplikasi memungkinkan pengguna untuk melakukan **input data secara manual**
    maupun melalui **unggah file CSV atau Excel**, sehingga dapat digunakan
    baik untuk data skala kecil maupun data historis dalam jumlah besar.
    """)

    st.markdown("""
    ### Parameter Kualitas Air
    Parameter kualitas air yang digunakan dalam aplikasi ini meliputi:
    - **Temperatur (°C)**
    - **pH**
    - **Dissolved Oxygen (DO)**
    - **Biochemical Oxygen Demand (BOD)**
    - **Chemical Oxygen Demand (COD)**
    - **Total Suspended Solid (TSS)**
    - **Total Dissolved Solid (TDS)**

    Parameter-parameter tersebut merupakan indikator utama yang umum digunakan
    dalam penilaian kualitas air sungai dan memiliki pengaruh langsung terhadap
    kondisi ekosistem perairan.
    """)

    st.markdown("""
    ### Dasar Regulasi dan Pedoman
    Penentuan status kualitas air dalam aplikasi ini mengacu pada
    **Peraturan Pemerintah Republik Indonesia Nomor 22 Tahun 2021**,
    khususnya **Baku Mutu Air Kelas II**.

    Berdasarkan regulasi tersebut, setiap data kualitas air diklasifikasikan
    ke dalam dua kategori, yaitu **Memenuhi Baku Mutu** dan
    **Tidak Memenuhi Baku Mutu**, sehingga hasil prediksi memiliki
    landasan regulatif yang jelas dan relevan dengan standar lingkungan di Indonesia.
    """)

    st.markdown("""
    ### Model Prediksi
    Aplikasi ini menggunakan algoritma **XGBoost (Extreme Gradient Boosting)**
    sebagai model utama untuk melakukan prediksi kualitas air.

    XGBoost dipilih karena memiliki kemampuan yang baik dalam menangani data
    numerik multivariat, performa klasifikasi yang tinggi, serta efisien
    dalam memproses data berukuran besar. Model dilatih menggunakan data historis
    kualitas air yang telah melalui proses preprocessing dan pelabelan
    berdasarkan baku mutu PP No. 22 Tahun 2021.
    """)

    st.markdown("""
    ### Evaluasi Model
    Kinerja model prediksi dievaluasi menggunakan beberapa metrik evaluasi klasifikasi,
    antara lain:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1-Score**

    Selain itu, evaluasi juga dilakukan menggunakan **confusion matrix**
    untuk melihat distribusi hasil prediksi antara kelas
    *Memenuhi Baku Mutu* dan *Tidak Memenuhi Baku Mutu*.
    """)

    st.markdown("""
    ### Visualisasi dan Analisis
    Aplikasi ini dilengkapi dengan fitur visualisasi interaktif, yaitu:
    - **Bar chart** untuk menampilkan perbandingan jumlah data yang
      memenuhi dan tidak memenuhi baku mutu pada setiap periode waktu.
    - **Line chart** untuk menampilkan tren perubahan parameter kualitas air
      secara temporal, dilengkapi dengan teknik smoothing agar pola tren
      lebih mudah dibaca.

    Visualisasi ini bertujuan untuk membantu pengguna memahami dinamika
    kualitas air sungai secara intuitif dan informatif.
    """)

    st.markdown("""
    ### Tujuan Pengembangan
    Tujuan utama pengembangan aplikasi ini adalah:
    1. Membantu analisis kualitas air sungai secara cepat dan berbasis data  
    2. Mendukung kegiatan akademik seperti penelitian dan tugas akhir  
    3. Menyediakan visualisasi tren kualitas air yang mudah dipahami  
    4. Mendukung pengambilan keputusan berbasis data terkait pengelolaan kualitas air  

    Dengan demikian, aplikasi ini diharapkan dapat menjadi **alat bantu analisis
    kualitas air sungai** yang praktis, informatif, dan sesuai dengan
    standar regulasi yang berlaku.
    """)


# =====================================================
# PROSES PELATIHAN MODEL (READ-ONLY, TANPA TRAINING ULANG)
# =====================================================
elif menu == "Proses Pelatihan Model":

    st.title("📚 Bukti Proses Pelatihan Model")

    st.info(
        "Halaman ini menampilkan bukti tahapan pelatihan model machine learning "
        "yang telah dilakukan sebelumnya. Seluruh tabel dan visualisasi bersifat "
        "read-only dan tidak melibatkan proses pelatihan ulang."
    )

    # =====================================================
    # ① DATA LATIH (5 BARIS)
    # =====================================================
    st.markdown("### ① Contoh Data yang Digunakan untuk Pelatihan")

    df_latih = pd.read_excel(
        "HASIL_PELATIHAN_MODEL/01_data_latih_5_baris.xlsx"
    )

    st.dataframe(df_latih, use_container_width=True)

    st.caption(
        "Tabel ini menampilkan lima baris contoh data parameter kualitas air "
        "yang digunakan sebagai input awal dalam proses pelatihan model."
    )

    # =====================================================
    # ② HASIL PREPROCESSING
    # =====================================================
    st.markdown("---")
    st.markdown("### ② Hasil Preprocessing Data")

    df_prep = pd.read_excel(
        "HASIL_PELATIHAN_MODEL/02_data_setelah_preprocessing.xlsx"
    )

    st.dataframe(df_prep, use_container_width=True)

    st.caption(
        "Tahap preprocessing bertujuan untuk memastikan seluruh parameter "
        "berada dalam format numerik dan bebas dari nilai hilang, "
        "sehingga data siap digunakan untuk pelatihan model."
    )

    # =====================================================
    # ③ DATA TIME-AWARE
    # =====================================================
    st.markdown("---")
    st.markdown("### ③ Pembentukan Data Time-Aware")

    df_time = pd.read_excel(
        "HASIL_PELATIHAN_MODEL/03_time_aware_fitur_label_target.xlsx"
    )

    st.dataframe(df_time, use_container_width=True)

    st.caption(
        "Pendekatan time-aware dilakukan dengan membentuk target berdasarkan "
        "kondisi kualitas air pada waktu berikutnya, sehingga model mampu "
        "menangkap pola perubahan secara temporal."
    )

    # =====================================================
    # ④ EVALUASI KINERJA MODEL
    # =====================================================
    st.markdown("---")
    st.markdown("### ④ Evaluasi Kinerja Model")

    col1, col2 = st.columns([1.1, 1])

    # ---- Classification Report ----
    with col1:
        st.markdown("**Classification Report**")

        report_df = pd.read_excel(
            "HASIL_PELATIHAN_MODEL/04_classification_report.xlsx"
        )

        st.dataframe(
            report_df,
            use_container_width=True,
            height=260
        )

        st.caption(
            "Classification report digunakan untuk mengevaluasi performa model "
            "berdasarkan metrik precision, recall, dan F1-score pada masing-masing kelas."
        )

    # ---- Confusion Matrix ----
    with col2:
        st.markdown("**Confusion Matrix**")

        st.image(
            "HASIL_PELATIHAN_MODEL/05_confusion_matrix.png",
            use_container_width=True
        )

        st.caption(
            "Confusion matrix menunjukkan distribusi prediksi benar dan salah "
            "yang dihasilkan oleh model terhadap data uji."
        )

    # =====================================================
    # ⑤ FEATURE IMPORTANCE
    # =====================================================
    st.markdown("---")
    st.markdown("### ⑤ Feature Importance")

    st.image(
        "HASIL_PELATIHAN_MODEL/06_feature_importance.png",
        use_container_width=True
    )

    st.caption(
        "Visualisasi feature importance menggambarkan tingkat kontribusi "
        "masing-masing parameter kualitas air dalam mempengaruhi keputusan model."
    )


