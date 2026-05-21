import streamlit as st              # Membuat aplikasi web
import pandas as pd                 # Mengolah data tabel
import numpy as np                  # operasi numerik dan perhitungan matematik
import joblib                       # menyimpan dan memuat model ML
import matplotlib.pyplot as plt     # Membuat grafik
from io import BytesIO              # Export file ke memori

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
model = joblib.load("model_xgboost_kualitas_air.pkl")
metadata = joblib.load("metadata_model.pkl")
fitur = metadata["fitur"]
scaler = joblib.load("scaler_kualitas_air.pkl")

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
# PARAMETER IP (Baku Mutu Kelas II Kepmen LH 115/2003)
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
C_MAKS_DO = 7.5  # Nilai teoritis kelarutan oksigen maksimum pada suhu ~28°C

def hitung_ip(row):
    rasio = []

    # 1. TEMPERATUR (Penyimpangan)
    dev_temp = abs(row["Temperatur"] - TEMP_ALAMI)
    rasio.append(dev_temp / BM["Temperatur"])

    # 2. pH (Parameter Non-Konvensional)
    L_min, L_max = BM["pH_min"], BM["pH_max"]
    L_mid = (L_min + L_max) / 2
    if row["pH"] < L_mid:
        r_ph = (L_mid - row["pH"]) / (L_mid - L_min)
    else:
        r_ph = (row["pH"] - L_mid) / (L_max - L_mid)
    rasio.append(abs(r_ph))

    # 3. DO (Dissolved Oxygen - Parameter Berbanding Terbalik)
    # Menggunakan rumus khusus DO dengan C_maks teoritis
    r_do = (C_MAKS_DO - row["DO"]) / (C_MAKS_DO - BM["DO"])
    rasio.append(r_do)

    # 4. PARAMETER LAIN (BOD, COD, TSS, TDS)
    rasio.append(row["BOD"] / BM["BOD"])
    rasio.append(row["COD"] / BM["COD"])
    rasio.append(row["TSS"] / BM["TSS"])
    rasio.append(row["TDS"] / BM["TDS"])

    # 5. PENYESUAIAN KEPUTUSAN MENTERI LH 115/2003 (PENTING!)
    # Jika Ci/Lij > 1.0, maka dilakukan transformasi logaritmik agar tidak bias
    rasio_terpenuhi = []
    for r in rasio:
        if r > 1.0:
            # Menggunakan konstanta P = 5 untuk parameter dominan/bebas
            r_baru = 1.0 + 5.0 * np.log10(r)
            rasio_terpenuhi.append(r_baru)
        else:
            rasio_terpenuhi.append(r)

    rasio_terpenuhi = np.array(rasio_terpenuhi, dtype=float)
    
    # Hitung nilai akhir IP
    return np.sqrt((rasio_terpenuhi.max()**2 + rasio_terpenuhi.mean()**2) / 2)

# =====================================================
# SESSION
# =====================================================
KOL = ["Tanggal","Lokasi","Temperatur","pH","DO","BOD","COD","TSS","TDS","Nilai IP","Hasil Prediksi"]

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
# WARNA PARAMETER MELEBIHI BAKU MUTU
# =====================================================
def warna_parameter(val, kolom):

    try:
        val = float(val)
    except:
        return ""

    # Temperatur
    if kolom == "Temperatur":
        if abs(val - TEMP_ALAMI) > BM["Temperatur"]:
            return "background-color: #ff4d4d; color: white; font-weight: bold;"

    # pH
    elif kolom == "pH":
        if val < BM["pH_min"] or val > BM["pH_max"]:
            return "background-color: #ff4d4d; color: white; font-weight: bold;"

    # DO
    elif kolom == "DO":
        if val < BM["DO"]:
            return "background-color: #ff4d4d; color: white; font-weight: bold;"

    # BOD
    elif kolom == "BOD":
        if val > BM["BOD"]:
            return "background-color: #ff4d4d; color: white; font-weight: bold;"

    # COD
    elif kolom == "COD":
        if val > BM["COD"]:
            return "background-color: #ff4d4d; color: white; font-weight: bold;"

    # TSS
    elif kolom == "TSS":
        if val > BM["TSS"]:
            return "background-color: #ff4d4d; color: white; font-weight: bold;"

    # TDS
    elif kolom == "TDS":
        if val > BM["TDS"]:
            return "background-color: #ff4d4d; color: white; font-weight: bold;"

    return ""

# =====================================================
# MENU 1
# =====================================================
if menu == "🏠 Input & Prediksi":

    st.title("📊 Input & Prediksi Data Kualitas Air")

    # ================= INPUT =================
    st.subheader("📝 Input Manual")

    c1, c2 = st.columns(2)

    with c1:
        temperatur = st.number_input(
            "Temperatur (°C) | Rentang normal: 25–31°C",
            0.0, 50.0, 25.0,
            help="Deviasi maksimal ±3°C dari temperatur alami"
        )

        ph = st.number_input(
            "pH | Rentang normal: 6–9",
            0.0, 14.0, 7.0,
            help="Nilai pH di luar rentang 6–9 dianggap melebihi baku mutu"
        )

        do = st.number_input(
            "DO - Dissolved Oxygen (mg/L) | Minimal 4 mg/L",
            0.0, 20.0, 4.0,
            help="Nilai DO kurang dari 4 mg/L tidak memenuhi baku mutu"
        )

        bod = st.number_input(
            "BOD (mg/L) | Maksimal 3 mg/L",
            0.0, 50.0, 3.0,
            help="Nilai BOD lebih dari 3 mg/L menunjukkan pencemaran organik"
        )

    with c2:
        cod = st.number_input(
            "COD (mg/L) | Maksimal 25 mg/L",
            0.0, 100.0, 25.0,
            help="Nilai COD tinggi menunjukkan banyak bahan pencemar kimia"
        )

        tss = st.number_input(
            "TSS (mg/L) | Maksimal 50 mg/L",
            0.0, 500.0, 50.0,
            help="Nilai TSS tinggi menunjukkan banyak partikel tersuspensi"
        )

        tds = st.number_input(
            "TDS (mg/L) | Maksimal 1000 mg/L",
            0.0, 5000.0, 1000.0,
            help="Nilai TDS tinggi menunjukkan banyak zat terlarut"
        )
        lokasi = st.text_input("Lokasi Pengambilan Sampel")
        tanggal = st.date_input("Tanggal")

    if st.button("🔍 Prediksi & Simpan"):
        # 1. Buat DataFrame data asli
        df_new = pd.DataFrame([{
            "Tanggal": tanggal,
            "Lokasi": lokasi,
            "Temperatur": temperatur,
            "pH": ph,
            "DO": do,
            "BOD": bod,
            "COD": cod,
            "TSS": tss,
            "TDS": tds
        }])

        # 2. Hitung IP (menggunakan data asli)
        df_new["Nilai IP"] = df_new.apply(hitung_ip, axis=1)

        # 3. STANDARISASI sebelum Prediksi
        # menstandarisasi kolom fitur saja
        data_untuk_prediksi = df_new[fitur].copy()
        data_scaled = scaler.transform(data_untuk_prediksi) 

        # 4. Prediksi menggunakan data yang sudah di-scale
        pred = model.predict(data_scaled)[0] 
        
        df_new["Hasil Prediksi"] = "Memenuhi Baku Mutu" if pred == 0 else "Tidak Memenuhi"

        # Simpan ke session state (simpan data
        st.session_state.data_all = pd.concat(
            [st.session_state.data_all, df_new[KOL]],
            ignore_index=True
        )
        st.success("Data berhasil ditambahkan")

    # ================= TEMPLATE =================
    st.markdown("---")
    st.subheader("📥 Download Template Excel")

    template = pd.DataFrame(columns=[
        "Tanggal","Lokasi","Temperatur","pH","DO","BOD","COD","TSS","TDS"
    ])
    download_excel(template, "template_kualitas_air.xlsx")

    # ================= UPLOAD =================
    st.markdown("---")
    st.subheader("📤 Upload Data")

    file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

    # ================= PROSES FILE =================
    if file is not None and "uploaded_file" not in st.session_state:

        # baca file
        df_up = (
            pd.read_csv(file)
            if file.name.endswith(".csv")
            else pd.read_excel(file)
        )

        # ================= VALIDASI KOLOM =================
        kolom_wajib = [
            "Tanggal","Lokasi","Temperatur","pH",
            "DO","BOD","COD","TSS","TDS"
        ]

        kolom_tidak_ada = [
            k for k in kolom_wajib if k not in df_up.columns
        ]

        if kolom_tidak_ada:
            st.error(f"Kolom berikut tidak ditemukan: {kolom_tidak_ada}")
            st.stop()

        # ================= KONVERSI NUMERIK =================
        kolom_numerik = [
            "Temperatur","pH","DO","BOD",
            "COD","TSS","TDS"
        ]

        for col in kolom_numerik:
            df_up[col] = pd.to_numeric(
                df_up[col],
                errors="coerce"
            )

        # ================= CEK NILAI KOSONG =================
        kosong = df_up[kolom_numerik].isnull().sum()

        if kosong.sum() > 0:

            st.warning("⚠️ Ditemukan data kosong atau format angka tidak valid")

            st.dataframe(
                kosong[kosong > 0]
                .reset_index()
                .rename(columns={
                    "index": "Parameter",
                    0: "Jumlah Kosong"
                }),
                use_container_width=True
            )

            st.info("""
            Pastikan seluruh parameter numerik terisi dengan benar sebelum dilakukan prediksi.

            Contoh kesalahan:
            - sel kosong
            - teks pada kolom angka
            - simbol atau format tidak sesuai
            """)

            st.stop()

        # ================= LANJUT PROSES =================
        df_up["Tanggal"] = pd.to_datetime(
            df_up["Tanggal"]
        ).dt.date

        df_up["Nilai IP"] = df_up.apply(
            hitung_ip,
            axis=1
        )

        # 1. Transform dulu fiturnya ke skala standarisasi
        data_up_scaled = scaler.transform(df_up[fitur])

        # 2. Masukkan data yang sudah di-scale ke model.predict
        df_up["Hasil Prediksi"] = np.where(
            model.predict(data_up_scaled) == 0,
            "Memenuhi Baku Mutu",
            "Tidak Memenuhi"
        )
        # ------------------------------------------

        st.session_state.data_all = pd.concat(
            [st.session_state.data_all, df_up[KOL]],
            ignore_index=True
        )

        st.success("Data berhasil diupload")
        st.session_state.uploaded_file = file.name

    # ================= OUTPUT =================
    if not st.session_state.data_all.empty:
        

        df = st.session_state.data_all.copy()
        # ================= FILTER DATA =================
        st.markdown("---")
        st.subheader("🔎 Filter Data")

        df["Tanggal"] = pd.to_datetime(df["Tanggal"])

        colf1, colf2, colf3, colf4 = st.columns(4)

        with colf1:

            tahun_list = sorted(df["Tanggal"].dt.year.unique())

            tahun_options = ["Semua"] + tahun_list

            tahun = st.multiselect(
                "Filter Tahun",
                tahun_options,
                default=["Semua"]
            )

            if "Semua" in tahun or len(tahun) == 0:
                tahun = tahun_list


        with colf2:

            bulan_list = list(range(1, 13))

            bulan_options = ["Semua"] + bulan_list

            bulan = st.multiselect(
                "Filter Bulan",
                bulan_options,
                default=["Semua"]
            )

            if "Semua" in bulan or len(bulan) == 0:
                bulan = bulan_list


        with colf3:

            lokasi_list = sorted(df["Lokasi"].dropna().unique())

            lokasi_options = ["Semua"] + lokasi_list

            lokasi_filter = st.multiselect(
                "Filter Lokasi",
                lokasi_options,
                default=["Semua"]
            )

            if "Semua" in lokasi_filter or len(lokasi_filter) == 0:
                lokasi_filter = lokasi_list

        with colf4:
            tanggal_range = st.date_input(
                "Filter Rentang Tanggal",
                [df["Tanggal"].min(), df["Tanggal"].max()]
            )
        
        # ================= APPLY FILTER =================
        df_filter = df[
            df["Tanggal"].dt.year.isin(tahun) &
            df["Tanggal"].dt.month.isin(bulan) &
            df["Lokasi"].isin(lokasi_filter)
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

        col_ket1, col_ket2 = st.columns(2)

        with col_ket1:
            st.caption("""
            Keterangan Parameter:

            • Temperatur : deviasi > 3°C dari kondisi alami  
            • pH : < 6 atau > 9  
            • DO : < 4 mg/L  
            • BOD : > 3 mg/L  
            • COD : > 25 mg/L  
            • TSS : > 50 mg/L  
            • TDS : > 1000 mg/L

            🟥 Sel merah menunjukkan parameter melebihi baku mutu.
            """)

        with col_ket2:
            st.caption("""
            Keterangan Status Mutu Air (IP):

            • IP ≤ 1,0 → Memenuhi Baku Mutu  
            • IP > 1,0 → Tidak Memenuhi Baku Mutu
            """)

        df_tampil = df_filter.copy()
        df_tampil["Tanggal"] = pd.to_datetime(df_tampil["Tanggal"]).dt.date

        # =====================================================
        # STYLE TABEL HASIL PREDIKSI
        # =====================================================
        kolom_parameter = [
            "Temperatur",
            "pH",
            "DO",
            "BOD",
            "COD",
            "TSS",
            "TDS"
        ]

        styled_df = df_tampil.style.format({
            "Temperatur": "{:.1f}",
            "pH": "{:.2f}",
            "DO": "{:.2f}",
            "BOD": "{:.2f}",
            "COD": "{:.2f}",
            "TSS": "{:.0f}",
            "TDS": "{:.0f}",
            "Nilai IP": "{:.2f}"
        })

        for col in kolom_parameter:
            styled_df = styled_df.map(
                lambda x, c=col: warna_parameter(x, c),
                subset=[col]
            )

        st.dataframe(
            styled_df,
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:
            download_excel(df_filter, "hasil_prediksi.xlsx")

        with col2:
            if st.button("🗑️ Hapus Semua Data"):
                st.session_state.data_all = pd.DataFrame(columns=KOL)

                if "uploaded_file" in st.session_state:
                    del st.session_state.uploaded_file

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

            bar1 = ax.bar(
                x - width/2,
                bar_df.get("Memenuhi Baku Mutu", 0),
                width,
                label="Memenuhi"
            )

            bar2 = ax.bar(
                x + width/2,
                bar_df.get("Tidak Memenuhi", 0),
                width,
                label="Tidak Memenuhi"
            )

            # ================= LABEL ANGKA =================
            for bars in [bar1, bar2]:

                for bar in bars:

                    height = bar.get_height()

                    if height > 0:

                        ax.text(
                            bar.get_x() + bar.get_width()/2,
                            height + 0.1,
                            f"{int(height)}",
                            ha="center",
                            va="bottom",
                            fontsize=9
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(bar_df.index, rotation=45)
            ax.set_ylabel("Jumlah Data")
            ax.set_title("Distribusi Status Kualitas Air")
            ax.legend()

            st.pyplot(fig)

        # ================= BAR CHART STATUS PER LOKASI =================
        if not df_filter.empty:

            df_filter = df_filter.copy()

            st.markdown("---")
            st.subheader("📊 Status Kualitas Air Berdasarkan Lokasi")

            # ================= PILIH PERIODE =================
            mode_periode = st.selectbox(
                "Tampilkan Berdasarkan",
                ["Per Bulan", "Per Tahun"]
            )

            # buat periode
            if mode_periode == "Per Bulan":

                df_filter["Periode"] = (
                    df_filter["Tanggal"]
                    .dt.to_period("M")
                    .astype(str)
                )

            else:

                df_filter["Periode"] = (
                    df_filter["Tanggal"]
                    .dt.year
                    .astype(str)
                )

            # gabungkan lokasi + periode
            df_filter["LokasiPeriode"] = (
                df_filter["Lokasi"] +
                " | " +
                df_filter["Periode"]
            )

            # hitung jumlah status
            lokasi_chart = (
                df_filter
                .groupby(["LokasiPeriode", "Hasil Prediksi"])
                .size()
                .unstack(fill_value=0)
            )

            # total lokasi
            total_lokasi = len(lokasi_chart)

            # jika hanya 1 lokasi
            if total_lokasi == 1:

                jumlah_lokasi = 1

                st.info("Hanya terdapat 1 lokasi/periode untuk ditampilkan")

            else:

                jumlah_lokasi = st.slider(
                    "Jumlah Lokasi yang Ditampilkan",
                    min_value=1,
                    max_value=total_lokasi,
                    value=total_lokasi
                )

            st.info(
                f"Menampilkan {jumlah_lokasi} dari total "
                f"{total_lokasi} lokasi dan periode"
            )

            # tampilkan sesuai slider
            lokasi_chart = lokasi_chart.head(jumlah_lokasi)

            # ukuran figure
            tinggi_fig = max(6, jumlah_lokasi * 0.6)

            fig3, ax3 = plt.subplots(
                figsize=(12, tinggi_fig)
            )

            # posisi batang
            y = np.arange(len(lokasi_chart))
            width = 0.35

            memenuhi = lokasi_chart.get("Memenuhi Baku Mutu", 0)
            tidak = lokasi_chart.get("Tidak Memenuhi", 0)

            # horizontal bar chart
            bar1 = ax3.barh(
                y - width/2,
                memenuhi,
                height=width,
                label="Memenuhi Baku Mutu"
            )

            bar2 = ax3.barh(
                y + width/2,
                tidak,
                height=width,
                label="Tidak Memenuhi"
            )

            # label angka
            for bars in [bar1, bar2]:

                for bar in bars:

                    nilai = bar.get_width()

                    if nilai > 0:

                        ax3.text(
                            nilai + 0.2,
                            bar.get_y() + bar.get_height()/2,
                            f"{int(nilai)}",
                            ha="left",
                            va="center",
                            fontsize=8
                        )

            ax3.set_yticks(y)

            ax3.set_yticklabels(
                lokasi_chart.index,
                fontsize=8
            )

            ax3.set_xlabel("Jumlah Data")
            ax3.set_ylabel("Lokasi dan Periode")

            ax3.set_title(
                "Distribusi Status Kualitas Air Berdasarkan Lokasi dan Periode"
            )

            ax3.legend()

            plt.tight_layout()

            st.pyplot(fig3)
                

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

        df_compare = df_filter.copy()

        df_compare["Status IP"] = np.where(
            df_compare["Nilai IP"] <= 1,
            "Memenuhi Baku Mutu",
            "Tidak Memenuhi"
        )

        df_compare["Tanggal"] = pd.to_datetime(
            df_compare["Tanggal"]
        ).dt.date

        st.dataframe(
            df_compare[[
                "Tanggal","Lokasi","Temperatur","pH","DO","BOD","COD","TSS","TDS",
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
    df_latih = pd.read_excel("HASIL_PELATIHAN_XGBOOST/01_data_mentah_5_baris.xlsx")

    df_latih_tampil = df_latih.copy()
    df_latih_tampil["Tanggal"] = pd.to_datetime(df_latih_tampil["Tanggal"]).dt.date

    st.dataframe(df_latih_tampil, use_container_width=True)

    # ================= EVALUASI MODEL =================
    st.markdown("---")
    st.subheader("Evaluasi Kinerja Model")

    # ================= CLASSIFICATION REPORT =================
    col1, col2 = st.columns(2)

    with col1:
        df_report = pd.read_excel("HASIL_PELATIHAN_XGBOOST/02_classification_report.xlsx")
        st.dataframe(df_report, use_container_width=True)

        st.markdown(
        """
        <div style='text-align: center; font-size:14px; color:gray; margin-top:5px;'>
            Classification Report Model XGBoost
        </div>
        """,
        unsafe_allow_html=True
        )

    with col2:
        st.markdown("""
        **Classification Report**
                    
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
            "HASIL_PELATIHAN_XGBOOST/03_confusion_matrix.png",
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

    # ===== ROC CURVE =====
    st.markdown("---")

    col_roc1, col_roc2 = st.columns([1.1, 1])

    with col_roc1:
        st.image(
            "HASIL_PELATIHAN_XGBOOST/17_roc_curve.png",
            caption="ROC Curve XGBoost"
        )

    with col_roc2:
        st.markdown("""
        **ROC Curve dan AUC**

        ROC Curve (Receiver Operating Characteristic Curve) digunakan untuk 
        mengevaluasi kemampuan model dalam membedakan dua kelas, yaitu 
        *Memenuhi Baku Mutu* dan *Tidak Memenuhi Baku Mutu*.

        Kurva ROC menunjukkan hubungan antara:
        - True Positive Rate (TPR)
        - False Positive Rate (FPR)

        Semakin mendekati sudut kiri atas, maka performa model semakin baik 
        dalam melakukan klasifikasi.

        Selain itu, nilai AUC (Area Under Curve) digunakan untuk mengukur 
        kualitas model secara keseluruhan. Nilai AUC yang mendekati 1 
        menunjukkan bahwa model memiliki kemampuan klasifikasi yang sangat baik.

        Berdasarkan hasil pengujian, model XGBoost mampu membedakan kondisi 
        kualitas air dengan performa yang baik.
        """)

    # ===== 2. FEATURE IMPORTANCE =====
    st.markdown("---")

    col3, col4 = st.columns([1.1, 1])

    with col3:
        st.image(
            "HASIL_PELATIHAN_XGBOOST/04_feature_importance.png",
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
            "HASIL_PELATIHAN_XGBOOST/11_split_data.png",
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
        st.image("HASIL_PELATIHAN_XGBOOST/12_max_depth.png")

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
        st.image("HASIL_PELATIHAN_XGBOOST/13_learning_rate.png")

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
        st.image("HASIL_PELATIHAN_XGBOOST/14_n_estimators.png")

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
        st.image("HASIL_PELATIHAN_XGBOOST/15_sampling.png")

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