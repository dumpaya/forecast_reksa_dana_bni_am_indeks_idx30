import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.preprocessing import MinMaxScaler 
import os # Tambahkan import os untuk pengecekan file

st.set_page_config(
    page_title="Dashboard Forecast Reksa Dana IDX30",
    layout="wide"
)

# 2. DEFINISI ARSITEKTUR MODEL PYTORCH (HARAP SESUAIKAN DENGAN MODEL ASLI ANDA!)
# WINDOW = 30, FEATURES = 1
WINDOW = 30

class BiLSTMModel(nn.Module):
    # Contoh arsitektur BiLSTM untuk input sequence (30, 1)
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size) # *2 karena bidirectional

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        # Ambil output dari langkah terakhir untuk prediksi satu langkah
        out = self.fc(h_lstm[:, -1, :])
        return out

class HybridCNNBiLSTMModel(nn.Module):
    # Contoh arsitektur CNN-BiLSTM
    def __init__(self, input_feature=1, cnn_filters=32, lstm_hidden=64):
        super(HybridCNNBiLSTMModel, self).__init__()
        # CNN Layer: Mengubah shape (N, 30, 1) -> (N, 1, 30) -> Conv1D -> (N, 32, 30)
        self.conv1d = nn.Conv1d(in_channels=input_feature, out_channels=cnn_filters, kernel_size=3, padding=1)
        
        # LSTM Layer: Input size adalah output dari CNN (cnn_filters = 32)
        # Perhatikan: ukuran input LSTM mungkin berbeda jika ada MaxPool
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        # 1. Permute untuk Conv1D: (N, seq_len, features) -> (N, features, seq_len)
        x = x.permute(0, 2, 1) 
        x = F.relu(self.conv1d(x))
        
        # 2. Permute kembali untuk LSTM: (N, features, seq_len) -> (N, seq_len, features)
        x = x.permute(0, 2, 1) 
        
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out

# Fungsi diagnostik untuk memeriksa file di direktori kerja
def check_file_in_root_dir():
    file_name = "return_reksa_dana_bni_fixed.xlsx"
    current_dir = os.getcwd()
    
    # Cek isi direktori kerja
    try:
        files_in_dir = os.listdir(current_dir)
        st.info(f"Direktori Kerja: {current_dir}")
        st.info(f"File ditemukan di direktori kerja: {files_in_dir}")
        if "model" in files_in_dir and os.path.isdir("model"):
            files_in_model = os.listdir("model")
            st.info(f"File ditemukan di folder 'model': {files_in_model}")
            
    except Exception as e:
        st.warning(f"Gagal membaca isi direktori: {e}")
        
    # Pengecekan status file Excel
    if file_name not in files_in_dir:
        st.error(f"❌ File Excel '{file_name}' tidak terdaftar di direktori root.")
        st.stop()
    
    return file_name

# 3. LOAD DATA
@st.cache_data
def load_excel():
    
    file_name = check_file_in_root_dir()

    try:
        # Menggunakan nama file langsung (relative path)
        df = pd.read_excel(file_name) 
        df["Date"] = pd.to_datetime(df["Date"])
        st.success(f"✅ File Excel '{file_name}' berhasil dimuat!")
        return df.sort_values("Date")
    except Exception as e:
        st.error(f"❌ Gagal memuat atau memproses file Excel: {e}")
        st.stop()

df = load_excel()


# 4. LOAD MODEL (Menggunakan Path Relatif) 
@st.cache_resource
def load_models():

    bilstm = BiLSTMModel()
    hybrid = HybridCNNBiLSTMModel()

    try:
        # PENTING: Gunakan path relatif ke folder 'model'
        bilstm.load_state_dict(
            torch.load("model/bilstm_model.pt", map_location="cpu")
        )
        bilstm.eval()

        hybrid.load_state_dict(
            torch.load("model/hybrid_cnn_bilstm.pt", map_location="cpu")
        )
        hybrid.eval()
        st.success("✅ Model PyTorch (BiLSTM & Hybrid) berhasil dimuat!")


    except Exception as e:
        st.error(f"❌ Error saat memuat model PyTorch. Pastikan arsitektur di kelas BiLSTMModel dan HybridCNNBiLSTMModel sudah benar. Error: {e}")
        st.stop()

    # PENTING: Gunakan path relatif ke folder 'model'
    try:
        with open("model/holt_winters_best.pkl", "rb") as f:
            hw = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Gagal memuat model Holt-Winters (holt_winters_best.pkl). Error: {e}")
        st.stop()

    try:
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Gagal memuat Scaler (scaler.pkl). Error: {e}")
        st.stop()

    return bilstm, hybrid, hw, scaler


bilstm_model, hybrid_model, hw_model, scaler = load_models()


# 5. PREPROCESS WINDOW
def make_sequence(data, window=WINDOW):
    seq = []
    for i in range(len(data) - window):
        seq.append(data[i: i + window])
    return np.array(seq)


# 6. PREDIKSI MODEL
def predict_bilstm(df):
    X = scaler.transform(df[["Return"]])
    seq = make_sequence(X)
    # Ubah ke PyTorch tensor (batch_size, seq_len, features)
    seq = torch.tensor(seq, dtype=torch.float32)

    with torch.no_grad():
        pred = bilstm_model(seq)

    pred = pred.numpy().reshape(-1, 1)
    return scaler.inverse_transform(pred).flatten()

def predict_hybrid(df):
    X = scaler.transform(df[["Return"]])
    seq = make_sequence(X)
    # Ubah ke PyTorch tensor (batch_size, seq_len, features)
    seq = torch.tensor(seq, dtype=torch.float32)

    with torch.no_grad():
        pred = hybrid_model(seq)

    pred = pred.numpy().reshape(-1, 1)
    return scaler.inverse_transform(pred).flatten()

def predict_hw(steps):
    # Asumsi model Holt-Winters dapat memprediksi langsung
    return hw_model.forecast(steps=steps)


# 7. SIDEBAR
st.sidebar.header("Pengaturan Forecast")
model_choice = st.sidebar.selectbox(
    "Pilih Model:",
    ["Holt-Winters", "BiLSTM", "Hybrid CNN-BiLSTM"]
)

steps = st.sidebar.slider("Horizon Prediksi (Hari):", 10, 90, 30)


# 8. MAIN UI
st.title("📈 Dashboard Forecast Return Reksa Dana BNI-AM IDX30")

st.write("Data terbaru (Return):")
st.dataframe(df.tail(10))


# 9. VISUALISASI
st.subheader(f"Hasil Prediksi Menggunakan {model_choice}")

# Inisialisasi variabel pred dan dates
pred = None
dates = None
forecast = None

try:
    if model_choice == "Holt-Winters":
        forecast = predict_hw(steps)
        # Tanggal prediksi dimulai setelah hari terakhir data aktual
        dates = pd.date_range(df["Date"].iloc[-1], periods=steps + 1, freq="D")[1:]

    elif model_choice == "BiLSTM":
        pred = predict_bilstm(df)
        
    else: # Hybrid CNN-BiLSTM
        pred = predict_hybrid(df)

except Exception as e:
    st.error(f"Gagal melakukan prediksi untuk model {model_choice}. Detail: {e}")
    st.stop()


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["Date"], df["Return"], label="Actual Return", color='tab:blue')

if model_choice == "Holt-Winters" and forecast is not None:
    # Plot forecast Holt-Winters
    ax.plot(dates, forecast, label=f"Forecast ({steps} Hari)", color='tab:orange', linestyle='--')
    
    # Tambahkan garis vertikal pemisah data historis dan prediksi
    ax.axvline(x=df["Date"].iloc[-1], color='gray', linestyle=':', linewidth=1, label='Start Forecast')
    
elif pred is not None:
    # Untuk model DL, plot prediksi dimulai setelah WINDOW
    ax.plot(df["Date"].iloc[WINDOW:], pred, label=f"Predicted (In-Sample)", color='tab:red')

# Styling plot
ax.set_title(f"Perbandingan Actual vs. {model_choice} Hasil", fontsize=16)
ax.set_xlabel("Tanggal")
ax.set_ylabel("Return Harian")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
st.pyplot(fig)


# 10. DOWNLOAD BUTTON
if st.button("Download Hasil Prediksi"):

    if model_choice == "Holt-Winters" and dates is not None:
        result = pd.DataFrame({"Date": dates, "Forecast": forecast})
    elif pred is not None:
        result = pd.DataFrame({
            "Date": df["Date"].iloc[WINDOW:],
            "Forecast": pred
        })
    else:
        st.warning("Tidak ada data prediksi untuk diunduh.")
        st.stop()

    st.download_button(
        "Download CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{model_choice.lower().replace('-', '_')}.csv",
        mime="text/csv"
    )
