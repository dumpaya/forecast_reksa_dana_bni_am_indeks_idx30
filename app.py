import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Dashboard Forecast Reksa Dana IDX30",
    layout="wide"
)

# 2. DUMMY MODEL (HANYA UNTUK LOAD STATE_DICT)
class EmptyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer kosong — akan ditimpa oleh state_dict
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        return x[:, -1:, :]  # hanya placeholder, tidak dipakai


# 3. LOAD DATA
@st.cache_data
def load_excel():
    df = pd.read_excel("return_reksa_dana_bni_fixed.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")

df = load_excel()


# 4. LOAD MODEL 
@st.cache_resource
def load_models():

    bilstm = EmptyModel()
    hybrid = EmptyModel()

    try:
        bilstm.load_state_dict(
            torch.load("/mnt/data/bilstm_model.pt", map_location="cpu"),
            strict=False
        )
        bilstm.eval()

        hybrid.load_state_dict(
            torch.load("/mnt/data/hybrid_cnn_bilstm.pt", map_location="cpu"),
            strict=False
        )
        hybrid.eval()

    except Exception as e:
        st.error(f"❌ Error load model: {e}")
        st.stop()

    with open("/mnt/data/holt_winters_best.pkl", "rb") as f:
        hw = pickle.load(f)

    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return bilstm, hybrid, hw, scaler


bilstm_model, hybrid_model, hw_model, scaler = load_models()


# 5. PREPROCESS WINDOW
WINDOW = 30

def make_sequence(data, window=WINDOW):
    seq = []
    for i in range(len(data) - window):
        seq.append(data[i: i + window])
    return np.array(seq)


# 6. PREDIKSI MODEL
def predict_bilstm(df):
    X = scaler.transform(df[["Return"]])
    seq = make_sequence(X)
    seq = torch.tensor(seq, dtype=torch.float32)

    with torch.no_grad():
        pred = bilstm_model(seq)

    pred = pred.numpy().reshape(-1, 1)
    return scaler.inverse_transform(pred).flatten()

def predict_hybrid(df):
    X = scaler.transform(df[["Return"]])
    seq = make_sequence(X)
    seq = torch.tensor(seq, dtype=torch.float32)

    with torch.no_grad():
        pred = hybrid_model(seq)

    pred = pred.numpy().reshape(-1, 1)
    return scaler.inverse_transform(pred).flatten()

def predict_hw(steps):
    return hw_model.forecast(steps)


# 7. SIDEBAR
st.sidebar.header("Pengaturan")
model_choice = st.sidebar.selectbox(
    "Pilih model:",
    ["Holt-Winters", "BiLSTM", "Hybrid CNN-BiLSTM"]
)

steps = st.sidebar.slider("Horizon prediksi:", 10, 90, 30)


# 8. MAIN UI
st.title("📈 Dashboard Forecast Return Reksa Dana BNI-AM IDX30")

st.write("Data terbaru:")
st.dataframe(df.tail(10))


# 9. VISUALISASI
if model_choice == "Holt-Winters":

    st.subheader("Forecast Holt-Winters")
    forecast = predict_hw(steps)
    dates = pd.date_range(df["Date"].iloc[-1], periods=steps + 1, freq="D")[1:]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Return"], label="Actual")
    ax.plot(dates, forecast, label="Forecast")
    ax.legend()
    st.pyplot(fig)

else:
    st.subheader(model_choice)

    if model_choice == "BiLSTM":
        pred = predict_bilstm(df)
    else:
        pred = predict_hybrid(df)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Return"], label="Actual")
    ax.plot(df["Date"].iloc[WINDOW:], pred, label=model_choice)
    ax.legend()
    st.pyplot(fig)


# 10. DOWNLOAD BUTTON
if st.button("Download Hasil Prediksi"):

    if model_choice == "Holt-Winters":
        result = pd.DataFrame({"Date": dates, "Forecast": forecast})
    else:
        result = pd.DataFrame({
            "Date": df["Date"].iloc[WINDOW:],
            "Forecast": pred
        })

    st.download_button(
        "Download CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{model_choice}.csv",
        mime="text/csv"
    )
