#1. config & import
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard Forecast Reksa Dana IDX30",
    layout="wide"
)


#2. load data excel
@st.cache_data
def load_excel():
    df = pd.read_excel("return_reksa_dana_bni_fixed.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

df = load_excel()


#3. load model
@st.cache_resource
def load_models():
    bilstm = load_model("model/bilstm_model.h5")
    hybrid = load_model("model/hybrid_cnn_bilstm.h5")

    with open("model/holt_winters_best.pkl", "rb") as f:
        hw = pickle.load(f)

    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return bilstm, hybrid, hw, scaler

bilstm_model, hybrid_model, hw_model, scaler = load_models()


#4. fungsi preprocess data
def prepare_sequence(data, window=30):
    seq = []
    for i in range(len(data) - window):
        seq.append(data[i:i+window])
    return np.array(seq)


#5. fungsi prediksi
def predict_bilstm(df):
    scaled = scaler.transform(df[["Return"]])
    seq = prepare_sequence(scaled)
    pred = bilstm_model.predict(seq)
    return scaler.inverse_transform(pred)

def predict_hybrid(df):
    scaled = scaler.transform(df[["Return"]])
    seq = prepare_sequence(scaled)
    pred = hybrid_model.predict(seq)
    return scaler.inverse_transform(pred)

def predict_hw(df, steps=30):
    return hw_model.forecast(steps)


#6. sidebar
st.sidebar.header("Pengaturan")
model_choice = st.sidebar.selectbox(
    "Pilih model:",
    ["Holt-Winters", "BiLSTM", "Hybrid CNN-BiLSTM"]
)

steps = st.sidebar.slider("Horizon prediksi (hari):", 10, 90, 30)


#7. main content
st.title("Dashboard Forecast Return Reksa Dana BNI-AM IDX30")

st.write("Data terbaru:")
st.dataframe(df.tail(10))


#8. visualisasi hasil prediksi
if model_choice == "Holt-Winters":
    st.subheader("Prediksi Holt-Winters")
    forecast = predict_hw(df, steps)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Return"], label="Actual")
    dates = pd.date_range(df["Date"].iloc[-1], periods=steps+1, freq="D")[1:]
    ax.plot(dates, forecast, label="Forecast")
    ax.legend()
    st.pyplot(fig)

elif model_choice == "BiLSTM":
    st.subheader("Prediksi BiLSTM")
    pred = predict_bilstm(df)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Return"], label="Actual")
    ax.plot(df["Date"][30:], pred, label="BiLSTM")
    ax.legend()
    st.pyplot(fig)

else:
    st.subheader("Prediksi Hybrid CNN-BiLSTM")
    pred = predict_hybrid(df)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["Return"], label="Actual")
    ax.plot(df["Date"][30:], pred, label="Hybrid CNN-BiLSTM")
    ax.legend()
    st.pyplot(fig)


#9. tombol download hasil
if st.button("Download Hasil Prediksi"):
    if model_choice == "Holt-Winters":
        result = pd.DataFrame({
            "Date": dates,
            "Forecast": forecast
        })
    else:
        result = pd.DataFrame({
            "Date": df["Date"][30:],
            "Forecast": pred.flatten()
        })

    st.download_button(
        label="Download CSV",
        data=result.to_csv(index=False),
        file_name=f"forecast_{model_choice.lower()}.csv",
        mime="text/csv"
    )
