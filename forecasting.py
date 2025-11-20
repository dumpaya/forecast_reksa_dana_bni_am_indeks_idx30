import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model

# Fungsi evaluasi tanpa sklearn
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

## 1. Data Loading dan Prepocessing
# Path file
path = r"E:\Tugas Kuliah\DasLit\return_reksa dana_bni_fixed.xlsx"

# Baca file Excel
df = pd.read_excel(path)

# Lihat isi data
print(df.head())

# cek info data
print(df.info())

#cek missing value
print(df.isnull().sum())

# Mengisi missing value dengan metode forward fill
df.fillna(method='ffill', inplace=True)

# konversi kolom 'Tanggal' ke tipe datatime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.fillna(method='ffill', inplace=True)

# menampilkan data setelah prepocessing
print("\n=== Data Setelah Preprocessing ===")
print(df.head(10))

# cek ulang missing value
print("\n Missing Value Setelah Cleaning")
print(df.isnull().sum())


## 2. Exploratory Data Analysis (EDA)
# Statistik deskriptif
print("\n=== Statistik Deskriptif ===")
print(df.describe())

# Visualisasi tren return reksa dana
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df['return'], color='blue', linewidth=1.5)
ax.set_title('Tren Return Reksa Dana BNI-AM Index IDX30')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Return')
ax.grid(True)

# Format tanggal
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)

# Tambahkan interaktivitas hover
cursor = mplcursors.cursor(ax, hover=True)
@cursor.connect("add")
def on_hover(sel):
    xdata = sel.target[0]
    x_datetime = mdates.num2date(xdata).replace(tzinfo=None)
    nearest_index = (abs(df.index - x_datetime)).argmin()
    nearest_date = df.index[nearest_index]
    y_value = df['return'].iloc[nearest_index]
    sel.annotation.set(text=f'Tanggal: {x_datetime.date()}\n Return: {y_value:.4f}')

plt.show()

# Visualisasi distribusi return reksa dana
plt.figure(figsize=(8,6))
plt.hist(df['return'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribusi Return Reksa Dana BNI-AM Indeks IDX30')
plt.xlabel('Return')
plt.ylabel('Frekuensi')
plt.grid()
plt.show()

# Uji Stationeritas menggunakan ADF Test
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['return'])
print('\n === Hasil ADF Test ===')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' %result[1])
for key, value in result[4].items():
    print('Critical Values:')
    print('\t%s: %.3f' % (key, value))
    if result[1] < 0.05:
        print('Data Stationer tolak H0 (Stationer)')
    else:
        print('Data Non-Stationer terima H0 (Non-Stationer)')

# Decompose time series
decomposition = seasonal_decompose(df['return'], model='additive', period=5)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

# single holt winters exponential smoothing model
seasonal_model = ExponentialSmoothing(
    df['return'],
    trend='add',
    seasonal=None
).fit(optimized=True)
print("\nPreview Holt-Winters Seasonal Model Fitted Values:")
print(seasonal_model.fittedvalues.head())
#plot seasonal model
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df["return"], label='Data Aktual', color='blue')
ax.plot(seasonal_model.fittedvalues.index, seasonal_model.fittedvalues, label='Holt-Winters Fitted Values', color='orange')
ax.set_title('Holt-Winters Exponential Smoothing Fitted Values')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Return')
ax.legend()
ax.grid(True)
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
# Tambahkan interaktivitas hover
cursor = mplcursors.cursor(ax, hover=True)
@cursor.connect("add")
def on_hover(sel):
    xdata = sel.target[0]
    x_datetime = mdates.num2date(xdata).replace(tzinfo=None)
    nearest_index = (abs(df.index - x_datetime)).argmin()
    nearest_date = df.index[nearest_index]
    y_value = df['return'].iloc[nearest_index]
    sel.annotation.set(text=f'Tanggal: {x_datetime.date()}\n Return: {y_value:.4f}')
plt.show()

# double holt winters exponential smoothing model
double_model = ExponentialSmoothing(
    df['return'],
    trend='add',
    seasonal=None
).fit(optimized=True)
print("\nPreview Holt-Winters Double Model Fitted Values:")
print(double_model.fittedvalues.head())
#plot double model
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df["return"], label='Data Aktual', color='blue')
ax.plot(double_model.fittedvalues.index, double_model.fittedvalues, label='Holt-Winters Double Fitted Values', color='orange')
ax.set_title('Holt-Winters Exponential Smoothing Double Fitted Values')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Return')
ax.legend()
ax.grid(True)
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
# Tambahkan interaktivitas hover
cursor = mplcursors.cursor(ax, hover=True)
@cursor.connect("add")
def on_hover(sel):
    xdata = sel.target[0]
    x_datetime = mdates.num2date(xdata).replace(tzinfo=None)
    nearest_index = (abs(df.index - x_datetime)).argmin()
    nearest_date = df.index[nearest_index]
    y_value = df['return'].iloc[nearest_index]
    sel.annotation.set(text=f'Tanggal: {x_datetime.date()}\n Return: {y_value:.4f}')
plt.show()


## 3. Splitting Data
df_filled = df.asfreq('B')
df_filled['return'].fillna(method='ffill', inplace=True)

# Split ulang karena index berubah
train_size = int(len(df_filled) * 0.8)
train, test = df_filled.iloc[:train_size], df_filled.iloc[train_size:]

## 4. Forecasting dengan Holt-Winters Exponential Smoothing
# Fit model Holt-Winters
hw_model = ExponentialSmoothing(
    train['return'],
    trend='add',
    seasonal=None
).fit(optimized=True)
# Forecasting
forecast_hw = pd.Series(hw_model.forecast(len(test)), index=test.index)

print("\nPreview Forecast Holt-Winters:")
print(forecast_hw.head())

# Plot Holt–Winters Forecast
fitted_values = hw_model.fittedvalues
full_forecast_hw_series= pd.concat([fitted_values, forecast_hw])
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df["return"], label='Data Aktual', color='blue')
ax.plot(full_forecast_hw_series.index, full_forecast_hw_series, label='Holt-Winters Forecast', color='orange')
ax.axvline(x=train.index[-1], color='gray', linestyle=':', label='Batas Train-Test')
ax.set_title('Holt-Winters Exponential Smoothing Forecast')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Return')
ax.legend()
ax.grid(True)
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
# Tambahkan interaktivitas hover
cursor = mplcursors.cursor(ax, hover=True)
@cursor.connect("add")
def on_hover(sel):
    xdata = sel.target[0]
    x_datetime = mdates.num2date(xdata).replace(tzinfo=None)
    nearest_index = (abs(df.index - x_datetime)).argmin()
    nearest_date = df.index[nearest_index]
    y_value = df['return'].iloc[nearest_index]
    sel.annotation.set(text=f'Tanggal: {x_datetime.date()}\n Return: {y_value:.4f}')
plt.show()

# Evaluasi model Holt-Winters
# Rentang parameter alpha dan beta
alphas = np.round(np.arange(0.05, 1.05, 0.05), 2)
betas  = np.round(np.arange(0.05, 1.05, 0.05), 2)
results = []
failed = []

for alpha in alphas:
    for beta in betas:
        try:
            model = ExponentialSmoothing(
                train['return'],
                trend='add',
                seasonal=None
            ).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)

            forecast = model.forecast(len(test))
            mape = mean_absolute_percentage_error(test['return'], forecast) * 100
            mse = mean_squared_error(test['return'], forecast)
            rmse = np.sqrt(mse)
            results.append([alpha, beta, mape, mse, rmse])
        except Exception as e:
            failed.append((alpha, beta))
            continue

forecast_results = pd.DataFrame(results, columns=['Alpha', 'Beta', 'MAPE', 'MSE', 'RMSE'])
print("\n=== Hasil Evaluasi Holt-Winters dengan Berbagai Kombinasi Alpha dan Beta ===")
print(forecast_results)

# Cari kombinasi terbaik berdasarkan MAPE
best_row = forecast_results.loc[forecast_results['MAPE'].idxmin()]
best_alpha = best_row['Alpha']
best_beta  = best_row['Beta']
print(f"\nKombinasi terbaik - Alpha: {best_alpha}, Beta: {best_beta}")
# Evaluasi dengan kombinasi terbaik
best_model = ExponentialSmoothing(
    train['return'],
    trend='add',
    seasonal=None
).fit(smoothing_level=best_alpha, smoothing_trend=best_beta, optimized=False)
best_forecast = best_model.forecast(len(test))
mape_best = mean_absolute_percentage_error(test['return'], best_forecast)
mse_best  = mean_squared_error(test['return'], best_forecast)
rmse_best = root_mean_squared_error(test['return'], best_forecast)

print(f"\nEvaluasi Holt-Winters terbaik:")
print(f"MAPE : {mape_best:.4f}")
print(f"MSE  : {mse_best:.6f}")
print(f"RMSE : {rmse_best:.4f}")
print(f"Gagal kombinasi: {len(failed)} dari total {len(alphas)*len(betas)} kombinasi.")


## 5. Forecasting hybrid dengan Holt-Winters dan GARCH
print("\nProses Hybrid Holt-Winters + GARCH")

# Fit Holt-Winters pada data TRAIN
hw_model = ExponentialSmoothing(
    train['return'],
    trend='add',
    seasonal=None
).fit(optimized=True)

# Hitung residual training
residuals = train['return'] - hw_model.fittedvalues
residuals = residuals.fillna(0)

# Ambil Fitted Values Holt-Winters untuk periode TRAIN
fitted_values_hw_hybrid = hw_model.fittedvalues 

# Fit model GARCH(1,1) pada residual Holt-Winters
garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
garch_fitted = garch_model.fit(disp='off')

# Forecast residual dengan horizon sepanjang test set
garch_forecast = garch_fitted.forecast(horizon=len(test))
garch_mean_forecast = garch_forecast.mean.iloc[-1].values[:len(test)]

# Forecast utama Holt-Winters sepanjang test
hw_forecast = pd.Series(hw_model.forecast(len(test)), index=test.index)

# Pastikan panjang sama
min_len = min(len(hw_forecast), len(garch_mean_forecast))
hw_forecast = hw_forecast.iloc[:min_len]
garch_mean_forecast = garch_mean_forecast[:min_len]
test = test.iloc[:min_len]

# Buat hybrid forecast (Out-of-sample)
hybrid_forecast = pd.Series(hw_forecast.values + garch_mean_forecast, index=test.index)

# Evaluasi
mask = (~test['return'].isna()) & (~hybrid_forecast.isna())
y_true = test['return'][mask]
y_pred = hybrid_forecast[mask]

mape_hybrid = mean_absolute_percentage_error(y_true, y_pred)
mse_hybrid  = mean_squared_error(y_true, y_pred)
rmse_hybrid = root_mean_squared_error(y_true, y_pred)

print("\nEvaluasi Hybrid Holt-Winters + GARCH")
print(f"MAPE : {mape_hybrid:.4f}")
print(f"MSE  : {mse_hybrid:.6f}")
print(f"RMSE : {rmse_hybrid:.4f}")

# Plot hasil hybrid forecast
in_sample_hybrid_forecast = fitted_values_hw_hybrid
full_hybrid_forecast_series = pd.concat([in_sample_hybrid_forecast, hybrid_forecast])
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df.index, df["return"], label='Data Aktual', color='blue')
ax.plot(full_hybrid_forecast_series.index, full_hybrid_forecast_series, label='Hybrid Holt-Winters x GARCH', color='red')
ax.axvline(x=train.index[-1], color='gray', linestyle=':', label='Batas Train/Test')
ax.set_title('Hybrid Holt-Winters + GARCH Forecast')
ax.set_xlabel('Tanggal')
ax.set_ylabel('Return')
ax.legend()
ax.grid(True)
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
# Tambahkan interaktivitas hover
cursor = mplcursors.cursor(ax, hover=True)
@cursor.connect("add")
def on_hover(sel):
    xdata = sel.target[0]
    x_datetime = mdates.num2date(xdata).replace(tzinfo=None)
    nearest_index = (abs(df.index - x_datetime)).argmin()
    nearest_date = df.index[nearest_index]
    y_value = df['return'].iloc[nearest_index]
    sel.annotation.set(text=f'Tanggal: {x_datetime.date()}\n Return: {y_value:.4f}')
plt.show()