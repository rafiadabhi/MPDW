import pandas as pd

# 1. Baca CSV
df = pd.read_csv(r"D:\Github\college_course\college_course\Time Series Forecasting Methods\AAPL_STOCK.csv")

# 2. Ubah Date jadi datetime
df['Date'] = pd.to_datetime(df['Date'])

# 3. Rename kolom sesuai template
df = df.rename(columns={
    'Price': 'Close',
    'Vol.': 'Volume',
    'Change %': 'ChangePercent'
})

# 4. Hapus kolom ChangePercent
df = df.drop(columns=['ChangePercent'])

# 5. Jadikan Date sebagai index
df.set_index('Date', inplace=True)

# 6. Ambil data akhir bulan khusus kolom Close (closing price terakhir tiap bulan)
monthly_close = df['Close'].resample('M').last()

# 7. Gabungkan dengan kolom lain (Open, High, Low, Volume) kalau perlu
monthly_ohlcv = df.resample('M').last()

# 8. Simpan hasil
monthly_close.to_csv(r"D:\Github\college_course\college_course\Time Series Forecasting Methods\AAPL_STOCK_monthly_close.csv")
monthly_ohlcv.to_csv(r"D:\Github\college_course\college_course\Time Series Forecasting Methods\AAPL_STOCK_monthly_all.csv")

print("✅ Data bulanan berhasil disimpan (close saja & full OHLCV)")
