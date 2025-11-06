import pandas as pd
import os

# Tentukan periode penuh
start_date = '2014-01-01'
end_date = '2023-12-31'

# 1. Buat deret tanggal dengan frekuensi mingguan (berakhir di hari Minggu: W-SUN)
# pd.to_datetime(start_date) memastikan kita bekerja dengan objek tanggal yang benar
weekly_index = pd.date_range(
    start=pd.to_datetime(start_date), 
    end=pd.to_datetime(end_date), 
    freq='W-SUN'
)

# 2. Buat Pandas DataFrame dari deret tanggal tersebut
# Kita buat satu kolom bernama 'Tanggal_Akhir_Minggu'
df_tanggal_weekly = pd.DataFrame(
    {'Tanggal_Akhir_Minggu': weekly_index}
)

# 3. Simpan DataFrame ke file CSV
nama_file_csv = 'tanggal_weekly_2014_2023.csv'
df_tanggal_weekly.to_csv(nama_file_csv, index=False)

print(f"✅ File CSV berhasil dibuat!")
print(f"Nama file: {nama_file_csv}")
print(f"Data ditampilkan 5 baris pertama:")
print(df_tanggal_weekly.head())
print(f"\nTotal titik data mingguan: {len(df_tanggal_weekly)}")