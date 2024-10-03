import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Judul aplikasi
st.title("Analisis Kualitas Udara di Shunyi")

# Membaca dataset
shunyi_data = pd.read_csv(r"C:\Users\ASUS\Documents\College\Bangkit\Course\PRSA_Data_20130301-20170228\PRSA_Data_Shunyi_20130301-20170228.csv")

st.subheader("Data Awal:")
st.write(shunyi_data.head())

# Informasi Dasar Dataset
st.subheader("Informasi Dasar Dataset:")
st.write(shunyi_data.info())

# Mengecek missing values
st.subheader("Missing Values:")
st.write(shunyi_data.isnull().sum())

# Melihat statistik deskriptif
st.subheader("Statistik Deskriptif:")
st.write(shunyi_data.describe())

# Mengecek apakah ada data yang duplikat
st.subheader("Data Duplikat:")
st.write(shunyi_data.duplicated().sum())

# Menghapus baris yang memiliki missing values
shunyi_cleaned = shunyi_data.dropna()

# Mengonversi kolom 'year', 'month', 'day', 'hour' menjadi datetime
if 'year' in shunyi_cleaned.columns and 'month' in shunyi_cleaned.columns and 'day' in shunyi_cleaned.columns and 'hour' in shunyi_cleaned.columns:
    shunyi_cleaned['date'] = pd.to_datetime(shunyi_cleaned[['year', 'month', 'day', 'hour']])
else:
    shunyi_cleaned['date'] = pd.to_datetime(shunyi_cleaned['date'])

# Menghapus kolom yang tidak diperlukan
shunyi_cleaned = shunyi_cleaned.drop(columns=['No'], errors='ignore')

# Menampilkan data setelah dibersihkan
st.subheader("Data Setelah Dibersihkan:")
st.write(shunyi_cleaned.info())

# Memilih beberapa kolom yang relevan untuk analisis multivariat
cols = ['PM2.5', 'TEMP', 'DEWP', 'WSPM']

# Heatmap untuk melihat korelasi antar variabel
st.subheader("Heatmap Korelasi antara Variabel Lingkungan dan Polusi Udara")
fig, ax = plt.subplots(figsize=(10, 6))
corr = shunyi_cleaned[cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
plt.title('Korelasi antara Variabel Lingkungan dan Polusi Udara')
st.pyplot(fig)  # Mengirimkan objek fig

# Diagram Batang untuk Rata-rata PM2.5 berdasarkan Kategori Suhu
shunyi_cleaned['TEMP_category'] = pd.cut(shunyi_cleaned['TEMP'], bins=[-10, 0, 10, 20, 30, 40], labels=['<0°C', '0-10°C', '10-20°C', '20-30°C', '>30°C'])
avg_pm25_temp = shunyi_cleaned.groupby('TEMP_category')['PM2.5'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='TEMP_category', y='PM2.5', data=avg_pm25_temp, palette='Set2', ax=ax)
plt.title('Rata-rata PM2.5 berdasarkan Kategori Suhu')
plt.xlabel('Kategori Suhu')
plt.ylabel('Rata-rata Konsentrasi PM2.5')
plt.xticks(rotation=45)
plt.grid(axis='y')
st.pyplot(fig)  # Mengirimkan objek fig

# Memvisualisasikan hubungan antara PM2.5 dan faktor lingkungan
plt.figure(figsize=(18, 5))

# Subplot 1: Hubungan antara PM2.5 dan Suhu (TEMP)
plt.subplot(1, 3, 1)
sns.lineplot(x='date', y='PM2.5', data=shunyi_cleaned, label='PM2.5', color='blue')
sns.lineplot(x='date', y='TEMP', data=shunyi_cleaned, label='Suhu (TEMP)', color='orange')
plt.title('Hubungan PM2.5 dan Suhu')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi PM2.5 dan Suhu')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Subplot 2: Hubungan antara PM2.5 dan Kelembaban (DEWP)
plt.subplot(1, 3, 2)
sns.lineplot(x='date', y='PM2.5', data=shunyi_cleaned, label='PM2.5', color='blue')
sns.lineplot(x='date', y='DEWP', data=shunyi_cleaned, label='Kelembaban (DEWP)', color='orange')
plt.title('Hubungan PM2.5 dan Kelembaban')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi PM2.5 dan Kelembaban')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Subplot 3: Hubungan antara PM2.5 dan Kecepatan Angin (WSPM)
plt.subplot(1, 3, 3)
sns.lineplot(x='date', y='PM2.5', data=shunyi_cleaned, label='PM2.5', color='blue')
sns.lineplot(x='date', y='WSPM', data=shunyi_cleaned, label='Kecepatan Angin (WSPM)', color='orange')
plt.title('Hubungan PM2.5 dan Kecepatan Angin')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi PM2.5 dan Kecepatan Angin')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

plt.tight_layout()

# Menampilkan plot di Streamlit
st.pyplot(plt)

# Memvisualisasikan hubungan antara PM10 dan faktor lingkungan dengan line chart
plt.figure(figsize=(18, 5))

# Subplot 1: Hubungan antara PM10 dan Suhu (TEMP)
plt.subplot(1, 3, 1)
sns.lineplot(x='date', y='PM10', data=shunyi_cleaned, label='PM10', color='blue')
sns.lineplot(x='date', y='TEMP', data=shunyi_cleaned, label='Suhu (TEMP)', color='orange')
plt.title('Hubungan PM10 dan Suhu')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi PM10 dan Suhu')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Subplot 2: Hubungan antara PM10 dan Kelembaban (DEWP)
plt.subplot(1, 3, 2)
sns.lineplot(x='date', y='PM10', data=shunyi_cleaned, label='PM10', color='blue')
sns.lineplot(x='date', y='DEWP', data=shunyi_cleaned, label='Kelembaban (DEWP)', color='orange')
plt.title('Hubungan PM10 dan Kelembaban')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi PM10 dan Kelembaban')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Subplot 3: Hubungan antara PM10 dan Kecepatan Angin (WSPM)
plt.subplot(1, 3, 3)
sns.lineplot(x='date', y='PM10', data=shunyi_cleaned, label='PM10', color='blue')
sns.lineplot(x='date', y='WSPM', data=shunyi_cleaned, label='Kecepatan Angin (WSPM)', color='orange')
plt.title('Hubungan PM10 dan Kecepatan Angin')
plt.xlabel('Tanggal')
plt.ylabel('Konsentrasi PM10 dan Kecepatan Angin')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

plt.tight_layout()

# Menampilkan plot di Streamlit
st.pyplot(plt)

# Mengelompokkan data berdasarkan bulan, kemudian menghitung rata-rata PM2.5 dan PM10
monthly_avg = shunyi_cleaned.groupby(['month'])[['PM2.5', 'PM10']].mean().reset_index()
monthly_avg['month'] = monthly_avg['month'].astype(str)

# Visualisasi Pola Bulanan dari PM2.5 dan PM10
st.subheader("Pola Bulanan dari PM2.5 dan PM10")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=monthly_avg, x='month', y='PM2.5', marker='o', label='PM2.5', ax=ax)
sns.lineplot(data=monthly_avg, x='month', y='PM10', marker='o', color='orange', label='PM10', ax=ax)
plt.title('Pola Bulanan dari PM2.5 dan PM10')
plt.xlabel('Bulan')
plt.ylabel('Rata-rata Konsentrasi')
plt.xticks(rotation=45)
plt.legend(title='Konsentrasi')
plt.grid(True)
st.pyplot(fig)  # Mengirimkan objek fig

# Visualisasi Pola Tahunan dan Musiman dari PM2.5 dan PM10
monthly_avg = shunyi_cleaned.groupby(['year', 'month'])[['PM2.5', 'PM10']].mean().reset_index()
monthly_avg['date'] = pd.to_datetime(monthly_avg[['year', 'month']].assign(day=1))

st.subheader("Pola Tahunan dan Musiman dari PM2.5 dan PM10")
fig, axs = plt.subplots(2, 1, figsize=(14, 7))

# Plot PM2.5
sns.lineplot(data=monthly_avg, x='date', y='PM2.5', marker='o', ax=axs[0])
axs[0].set_title('Pola Tahunan dan Musiman dari PM2.5')
axs[0].set_xlabel('Tanggal')
axs[0].set_ylabel('Rata-rata PM2.5')
axs[0].tick_params(axis='x', rotation=45)

# Plot PM10
sns.lineplot(data=monthly_avg, x='date', y='PM10', marker='o', color='orange', ax=axs[1])
axs[1].set_title('Pola Tahunan dan Musiman dari PM10')
axs[1].set_xlabel('Tanggal')
axs[1].set_ylabel('Rata-rata PM10')
axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig)  # Mengirimkan objek fig