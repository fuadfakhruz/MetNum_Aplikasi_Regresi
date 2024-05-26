import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Membaca data dari file yang diunggah
file_path = 'Student_Performance.csv'
data = pd.read_csv(file_path)

# Menampilkan nama kolom untuk memastikan kesesuaian
print(data.columns)
print(data.head())

# Memilih kolom yang relevan
# Pastikan nama kolom sesuai dengan yang ada di file
NL = data['Sample Question Papers Practiced'].values
NT = data['Performance Index'].values

# Model Linear untuk Problem 2
X_NL = NL.reshape(-1, 1)
linear_model_NL = LinearRegression()
linear_model_NL.fit(X_NL, NT)
NT_pred_linear_NL = linear_model_NL.predict(X_NL)

# Plot hasil regresi linear
plt.figure(figsize=(12, 6))
plt.scatter(NL, NT, color='red', label='Data Asli')
plt.plot(NL, NT_pred_linear_NL, color='blue', label='Regresi Linear')
plt.title('Regresi Linear (Jumlah Latihan Soal vs Nilai Ujian)')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.grid(True)
plt.show()

# Menghitung RMS untuk model linear (NL)
rms_linear_NL = np.sqrt(mean_squared_error(NT, NT_pred_linear_NL))
print(f'RMS untuk model linear (NL): {rms_linear_NL}')

# Model Eksponensial untuk Problem 2
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

popt, _ = curve_fit(exp_func, NL, NT)
NT_pred_exp_NL = exp_func(NL, *popt)

# Plot hasil regresi eksponensial
plt.figure(figsize=(12, 6))
plt.scatter(NL, NT, color='red', label='Data Asli')
plt.plot(NL, NT_pred_exp_NL, color='blue', label='Regresi Eksponensial')
plt.title('Regresi Eksponensial (Jumlah Latihan Soal vs Nilai Ujian)')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.grid(True)
plt.show()

# Menghitung RMS untuk model eksponensial (NL)
rms_exp_NL = np.sqrt(mean_squared_error(NT, NT_pred_exp_NL))
print(f'RMS untuk model eksponensial (NL): {rms_exp_NL}')
