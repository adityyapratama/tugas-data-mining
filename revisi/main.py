import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc




def load_and_clean_data(file_path):
    """
    1. Memuat dataset dan menghapus missing values (?).
    """
    df = pd.read_csv(file_path)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.apply(pd.to_numeric)
    return df

def discretize_data(df):
    """
    2. Mengubah atribut numerik menjadi kategori (0 dan 1) sesuai jurnal.
    """
    df_disc = df.copy()
    df_disc['age'] = df_disc['age'].apply(lambda x: 0 if x < 54 else 1)
    df_disc['trestbps'] = df_disc['trestbps'].apply(lambda x: 0 if x < 148 else 1)
    df_disc['chol'] = df_disc['chol'].apply(lambda x: 0 if x < 346 else 1)
    df_disc['thalach'] = df_disc['thalach'].apply(lambda x: 0 if x < 136 else 1)
    df_disc['oldpeak'] = df_disc['oldpeak'].apply(lambda x: 0 if x < 3.2 else 1)
    return df_disc


# FUNGSI-FUNGSI VISUALISASI
def plot_distribusi_target(df):
    """
    Membuat Diagram Batang (Bar Chart) untuk melihat proporsi 
    pasien Sehat (0) dan Sakit (1).
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='Set2')
    plt.title('Distribusi Pasien: Sehat (0) vs Sakit (1)')
    plt.xlabel('Kondisi Pasien (Target)')
    plt.ylabel('Jumlah')
    plt.xticks([0, 1], ['Sehat (0)', 'Sakit (1)'])
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """
    Membuat visualisasi Confusion Matrix menggunakan Heatmap.
    Menampilkan nilai TP, TN, FP, dan FN.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Prediksi Sehat', 'Prediksi Sakit'],
                yticklabels=['Aktual Sehat', 'Aktual Sakit'])
    plt.title('Confusion Matrix - Naive Bayes\n')
    plt.ylabel('Kondisi Aktual (Sebenarnya)')
    plt.xlabel('Prediksi Model')
    plt.show()


# ALUR UTAMA PROGRAM
def main():
    file_path = 'heart.csv'
    
    print("--- Memulai Proses Data Mining ---")
    
    # 1 & 2: Preprocessing
    df_clean = load_and_clean_data(file_path)
    df_ready = discretize_data(df_clean)
    
    # Memisahkan Fitur (X) dan Target (y)
    X = df_ready.drop('target', axis=1)
    y = df_ready['target']
    
    # Visualisasi 1: Distribusi Target
    print("\n-> Menampilkan Grafik Distribusi Pasien...")
    plot_distribusi_target(df_ready)
    
    # 3. Model & Evaluasi
    model = CategoricalNB()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    print("-> Sedang Melatih Model Naive Bayes dengan 10-Fold CV...")
    
    # Menghitung akurasi rata-rata
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    akurasi = scores.mean() * 100
    
    # Untuk mendapatkan visualisasi, kita prediksi seluruh dataset dengan metode cross-validation
    y_pred = cross_val_predict(model, X, y, cv=kf)
    

    print(f"\nSelesai! Akurasi Rata-rata Model: {akurasi:.2f}%")
    
    # Visualisasi 2: Confusion Matrix
    print("-> Menampilkan Confusion Matrix...")
    plot_confusion_matrix(y, y_pred)
    

if __name__ == "__main__":
    main()