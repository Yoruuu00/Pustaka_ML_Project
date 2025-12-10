import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
from sklearn.preprocessing import LabelEncoder

# Matikan warning merah yang mengganggu
warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI
# ============================================================================
OUTPUT_PATH = 'output'
MODEL_DIR = os.path.join(OUTPUT_PATH, 'models')
DATA_FILE = os.path.join(OUTPUT_PATH, 'ML_Dataset_Enriched_Google.csv')

def load_resources():
    print("⏳ Memuat Database & Model AI...", end="\r")
    
    if not os.path.exists(DATA_FILE):
        print("\n❌ Error: File dataset tidak ditemukan.")
        return None, None, None, None

    df = pd.read_csv(DATA_FILE)
    
    # Bersihkan nama kolom
    df.columns = [c.strip().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for c in df.columns]
    
    # Setup Encoder
    cat_col_source = None
    for col in ['Category', 'Category_OLTP', 'category', 'Genre']:
        if col in df.columns:
            cat_col_source = col
            break
            
    if cat_col_source:
        df['Category_Source'] = df[cat_col_source]
    else:
        df['Category_Source'] = 'General'

    if 'google_categories' in df.columns:
        df['Final_Category'] = df['google_categories'].replace('Unknown', np.nan).fillna(df['Category_Source'])
    else:
        df['Final_Category'] = df['Category_Source']
        
    le = LabelEncoder()
    le.fit(df['Final_Category'].astype(str))
    
    defaults = {
        'rating': df['google_rating'].mean() if 'google_rating' in df.columns else 4.0,
        'pages': df['google_page_count'].mean() if 'google_page_count' in df.columns else 250,
        'age': 5, 
        'month': 12 
    }

    # Load Model (Prioritas Optimized)
    model = None
    model_files = [
        'regression_gradient_boosting_optimized.pkl',
        'regression_random_forest_optimized.pkl',
    ]
    
    for fname in model_files:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            # print(f"✅ Model Loaded: {fname}" + " "*20)
            break
            
    if model is None:
        print("\n❌ Error: Model tidak ditemukan! Jalankan ml_training.py dulu.")
        return None, None, None, None

    return df, le, model, defaults

def predict_sales():
    df, le, model, defaults = load_resources()
    if df is None: return

    print("\n" + "="*60)
    print(" 🔮 DEMO PREDIKSI PENJUALAN TOKO BUKU PUSTAKA NUSANTARA")
    print("    Powered by: Machine Learning & Google Books API")
    print("="*60)

    # Cek Skala Harga Dataset (Untuk Konversi Otomatis)
    avg_price_dataset = df['Item_Price'].mean()
    print(f"ℹ️ Info: Rata-rata harga di database: {avg_price_dataset:,.0f} (Mata Uang Asing/INR)")
    print("-" * 60)

    while True:
        print("\nINPUT DATA BUKU BARU (Ketik 'exit' untuk keluar)")
        
        # 1. Input Judul
        title = input("📚 Judul Buku: ").strip()
        if title.lower() == 'exit': break
        
        # Cek Database
        existing_book = df[df['Title'].str.contains(title, case=False, na=False)]
        
        rating_val = defaults['rating']
        pages_val = defaults['pages']
        cat_val = 'General'
        
        if not existing_book.empty:
            data = existing_book.iloc[0]
            print(f"   ✓ Buku ditemukan! ('{data['Title']}')")
            if 'google_rating' in data and data['google_rating'] > 0:
                rating_val = data['google_rating']
            if 'google_page_count' in data and data['google_page_count'] > 0:
                pages_val = data['google_page_count']
            if 'Final_Category' in data:
                cat_val = data['Final_Category']
        else:
            print("   ℹ️ Buku Baru. Menggunakan nilai default.")
        
        # 2. Input Harga
        while True:
            try:
                price_input_raw = input("💰 Harga Jual (Rp): ").replace('.', '').replace(',', '')
                price_rp = float(price_input_raw)
                break
            except ValueError:
                print("   ❌ Masukkan angka valid!")

        # --- LOGIKA KONVERSI MATA UANG ---
        # Jika user input > 10.000 (Rupiah), tapi data latih rata-ratanya < 5.000 (Asing/INR)
        # Maka kita bagi harga input dengan kurs asumsi (misal 1 INR = 200 Perak)
        # Agar model 'mengerti' angkanya.
        
        model_input_price = price_rp
        conversion_factor = 1.0
        
        if price_rp > 10000 and avg_price_dataset < 5000:
            conversion_factor = 200.0 # Asumsi Konversi Kasar Rupiah -> INR
            model_input_price = price_rp / conversion_factor
            print(f"   (Konversi Otomatis ke Skala Model: {model_input_price:.1f})")

        # 3. Encoding Kategori
        try:
            if cat_val in le.classes_:
                cat_encoded = le.transform([cat_val])[0]
            else:
                cat_encoded = le.transform([le.classes_[0]])[0]
        except:
            cat_encoded = 0

        # 4. Siapkan Input DataFrame (Biar Gak Warning Merah)
        feature_names = ['Item_Price', 'Category_Encoded', 'Book_Age', 'Month', 'google_rating_clean', 'google_page_count_clean']
        
        input_data = pd.DataFrame([[
            model_input_price, # Harga yg sudah disesuaikan
            cat_encoded,
            defaults['age'],
            12,
            rating_val,
            pages_val
        ]], columns=feature_names)

        # 5. Prediksi
        print("\n🤖 AI sedang menghitung...", end="\r")
        time.sleep(1)
        
        # Prediksi Log -> Real
        pred_log = model.predict(input_data)[0]
        pred_omzet_model_currency = np.expm1(pred_log)
        
        # Kembalikan Omzet ke Rupiah
        pred_omzet_rp = pred_omzet_model_currency * conversion_factor
        
        # Hitung Quantity
        # Jika prediksinya aneh (lebih kecil dari harga), minimal dianggap laku 1 (jika rating bagus)
        est_qty = pred_omzet_rp / price_rp
        
        if est_qty < 1.0 and rating_val > 4.0:
            est_qty = 1.0 # Bonus untuk buku rating tinggi
            pred_omzet_rp = price_rp

        # 6. Tampilkan Hasil
        print(" "*30)
        print("📊 HASIL PREDIKSI AI:")
        print(f"   Harga Jual      : Rp {price_rp:,.0f}")
        print(f"   Kualitas (Rating): {rating_val:.1f} / 5.0")
        print(f"   ----------------------------------")
        print(f"   📈 POTENSI OMZET : Rp {pred_omzet_rp:,.0f}")
        print(f"   📦 ESTIMASI LAKU : {est_qty:.1f} pcs / bulan")
        
        if est_qty < 1:
            print("\n   ⚠️ Catatan: Prediksi rendah. Mungkin harga terlalu mahal untuk pasar ini?")
        elif est_qty > 10:
            print("\n   🔥 Catatan: Potensi Best Seller!")
            
        print("="*60)

if __name__ == "__main__":
    try:
        predict_sales()
    except KeyboardInterrupt:
        print("\n\n👋 Bye bye!")