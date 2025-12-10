import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Matikan warning agar terminal bersih
warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI
# ============================================================================
OUTPUT_PATH = 'output'
MODEL_PATH = os.path.join(OUTPUT_PATH, 'models')
os.makedirs(MODEL_PATH, exist_ok=True)

DATA_FILE = os.path.join(OUTPUT_PATH, 'ML_Dataset_Enriched_Google.csv')

def auto_process_data(df):
    print("   🔧 Feature Engineering & Log Transformation...")
    
    # 1. Bersihkan Nama Kolom
    df.columns = [c.strip().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for c in df.columns]
    
    # 2. Deteksi Kolom Kategori (PERBAIKAN DI SINI)
    # Kita cari mana kolom kategori yang tersedia di CSV
    cat_col_source = None
    possible_names = ['Category', 'Category_OLTP', 'category', 'Genre']
    
    for col in possible_names:
        if col in df.columns:
            cat_col_source = col
            break
            
    # Jika tidak ketemu, buat dummy
    if cat_col_source is None:
        print("      ⚠️ Warning: Kolom Kategori tidak ditemukan. Menggunakan default 'General'.")
        df['Category_Source'] = 'General'
    else:
        df['Category_Source'] = df[cat_col_source]

    # 3. Gabungkan dengan Kategori Google
    if 'google_categories' in df.columns:
        # Prioritas: Google -> Data Toko
        df['Final_Category'] = df['google_categories'].replace('Unknown', np.nan).fillna(df['Category_Source'])
    else:
        df['Final_Category'] = df['Category_Source']
    
    # Encode jadi angka
    le = LabelEncoder()
    df['Category_Encoded'] = le.fit_transform(df['Final_Category'].astype(str))

    # 4. Imputasi Data Google
    if 'google_rating' in df.columns:
        mean_rating = df[df['google_rating'] > 0]['google_rating'].mean()
        if pd.isna(mean_rating): mean_rating = 4.0
        df['google_rating_clean'] = df['google_rating'].replace(0, mean_rating)
    else:
        df['google_rating_clean'] = 4.0

    if 'google_page_count' in df.columns:
        mean_pages = df[df['google_page_count'] > 0]['google_page_count'].mean()
        if pd.isna(mean_pages): mean_pages = 250
        df['google_page_count_clean'] = df['google_page_count'].replace(0, mean_pages)
    else:
        df['google_page_count_clean'] = 250

    # 5. Month
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'purchase' in c.lower()]
    if date_cols:
        tgl_col = date_cols[0]
        df[tgl_col] = pd.to_datetime(df[tgl_col], errors='coerce')
        df['Month'] = df[tgl_col].dt.month.fillna(6).astype(int)
    else:
        df['Month'] = 6
            
    # 6. Book Age
    if 'Book_Age' not in df.columns:
        df['Book_Age'] = 5

    return df

def train_models():
    print("\n" + "="*70)
    print(" 🚀 ML TRAINING: LOG-TRANSFORMED OPTIMIZATION")
    print(" Strategi: Menggunakan Skala Logaritma untuk menstabilkan prediksi")
    print("="*70)

    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: File {DATA_FILE} belum ada! Jalankan get_google_data.py dulu.")
        return

    try:
        df = pd.read_csv(DATA_FILE)
        print(f"✓ Data loaded: {len(df)} rows")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    df = auto_process_data(df)

    # 2. SETTING FITUR
    features = [
        'Item_Price',              # Ekonomi
        'Category_Encoded',        # Genre
        'Book_Age',                # Tren
        'Month',                   # Musim
        'google_rating_clean',     # Kualitas
        'google_page_count_clean'  # Spesifikasi
    ]
    
    target_col = 'Total_Amount' # Target Asli (Rupiah)

    # Pastikan semua fitur ada
    features = [f for f in features if f in df.columns]
    print(f"\nFeatures (X): {features}")
    
    # Bersihkan NaN
    df = df.dropna(subset=features + [target_col])
    
    X = df[features]
    y_original = df[target_col]

    # --- TEKNIK OPTIMASI: LOG TRANSFORMATION ---
    y_log = np.log1p(y_original) 
    
    print("✓ Target transformed using Log1p (Normalizing distribution)")

    # Split Data
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

    # 3. Training
    models = {
        "Random Forest (Optimized)": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
        "Gradient Boosting (Optimized)": GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    }

    results = {}
    best_model_name = ""
    best_score = -999

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # A. CROSS VALIDATION
        cv_scores = cross_val_score(model, X_train, y_train_log, cv=5, scoring='r2')
        print(f"   📊 Cross-Validation Scores (5-Fold): {cv_scores}")
        print(f"   📊 Rata-rata Stabilitas: {cv_scores.mean():.4f}")

        # B. Final Training
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
        
        # Kembalikan Log ke Rupiah Asli
        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test_log)
        
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        r2 = r2_score(y_test_real, y_pred_real)
        
        print(f"   RMSE Real: {rmse:,.2f}")
        print(f"   R2 Score Real: {r2:.4f}")
        
        # Simpan Model
        filename = f"regression_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
        pickle.dump(model, open(os.path.join(MODEL_PATH, filename), 'wb'))
        
        results[name] = {"RMSE": rmse, "R2": r2}
        
        if r2 > best_score:
            best_score = r2
            best_model_name = name

    # 4. Kesimpulan
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   Akurasi Akhir (R2): {best_score:.4f}")
    print("="*70)
    
    with open(os.path.join(OUTPUT_PATH, 'ml_models_summary.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    train_models()