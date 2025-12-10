import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI
# ============================================================================

if os.path.exists('/app'):
    BASE_PATH = '/app'
else:
    BASE_PATH = '.'

OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
LOG_PATH = os.path.join(BASE_PATH, 'logs')

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

def log_message(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"[{timestamp}] [{level}] {message}"
    print(log_text)
    
    log_file = os.path.join(LOG_PATH, f"preprocessing_{datetime.now().strftime('%Y%m%d')}.txt")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_text + '\n')

# ============================================================================
# 1. DATA LOADING & INITIAL INSPECTION
# ============================================================================

def load_dataset():
    """Load dataset with automatic path detection"""
    log_message("="*70)
    log_message("STEP 1: LOADING DATASET")
    log_message("="*70)
    
    # Prioritas pencarian file
    POSSIBLE_FILES = [
        os.path.join(OUTPUT_PATH, 'ML_Dataset_Enriched.csv'),     # 1. Hasil API
        'ML_Dataset_Enriched.csv',                                 # 2. Hasil API di root
        'data/Merged_OLTP_Books_Cleaned0.csv',                     # 3. Raw Data di folder data
        'Merged_OLTP_Books_Cleaned0.csv'                           # 4. Raw Data di root
    ]
    
    df = None
    loaded_path = ""
    
    for path in POSSIBLE_FILES:
        if os.path.exists(path):
            df = pd.read_csv(path)
            loaded_path = path
            break
            
    if df is None:
        log_message("❌ No dataset found! Please put CSV file in folder.", "ERROR")
        return None
    
    log_message(f"✓ Loaded: {loaded_path}")
    log_message(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # --- AUTO FIX: Add Dummy API Columns if missing ---
    # Ini agar feature engineering tidak error kalau pakai Raw Data
    if 'api_first_publish_year' not in df.columns:
        log_message("⚠ API columns missing. Injecting default values for pipeline compatibility...")
        df['api_first_publish_year'] = 2020
        df['api_number_of_pages'] = 200
        df['api_edition_count'] = 1
        df['api_found'] = False
    
    return df

def initial_inspection(df):
    """Inspect dataset structure"""
    log_message("\n" + "="*70)
    log_message("INITIAL DATA INSPECTION")
    log_message("="*70)
    
    log_message(f"\n1. COLUMNS ({len(df.columns)}):")
    # Tampilkan 5 kolom pertama saja biar log gak kepanjangan
    for i, col in enumerate(df.columns[:5], 1):
        log_message(f"   {i}. {col} ({df[col].dtype})")
    log_message("   ... (and more)")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        log_message(f"\n2. MISSING VALUES FOUND: {missing.sum()}")
    else:
        log_message("   ✓ No missing values!")

# ============================================================================
# 2. HANDLE MISSING VALUES
# ============================================================================

def handle_missing_values(df):
    """Handle missing values based on column type"""
    log_message("\n" + "="*70)
    log_message("STEP 2: HANDLING MISSING VALUES")
    log_message("="*70)
    
    df_clean = df.copy()
    
    # 1. Bersihkan Format Uang (Rp) Dulu (PENTING)
    if 'Item_Price' in df_clean.columns and df_clean['Item_Price'].dtype == 'object':
        def clean_currency(x):
            if isinstance(x, str):
                return float(x.replace('Rp', '').replace('.', '').replace(',', '').strip())
            return float(x)
        
        df_clean['Item_Price'] = df_clean['Item_Price'].apply(clean_currency)
        log_message("✓ Cleaned 'Item_Price' format")

    # 2. Hitung Total & Profit (Jika kosong)
    if 'Total_Amount' not in df_clean.columns:
         if 'Quantity' in df_clean.columns and 'Item_Price' in df_clean.columns:
             df_clean['Total_Amount'] = df_clean['Quantity'] * df_clean['Item_Price']
             
    if 'Profit' not in df_clean.columns and 'Total_Amount' in df_clean.columns:
        df_clean['Profit'] = df_clean['Total_Amount'] * 0.3

    # 3. Isi Missing Values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Fill numeric with median/0
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            # Jika median NaN (kolom kosong total), isi 0
            if pd.isna(median_val): median_val = 0
            df_clean[col].fillna(median_val, inplace=True)
            log_message(f"✓ Filled numeric {col} with: {median_val}")
    
    # Fill categorical with mode or 'Unknown'
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            if not df_clean[col].mode().empty:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
            else:
                df_clean[col].fillna('Unknown', inplace=True)
            log_message(f"✓ Filled text {col}")
    
    return df_clean

# ============================================================================
# 3. HANDLE OUTLIERS
# ============================================================================

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def handle_outliers(df):
    log_message("\n" + "="*70)
    log_message("STEP 3: HANDLING OUTLIERS")
    log_message("="*70)
    
    df_clean = df.copy()
    numeric_cols = ['Quantity', 'Item_Price', 'Total_Amount', 'Profit']
    
    for col in numeric_cols:
        if col in df_clean.columns:
            outliers, lower, upper = detect_outliers(df_clean, col)
            if len(outliers) > 0:
                df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                log_message(f"✓ Capped {len(outliers)} outliers in '{col}'")
            
    return df_clean

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

def create_features(df):
    log_message("\n" + "="*70)
    log_message("STEP 4: FEATURE ENGINEERING")
    log_message("="*70)
    
    df_features = df.copy()
    
    # 1. Date Features
    if 'Purchase_Date' in df_features.columns:
        df_features['Purchase_Date'] = pd.to_datetime(df_features['Purchase_Date'], errors='coerce')
        df_features['Purchase_Date'] = df_features['Purchase_Date'].fillna(pd.Timestamp.now())
        
        df_features['Year'] = df_features['Purchase_Date'].dt.year
        df_features['Month'] = df_features['Purchase_Date'].dt.month
        df_features['Day'] = df_features['Purchase_Date'].dt.day
        df_features['DayOfWeek'] = df_features['Purchase_Date'].dt.dayofweek
        df_features['Quarter'] = df_features['Purchase_Date'].dt.quarter
        df_features['IsWeekend'] = (df_features['DayOfWeek'] >= 5).astype(int)
        log_message("✓ Created date features")
    
    # 2. Profit Features
    if 'Profit' in df_features.columns and 'Total_Amount' in df_features.columns:
        # Hindari pembagian dengan nol
        df_features['Profit_Margin'] = df_features.apply(
            lambda x: (x['Profit'] / x['Total_Amount'] * 100) if x['Total_Amount'] > 0 else 0, axis=1
        )
        
        # Buat kategori profit (Low/Medium/High)
        try:
            df_features['Profit_Category'] = pd.qcut(
                df_features['Profit'].rank(method='first'), 
                q=3, 
                labels=['Low', 'Medium', 'High']
            )
        except:
             # Fallback jika qcut gagal (data terlalu sedikit/sama)
             df_features['Profit_Category'] = 'Medium'

        log_message("✓ Created Profit metrics")
    
    # 3. Book Age
    if 'api_first_publish_year' in df_features.columns:
        df_features['Book_Age'] = 2025 - df_features['api_first_publish_year']
        df_features['Book_Age'] = df_features['Book_Age'].apply(lambda x: max(x, 0))
        log_message("✓ Created Book_Age")
    
    # 4. Has API Data Flag
    if 'api_found' in df_features.columns:
        df_features['Has_API_Data'] = df_features['api_found'].fillna(False).astype(int)
    else:
        df_features['Has_API_Data'] = 0

    return df_features

# ============================================================================
# 5. ENCODING CATEGORICAL VARIABLES
# ============================================================================

def encode_categorical(df):
    log_message("\n" + "="*70)
    log_message("STEP 5: ENCODING CATEGORICAL VARIABLES")
    log_message("="*70)
    
    df_encoded = df.copy()
    categorical_cols = ['Category'] # Fokus kategori penting
    
    le_dict = {}
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[f'{col}_Encoded'] = le.fit_transform(df_encoded[col].astype(str))
            le_dict[col] = le
            log_message(f"✓ Encoded '{col}'")
    
    # Save encoders
    with open(os.path.join(OUTPUT_PATH, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(le_dict, f)
    
    return df_encoded, le_dict

# ============================================================================
# 6. FEATURE SCALING
# ============================================================================

def scale_features(df, method='standard'):
    log_message("\n" + "="*70)
    log_message("STEP 6: FEATURE SCALING")
    log_message("="*70)
    
    df_scaled = df.copy()
    scale_cols = ['Quantity', 'Item_Price', 'Total_Amount', 'Profit', 'Book_Age']
    scale_cols = [c for c in scale_cols if c in df_scaled.columns]
    
    scaler = StandardScaler()
    
    if scale_cols:
        scaled_data = scaler.fit_transform(df_scaled[scale_cols])
        for i, col in enumerate(scale_cols):
            df_scaled[f'{col}_Scaled'] = scaled_data[:, i]
            
    # Save scaler
    with open(os.path.join(OUTPUT_PATH, 'standard_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    log_message(f"✓ Scaled {len(scale_cols)} columns")
    return df_scaled, scaler

# ============================================================================
# 7. TRAIN-TEST SPLIT & SAVE
# ============================================================================

def split_and_save(df):
    log_message("\n" + "="*70)
    log_message("STEP 7: SPLIT & SAVE")
    log_message("="*70)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
    
    full_path = os.path.join(OUTPUT_PATH, 'ML_Dataset_Processed_Full.csv')
    train_path = os.path.join(OUTPUT_PATH, 'ML_Dataset_Train.csv')
    test_path = os.path.join(OUTPUT_PATH, 'ML_Dataset_Test.csv')
    
    df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    log_message(f"✓ Files saved in 'output/' folder.")
    log_message(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # DATA QUALITY SCORE
    completeness = (1 - df.isnull().sum().sum() / df.size) * 100
    log_message(f"\n✨ FINAL DATA QUALITY SCORE (COMPLETENESS): {completeness:.2f}%")
    
    return train_df, test_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    log_message("\n" + "="*70)
    log_message(" PREPROCESSING PIPELINE (FINAL)")
    log_message("="*70)
    
    df = load_dataset()
    if df is None: return
    
    initial_inspection(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = create_features(df)
    df, _ = encode_categorical(df)
    df, _ = scale_features(df)
    
    split_and_save(df)
    log_message("\n✓ DONE!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_message(f"\n❌ ERROR: {e}", "ERROR")
        import traceback
        traceback.print_exc()