import pandas as pd
from sqlalchemy import create_engine
import os

# Settingan Laragon (Default)
DB_USER = 'root'
DB_PASS = '' 
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'pustaka_dw'

CSV_FILE = 'output/ML_Dataset_Processed_Full.csv'

def run_etl():
    print("🚀 MEMULAI PROSES IMPORT KE LARAGON...")

    # 1. Cek File
    if not os.path.exists(CSV_FILE):
        print("❌ File CSV tidak ketemu!")
        return
    
    df = pd.read_csv(CSV_FILE)
    print(f"📖 Membaca {len(df)} baris data...")

    # 2. Koneksi Mesin
    # Format: mysql+mysqlconnector://user:pass@host:port/dbname
    str_koneksi = f'mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    
    try:
        engine = create_engine(str_koneksi)
        
        # 3. MASUKKAN DATA (Otomatis Buat Tabel!)
        print("💾 Sedang membuat tabel 'fact_sales' dan memasukkan data...")
        df.to_sql('fact_sales', con=engine, if_exists='replace', index=False)
        
        print("✅ SUKSES! Cek HeidiSQL sekarang.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Pastikan Laragon sudah START dan database 'pustaka_dw' sudah dibuat.")

if __name__ == "__main__":
    run_etl()