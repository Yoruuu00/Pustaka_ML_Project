import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# 1. KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="Pustaka Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #c8e6c9;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 2. FUNGSI LOAD DATA & MODEL
# ============================================================================
@st.cache_resource
def load_resources():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_PATH = os.path.join(BASE_DIR, 'output')
    MODEL_DIR = os.path.join(OUTPUT_PATH, 'models')
    DATA_FILE = os.path.join(OUTPUT_PATH, 'ML_Dataset_Enriched_Google.csv')

    if not os.path.exists(DATA_FILE):
        return None, None, None, None

    try:
        df = pd.read_csv(DATA_FILE)
        df.columns = [c.strip().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for c in df.columns]
        
        # Setup Encoder & Kategori
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
        df['Final_Category'] = df['Final_Category'].astype(str)
        le.fit(df['Final_Category'])
        
        # --- PERBAIKAN: HITUNG RATA-RATA PER KATEGORI ---
        # Kita buat kamus harga: {'Fiction': 300, 'Science': 500, ...}
        cat_avg_prices = df.groupby('Final_Category')['Item_Price'].mean().to_dict()
        
        # Rata-rata Global (Cadangan jika kategori baru)
        global_avg = df['Item_Price'].mean() if 'Item_Price' in df.columns else 327

        defaults = {
            'rating': df['google_rating'].mean() if 'google_rating' in df.columns else 4.0,
            'pages': df['google_page_count'].mean() if 'google_page_count' in df.columns else 250,
            'global_avg_price': global_avg,
            'cat_avg_prices': cat_avg_prices  # Simpan data harga per kategori
        }

        model = None
        model_files = [
            'regression_gradient_boosting_optimized.pkl',
            'regression_random_forest_optimized.pkl',
            'regression_gradient_boosting_enriched.pkl'
        ]
        
        for fname in model_files:
            path = os.path.join(MODEL_DIR, fname)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                break
        
        return df, le, model, defaults

    except Exception as e:
        return None, None, None, None

# ============================================================================
# 3. INTERFACE UTAMA
# ============================================================================
def main():
    df, le, model, defaults = load_resources()

    if df is None:
        st.error("❌ Data tidak ditemukan.")
        return

    # SIDEBAR
    st.sidebar.title("🎛️ Panel Kontrol")
    mode = st.sidebar.radio("Pilih Mode:", ["🤖 Simulasi Prediksi", "📊 Dashboard Analitik"])
    st.sidebar.markdown("---")

    # ========================================================================
    # TAB 1: SIMULASI AI
    # ========================================================================
    if mode == "🤖 Simulasi Prediksi":
        st.title("🤖 Simulasi Harga & Penjualan Cerdas")
        
        # --- INPUT ---
        st.sidebar.subheader("📝 Parameter Buku")
        book_titles = sorted(df['Title'].unique().tolist())
        selected_title = st.sidebar.selectbox("🔍 Referensi:", ["(Input Manual)"] + book_titles)
        
        d_rating, d_pages, d_cat = 4.0, 250, "General"
        
        if selected_title != "(Input Manual)":
            row = df[df['Title'] == selected_title].iloc[0]
            if 'google_rating' in row: 
                r = float(row['google_rating'])
                d_rating = r if r >= 1.0 else 1.0
            if 'google_page_count' in row: 
                p = int(row['google_page_count'])
                d_pages = p if p >= 10 else 10
            if 'Final_Category' in row: d_cat = row['Final_Category']
            st.sidebar.success("✅ Data dimuat!")

        # Pilih Kategori
        category = st.sidebar.selectbox("📂 Kategori:", options=le.classes_, index=0 if d_cat not in le.classes_ else list(le.classes_).index(d_cat))
        
        # --- LOGIKA HARGA REFERENSI (DINAMIS PER KATEGORI) ---
        # Ambil rata-rata harga untuk kategori yang dipilih
        cat_avg_inr = defaults['cat_avg_prices'].get(category, defaults['global_avg_price'])
        market_ref_price_rp = cat_avg_inr * 200.0 # Konversi ke Rupiah
        
        # Input Harga dengan default mengikuti pasar kategori
        price = st.sidebar.number_input("💰 Harga Jual (Rp):", min_value=1000, value=int(market_ref_price_rp), step=5000)
        
        # Info kecil di bawah input harga
        st.sidebar.caption(f"ℹ️ Rata-rata pasar kategori **{category}**: Rp {market_ref_price_rp:,.0f}")
        
        rating = st.sidebar.slider("⭐ Rating Google:", 1.0, 5.0, float(d_rating))
        pages = st.sidebar.number_input("📖 Jumlah Halaman:", 10, 5000, d_pages)
        
        if st.sidebar.button("🚀 HITUNG PREDIKSI", type="primary"):
            
            # --- 1. LOGIKA EKONOMI (Manual Adjustment) ---
            # Bandingkan harga user vs Rata-rata KATEGORI (Bukan Global lagi)
            price_ratio = price / market_ref_price_rp
            
            # Faktor Elastisitas: Jika Murah -> Laku Keras
            elasticity_boost = 1.0
            if price_ratio < 1.0:
                elasticity_boost = 1.0 + (1.0 - price_ratio) * 1.8  # Boost lebih agresif
            else:
                elasticity_boost = 1.0 - (price_ratio - 1.0) * 0.6 
                if elasticity_boost < 0.2: elasticity_boost = 0.2

            # --- 2. PREDIKSI MODEL AI ---
            conversion_rate = 200.0
            model_input_price = price / conversion_rate
            
            try: cat_enc = le.transform([category])[0]
            except: cat_enc = 0
            
            input_df = pd.DataFrame([[model_input_price, cat_enc, 5, 12, rating, pages]], 
                                    columns=['Item_Price', 'Category_Encoded', 'Book_Age', 'Month', 'google_rating_clean', 'google_page_count_clean'])
            
            with st.spinner('🤖 AI menghitung probabilitas pasar...'):
                time.sleep(0.5)
                pred_log = model.predict(input_df)[0]
                pred_omzet_inr = np.expm1(pred_log)
                
                # Kembalikan ke Rupiah
                base_omzet_rp = pred_omzet_inr * conversion_rate
                
                # --- 3. GABUNGKAN AI + LOGIKA EKONOMI ---
                final_omzet_rp = base_omzet_rp * elasticity_boost
                
                # Hitung Qty
                est_qty = final_omzet_rp / price
                
                # Logic Bonus Rating Bagus
                if est_qty < 1.0 and rating >= 4.0:
                    est_qty = 1.2 # Minimal laku 1 buku
                    final_omzet_rp = price * est_qty
            
            # --- HASIL ---
            st.markdown("### 🎯 Hasil Analisis Bisnis")
            c1, c2, c3 = st.columns(3)
            c1.metric("🏷️ Harga Input", f"Rp {price:,.0f}")
            c2.metric("📦 Estimasi Laku", f"{est_qty:.1f} Pcs/Bln")
            c3.metric("📈 Potensi Omzet", f"Rp {final_omzet_rp:,.0f}", delta=f"{elasticity_boost:.2f}x Demand" if elasticity_boost > 1 else "Low Demand")
            
            st.markdown("---")
            
            # --- ANALISIS DINAMIS ---
            st.subheader("📊 Insight Pakar:")
            
            # Cek 1: Bandingkan Harga User vs Pasar Kategori
            is_cheap = price < market_ref_price_rp
            price_status = "LEBIH MURAH" if is_cheap else "LEBIH MAHAL"
            diff_percent = abs(1 - price_ratio) * 100
            
            if est_qty < 2:
                # KASUS: Penjualan Rendah
                if is_cheap:
                    st.warning(f"""
                    **⚠️ Penjualan Rendah, Padahal Harga Murah.**
                    - Harga Anda (Rp {price:,.0f}) sudah **{diff_percent:.0f}% {price_status}** dibanding rata-rata kategori **{category}** (Rp {market_ref_price_rp:,.0f}).
                    - Masalahnya mungkin **Kategori ini sepi peminat** atau **Rating {rating} kurang meyakinkan**.
                    - *Saran: Coba naikkan Rating atau ganti strategi promosi.*
                    """)
                else:
                    st.error(f"""
                    **⛔ Harga Terlalu Tinggi!**
                    - Harga Anda (Rp {price:,.0f}) lebih mahal **{diff_percent:.0f}%** dari standar kategori **{category}** (Rp {market_ref_price_rp:,.0f}).
                    - Pasar sensitif terhadap harga. Coba turunkan ke kisaran Rp {market_ref_price_rp:,.0f}.
                    """)
            
            elif est_qty > 10:
                # KASUS: Penjualan Tinggi
                st.success(f"""
                **🔥 BEST SELLER! Strategi Tepat.**
                - Dengan harga Rp {price:,.0f} ({price_status} dari pasar kategori **{category}**), volume penjualan melonjak.
                - Kualitas buku (Rating {rating}) juga mendukung. Pertahankan stok!
                """)
            else:
                # KASUS: Standar
                st.info(f"""
                **✅ Performa Stabil.**
                - Harga Rp {price:,.0f} kompetitif untuk kategori **{category}** (Rata-rata: Rp {market_ref_price_rp:,.0f}).
                - Penjualan {est_qty:.1f} pcs adalah angka yang wajar.
                """)

    # ========================================================================
    # TAB 2: VISUALISASI
    # ========================================================================
    elif mode == "📊 Dashboard Analitik":
        st.title("📊 Business Insights")
        if 'Total_Amount' in df.columns:
            df_disp = df.copy()
            df_disp['Total_IDR'] = df_disp['Total_Amount'] * 200
            
            st.subheader("🏆 Top 10 Kategori")
            fig1 = px.bar(df_disp.groupby('Final_Category')['Total_IDR'].sum().reset_index().sort_values('Total_IDR', ascending=False).head(10), 
                          x='Final_Category', y='Total_IDR', template='plotly_white', color='Total_IDR')
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("💰 Distribusi Harga")
            fig2 = px.histogram(df_disp, x=df_disp['Item_Price']*200, nbins=30, labels={'x':'Harga (Rp)'}, template='plotly_white')
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()