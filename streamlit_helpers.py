import pandas as pd
import numpy as np
from datetime import datetime

def create_features_for_prediction(data):
    """Create features for a single prediction."""
    df = pd.DataFrame([data])
    
    # Date Features
    df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], errors='coerce')
    df['Year'] = df['Purchase_Date'].dt.year
    df['Month'] = df['Purchase_Date'].dt.month
    df['Day'] = df['Purchase_Date'].dt.day
    df['DayOfWeek'] = df['Purchase_Date'].dt.dayofweek
    df['Quarter'] = df['Purchase_Date'].dt.quarter
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Price-based Features
    df['Price_Per_Unit'] = df['Total_Amount'] / df['Quantity']
    
    # Profit Features
    df['Profit_Margin'] = (df['Profit'] / df['Total_Amount']) * 100
    df['Profit_Category'] = pd.cut(
        df['Profit'], 
        bins=[0, 50, 100, np.inf], 
        labels=['Low', 'Medium', 'High']
    )
    
    # Book Age
    current_year = datetime.now().year
    df['Book_Age'] = current_year - df['api_first_publish_year']
    
    # Has API Data Flag
    df['Has_API_Data'] = df['api_found'].fillna(False).astype(int)
    
    # Price Category
    df['Price_Category'] = pd.cut(
        df['Item_Price'],
        bins=[0, 200, 500, 1000, np.inf],
        labels=['Budget', 'Mid-Range', 'Premium', 'Luxury']
    )
    
    return df

def encode_categorical_for_prediction(df, encoders):
    """Encode categorical variables for a single prediction."""
    df_encoded = df.copy()
    for col, le in encoders.items():
        if col in df_encoded.columns:
            # Handle unseen labels by assigning a default value (e.g., -1 or a new class)
            df_encoded[f'{col}_Encoded'] = df_encoded[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    return df_encoded

def scale_features_for_prediction(df, scaler):
    """Scale numerical features for a single prediction."""
    df_scaled = df.copy()
    scale_cols = ['Quantity', 'Item_Price', 'Total_Amount', 'Profit', 'Profit_Margin', 
                  'api_first_publish_year', 'api_number_of_pages', 'api_edition_count', 'Book_Age']
    
    existing_cols = [col for col in scale_cols if col in df_scaled.columns]
    
    if existing_cols:
        df_scaled[existing_cols] = scaler.transform(df_scaled[existing_cols])
        for col in existing_cols:
            df_scaled.rename(columns={col: f'{col}_Scaled'}, inplace=True)

    return df_scaled
