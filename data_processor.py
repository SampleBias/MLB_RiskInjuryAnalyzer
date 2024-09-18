import pandas as pd
import numpy as np
from pybaseball import batting_stats, pitching_stats, statcast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_PATH = 'mlb_data.pkl'
MODEL_PATH = 'mlb_injury_model.joblib'
SCALER_PATH = 'mlb_injury_scaler.joblib'
PROCESSED_DATA_PATH = 'mlb_processed_injury_data.pkl'

def load_and_process_data():
    logging.info("Starting data loading and processing")
    seasons = [2020, 2021, 2022, 2023]
    
    # Load batting and pitching data
    batting_data = pd.concat([batting_stats(season) for season in seasons])
    pitching_data = pd.concat([pitching_stats(season) for season in seasons])
    
    # Combine batting and pitching data
    df = pd.concat([batting_data, pitching_data], axis=0, ignore_index=True)
    
    # Ensure 'player_name' column exists
    if 'Name' in df.columns:
        df['player_name'] = df['Name']
    elif 'name' in df.columns:
        df['player_name'] = df['name']
    else:
        logging.warning("No 'Name' or 'name' column found. Using index as player_name.")
        df['player_name'] = df.index.astype(str)
    
    # Add Statcast data (for demonstration, we'll just use the last season)
    start_date = f"{seasons[-1]}-03-01"
    end_date = f"{seasons[-1]}-11-30"
    
    try:
        statcast_data = statcast(start_dt=start_date, end_dt=end_date)
        statcast_agg = statcast_data.groupby('player_name').agg({
            'launch_speed': 'mean',
            'launch_angle': 'mean'
        }).reset_index()
        df = pd.merge(df, statcast_agg, on='player_name', how='left')
    except Exception as e:
        logging.warning(f"Error processing Statcast data: {e}")
        logging.warning("Continuing without Statcast data")
    
    # Simulate injury data (replace this with real injury data if available)
    df['Injured'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])  # 10% injury rate
    
    logging.info("Data loading and initial processing completed")
    return df

def preprocess_data(df):
    logging.info("Starting data preprocessing")
    features = [
        'player_name', 'Age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI',
        'BB', 'SO', 'BA', 'OBP', 'SLG', 'OPS', 'WAR',
        'ERA', 'W', 'L', 'SV', 'IP', 'WHIP', 'SO9', 'launch_speed', 'launch_angle',
        'Injured'
    ]
    available_features = [col for col in features if col in df.columns]
    df_processed = df[available_features].copy()
    
    # Fill NA values
    df_processed = df_processed.fillna(0)
    
    # Create position category
    if 'IP' in df_processed.columns:
        df_processed['Position'] = np.where(df_processed['IP'] > 0, 'Pitcher', 'Position Player')
    else:
        logging.warning("IP column not found. Using random values for Position.")
        df_processed['Position'] = np.random.choice(['Pitcher', 'Position Player'], size=len(df_processed))
    
    df_processed = pd.get_dummies(df_processed, columns=['Position'], drop_first=True)
    
    # Ensure 'player_name' is in the dataframe
    if 'player_name' not in df_processed.columns:
        logging.warning("player_name column not found. Using index as player_name.")
        df_processed['player_name'] = df_processed.index.astype(str)
    
    X = df_processed.drop(['Injured', 'player_name'], axis=1, errors='ignore')
    y = df_processed['Injured']
    
    logging.info("Data preprocessing completed")
    return X, y, df_processed

def train_model(X, y):
    logging.info("Starting model training")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    logging.info("Model training completed")
    return model, scaler

def main():
    # Load and process data
    df = load_and_process_data()
    df.to_pickle(DATA_PATH)
    logging.info(f"Raw data saved to {DATA_PATH}")
    
    # Preprocess data
    X, y, df_processed = preprocess_data(df)
    df_processed.to_pickle(PROCESSED_DATA_PATH)
    logging.info(f"Processed data saved to {PROCESSED_DATA_PATH}")
    
    # Train model
    model, scaler = train_model(X, y)
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    logging.info(f"Scaler saved to {SCALER_PATH}")

if __name__ == "__main__":
    main()