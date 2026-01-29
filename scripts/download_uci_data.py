#!/usr/bin/env python
"""
Script to download and preprocess the UCI Household Power Consumption dataset.
"""

import io
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# URL for the dataset
DATASET_URL = "https://d396qusza40orc.cloudfront.net/exdata%2Fdata%2Fhousehold_power_consumption.zip"
OUTPUT_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def download_and_process():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Download
    print(f"Downloading dataset from {DATASET_URL}...")
    response = requests.get(DATASET_URL)
    response.raise_for_status()
    
    # 2. Extract
    print("Extracting zip file...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # The file inside is usually "household_power_consumption.txt"
        file_name = z.namelist()[0]
        print(f"Found file: {file_name}")
        
        with z.open(file_name) as f:
            # 3. Read and Parse
            print("Reading CSV (this may take a moment)...")
            # The file uses semi-colon separator and '?' for missing values
            df = pd.read_csv(
                f, 
                sep=';', 
                na_values=['?'],
                low_memory=False,
                infer_datetime_format=False
            )
    
    print(f"Raw shape: {df.shape}")
    
    # 4. Preprocess
    print("Preprocessing dates and times...")
    # Combine Date and Time into a single timestamp
    # Date format is dd/mm/yyyy
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    
    print("Cleaning data...")
    # Select relevant columns
    # Global_active_power is in kilowatts
    df = df[['timestamp', 'Global_active_power']]
    
    # Drop missing values
    df = df.dropna()
    
    # Set index
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # 5. Resample to Hourly
    print("Resampling to hourly frequency...")
    # Resample to hourly means (kW averaged over hour = kWh)
    df_hourly = df.resample('h').mean()
    
    # Rename to schema expected by our system
    df_hourly = df_hourly.rename(columns={'Global_active_power': 'consumption_kwh'})
    
    # Fill any gaps created by resampling (though Dropna handled most, resampling might introduce NaNs for empty hours)
    df_hourly = df_hourly.fillna(method='ffill')
    
    # 6. Save
    output_file = PROCESSED_DIR / "uci_household_hourly.csv"
    df_hourly.reset_index().to_csv(output_file, index=False)
    
    print(f"Processed shape: {df_hourly.shape}")
    print(f"Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")
    print(f"Saved to {output_file}")
    
    # Also update the main DB/Ingestion source? 
    # For now, we'll just leave it as a CSV that run_training.py can pick up via --data-file argument
    
    return output_file

if __name__ == "__main__":
    download_and_process()
