#!/usr/bin/env python
"""
Generate synthetic electricity consumption data for testing.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(
    start_date: datetime,
    end_date: datetime,
    output_path: Path,
    base_consumption: float = 1000.0,
    noise_level: float = 0.1,
) -> pd.DataFrame:
    """
    Generate synthetic electricity consumption data.
    
    Includes patterns for:
    - Daily seasonality (peak hours)
    - Weekly seasonality (weekday vs weekend)
    - Yearly seasonality (summer vs winter)
    - Random noise
    """
    # Generate hourly timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    n = len(timestamps)
    
    # Initialize consumption array
    consumption = np.zeros(n)
    
    # Base consumption
    consumption += base_consumption
    
    # Daily pattern (higher during day, peak in evening)
    hours = timestamps.hour
    daily_pattern = np.where(
        (hours >= 6) & (hours <= 22),
        0.3 * np.sin(2 * np.pi * (hours - 6) / 16),  # Daytime
        -0.2,  # Nighttime reduction
    )
    # Evening peak
    daily_pattern += np.where(
        (hours >= 17) & (hours <= 21),
        0.2,
        0,
    )
    consumption += base_consumption * daily_pattern
    
    # Weekly pattern (lower on weekends)
    weekday = timestamps.dayofweek
    weekly_pattern = np.where(weekday < 5, 0.1, -0.15)
    consumption += base_consumption * weekly_pattern
    
    # Yearly pattern (higher in summer due to AC, higher in winter due to heating)
    day_of_year = timestamps.dayofyear
    # Bimodal pattern - peaks in summer and winter
    yearly_pattern = 0.15 * np.cos(2 * np.pi * day_of_year / 365)  # Winter peak
    yearly_pattern += 0.1 * np.abs(np.sin(2 * np.pi * (day_of_year - 180) / 365))  # Summer peak
    consumption += base_consumption * yearly_pattern
    
    # Add trend (slight increase over time)
    trend = np.linspace(0, 0.1, n)
    consumption += base_consumption * trend
    
    # Add random noise
    noise = np.random.normal(0, noise_level * base_consumption, n)
    consumption += noise
    
    # Ensure non-negative values
    consumption = np.maximum(consumption, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "consumption_kwh": consumption,
    })
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Consumption range: {df['consumption_kwh'].min():.2f} to {df['consumption_kwh'].max():.2f}")
    print(f"Saved to: {output_path}")
    
    return df


def generate_weather_data(
    start_date: datetime,
    end_date: datetime,
    output_path: Path,
) -> pd.DataFrame:
    """Generate synthetic weather data."""
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    n = len(timestamps)
    
    # Base temperature varies by season
    day_of_year = timestamps.dayofyear
    base_temp = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer
    
    # Daily variation
    hours = timestamps.hour
    daily_temp = 5 * np.sin(2 * np.pi * (hours - 6) / 24)
    
    temperature = base_temp + daily_temp + np.random.normal(0, 2, n)
    
    # Humidity (inverse correlation with temperature)
    humidity = 60 - 0.5 * (temperature - 25) + np.random.normal(0, 10, n)
    humidity = np.clip(humidity, 20, 100)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_celsius": temperature,
        "humidity_percent": humidity,
    })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} weather records")
    print(f"Saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sample electricity data")
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Number of years of data to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Output directory",
    )
    parser.add_argument(
        "--include-weather",
        action="store_true",
        help="Also generate weather data",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=365 * args.years)
    
    print("Generating electricity consumption data...")
    generate_sample_data(
        start_date=start_date,
        end_date=end_date,
        output_path=output_dir / "electricity.csv",
    )
    
    if args.include_weather:
        print("\nGenerating weather data...")
        generate_weather_data(
            start_date=start_date,
            end_date=end_date,
            output_path=output_dir / "weather.csv",
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
