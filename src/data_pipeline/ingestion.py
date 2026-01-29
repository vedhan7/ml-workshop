"""
Data ingestion module for loading electricity consumption data.
Supports CSV/Parquet files and weather APIs.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
import structlog

from src.config import get_settings
from src.database.connection import get_db_context
from src.database.models import ElectricityConsumption, WeatherData

logger = structlog.get_logger(__name__)


class DataIngestionService:
    """Service for ingesting data from various sources."""
    
    def __init__(self, location_id: Optional[str] = None):
        self.settings = get_settings()
        self.location_id = location_id
    
    def ingest_csv(
        self,
        file_path: str | Path,
        timestamp_col: str = "timestamp",
        consumption_col: str = "consumption_kwh",
        meter_id: Optional[str] = None,
    ) -> int:
        """
        Ingest electricity consumption data from CSV file.
        
        Args:
            file_path: Path to CSV file
            timestamp_col: Name of timestamp column
            consumption_col: Name of consumption column
            meter_id: Optional meter identifier
            
        Returns:
            Number of records ingested
        """
        logger.info("ingesting_csv", file_path=str(file_path))
        
        df = pd.read_csv(file_path, parse_dates=[timestamp_col])
        
        records = []
        for _, row in df.iterrows():
            record = ElectricityConsumption(
                timestamp=row[timestamp_col],
                consumption_kwh=float(row[consumption_col]),
                location_id=self.location_id,
                meter_id=meter_id,
            )
            records.append(record)
        
        with get_db_context() as db:
            # Use bulk insert with upsert logic
            for record in records:
                existing = db.query(ElectricityConsumption).filter(
                    ElectricityConsumption.timestamp == record.timestamp,
                    ElectricityConsumption.location_id == record.location_id,
                ).first()
                
                if existing:
                    existing.consumption_kwh = record.consumption_kwh
                    existing.meter_id = record.meter_id
                else:
                    db.add(record)
            
            db.commit()
        
        logger.info("csv_ingestion_complete", records=len(records))
        return len(records)
    
    def ingest_parquet(
        self,
        file_path: str | Path,
        timestamp_col: str = "timestamp",
        consumption_col: str = "consumption_kwh",
    ) -> int:
        """
        Ingest electricity consumption data from Parquet file.
        
        Args:
            file_path: Path to Parquet file
            timestamp_col: Name of timestamp column
            consumption_col: Name of consumption column
            
        Returns:
            Number of records ingested
        """
        logger.info("ingesting_parquet", file_path=str(file_path))
        
        df = pd.read_parquet(file_path)
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        records_count = 0
        with get_db_context() as db:
            for _, row in df.iterrows():
                record = ElectricityConsumption(
                    timestamp=row[timestamp_col],
                    consumption_kwh=float(row[consumption_col]),
                    location_id=self.location_id,
                )
                db.merge(record)
                records_count += 1
            db.commit()
        
        logger.info("parquet_ingestion_complete", records=records_count)
        return records_count
    
    def fetch_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch weather data from OpenWeatherMap API.
        
        Note: Free tier has limitations. Consider using historical data files
        for production use.
        """
        api_key = self.settings.openweather_api_key
        if not api_key:
            logger.warning("no_api_key", message="OpenWeatherMap API key not configured")
            return pd.DataFrame()
        
        # OpenWeatherMap One Call API (historical data requires paid plan)
        # For development, we'll use current weather as a sample
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }
        
        try:
            response = httpx.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            weather_record = WeatherData(
                timestamp=datetime.utcnow(),
                temperature_celsius=data["main"]["temp"],
                humidity_percent=data["main"]["humidity"],
                feels_like_celsius=data["main"]["feels_like"],
                wind_speed_mps=data["wind"]["speed"],
                cloud_cover_percent=data["clouds"]["all"],
                location_id=self.location_id,
            )
            
            with get_db_context() as db:
                db.add(weather_record)
                db.commit()
            
            return pd.DataFrame([{
                "timestamp": weather_record.timestamp,
                "temperature_celsius": weather_record.temperature_celsius,
                "humidity_percent": weather_record.humidity_percent,
            }])
            
        except httpx.HTTPError as e:
            logger.error("weather_api_error", error=str(e))
            return pd.DataFrame()
    
    def load_from_database(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load electricity consumption data from database.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with consumption data
        """
        with get_db_context() as db:
            query = db.query(ElectricityConsumption)
            
            if self.location_id:
                query = query.filter(ElectricityConsumption.location_id == self.location_id)
            if start_date:
                query = query.filter(ElectricityConsumption.timestamp >= start_date)
            if end_date:
                query = query.filter(ElectricityConsumption.timestamp <= end_date)
            
            query = query.order_by(ElectricityConsumption.timestamp)
            
            records = query.all()
            
            if not records:
                return pd.DataFrame()
            
            return pd.DataFrame([
                {
                    "timestamp": r.timestamp,
                    "consumption_kwh": r.consumption_kwh,
                    "location_id": r.location_id,
                    "meter_id": r.meter_id,
                    "is_interpolated": r.is_interpolated,
                    "is_outlier": r.is_outlier,
                }
                for r in records
            ])


# Convenience functions
def ingest_csv(file_path: str | Path, **kwargs) -> int:
    """Convenience function for CSV ingestion."""
    service = DataIngestionService()
    return service.ingest_csv(file_path, **kwargs)


def ingest_weather_api(latitude: float, longitude: float, **kwargs) -> pd.DataFrame:
    """Convenience function for weather API ingestion."""
    service = DataIngestionService()
    return service.fetch_weather_data(latitude, longitude, **kwargs)
