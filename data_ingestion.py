import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import streamlit as st

class MarketDataIngestion:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
    def get_market_data(self, symbol, start_date, end_date, multiplier="1"):
        """
        Fetch market data from Polygon API
        """
        try:
            # Convert dates to string format
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Build API URL for aggregates (OHLCV bars)
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/minute/{start_str}/{end_str}"
            
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000,
                "apikey": self.api_key
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            
            if data.get("status") != "OK":
                st.error(f"API returned status: {data.get('status')} - {data.get('error', 'Unknown error')}")
                return None
            
            if "results" not in data or not data["results"]:
                st.warning("No data returned from API")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data["results"])
            
            # Rename columns to standard format
            df = df.rename(columns={
                "o": "open",
                "h": "high", 
                "l": "low",
                "c": "close",
                "v": "volume",
                "t": "timestamp"
            })
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Ensure we have the required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in df.columns:
                    st.error(f"Missing required column: {col}")
                    return None
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching market data: {str(e)}")
            return None
    
    def get_real_time_quote(self, symbol):
        """
        Get real-time quote data
        """
        try:
            url = f"{self.base_url}/v2/last/trade/{symbol}"
            params = {"apikey": self.api_key}
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "OK":
                    return data.get("results", {})
            
            return None
            
        except Exception as e:
            st.error(f"Error fetching real-time quote: {str(e)}")
            return None
    
    def get_market_hours(self):
        """
        Check if market is open
        """
        try:
            url = f"{self.base_url}/v1/marketstatus/now"
            params = {"apikey": self.api_key}
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("market", "closed") == "open"
            
            return False
            
        except Exception as e:
            return False
    
    def validate_symbol(self, symbol):
        """
        Validate if symbol exists
        """
        try:
            url = f"{self.base_url}/v3/reference/tickers/{symbol}"
            params = {"apikey": self.api_key}
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "OK"
            
            return False
            
        except Exception as e:
            return False
