import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import streamlit as st
import os

class MarketDataIngestion:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.demo_mode = False
        
    def get_market_data(self, symbol, start_date, end_date, multiplier="1"):
        """
        Fetch market data from Polygon API with rate limiting and error handling
        """
        try:
            # Demo mode: Generate realistic market data for demonstration
            if self.api_key == "demo_mode":
                return self._generate_demo_data(symbol, start_date, end_date, multiplier)
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
            
            # Add retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 429:  # Rate limited
                        wait_time = (2 ** attempt) * 15  # 15, 30, 60 seconds
                        st.warning(f"Rate limited. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    
                    if response.status_code != 200:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        return None
                    
                    data = response.json()
                    
                    if data.get("status") == "DELAYED":
                        st.info(f"Note: Data is delayed (free tier limitation)")
                        # For free tier, try to work with delayed data if available
                        if "results" not in data or not data["results"]:
                            st.warning("API returned no data. This may be due to free tier limitations or weekend/holiday periods when markets are closed.")
                            st.info("To continue with the demonstration, please try selecting a different date range (try going back 2-3 weeks) or contact your API provider for data access verification.")
                            return None
                    elif data.get("status") != "OK":
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
                    
                    # Add small delay to respect rate limits
                    time.sleep(2)
                    return df
                    
                except requests.exceptions.Timeout:
                    st.warning(f"Request timeout on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        st.error("Request timed out after multiple attempts")
                        return None
                    time.sleep(5)
                    
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
    
    def _generate_demo_data(self, symbol, start_date, end_date, multiplier):
        """
        Generate realistic demo market data for demonstration purposes
        """
        # Calculate the number of periods based on timeframe
        if multiplier == "1":
            freq = "1min"
            periods_per_day = 390  # Market hours: 6.5 hours * 60 minutes
        elif multiplier == "5":
            freq = "5min"
            periods_per_day = 78   # 390 / 5
        else:
            freq = "15min"
            periods_per_day = 26   # 390 / 15
        
        # Create date range (only weekdays)
        date_range = pd.bdate_range(start=start_date, end=end_date, freq='D')
        
        # Base price for different symbols
        base_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2500.0,
            'TSLA': 200.0,
            'SPY': 400.0,
            'QQQ': 350.0
        }
        
        base_price = base_prices.get(symbol, 150.0)
        
        # Generate timestamps for market hours
        timestamps = []
        for date in date_range:
            market_open = pd.Timestamp(date).replace(hour=9, minute=30)
            market_close = pd.Timestamp(date).replace(hour=16, minute=0)
            
            if multiplier == "1":
                time_range = pd.date_range(market_open, market_close, freq='1min')
            elif multiplier == "5":
                time_range = pd.date_range(market_open, market_close, freq='5min')
            else:
                time_range = pd.date_range(market_open, market_close, freq='15min')
            
            timestamps.extend(time_range)
        
        n_periods = len(timestamps)
        
        # Generate realistic price movements using Geometric Brownian Motion
        dt = 1/252  # Daily time step
        mu = 0.05   # Annual drift (5%)
        sigma = 0.2 # Annual volatility (20%)
        
        # Scale for intraday periods
        if multiplier == "1":
            dt = dt / 390
        elif multiplier == "5":
            dt = dt / 78
        else:
            dt = dt / 26
        
        # Generate price movements
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_periods)
        
        # Add microstructure effects
        microstructure_noise = np.random.normal(0, 0.0005, n_periods)
        mean_reversion = -0.1 * np.diff(np.concatenate([[0], returns]), prepend=0)
        
        adjusted_returns = returns + microstructure_noise + mean_reversion
        
        # Generate prices
        prices = [base_price]
        for i in range(1, n_periods):
            new_price = prices[-1] * (1 + adjusted_returns[i])
            prices.append(max(new_price, 0.01))
        
        # Generate OHLCV data
        data = []
        for i, timestamp in enumerate(timestamps):
            base = prices[i]
            
            # Generate realistic intraday range
            volatility_factor = np.random.uniform(0.001, 0.003)
            range_size = base * volatility_factor
            
            # OHLC generation
            if i == 0:
                open_price = base
            else:
                gap = np.random.normal(0, base * 0.0001)
                open_price = max(prices[i-1] + gap, 0.01)
            
            close_price = base
            high = max(open_price, close_price) + np.random.uniform(0, range_size)
            low = min(open_price, close_price) - np.random.uniform(0, range_size)
            
            # Ensure price consistency
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Generate volume with realistic patterns
            hour = timestamp.hour
            if hour in [9, 15]:  # Opening and closing volume spikes
                base_volume = np.random.lognormal(12, 1)
            else:
                base_volume = np.random.lognormal(11, 0.8)
            
            volume = int(base_volume * 100)
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data, index=timestamps)
        df.index.name = 'timestamp'
        
        return df
