import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_realistic_market_data(symbol="AAPL", days=7, start_price=150.0):
    """
    Generate realistic market data for demonstration purposes
    """
    # Create timestamp range (1-minute intervals during market hours)
    start_date = datetime.now() - timedelta(days=days)
    
    # Market hours: 9:30 AM to 4:00 PM EST
    timestamps = []
    current_date = start_date.date()
    end_date = datetime.now().date()
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            market_open = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=9, minutes=30)
            market_close = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=16)
            
            current_time = market_open
            while current_time <= market_close:
                timestamps.append(current_time)
                current_time += timedelta(minutes=1)
        
        current_date += timedelta(days=1)
    
    n_periods = len(timestamps)
    
    # Generate realistic price movements using GBM (Geometric Brownian Motion)
    dt = 1/390  # 1 minute in trading day (390 minutes)
    mu = 0.05  # Annual drift (5%)
    sigma = 0.2  # Annual volatility (20%)
    
    # Generate random price movements
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_periods)
    
    # Add some microstructure noise and mean reversion
    microstructure_noise = np.random.normal(0, 0.001, n_periods)
    mean_reversion = -0.1 * np.diff(np.concatenate([[0], returns]), prepend=0)
    
    adjusted_returns = returns + microstructure_noise + mean_reversion
    
    # Calculate prices
    prices = [start_price]
    for i in range(1, n_periods):
        new_price = prices[-1] * (1 + adjusted_returns[i])
        prices.append(max(new_price, 0.01))  # Ensure positive prices
    
    # Generate OHLCV data
    data = []
    for i, timestamp in enumerate(timestamps):
        base_price = prices[i]
        
        # Generate realistic high, low within the period
        volatility_factor = np.random.uniform(0.0005, 0.002)
        spread = base_price * volatility_factor
        
        high = base_price + np.random.uniform(0, spread)
        low = base_price - np.random.uniform(0, spread)
        
        # Open is previous close + small gap
        if i == 0:
            open_price = base_price
        else:
            gap = np.random.normal(0, base_price * 0.0001)
            open_price = max(prices[i-1] + gap, 0.01)
        
        close_price = base_price
        
        # Ensure OHLC consistency
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Generate volume (higher volume during market open/close)
        hour = timestamp.hour
        if hour in [9, 15]:  # Opening and closing hours
            base_volume = np.random.lognormal(12, 1)  # Higher volume
        else:
            base_volume = np.random.lognormal(11, 0.8)  # Normal volume
        
        volume = int(base_volume * 100)  # Round to reasonable values
        
        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def save_demo_data():
    """Save demo data to file for the application"""
    demo_data = generate_realistic_market_data("AAPL", days=7, start_price=150.0)
    demo_data.to_csv('demo_market_data.csv')
    return demo_data

if __name__ == "__main__":
    data = save_demo_data()
    print(f"Generated {len(data)} records of demo market data")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"Average volume: {data['volume'].mean():,.0f}")