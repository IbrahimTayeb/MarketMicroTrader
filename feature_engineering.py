import pandas as pd
import numpy as np
import talib as ta
from scipy import stats

class FeatureEngineering:
    def __init__(self):
        self.lookback_periods = [5, 10, 20, 50]
        
    def engineer_features(self, data):
        """
        Engineer microstructure and technical features
        """
        try:
            df = data.copy()
            
            # Basic price features
            df = self._add_price_features(df)
            
            # Technical indicators
            df = self._add_technical_indicators(df)
            
            # Microstructure features
            df = self._add_microstructure_features(df)
            
            # Volume features
            df = self._add_volume_features(df)
            
            # Statistical features
            df = self._add_statistical_features(df)
            
            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in feature engineering: {str(e)}")
    
    def _add_price_features(self, df):
        """Add basic price-based features"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price movements
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open']
        
        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # OHLC relationships
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['body_size'] = np.abs(df['close'] - df['open'])
        
        return df
    
    def _add_technical_indicators(self, df):
        """Add technical analysis indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Moving averages
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = ta.SMA(close, timeperiod=period)
            df[f'ema_{period}'] = ta.EMA(close, timeperiod=period)
            df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # RSI
        df['rsi_14'] = ta.RSI(close, timeperiod=14)
        df['rsi_7'] = ta.RSI(close, timeperiod=7)
        
        # MACD
        macd, macd_signal, macd_hist = ta.MACD(close)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = ta.BBANDS(close, timeperiod=20)
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # ATR (Average True Range)
        df['atr_14'] = ta.ATR(high, low, close, timeperiod=14)
        
        # Stochastic
        slowk, slowd = ta.STOCH(high, low, close)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        return df
    
    def _add_microstructure_features(self, df):
        """Add microstructure-specific features"""
        # Bid-Ask spread approximation (using high-low as proxy)
        df['spread'] = df['high'] - df['low']
        df['spread_pct'] = df['spread'] / df['close']
        
        # Rolling spread statistics
        for period in [5, 10, 20]:
            df[f'spread_ma_{period}'] = df['spread_pct'].rolling(period).mean()
            df[f'spread_std_{period}'] = df['spread_pct'].rolling(period).std()
            df[f'spread_percentile_{period}'] = df['spread_pct'].rolling(period).rank(pct=True)
        
        # Order book imbalance proxy
        # Using volume and price movements as proxy for imbalance
        df['volume_imbalance'] = np.where(
            df['close'] > df['open'], 
            df['volume'], 
            -df['volume']
        )
        
        for period in [5, 10]:
            df[f'volume_imbalance_ma_{period}'] = df['volume_imbalance'].rolling(period).mean()
        
        # Trade aggressor side (approximation)
        df['aggressor_side'] = np.where(df['close'] > df['open'], 1, -1)
        
        # Microstructure patterns
        df['doji'] = np.where(np.abs(df['close'] - df['open']) < 0.1 * df['hl_range'], 1, 0)
        df['hammer'] = np.where(
            (df['lower_shadow'] > 2 * df['body_size']) & 
            (df['upper_shadow'] < 0.5 * df['body_size']), 1, 0
        )
        
        return df
    
    def _add_volume_features(self, df):
        """Add volume-based features"""
        # Volume statistics
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_std_{period}'] = df['volume'].rolling(period).std()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * np.sign(df['returns'])
        
        # On-Balance Volume (OBV)
        df['obv'] = (df['volume'] * np.sign(df['returns'])).cumsum()
        
        # Volume Rate of Change
        df['volume_roc_5'] = df['volume'].pct_change(5)
        df['volume_roc_10'] = df['volume'].pct_change(10)
        
        # Money Flow Index approximation
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(df['returns'] > 0, 0).rolling(14).sum()
        negative_flow = money_flow.where(df['returns'] < 0, 0).rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + positive_flow / negative_flow.abs()))
        
        return df
    
    def _add_statistical_features(self, df):
        """Add statistical features"""
        # Rolling volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volatility_percentile_{period}'] = df[f'volatility_{period}'].rolling(50).rank(pct=True)
        
        # Skewness and Kurtosis
        for period in [10, 20]:
            df[f'skewness_{period}'] = df['returns'].rolling(period).skew()
            df[f'kurtosis_{period}'] = df['returns'].rolling(period).kurt()
        
        # Z-score of returns
        for period in [20, 50]:
            df[f'returns_zscore_{period}'] = (
                df['returns'] - df['returns'].rolling(period).mean()
            ) / df['returns'].rolling(period).std()
        
        # Autocorrelation
        for lag in [1, 2, 5]:
            df[f'autocorr_lag_{lag}'] = df['returns'].rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Price momentum
        for period in [3, 5, 10]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        return df
    
    def get_feature_importance_groups(self):
        """Return feature groups for analysis"""
        return {
            'price': ['returns', 'log_returns', 'price_change_pct', 'hl_range_pct'],
            'technical': ['rsi_14', 'macd', 'bb_position', 'atr_14'],
            'microstructure': ['spread_pct', 'volume_imbalance', 'aggressor_side'],
            'volume': ['volume_ratio_10', 'obv', 'mfi'],
            'statistical': ['volatility_10', 'skewness_10', 'momentum_5']
        }
