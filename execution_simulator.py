import pandas as pd
import numpy as np
from datetime import datetime

class ExecutionSimulator:
    def __init__(self):
        self.execution_styles = ['Market', 'Limit', 'TWAP', 'VWAP']
        
    def simulate_execution(self, data, signals, execution_type='Market Order', slippage=0.0005):
        """
        Simulate trade execution with different strategies
        """
        try:
            df = data.copy()
            df['signal'] = signals
            
            execution_results = []
            
            for i, row in df.iterrows():
                if row['signal'] in [0, 2]:  # Only execute on buy/sell signals
                    execution = self._execute_trade(
                        row, execution_type, slippage
                    )
                    if execution:
                        execution['timestamp'] = i
                        execution_results.append(execution)
            
            if execution_results:
                results_df = pd.DataFrame(execution_results)
                results_df.set_index('timestamp', inplace=True)
                return results_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            raise Exception(f"Error simulating execution: {str(e)}")
    
    def _execute_trade(self, market_data, execution_type, base_slippage):
        """
        Execute individual trade based on strategy
        """
        try:
            signal = market_data['signal']
            price = market_data['close']
            volume = market_data['volume']
            
            # Determine trade direction
            side = 'BUY' if signal == 2 else 'SELL'
            
            # Calculate execution parameters
            execution = {
                'signal': signal,
                'side': side,
                'target_price': price,
                'volume': volume
            }
            
            if execution_type == 'Market Order':
                execution.update(self._market_order_execution(market_data, base_slippage))
            elif execution_type == 'Limit Order':
                execution.update(self._limit_order_execution(market_data, base_slippage))
            elif execution_type == 'TWAP':
                execution.update(self._twap_execution(market_data, base_slippage))
            elif execution_type == 'VWAP':
                execution.update(self._vwap_execution(market_data, base_slippage))
            
            return execution
            
        except Exception as e:
            return None
    
    def _market_order_execution(self, market_data, base_slippage):
        """
        Simulate market order execution
        """
        price = market_data['close']
        volume = market_data['volume']
        
        # Market orders have higher slippage but immediate execution
        market_impact = self._calculate_market_impact(volume)
        total_slippage = base_slippage + market_impact
        
        executed_price = price * (1 + total_slippage)
        
        return {
            'execution_type': 'Market',
            'executed_price': executed_price,
            'slippage': total_slippage,
            'filled': True,
            'fill_ratio': 1.0,
            'latency': np.random.normal(10, 2),  # 10ms average latency
            'execution_cost': abs(executed_price - price)
        }
    
    def _limit_order_execution(self, market_data, base_slippage):
        """
        Simulate limit order execution
        """
        price = market_data['close']
        high = market_data['high']
        low = market_data['low']
        signal = market_data['signal']
        
        # Set limit price based on signal
        if signal == 2:  # Buy
            limit_price = price * 0.999  # Slightly below market
            filled = low <= limit_price
        else:  # Sell
            limit_price = price * 1.001  # Slightly above market
            filled = high >= limit_price
        
        if filled:
            executed_price = limit_price
            slippage = abs(executed_price - price) / price
        else:
            executed_price = 0
            slippage = 0
        
        return {
            'execution_type': 'Limit',
            'executed_price': executed_price,
            'slippage': slippage,
            'filled': filled,
            'fill_ratio': 1.0 if filled else 0.0,
            'latency': np.random.normal(50, 10),  # Higher latency for limit orders
            'execution_cost': abs(executed_price - price) if filled else 0
        }
    
    def _twap_execution(self, market_data, base_slippage):
        """
        Simulate TWAP (Time-Weighted Average Price) execution
        """
        price = market_data['close']
        
        # TWAP reduces market impact but may have timing risk
        market_impact_reduction = 0.3  # 30% reduction in market impact
        timing_risk = np.random.normal(0, 0.0002)  # Small timing risk
        
        adjusted_slippage = base_slippage * (1 - market_impact_reduction) + abs(timing_risk)
        executed_price = price * (1 + adjusted_slippage)
        
        return {
            'execution_type': 'TWAP',
            'executed_price': executed_price,
            'slippage': adjusted_slippage,
            'filled': True,
            'fill_ratio': 1.0,
            'latency': np.random.normal(30, 5),
            'execution_cost': abs(executed_price - price)
        }
    
    def _vwap_execution(self, market_data, base_slippage):
        """
        Simulate VWAP (Volume-Weighted Average Price) execution
        """
        price = market_data['close']
        volume = market_data['volume']
        
        # VWAP considers volume patterns
        volume_factor = min(1.0, volume / 100000)  # Normalize volume
        market_impact_reduction = 0.4 * volume_factor  # Better execution with higher volume
        
        adjusted_slippage = base_slippage * (1 - market_impact_reduction)
        executed_price = price * (1 + adjusted_slippage)
        
        return {
            'execution_type': 'VWAP',
            'executed_price': executed_price,
            'slippage': adjusted_slippage,
            'filled': True,
            'fill_ratio': 1.0,
            'latency': np.random.normal(25, 5),
            'execution_cost': abs(executed_price - price)
        }
    
    def _calculate_market_impact(self, volume):
        """
        Calculate market impact based on volume
        """
        # Simple square root model for market impact
        normalized_volume = volume / 1000000  # Normalize to millions
        market_impact = 0.0001 * np.sqrt(max(0.01, normalized_volume))
        return min(market_impact, 0.002)  # Cap at 20 bps
    
    def analyze_execution_quality(self, execution_results):
        """
        Analyze execution quality metrics
        """
        if execution_results.empty:
            return {}
        
        analysis = {
            'total_trades': len(execution_results),
            'fill_rate': execution_results['filled'].mean(),
            'avg_slippage_bps': execution_results['slippage'].mean() * 10000,
            'avg_latency_ms': execution_results['latency'].mean(),
            'total_execution_cost': execution_results['execution_cost'].sum(),
            'slippage_volatility': execution_results['slippage'].std() * 10000
        }
        
        # Execution type breakdown
        if 'execution_type' in execution_results.columns:
            type_analysis = execution_results.groupby('execution_type').agg({
                'slippage': ['mean', 'std'],
                'latency': 'mean',
                'filled': 'mean',
                'execution_cost': 'sum'
            }).round(6)
            analysis['by_execution_type'] = type_analysis
        
        # Signal type breakdown
        signal_analysis = execution_results.groupby('signal').agg({
            'slippage': ['mean', 'std'],
            'latency': 'mean',
            'filled': 'mean',
            'execution_cost': 'sum'
        }).round(6)
        analysis['by_signal'] = signal_analysis
        
        return analysis
    
    def simulate_smart_order_routing(self, market_data, order_size, dark_pool_ratio=0.3):
        """
        Simulate smart order routing across venues
        """
        venues = {
            'NYSE': {'fee': 0.0003, 'liquidity': 0.4, 'speed': 'fast'},
            'NASDAQ': {'fee': 0.0002, 'liquidity': 0.3, 'speed': 'fast'},
            'Dark Pool': {'fee': 0.0001, 'liquidity': dark_pool_ratio, 'speed': 'medium'},
            'ECN': {'fee': 0.0004, 'liquidity': 0.2, 'speed': 'fastest'}
        }
        
        routing_decision = []
        remaining_size = order_size
        
        # Route to dark pool first if available
        if dark_pool_ratio > 0:
            dark_fill = min(remaining_size, order_size * dark_pool_ratio)
            routing_decision.append({
                'venue': 'Dark Pool',
                'size': dark_fill,
                'expected_fee': venues['Dark Pool']['fee'] * dark_fill
            })
            remaining_size -= dark_fill
        
        # Route remaining to lit venues based on liquidity
        if remaining_size > 0:
            lit_venues = ['NYSE', 'NASDAQ', 'ECN']
            for venue in lit_venues:
                venue_size = remaining_size * venues[venue]['liquidity']
                if venue_size > 0:
                    routing_decision.append({
                        'venue': venue,
                        'size': venue_size,
                        'expected_fee': venues[venue]['fee'] * venue_size
                    })
        
        return routing_decision
    
    def calculate_implementation_shortfall(self, execution_results, benchmark_price):
        """
        Calculate implementation shortfall vs benchmark
        """
        if execution_results.empty:
            return {}
        
        filled_trades = execution_results[execution_results['filled'] == True]
        
        if filled_trades.empty:
            return {'implementation_shortfall': 0, 'trade_count': 0}
        
        # Calculate shortfall for each trade
        shortfalls = []
        for _, trade in filled_trades.iterrows():
            if trade['side'] == 'BUY':
                shortfall = trade['executed_price'] - benchmark_price
            else:
                shortfall = benchmark_price - trade['executed_price']
            
            shortfalls.append(shortfall)
        
        total_shortfall = sum(shortfalls)
        avg_shortfall = np.mean(shortfalls)
        shortfall_volatility = np.std(shortfalls)
        
        return {
            'implementation_shortfall': total_shortfall,
            'avg_shortfall': avg_shortfall,
            'shortfall_volatility': shortfall_volatility,
            'trade_count': len(filled_trades)
        }
