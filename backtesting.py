import pandas as pd
import numpy as np
from datetime import datetime

class BacktestingEngine:
    def __init__(self):
        self.trades = []
        self.positions = []
        
    def run_backtest(self, data, signals, initial_capital=100000, commission=1.0, position_size=0.1):
        """
        Run backtest simulation
        """
        try:
            df = data.copy()
            df['signal'] = signals
            
            # Initialize portfolio
            portfolio = {
                'cash': initial_capital,
                'position': 0,
                'portfolio_value': initial_capital,
                'returns': 0
            }
            
            results = []
            trades = []
            current_position = 0
            entry_price = 0
            
            for i, row in df.iterrows():
                signal = row['signal']
                price = row['close']
                
                # Calculate portfolio value
                portfolio_value = portfolio['cash'] + current_position * price
                
                # Generate trading decision
                trade_decision = self._get_trade_decision(signal, current_position)
                
                if trade_decision != 0:  # Execute trade
                    trade_size = self._calculate_position_size(
                        portfolio_value, price, position_size, trade_decision
                    )
                    
                    if trade_size != 0:
                        # Execute trade
                        cost = abs(trade_size) * price + commission
                        
                        if portfolio['cash'] >= cost:  # Check if we have enough cash
                            # Record trade
                            trade = {
                                'timestamp': i,
                                'signal': signal,
                                'side': 'BUY' if trade_size > 0 else 'SELL',
                                'size': abs(trade_size),
                                'price': price,
                                'cost': cost,
                                'portfolio_value_before': portfolio_value
                            }
                            trades.append(trade)
                            
                            # Update portfolio
                            portfolio['cash'] -= trade_size * price + commission
                            current_position += trade_size
                            
                            # Track entry price for PnL calculation
                            if current_position != 0:
                                entry_price = price
                
                # Calculate current portfolio value and returns
                portfolio_value = portfolio['cash'] + current_position * price
                
                if len(results) > 0:
                    prev_value = results[-1]['portfolio_value']
                    returns = (portfolio_value - prev_value) / prev_value
                else:
                    returns = 0
                
                # Store results
                result = {
                    'cash': portfolio['cash'],
                    'position': current_position,
                    'price': price,
                    'portfolio_value': portfolio_value,
                    'returns': returns,
                    'signal': signal
                }
                results.append(result)
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results, index=df.index)
            
            # Store trades for analysis
            self.trades = trades
            
            return results_df
            
        except Exception as e:
            raise Exception(f"Error running backtest: {str(e)}")
    
    def _get_trade_decision(self, signal, current_position):
        """
        Determine trade decision based on signal and current position
        """
        if signal == 2 and current_position <= 0:  # Buy signal
            return 1
        elif signal == 0 and current_position >= 0:  # Sell signal
            return -1
        elif signal == 1 and current_position != 0:  # Hold signal - close position
            return -np.sign(current_position)
        else:
            return 0  # No trade
    
    def _calculate_position_size(self, portfolio_value, price, max_position_size, direction):
        """
        Calculate position size based on portfolio value and risk management
        """
        max_dollar_amount = portfolio_value * max_position_size
        max_shares = int(max_dollar_amount / price)
        
        if direction > 0:  # Buy
            return max_shares
        else:  # Sell
            return -max_shares
    
    def calculate_performance_metrics(self, results_df, initial_capital):
        """
        Calculate comprehensive performance metrics
        """
        try:
            metrics = {}
            
            # Basic returns
            total_return = (results_df['portfolio_value'].iloc[-1] / initial_capital) - 1
            metrics['total_return'] = total_return
            
            # Annualized return (assuming 252 trading days)
            days = len(results_df)
            if days > 0:
                annualized_return = ((1 + total_return) ** (252 / days)) - 1
                metrics['annualized_return'] = annualized_return
            
            # Volatility
            returns = results_df['returns'].dropna()
            if len(returns) > 1:
                daily_vol = returns.std()
                annualized_vol = daily_vol * np.sqrt(252)
                metrics['volatility'] = annualized_vol
                
                # Sharpe ratio (assuming 0% risk-free rate)
                if annualized_vol > 0:
                    sharpe_ratio = annualized_return / annualized_vol
                    metrics['sharpe_ratio'] = sharpe_ratio
            
            # Maximum drawdown
            portfolio_values = results_df['portfolio_value']
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = drawdown.min()
            metrics['max_drawdown'] = max_drawdown
            
            # Win rate
            if len(returns) > 0:
                win_rate = (returns > 0).mean()
                metrics['win_rate'] = win_rate
            
            # Number of trades
            metrics['num_trades'] = len(self.trades)
            
            # Average trade return
            if len(self.trades) > 0:
                trade_returns = []
                for i, trade in enumerate(self.trades):
                    if i > 0:  # Skip first trade
                        prev_trade = self.trades[i-1]
                        trade_return = (trade['price'] - prev_trade['price']) / prev_trade['price']
                        trade_returns.append(trade_return)
                
                if trade_returns:
                    metrics['avg_trade_return'] = np.mean(trade_returns)
                    metrics['trade_win_rate'] = np.mean([r > 0 for r in trade_returns])
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Error calculating performance metrics: {str(e)}")
    
    def get_trade_analysis(self):
        """
        Analyze individual trades
        """
        if not self.trades:
            return None
        
        trade_df = pd.DataFrame(self.trades)
        
        # Group by signal type
        signal_analysis = trade_df.groupby('signal').agg({
            'size': ['count', 'mean'],
            'price': 'mean',
            'cost': 'sum'
        }).round(4)
        
        return trade_df, signal_analysis
    
    def simulate_slippage(self, trades, slippage_bps=5):
        """
        Simulate market impact and slippage
        """
        slippage_factor = slippage_bps / 10000  # Convert basis points to decimal
        
        adjusted_trades = []
        for trade in trades:
            slippage_amount = trade['price'] * slippage_factor
            
            if trade['side'] == 'BUY':
                adjusted_price = trade['price'] * (1 + slippage_factor)
            else:
                adjusted_price = trade['price'] * (1 - slippage_factor)
            
            adjusted_trade = trade.copy()
            adjusted_trade['executed_price'] = adjusted_price
            adjusted_trade['slippage'] = slippage_amount
            adjusted_trade['adjusted_cost'] = adjusted_trade['size'] * adjusted_price
            
            adjusted_trades.append(adjusted_trade)
        
        return adjusted_trades
    
    def compare_strategies(self, results_list, strategy_names):
        """
        Compare multiple strategy results
        """
        comparison = {}
        
        for i, (results, name) in enumerate(zip(results_list, strategy_names)):
            initial_capital = results['portfolio_value'].iloc[0]
            metrics = self.calculate_performance_metrics(results, initial_capital)
            comparison[name] = metrics
        
        return pd.DataFrame(comparison).T
