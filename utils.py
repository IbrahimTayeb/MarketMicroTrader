import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

def create_candlestick_chart(data, title="Price Chart"):
    """
    Create candlestick chart with volume
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(title, 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            name="Volume",
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False
    )
    
    return fig

def create_performance_chart(results_df, initial_capital, benchmark_data=None):
    """
    Create performance comparison chart
    """
    fig = go.Figure()
    
    # Portfolio performance
    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df['portfolio_value'],
        mode='lines',
        name='Strategy',
        line=dict(color='blue', width=2)
    ))
    
    # Benchmark (if provided)
    if benchmark_data is not None:
        benchmark_value = initial_capital * (benchmark_data / benchmark_data.iloc[0])
        fig.add_trace(go.Scatter(
            x=benchmark_value.index,
            y=benchmark_value.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', dash='dash')
        ))
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_drawdown_chart(results_df):
    """
    Create drawdown chart
    """
    portfolio_values = results_df['portfolio_value']
    peak = portfolio_values.expanding().max()
    drawdown = (portfolio_values - peak) / peak * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode='lines',
        fill='tonexty',
        name='Drawdown',
        line=dict(color='red'),
        fillcolor='rgba(255,0,0,0.3)'
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        height=300,
        yaxis=dict(ticksuffix='%')
    )
    
    return fig

def create_returns_distribution(results_df):
    """
    Create returns distribution histogram
    """
    returns = results_df['returns'].dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Returns',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add normal distribution overlay
    mean_ret = returns.mean()
    std_ret = returns.std()
    x_norm = np.linspace(returns.min(), returns.max(), 100)
    y_norm = (1/(std_ret * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mean_ret) / std_ret) ** 2)
    
    # Scale normal distribution to match histogram
    y_norm = y_norm * len(returns) * (returns.max() - returns.min()) / 50
    
    fig.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Returns",
        yaxis_title="Frequency",
        height=400,
        bargap=0.1
    )
    
    return fig

def create_signal_analysis_chart(data, signals):
    """
    Create signal analysis visualization
    """
    df = data.copy()
    df['signal'] = signals
    
    # Calculate forward returns for each signal
    forward_returns = {}
    for signal in [0, 1, 2]:
        signal_data = df[df['signal'] == signal]
        if not signal_data.empty:
            fwd_returns = signal_data['returns'].shift(-1).dropna()
            forward_returns[signal] = fwd_returns
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Sell Signals', 'Hold Signals', 'Buy Signals')
    )
    
    signal_names = ['Sell', 'Hold', 'Buy']
    colors = ['red', 'gray', 'green']
    
    for i, (signal, name, color) in enumerate(zip([0, 1, 2], signal_names, colors)):
        if signal in forward_returns:
            fig.add_trace(
                go.Histogram(
                    x=forward_returns[signal],
                    name=f'{name} Forward Returns',
                    marker_color=color,
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title="Forward Returns by Signal Type",
        height=400
    )
    
    return fig

def calculate_risk_metrics(returns):
    """
    Calculate comprehensive risk metrics
    """
    if len(returns) == 0:
        return {}
    
    returns_clean = returns.dropna()
    
    metrics = {
        'volatility': returns_clean.std() * np.sqrt(252),
        'skewness': returns_clean.skew(),
        'kurtosis': returns_clean.kurtosis(),
        'var_95': returns_clean.quantile(0.05),
        'var_99': returns_clean.quantile(0.01),
        'max_daily_loss': returns_clean.min(),
        'max_daily_gain': returns_clean.max(),
    }
    
    # Calculate VaR and CVaR
    metrics['cvar_95'] = returns_clean[returns_clean <= metrics['var_95']].mean()
    metrics['cvar_99'] = returns_clean[returns_clean <= metrics['var_99']].mean()
    
    return metrics

def format_percentage(value, decimals=2):
    """
    Format value as percentage
    """
    return f"{value * 100:.{decimals}f}%"

def format_currency(value, decimals=2):
    """
    Format value as currency
    """
    return f"${value:,.{decimals}f}"

def validate_data_quality(data):
    """
    Validate data quality and return issues
    """
    issues = []
    
    if data is None or data.empty:
        issues.append("Data is empty")
        return issues
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        issues.append(f"Missing columns: {missing_columns}")
    
    # Check for negative prices
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in data.columns and (data[col] <= 0).any():
            issues.append(f"Negative or zero prices found in {col}")
    
    # Check for NaN values
    nan_columns = data.columns[data.isna().any()].tolist()
    if nan_columns:
        issues.append(f"NaN values found in: {nan_columns}")
    
    # Check price consistency (high >= low, etc.)
    if all(col in data.columns for col in ['high', 'low', 'open', 'close']):
        if (data['high'] < data['low']).any():
            issues.append("High prices below low prices detected")
        if (data['high'] < data['open']).any() or (data['high'] < data['close']).any():
            issues.append("High prices below open/close prices detected")
        if (data['low'] > data['open']).any() or (data['low'] > data['close']).any():
            issues.append("Low prices above open/close prices detected")
    
    return issues

def create_correlation_matrix(features_df):
    """
    Create correlation matrix heatmap
    """
    # Select only numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = features_df[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    
    fig.update_layout(height=600)
    return fig

def display_data_summary(data):
    """
    Display comprehensive data summary
    """
    if data is None or data.empty:
        st.error("No data to summarize")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(data))
        st.metric("Date Range", f"{len(data)} periods")
    
    with col2:
        if 'close' in data.columns:
            st.metric("Price Range", f"${data['close'].min():.2f} - ${data['close'].max():.2f}")
        if 'volume' in data.columns:
            st.metric("Avg Volume", f"{data['volume'].mean():,.0f}")
    
    with col3:
        data_quality_issues = validate_data_quality(data)
        if data_quality_issues:
            st.error(f"Data Quality Issues: {len(data_quality_issues)}")
            for issue in data_quality_issues:
                st.write(f"• {issue}")
        else:
            st.success("Data Quality: Good")

def export_results(results_df, filename="trading_results.csv"):
    """
    Export results to CSV
    """
    try:
        csv = results_df.to_csv()
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        return True
    except Exception as e:
        st.error(f"Error exporting results: {str(e)}")
        return False

def create_execution_analysis_chart(execution_results):
    """
    Create execution analysis visualization
    """
    if execution_results.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Slippage Over Time',
            'Latency Distribution', 
            'Fill Rate by Signal',
            'Execution Cost Breakdown'
        )
    )
    
    # Slippage over time
    fig.add_trace(
        go.Scatter(
            x=execution_results.index,
            y=execution_results['slippage'] * 10000,
            mode='lines+markers',
            name='Slippage (bps)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Latency distribution
    fig.add_trace(
        go.Histogram(
            x=execution_results['latency'],
            name='Latency (ms)',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Fill rate by signal
    fill_by_signal = execution_results.groupby('signal')['filled'].mean()
    fig.add_trace(
        go.Bar(
            x=['Sell', 'Buy'],
            y=[fill_by_signal.get(0, 0), fill_by_signal.get(2, 0)],
            name='Fill Rate',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Execution cost breakdown
    cost_by_signal = execution_results.groupby('signal')['execution_cost'].sum()
    fig.add_trace(
        go.Bar(
            x=['Sell', 'Buy'],
            y=[cost_by_signal.get(0, 0), cost_by_signal.get(2, 0)],
            name='Execution Cost',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title="Execution Analysis")
    return fig
