import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from data_ingestion import MarketDataIngestion
from feature_engineering import FeatureEngineering
from alpha_signals import AlphaSignalGenerator
from backtesting import BacktestingEngine
from execution_simulator import ExecutionSimulator
import warnings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Systematic Trading Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

def main():
    st.title("🚀 Systematic Trading Engine")
    st.markdown("Real-time market microstructure analysis with ML-driven alpha signals")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Configuration
    st.sidebar.subheader("Data Source")
    
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Live API Data", "Demo Data"],
        help="Use Live API Data for real market data or Demo Data for demonstration purposes"
    )
    
    if data_source == "Live API Data":
        polygon_api_key = os.getenv("API_KEY")
        
        if not polygon_api_key:
            st.error("Invalid Polygon API Key")
            return
    else:
        polygon_api_key = "demo_mode"
        st.sidebar.info("Using demo data for demonstration purposes")
    
    # Symbol selection
    symbol = st.sidebar.selectbox(
        "Select Symbol",
        ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"],
        index=0
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1 minute", "5 minutes", "15 minutes"],
        index=0
    )
    
    # Date range
    end_date = datetime.date.today()
    start_date = st.sidebar.date_input(
        "Start Date",
        value=end_date - datetime.timedelta(days=7),
        max_value=end_date
    )
    
    # Initialize components
    data_ingestion = MarketDataIngestion(polygon_api_key)
    feature_eng = FeatureEngineering()
    signal_gen = AlphaSignalGenerator()
    backtester = BacktestingEngine()
    executor = ExecutionSimulator()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Data", 
        "🔧 Features", 
        "🎯 Alpha Signals", 
        "📈 Backtesting", 
        "⚡ Execution"
    ])
    
    with tab1:
        st.header("Market Data Ingestion")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("Fetch Data", type="primary"):
                with st.spinner("Fetching market data..."):
                    try:
                        # Convert timeframe
                        tf_map = {"1 minute": "1", "5 minutes": "5", "15 minutes": "15"}
                        multiplier = tf_map[timeframe]
                        
                        data = data_ingestion.get_market_data(
                            symbol, start_date, end_date, multiplier
                        )
                        
                        if data is not None and not data.empty:
                            st.session_state.data_cache[f"{symbol}_{timeframe}"] = data
                            st.success(f"Fetched {len(data)} records for {symbol}")
                        else:
                            st.error("No data received. Check your API key and symbol.")
                            
                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")
        
        # Display data if available
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in st.session_state.data_cache:
            data = st.session_state.data_cache[cache_key]
            
            # Display basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Records", len(data))
            with col2:
                st.metric("Price Range", f"${data['low'].min():.2f} - ${data['high'].max():.2f}")
            with col3:
                st.metric("Avg Volume", f"{data['volume'].mean():,.0f}")
            with col4:
                st.metric("Avg Spread", f"{((data['high'] - data['low']) / data['close'] * 100).mean():.3f}%")
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=symbol
            ))
            fig.update_layout(
                title=f"{symbol} Price Chart",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume chart
            fig_vol = px.bar(
                x=data.index,
                y=data['volume'],
                title=f"{symbol} Volume"
            )
            fig_vol.update_layout(height=300)
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Data table
            st.subheader("Recent Data")
            st.dataframe(data.tail(20), use_container_width=True)
    
    with tab2:
        st.header("Feature Engineering")
        
        cache_key = f"{symbol}_{timeframe}"
        if cache_key not in st.session_state.data_cache:
            st.warning("Please fetch market data first in the Market Data tab.")
        else:
            data = st.session_state.data_cache[cache_key]
            
            with st.spinner("Engineering features..."):
                try:
                    features_df = feature_eng.engineer_features(data)
                    
                    if features_df is not None:
                        st.success("Features engineered successfully!")
                        
                        # Feature summary
                        st.subheader("Feature Summary")
                        feature_cols = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Features", len(feature_cols))
                        with col2:
                            st.metric("Data Points", len(features_df))
                        
                        # Feature correlation heatmap
                        if len(feature_cols) > 1:
                            corr_matrix = features_df[feature_cols].corr()
                            fig_corr = px.imshow(
                                corr_matrix,
                                title="Feature Correlation Matrix",
                                color_continuous_scale="RdBu_r",
                                aspect="auto"
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Feature distributions
                        st.subheader("Feature Distributions")
                        selected_features = st.multiselect(
                            "Select features to visualize",
                            feature_cols,
                            default=feature_cols[:4] if len(feature_cols) >= 4 else feature_cols
                        )
                        
                        if selected_features:
                            fig_dist = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=selected_features[:4],
                                vertical_spacing=0.1
                            )
                            
                            for i, feature in enumerate(selected_features[:4]):
                                row = (i // 2) + 1
                                col = (i % 2) + 1
                                fig_dist.add_trace(
                                    go.Histogram(x=features_df[feature], name=feature, showlegend=False),
                                    row=row, col=col
                                )
                            
                            fig_dist.update_layout(height=500, title="Feature Distributions")
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Store features for next steps
                        st.session_state.data_cache[f"{cache_key}_features"] = features_df
                        
                        # Display feature data
                        st.subheader("Feature Data Preview")
                        st.dataframe(features_df.tail(20), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error engineering features: {str(e)}")
    
    with tab3:
        st.header("Alpha Signal Generation")
        
        cache_key = f"{symbol}_{timeframe}"
        features_key = f"{cache_key}_features"
        
        if features_key not in st.session_state.data_cache:
            st.warning("Please engineer features first in the Features tab.")
        else:
            features_df = st.session_state.data_cache[features_key]
            
            # Model selection
            col1, col2 = st.columns([3, 1])
            
            with col1:
                model_type = st.selectbox(
                    "Select Model Type",
                    ["Random Forest", "Logistic Regression", "XGBoost"],
                    index=0
                )
                
                lookback_periods = st.slider(
                    "Prediction Horizon (periods)",
                    min_value=1,
                    max_value=10,
                    value=3
                )
            
            with col2:
                if st.button("Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        try:
                            model, predictions, metrics = signal_gen.train_and_predict(
                                features_df, model_type, lookback_periods
                            )
                            
                            if model is not None:
                                st.session_state.model_trained = True
                                st.session_state.trained_model = model
                                st.session_state.predictions = predictions
                                st.session_state.model_metrics = metrics
                                st.success("Model trained successfully!")
                            
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
            
            # Display results if model is trained
            if st.session_state.model_trained and 'model_metrics' in st.session_state:
                metrics = st.session_state.model_metrics
                predictions = st.session_state.predictions
                
                # Model performance metrics
                st.subheader("Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                with col4:
                    st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
                
                # Confusion matrix
                if 'confusion_matrix' in metrics:
                    cm = metrics['confusion_matrix']
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Down', 'Flat', 'Up'],
                        y=['Down', 'Flat', 'Up'],
                        title="Confusion Matrix",
                        color_continuous_scale="Blues"
                    )
                    fig_cm.update_traces(text=cm, texttemplate="%{text}")
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                # Feature importance
                if 'feature_importance' in metrics:
                    importance_df = pd.DataFrame(metrics['feature_importance'])
                    fig_imp = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Feature Importance"
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # Signals visualization
                st.subheader("Generated Signals")
                if predictions is not None and len(predictions) > 0:
                    # Prepare data for visualization
                    viz_data = features_df.copy()
                    viz_data['signal'] = predictions
                    viz_data['signal_color'] = viz_data['signal'].map({0: 'red', 1: 'gray', 2: 'green'})
                    
                    fig_signals = go.Figure()
                    
                    # Price line
                    fig_signals.add_trace(go.Scatter(
                        x=viz_data.index,
                        y=viz_data['close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='blue')
                    ))
                    
                    # Signal markers
                    for signal_val, color, label in [(0, 'red', 'Sell'), (2, 'green', 'Buy')]:
                        signal_data = viz_data[viz_data['signal'] == signal_val]
                        if not signal_data.empty:
                            fig_signals.add_trace(go.Scatter(
                                x=signal_data.index,
                                y=signal_data['close'],
                                mode='markers',
                                name=f'{label} Signal',
                                marker=dict(color=color, size=8, symbol='triangle-up' if signal_val == 2 else 'triangle-down')
                            ))
                    
                    fig_signals.update_layout(
                        title="Price with Trading Signals",
                        xaxis_title="Time",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    st.plotly_chart(fig_signals, use_container_width=True)
                
                # Store predictions for backtesting
                st.session_state.data_cache[f"{cache_key}_signals"] = predictions
    
    with tab4:
        st.header("Backtesting")
        
        cache_key = f"{symbol}_{timeframe}"
        signals_key = f"{cache_key}_signals"
        
        if cache_key not in st.session_state.data_cache or signals_key not in st.session_state.data_cache:
            st.warning("Please generate alpha signals first.")
        else:
            data = st.session_state.data_cache[cache_key]
            signals = st.session_state.data_cache[signals_key]
            
            # Backtesting parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                initial_capital = st.number_input(
                    "Initial Capital ($)",
                    min_value=1000,
                    max_value=1000000,
                    value=100000,
                    step=1000
                )
            
            with col2:
                commission = st.number_input(
                    "Commission per Trade ($)",
                    min_value=0.0,
                    max_value=100.0,
                    value=1.0,
                    step=0.1
                )
            
            with col3:
                position_size = st.slider(
                    "Position Size (%)",
                    min_value=1,
                    max_value=100,
                    value=10
                )
            
            if st.button("Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    try:
                        results = backtester.run_backtest(
                            data, signals, initial_capital, commission, position_size / 100
                        )
                        
                        if results is not None:
                            st.session_state.backtest_results = results
                            st.success("Backtest completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error running backtest: {str(e)}")
            
            # Display backtest results
            if st.session_state.backtest_results is not None:
                results = st.session_state.backtest_results
                
                # Performance metrics
                st.subheader("Performance Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_return = (results['portfolio_value'].iloc[-1] / initial_capital - 1) * 100
                    st.metric("Total Return", f"{total_return:.2f}%")
                
                with col2:
                    annual_return = ((results['portfolio_value'].iloc[-1] / initial_capital) ** (252 / len(results)) - 1) * 100
                    st.metric("Annualized Return", f"{annual_return:.2f}%")
                
                with col3:
                    returns = results['returns'].dropna()
                    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                with col4:
                    max_dd = ((results['portfolio_value'] / results['portfolio_value'].cummax()) - 1).min() * 100
                    st.metric("Max Drawdown", f"{max_dd:.2f}%")
                
                # Equity curve
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=results.index,
                    y=results['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue')
                ))
                
                # Add benchmark (buy and hold)
                buy_hold_value = initial_capital * (data['close'] / data['close'].iloc[0])
                fig_equity.add_trace(go.Scatter(
                    x=buy_hold_value.index,
                    y=buy_hold_value.values,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='gray', dash='dash')
                ))
                
                fig_equity.update_layout(
                    title="Portfolio Performance vs Buy & Hold",
                    xaxis_title="Time",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Returns distribution
                if 'returns' in results.columns:
                    fig_returns = px.histogram(
                        results['returns'].dropna(),
                        title="Returns Distribution",
                        nbins=50
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
                
                # Detailed results table
                st.subheader("Detailed Results")
                st.dataframe(results.tail(50), use_container_width=True)
    
    with tab5:
        st.header("Execution Simulation")
        
        if st.session_state.backtest_results is None:
            st.warning("Please run a backtest first.")
        else:
            results = st.session_state.backtest_results
            
            # Execution parameters
            col1, col2 = st.columns(2)
            
            with col1:
                execution_type = st.selectbox(
                    "Execution Strategy",
                    ["Market Order", "Limit Order", "TWAP", "VWAP"],
                    index=0
                )
            
            with col2:
                slippage_bps = st.slider(
                    "Slippage (bps)",
                    min_value=0,
                    max_value=50,
                    value=5
                )
            
            if st.button("Simulate Execution", type="primary"):
                with st.spinner("Simulating execution..."):
                    try:
                        cache_key = f"{symbol}_{timeframe}"
                        data = st.session_state.data_cache[cache_key]
                        signals = st.session_state.data_cache[f"{cache_key}_signals"]
                        
                        execution_results = executor.simulate_execution(
                            data, signals, execution_type, slippage_bps / 10000
                        )
                        
                        if execution_results is not None:
                            st.success("Execution simulation completed!")
                            
                            # Execution metrics
                            st.subheader("Execution Analytics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                avg_slippage = execution_results['slippage'].mean() * 10000
                                st.metric("Avg Slippage (bps)", f"{avg_slippage:.1f}")
                            
                            with col2:
                                fill_rate = (execution_results['filled'] == True).mean() * 100
                                st.metric("Fill Rate", f"{fill_rate:.1f}%")
                            
                            with col3:
                                total_costs = execution_results['execution_cost'].sum()
                                st.metric("Total Execution Cost ($)", f"{total_costs:.2f}")
                            
                            with col4:
                                avg_latency = execution_results.get('latency', pd.Series([0])).mean()
                                st.metric("Avg Latency (ms)", f"{avg_latency:.1f}")
                            
                            # Slippage over time
                            fig_slippage = px.line(
                                x=execution_results.index,
                                y=execution_results['slippage'] * 10000,
                                title="Slippage Over Time (bps)"
                            )
                            st.plotly_chart(fig_slippage, use_container_width=True)
                            
                            # Execution cost breakdown
                            execution_summary = execution_results.groupby('signal').agg({
                                'execution_cost': 'sum',
                                'slippage': 'mean',
                                'filled': 'mean'
                            }).round(4)
                            
                            st.subheader("Execution Summary by Signal")
                            st.dataframe(execution_summary, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error simulating execution: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit | Market data powered by Polygon.io")

if __name__ == "__main__":
    main()
