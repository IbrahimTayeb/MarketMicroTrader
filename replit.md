# Systematic Trading Engine

## Overview

This is a real-time systematic trading engine that leverages machine learning to generate alpha signals from market microstructure data. The application uses high-frequency market data from the Polygon API to create predictive models and simulate algorithmic trade execution strategies.

The system is built as a Streamlit web application that provides an interactive interface for configuring trading parameters, visualizing market data, training ML models, and analyzing backtest results.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Interactive dashboard for configuration, visualization, and analysis
- **Plotly Visualizations**: Real-time charts including candlestick charts, volume analysis, and performance metrics
- **Session State Management**: Caches data and model states to improve performance

### Backend Architecture
- **Modular Python Design**: Separate modules for each major component (data ingestion, feature engineering, signal generation, backtesting, execution simulation)
- **Scikit-learn ML Pipeline**: Uses Random Forest, Logistic Regression, and XGBoost for signal generation
- **Real-time Data Processing**: Processes high-frequency market data into meaningful trading signals

### Data Processing Pipeline
1. **Data Ingestion**: Fetches OHLCV data from Polygon API
2. **Feature Engineering**: Creates technical indicators, microstructure features, and statistical measures
3. **Signal Generation**: ML models predict price direction (Up/Down/Flat)
4. **Backtesting**: Simulates strategy performance with realistic execution costs
5. **Execution Simulation**: Models different order types and execution strategies

## Key Components

### Data Ingestion (`data_ingestion.py`)
- **Purpose**: Connects to Polygon API and fetches market data
- **Features**: 
  - Minute-level OHLCV data retrieval
  - Error handling for API responses
  - Date range specification
- **Dependencies**: Polygon.io API key required

### Feature Engineering (`feature_engineering.py`)
- **Purpose**: Transforms raw market data into ML-ready features
- **Features**:
  - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
  - Microstructure features (spreads, imbalances, volatility)
  - Volume-based features (VWAP, volume ratios)
  - Statistical features (rolling correlations, momentum)
- **Technology**: Uses TA-Lib for technical analysis

### Alpha Signal Generation (`alpha_signals.py`)
- **Purpose**: Generates trading signals using machine learning
- **Models**: Random Forest, Logistic Regression, XGBoost
- **Target**: Predicts price direction over 3-period lookback
- **Features**: Dynamic threshold-based classification using volatility

### Backtesting Engine (`backtesting.py`)
- **Purpose**: Simulates trading strategy performance
- **Features**:
  - Portfolio management with cash and position tracking
  - Commission and slippage modeling
  - Performance metrics calculation
- **Configuration**: Configurable initial capital, commission rates, position sizing

### Execution Simulator (`execution_simulator.py`)
- **Purpose**: Models different execution strategies
- **Strategies**: Market orders, limit orders, TWAP, VWAP
- **Features**: Slippage modeling, execution timing analysis

### Utilities (`utils.py`)
- **Purpose**: Shared visualization and utility functions
- **Features**: Candlestick chart generation with volume overlay

## Data Flow

1. **Configuration**: User provides Polygon API key and trading parameters
2. **Data Retrieval**: System fetches historical market data for specified symbols and date ranges
3. **Feature Engineering**: Raw OHLCV data is transformed into 100+ technical and microstructure features
4. **Model Training**: ML models are trained on engineered features to predict price direction
5. **Signal Generation**: Trained models generate buy/sell/hold signals on new data
6. **Backtesting**: Signals are used to simulate trading with realistic execution costs
7. **Visualization**: Results are displayed through interactive charts and performance metrics

## External Dependencies

### Required APIs
- **Polygon.io**: Primary market data source (free tier available)
- API key must be provided for data access

### Python Libraries
- **Core**: pandas, numpy, streamlit
- **ML**: scikit-learn, xgboost
- **Technical Analysis**: TA-Lib (implied but may need installation)
- **Visualization**: plotly
- **Data**: requests for API calls

### System Requirements
- Python 3.11+
- Internet connection for API access
- Streamlit server capability

## Deployment Strategy

### Replit Configuration
- **Runtime**: Python 3.11 with Nix package management
- **Deployment**: Autoscale deployment target
- **Port**: Application runs on port 5000
- **Entry Point**: `streamlit run app.py --server.port 5000`

### Package Management
- Uses `pyproject.toml` for Python dependency management
- Core dependency: streamlit>=1.46.0
- Additional packages managed through Nix for system-level dependencies

### Configuration
- Streamlit configured for headless operation
- Server bound to 0.0.0.0:5000 for external access
- Light theme by default

## Changelog

```
Changelog:
- June 25, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```