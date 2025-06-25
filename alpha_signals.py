import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class AlphaSignalGenerator:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_target_variable(self, data, lookback_periods=3):
        """
        Create target variable for price direction prediction
        """
        try:
            df = data.copy()
            
            # Calculate future returns
            future_returns = df['close'].shift(-lookback_periods) / df['close'] - 1
            
            # Define thresholds for classification
            # Using dynamic thresholds based on volatility
            volatility = df['returns'].rolling(20).std()
            threshold = volatility * 0.5  # Half a standard deviation
            
            # Create target classes
            # 0: Down (negative return beyond threshold)
            # 1: Flat (within threshold range)  
            # 2: Up (positive return beyond threshold)
            target = np.where(
                future_returns > threshold, 2,
                np.where(future_returns < -threshold, 0, 1)
            )
            
            df['target'] = target
            df['future_returns'] = future_returns
            
            return df
            
        except Exception as e:
            raise Exception(f"Error preparing target variable: {str(e)}")
    
    def select_features(self, data):
        """
        Select relevant features for model training
        """
        try:
            # Exclude non-feature columns
            exclude_cols = [
                'open', 'high', 'low', 'close', 'volume', 
                'target', 'future_returns', 'typical_price'
            ]
            
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            # Remove features with too many NaN values
            nan_threshold = 0.1
            valid_features = []
            
            for col in feature_cols:
                nan_ratio = data[col].isna().sum() / len(data)
                if nan_ratio < nan_threshold:
                    valid_features.append(col)
            
            return valid_features
            
        except Exception as e:
            raise Exception(f"Error selecting features: {str(e)}")
    
    def train_and_predict(self, data, model_type="Random Forest", lookback_periods=3):
        """
        Train model and generate predictions
        """
        try:
            # Prepare data
            df = self.prepare_target_variable(data, lookback_periods)
            
            # Select features
            feature_columns = self.select_features(df)
            self.feature_columns = feature_columns
            
            if len(feature_columns) == 0:
                raise Exception("No valid features found")
            
            # Prepare training data
            X = df[feature_columns].copy()
            y = df['target'].copy()
            
            # Remove rows with NaN values
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 50:
                raise Exception("Not enough valid data points for training")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            test_size = min(0.3, max(0.1, 50 / len(X)))  # Adaptive test size
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Initialize model
            if model_type == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            elif model_type == "Logistic Regression":
                model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                )
            elif model_type == "XGBoost":
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                raise Exception(f"Unknown model type: {model_type}")
            
            # Train model
            model.fit(X_train, y_train)
            self.model = model
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                metrics['feature_importance'] = importance_df
            
            # Generate predictions for entire dataset
            full_predictions = model.predict(X_scaled)
            
            # Create prediction array aligned with original data
            predictions = np.full(len(df), 1)  # Default to flat
            predictions[valid_mask] = full_predictions
            
            return model, predictions, metrics
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")
    
    def predict_live(self, current_data):
        """
        Generate live predictions for new data
        """
        try:
            if self.model is None:
                raise Exception("Model not trained")
            
            if self.feature_columns is None:
                raise Exception("Feature columns not defined")
            
            # Prepare features
            X = current_data[self.feature_columns].copy()
            
            # Handle missing features
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            
            # Remove rows with NaN values
            X = X.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)
            prediction_proba = self.model.predict_proba(X_scaled)
            
            return prediction[0], prediction_proba[0]
            
        except Exception as e:
            raise Exception(f"Error making live prediction: {str(e)}")
    
    def get_signal_interpretation(self, signal):
        """
        Convert signal to human-readable interpretation
        """
        signal_map = {
            0: "SELL",
            1: "HOLD", 
            2: "BUY"
        }
        return signal_map.get(signal, "UNKNOWN")
    
    def analyze_signal_performance(self, data, signals):
        """
        Analyze historical signal performance
        """
        try:
            df = data.copy()
            df['signal'] = signals
            
            # Calculate forward returns for each signal
            forward_returns = {}
            signal_counts = {}
            
            for signal in [0, 1, 2]:
                signal_mask = df['signal'] == signal
                if signal_mask.any():
                    signal_data = df[signal_mask]
                    forward_ret = signal_data['returns'].shift(-1).dropna()
                    
                    forward_returns[signal] = {
                        'mean': forward_ret.mean(),
                        'std': forward_ret.std(),
                        'sharpe': forward_ret.mean() / forward_ret.std() if forward_ret.std() > 0 else 0,
                        'win_rate': (forward_ret > 0).mean()
                    }
                    signal_counts[signal] = len(signal_data)
            
            return forward_returns, signal_counts
            
        except Exception as e:
            raise Exception(f"Error analyzing signal performance: {str(e)}")
