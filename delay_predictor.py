import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from calibration import ProbabilityCalibrator
import warnings
warnings.filterwarnings('ignore')

class DelayPredictor:
    """
    Machine learning model for transit delay prediction
    """
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.shap_explainer = None
        self.calibrator = None
        self.use_calibrated = True
        self.is_trained = False
        self._X_valid = None
        self._y_valid = None
        self.calibration_curve_data = None
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Prepare training and testing data"""
        print("Preparing training and testing data...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale numerical features (excluding binary and categorical)
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")
        print(f"Delay rate: {y.mean():.2%}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        # Compute class imbalance weight
        num_pos = int((y_train == 1).sum())
        num_neg = int((y_train == 0).sum())
        scale_pos_weight = (num_neg / max(1, num_pos)) if num_pos > 0 else 1.0
        
        # Define parameter grid for hyperparameter tuning (kept small for speed)
        param_grid = {
            'n_estimators': [200],
            'max_depth': [6, 8],
            'learning_rate': [0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8]
        }
        
        # Initialize base model (no early stopping inside CV)
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            eval_metric='aucpr',
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight
        )
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='average_precision', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        # Train final model with early stopping on validation set
        self.model.set_params(early_stopping_rounds=10)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        print(f"Best parameters: {grid_search.best_params_}")
        
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        print("Training LightGBM model...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # Initialize base model
        base_model = lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            verbose=-1
        )
        
        # Grid search for best parameters
        grid_search = GridSearchCV(
            base_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        # Train final model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        print(f"Best parameters: {grid_search.best_params_}")
        
    def train(self, X, y, test_size=0.2):
        """Train the delay prediction model"""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, test_size)
        # Save validation for threshold tuning and calibration
        self._X_valid, self._y_valid = X_test.copy(), y_test.copy()
        
        # Train model based on type
        if self.model_type == 'xgboost':
            self.train_xgboost(X_train, y_train, X_test, y_test)
        elif self.model_type == 'lightgbm':
            self.train_lightgbm(X_train, y_train, X_test, y_test)
        else:
            raise ValueError("Model type must be 'xgboost' or 'lightgbm'")
        
        # Evaluate model
        self.evaluate_model(X_test, y_test)

        # Probability calibration on validation set for better probabilities
        try:
            self.calibrator = ProbabilityCalibrator(method='isotonic', cv='prefit').fit(self.model, X_test, y_test)
            # Recompute calibrated AUC
            from sklearn.metrics import roc_auc_score
            y_pred_proba_cal = self.calibrator.predict_proba(X_test)[:, 1]
            self.metrics['auc_score_calibrated'] = roc_auc_score(y_test, y_pred_proba_cal)
        except Exception:
            self.calibrator = None

        # Suggest operating threshold using PR curve (maximize F1)
        try:
            best_thr, best_f1 = self.suggest_threshold()
            self.metrics['best_threshold'] = float(best_thr)
            self.metrics['best_f1'] = float(best_f1)
        except Exception:
            pass

        # Compute calibration curve and Brier score (using calibrated probs if available)
        try:
            from sklearn.metrics import brier_score_loss
            from sklearn.calibration import calibration_curve
            probs_valid = self.predict_delay_risk(self._X_valid)
            self.metrics['brier_score'] = float(brier_score_loss(self._y_valid, probs_valid))
            frac_pos, mean_pred = calibration_curve(self._y_valid, probs_valid, n_bins=10, strategy='uniform')
            self.calibration_curve_data = {
                'fraction_of_positives': frac_pos.tolist(),
                'mean_predicted_value': mean_pred.tolist()
            }
        except Exception:
            self.calibration_curve_data = None

        # Create SHAP explainer
        self.create_shap_explainer(X_train)
        
        self.is_trained = True
        print("Model training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nModel Evaluation:")
        print("=" * 50)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'confusion_matrix': cm
        }
        
    def create_shap_explainer(self, X_train, sample_size=1000):
        """Create SHAP explainer for feature importance"""
        print("Creating SHAP explainer...")
        
        # Sample data for SHAP (to avoid memory issues)
        if len(X_train) > sample_size:
            X_sample = X_train.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_train
            
        if self.model_type == 'xgboost':
            self.shap_explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'lightgbm':
            self.shap_explainer = shap.TreeExplainer(self.model)
            
        print("SHAP explainer created")
        
    def get_feature_importance(self, top_n=20):
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        elif self.model_type == 'lightgbm':
            importance = self.model.feature_importances_
            
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)
        
    def get_shap_values(self, X, sample_size=1000):
        """Get SHAP values for feature explanation"""
        if not self.is_trained or self.shap_explainer is None:
            raise ValueError("Model must be trained and SHAP explainer must be created")
            
        # Sample data if too large
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
            
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X_sample)
        
        return shap_values, X_sample
        
    def predict_delay_risk(self, X):
        """Predict delay risk for new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Scale features
        X_scaled = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X_scaled[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        # Make predictions
        if self.use_calibrated and self.calibrator is not None:
            delay_prob = self.calibrator.predict_proba(X_scaled)[:, 1]
        else:
            delay_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        return delay_prob

    def suggest_threshold(self):
        """Return threshold that maximizes F1 on validation set using calibrated probs if available."""
        from sklearn.metrics import precision_recall_curve
        if self._X_valid is None or self._y_valid is None:
            raise ValueError("No validation set stored")
        probs = self.predict_delay_risk(self._X_valid)
        precision, recall, thresholds = precision_recall_curve(self._y_valid, probs)
        f1 = 2 * (precision * recall) / np.clip(precision + recall, 1e-9, None)
        idx = int(np.nanargmax(f1))
        best_thr = float(thresholds[max(0, idx-1)]) if len(thresholds) else 0.5
        best_f1 = float(np.nanmax(f1))
        return best_thr, best_f1
        
    def identify_high_risk_segments(self, stop_times_data, threshold=0.7):
        """Identify high-risk segments based on predicted delay probability"""
        print("Identifying high-risk segments...")
        
        # Prepare features for prediction - must mirror training features
        feature_columns = [
            'route_id_code', 'stop_id_code', 'route_freq', 'stop_freq',
            'stop_sequence', 'stop_sequence_normalized',
            'arrival_hour', 'arrival_minute', 'departure_hour', 'departure_minute',
            'dwell_time_seconds', 'dwell_time_normalized', 'travel_time_seconds',
            'distance_km', 'speed_kmh', 'speed_normalized',
            'total_stops', 'total_travel_time', 'total_distance', 'total_dwell_time',
            'is_peak_hour', 'is_bus', 'is_train', 'is_first_stop', 'is_last_stop',
            'long_dwell', 'slow_speed', 'trip_length_category'
        ]
        
        X_pred = stop_times_data[feature_columns].copy()
        
        # Handle categorical variables (only small one)
        categorical_cols = ['trip_length_category']
        X_pred_encoded = pd.get_dummies(X_pred, columns=categorical_cols, drop_first=True)
        
        # Ensure all columns from training are present
        missing_cols = set(self.feature_names) - set(X_pred_encoded.columns)
        for col in missing_cols:
            X_pred_encoded[col] = 0
        
        # Reorder columns to match training data
        X_pred_encoded = X_pred_encoded[self.feature_names]
        
        # Fill missing values
        X_pred_encoded = X_pred_encoded.fillna(0)
        
        # Predict delay risk
        delay_risk = self.predict_delay_risk(X_pred_encoded)
        
        # Add predictions to original data
        result_data = stop_times_data.copy()
        result_data['delay_risk'] = delay_risk
        result_data['high_risk'] = delay_risk > threshold
        
        # Identify high-risk segments
        high_risk_segments = result_data[result_data['high_risk']].copy()
        
        print(f"Identified {len(high_risk_segments)} high-risk segments")
        
        return result_data, high_risk_segments
        
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'trained_at': datetime.now()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.metrics = model_data.get('metrics', {})
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
        
    def plot_feature_importance(self, top_n=20, figsize=(12, 8)):
        """Plot feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        feature_importance = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - {self.model_type.upper()}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
    def plot_roc_curve(self, X_test, y_test, figsize=(8, 6)):
        """Plot ROC curve"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show() 