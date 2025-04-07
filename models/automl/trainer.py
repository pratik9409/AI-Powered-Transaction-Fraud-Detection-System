# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.ensemble import IsolationForest
# from sklearn.metrics import roc_auc_score
# import joblib
# import mlflow
# from datetime import datetime
# import time
# import os
# import logging

# class AutoMLTrainer:
#     def __init__(self, data_path="data/bank_transactions_data_2.csv", experiment_name="fraud_detection"):
#         self.data_path = data_path
#         self.experiment_name = experiment_name
#         self.logger = logging.getLogger(__name__)
        
#         # Ensure data directory exists
#         os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
#         # Initialize MLflow
#         self._init_mlflow()
        
#     def _init_mlflow(self):
#         """Initialize MLflow tracking"""
#         try:
#             # Set tracking URI (default to local if not set)
#             mlflow.set_tracking_uri("http://localhost:5000")
            
#             # Create experiment if it doesn't exist
#             if not mlflow.get_experiment_by_name(self.experiment_name):
#                 mlflow.create_experiment(self.experiment_name)
            
#             mlflow.set_experiment(self.experiment_name)
#             self.logger.info(f"MLflow experiment '{self.experiment_name}' initialized")
#         except Exception as e:
#             self.logger.warning(f"Failed to initialize MLflow: {str(e)}")
#             # Fallback to local tracking
#             mlflow.set_tracking_uri("file:///tmp/mlruns")
#             if not mlflow.get_experiment_by_name(self.experiment_name):
#                 mlflow.create_experiment(self.experiment_name)
#             mlflow.set_experiment(self.experiment_name)

#     def _load_data(self):
#         """Load or create sample data if none exists"""
#         if not os.path.exists(self.data_path):
#             self.logger.warning(f"Data file not found at {self.data_path}, creating sample data")
#             # Create sample data
#             sample_data = {
#                 'TransactionAmount': [100.0, 200.0, 50.0, 1000.0, 150.0],
#                 'TransactionDuration': [60, 120, 30, 180, 90],
#                 'LoginAttempts': [1, 2, 1, 3, 1],
#                 'is_fraud': [0, 0, 1, 1, 0]
#             }
#             df = pd.DataFrame(sample_data)
#             df.to_csv(self.data_path, index=False)
#         return pd.read_csv(self.data_path)

#     def preprocess_data(self):
#         """Preprocess the data for training"""
#         try:
#             df = self._load_data()
            
#             # Basic validation
#             if 'is_fraud' not in df.columns:
#                 raise ValueError("Target column 'is_fraud' missing in dataset")
                
#             X = df.drop(['is_fraud'], axis=1)
#             y = df['is_fraud']
#             return train_test_split(X, y, test_size=0.2, random_state=42)
            
#         except Exception as e:
#             self.logger.error(f"Error in data preprocessing: {str(e)}")
#             raise

#     def train_models(self):
#         """Train and evaluate all models"""
#         try:
#             X_train, X_test, y_train, y_test = self.preprocess_data()
            
#             models = {
#                 "xgboost": XGBClassifier(),
#                 "random_forest": RandomForestClassifier(),
#                 "isolation_forest": IsolationForest()
#             }
            
#             best_score = 0
#             best_model = None
            
#             for name, model in models.items():
#                 with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
#                     try:
#                         # Train model
#                         model.fit(X_train, y_train)
                        
#                         # Evaluate
#                         if name == "isolation_forest":
#                             y_pred = -model.decision_function(X_test)
#                         else:
#                             y_pred = model.predict_proba(X_test)[:, 1]
                        
#                         score = roc_auc_score(y_test, y_pred)
                        
#                         # Log metrics and model
#                         mlflow.log_metric("roc_auc", score)
#                         mlflow.sklearn.log_model(model, name)
                        
#                         if score > best_score:
#                             best_score = score
#                             best_model = model
#                             best_name = name
                            
#                     except Exception as e:
#                         self.logger.error(f"Error training {name}: {str(e)}")
#                         continue
            
#             if best_model is None:
#                 raise RuntimeError("All model training attempts failed")
                
#             # Save the best model
#             os.makedirs("models", exist_ok=True)
#             joblib.dump(best_model, f"models/{best_name}.pkl")
#             self.logger.info(f"Best model: {best_name} with score: {best_score:.4f}")
#             return best_model, best_score
            
#         except Exception as e:
#             self.logger.error(f"Training failed: {str(e)}")
#             raise

#     def retrain_schedule(self, interval_days=7):
#         """Schedule periodic retraining"""
#         while True:
#             try:
#                 self.logger.info("Starting scheduled retraining...")
#                 self.train_models()
#                 self.logger.info(f"Retraining completed. Next in {interval_days} days")
#             except Exception as e:
#                 self.logger.error(f"Retraining failed: {str(e)}")
#             finally:
#                 time.sleep(interval_days * 24 * 60 * 60)



# automl/trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import joblib
import mlflow
from datetime import datetime
import time
import os
import logging
import numpy as np

class AutoMLTrainer:
    def __init__(self, data_path="data/bank_transactions_data_2.csv", experiment_name="fraud_detection"):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Initialize MLflow
        self._init_mlflow()

    def _init_mlflow(self):
        """Initialize MLflow tracking"""
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            if not mlflow.get_experiment_by_name(self.experiment_name):
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            self.logger.warning(f"MLflow initialization failed, using local tracking: {str(e)}")
            mlflow.set_tracking_uri("file:///tmp/mlruns")

    def _generate_fraud_labels(self, df):
        """
        Generate synthetic fraud labels based on transaction patterns
        Modify these rules based on your actual fraud patterns
        """
        # Example rules (adjust based on your domain knowledge)
        conditions = [
            (df['TransactionAmount'] > 500) & (df['TransactionType'] == 'Debit'),
            (df['LoginAttempts'] > 2),
            (df['Channel'] == 'Online') & (df['TransactionDuration'] < 30),
            (df['Location'].str.contains('Foreign'))  # If you have international transactions
        ]
        
        # Start with all non-fraud (0)
        fraud_labels = np.zeros(len(df))
        
        # Apply conditions to flag potential fraud
        for condition in conditions:
            fraud_labels[condition] = 1
            
        # Ensure we have at least some fraud cases
        if sum(fraud_labels) == 0:
            fraud_labels[:min(3, len(df))] = 1  # Mark first 3 as fraud if none detected
            
        return fraud_labels

    def _prepare_data(self, df):
        """Prepare the data for training"""
        # Generate fraud labels if not present
        if 'is_fraud' not in df.columns:
            self.logger.warning("'is_fraud' column not found, generating synthetic labels")
            df['is_fraud'] = self._generate_fraud_labels(df)
        
        # Feature engineering
        df['TransactionSpeed'] = df['TransactionAmount'] / df['TransactionDuration']
        
        # Select features - modify based on what's available in your data
        feature_cols = [
            'TransactionAmount', 'TransactionDuration', 'LoginAttempts',
            'AccountBalance', 'TransactionSpeed', 'CustomerAge'
        ]
        
        # Only use columns that exist in the data
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols]
        y = df['is_fraud']
        
        return X, y

    def preprocess_data(self):
        """Load and preprocess the data"""
        try:
            df = pd.read_csv(self.data_path)
            
            # Convert date columns to datetime
            date_cols = ['TransactionDate', 'PreviousTransactionDate']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Calculate days since last transaction if possible
            if 'TransactionDate' in df.columns and 'PreviousTransactionDate' in df.columns:
                df['DaysSinceLastTransaction'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.days
            
            return self._prepare_data(df)
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def train_models(self):
        """Train and evaluate models"""
        try:
            X, y = self.preprocess_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            models = {
                "xgboost": XGBClassifier(),
                "random_forest": RandomForestClassifier(),
                "isolation_forest": IsolationForest(contamination=float(y_train.mean()))
            }
            
            best_score = 0
            best_model = None
            
            for name, model in models.items():
                with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    try:
                        model.fit(X_train, y_train)
                        
                        if name == "isolation_forest":
                            y_pred = -model.decision_function(X_test)
                        else:
                            y_pred = model.predict_proba(X_test)[:, 1]
                        
                        score = roc_auc_score(y_test, y_pred)
                        mlflow.log_metric("roc_auc", score)
                        mlflow.sklearn.log_model(model, name)
                        
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_name = name
                            
                    except Exception as e:
                        self.logger.error(f"Error training {name}: {str(e)}")
                        continue
            
            if best_model is None:
                raise RuntimeError("All model training attempts failed")
                
            # Save the best model
            model_path = f"models/{best_name}.pkl"
            joblib.dump(best_model, model_path)
            self.logger.info(f"Saved best model ({best_name}) to {model_path}")
            
            return best_model, best_score
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def retrain_schedule(self, interval_days=7):
        """Schedule periodic retraining"""
        while True:
            try:
                self.logger.info("Starting scheduled retraining...")
                self.train_models()
                self.logger.info(f"Retraining completed. Next in {interval_days} days")
            except Exception as e:
                self.logger.error(f"Retraining failed: {str(e)}")
            finally:
                time.sleep(interval_days * 24 * 60 * 60)