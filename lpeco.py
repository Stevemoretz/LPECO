"""
Basic Lion Optimizer Implementation
Author: Siavash Mobarhan
"""

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import time
import logging
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb

class LionPredictiveErrorCorrectionOptimizer(keras.optimizers.Optimizer):
    """Lion optimizer with predictive error correction based on PI control theory."""
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.99, 
                 error_decay=0.99, correction_strength=0.1, name="LPECO", **kwargs):
        super().__init__(name=name, **kwargs)
        self._learning_rate = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._error_decay = error_decay
        self._correction_strength = correction_strength
        
        # Register hyperparameters
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("error_decay", error_decay)
        self._set_hyper("correction_strength", correction_strength)
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "I")  # Integral term for error correction
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._get_hyper("learning_rate", var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        error_decay_t = self._get_hyper("error_decay", var_dtype)
        correction_strength_t = self._get_hyper("correction_strength", var_dtype)
        
        m = self.get_slot(var, "m")
        I = self.get_slot(var, "I")
        
        # Calculate prediction error (excess momentum)
        e_t = grad + m
        
        # Update error EMA (Integral term)
        I_update = error_decay_t * I + (1 - error_decay_t) * e_t
        
        # Lion-style interpolation (Proportional term)
        u_t = beta_1_t * m + (1 - beta_1_t) * grad
        
        # Correct the update with integral term
        u_t_corrected = u_t - correction_strength_t * I_update
        
        # Final update direction
        d_t = tf.sign(u_t_corrected)
        
        var_update = var - lr_t * d_t
        m_update = beta_2_t * m + (1 - beta_2_t) * grad
        
        return tf.group(
            var.assign(var_update),
            m.assign(m_update),
            I.assign(I_update)
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "error_decay": self._serialize_hyperparameter("error_decay"),
            "correction_strength": self._serialize_hyperparameter("correction_strength"),
        })
        return config

def create_simple_model(input_dim, num_classes):
    """Create a simple neural network model."""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100):
    """Train a model and return training history."""
    model.compile(
        optimizer=LionPredictiveErrorCorrectionOptimizer(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        verbose=0
    )
    
    return history.history

def load_openml_tabular_datasets():
    """Load tabular datasets from OpenML for benchmarking."""
    try:
        import openml
        datasets = {}
        
        # Load small datasets for initial testing
        dataset_ids = [31, 37, 44, 46, 50, 54, 60, 61]  # Common tabular datasets
        
        for dataset_id in dataset_ids:
            try:
                dataset = openml.datasets.get_dataset(dataset_id)
                X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
                datasets[dataset.name] = {'X': X, 'y': y}
        except Exception as e:
                print(f"Failed to load dataset {dataset_id}: {e}")
                continue
    
    return datasets
    except ImportError:
        print("OpenML not available, using synthetic data")
        return {}

def preprocess_data(X, y):
    """Basic data preprocessing for tabular data."""
    # Handle missing values
    if hasattr(X, 'fillna'):
        X = X.fillna(X.mean())
    
    # Encode categorical variables
    le = LabelEncoder()
    if y.dtype == 'object':
        y = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def benchmark_optimizer(optimizer_name, X, y, cv_folds=5):
    """Benchmark an optimizer on a dataset."""
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    times = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Preprocess data
        X_train, y_train = preprocess_data(X_train, y_train)
        X_val, y_val = preprocess_data(X_val, y_val)
        
        # Create and train model
        model = create_simple_model(X_train.shape[1], len(np.unique(y_train)))
        
        start_time = time.time()
        history = train_model(model, X_train, y_train, X_val, y_val)
        end_time = time.time()
        
        # Get best validation accuracy
        best_val_acc = max(history['val_accuracy'])
        scores.append(best_val_acc)
        times.append(end_time - start_time)
    
    return np.mean(scores), np.std(scores), np.mean(times)

if __name__ == "__main__":
    print("LPECO: Lion with Predictive Error Correction")
    print("Added comprehensive benchmarking framework")
