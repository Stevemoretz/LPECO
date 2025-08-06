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
import optuna
import json
import hashlib
import pingouin as pg
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

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

def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost model with given parameters."""
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy, model

def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM model with given parameters."""
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'verbose': -1
        }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy, model

def benchmark_gbdt(dataset_name, X, y, cv_folds=5):
    """Benchmark GBDT models on a dataset."""
    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    xgb_scores = []
    lgb_scores = []
    xgb_times = []
    lgb_times = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Preprocess data
        X_train, y_train = preprocess_data(X_train, y_train)
        X_val, y_val = preprocess_data(X_val, y_val)
        
        # XGBoost
        start_time = time.time()
        xgb_acc, _ = train_xgboost(X_train, y_train, X_val, y_val)
        xgb_times.append(time.time() - start_time)
        xgb_scores.append(xgb_acc)
        
        # LightGBM
        start_time = time.time()
        lgb_acc, _ = train_lightgbm(X_train, y_train, X_val, y_val)
        lgb_times.append(time.time() - start_time)
        lgb_scores.append(lgb_acc)
    
    return {
        'XGBoost': (np.mean(xgb_scores), np.std(xgb_scores), np.mean(xgb_times)),
        'LightGBM': (np.mean(lgb_scores), np.std(lgb_scores), np.mean(lgb_times))
    }

def load_cache():
    """Load cached results if available."""
    cache_file = 'cached_algorithms_results'
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_to_cache(opt_hyperparams, opt_name, dataset, model_type, score, training_time):
    """Save results to cache."""
    cache_file = 'cached_algorithms_results'
    cache = load_cache()
    
    key = f"{opt_name}_{dataset}_{model_type}_{opt_hyperparams}"
    cache[key] = {
        'score': score,
        'training_time': training_time,
        'hyperparams': opt_hyperparams
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache, f)

def generate_hyperparams_str(param_dict):
    """Generate a stable hash string from hyperparameters."""
    sorted_params = sorted(param_dict.items())
    param_str = '_'.join([f"{k}_{v}" for k, v in sorted_params])
    return hashlib.md5(param_str.encode()).hexdigest()

def optimize_lpeco_hyperparams(X, y, n_trials=50):
    """Optimize LPECO hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'beta_1': trial.suggest_float('beta_1', 0.85, 0.999),
            'beta_2': trial.suggest_float('beta_2', 0.98, 0.9999),
            'error_decay': trial.suggest_float('error_decay', 0.98, 1.0),
            'correction_strength': trial.suggest_float('correction_strength', 0.01, 0.5)
        }
        
        # Quick evaluation
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            X_train, y_train = preprocess_data(X_train, y_train)
            X_val, y_val = preprocess_data(X_val, y_val)
            
            model = create_simple_model(X_train.shape[1], len(np.unique(y_train)))
            model.compile(
                optimizer=LionPredictiveErrorCorrectionOptimizer(**params),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                              epochs=50, verbose=0)
            
            scores.append(max(history.history['val_accuracy']))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value

def friedman_test(results_df):
    """Perform Friedman test for multiple algorithm comparison."""
    try:
        # Prepare data for Friedman test
        algorithms = results_df['algorithm'].unique()
        datasets = results_df['dataset'].unique()
        
        # Create matrix: datasets x algorithms
        data_matrix = []
        for dataset in datasets:
            dataset_scores = []
            for alg in algorithms:
                score = results_df[(results_df['dataset'] == dataset) & 
                                 (results_df['algorithm'] == alg)]['score'].iloc[0]
                dataset_scores.append(score)
            data_matrix.append(dataset_scores)
        
        data_matrix = np.array(data_matrix)
        
        # Perform Friedman test
        statistic, p_value = stats.friedmanchisquare(*data_matrix.T)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'algorithms': algorithms.tolist(),
            'datasets': datasets.tolist()
        }
    except Exception as e:
        print(f"Friedman test failed: {e}")
        return None

def equivalence_test(scores1, scores2, margin=0.02):
    """Perform Two One-Sided Tests (TOST) for equivalence."""
    try:
        # Calculate means and standard errors
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        se1, se2 = np.std(scores1) / np.sqrt(len(scores1)), np.std(scores2) / np.sqrt(len(scores2))
        
        # Calculate difference and its standard error
        diff = mean1 - mean2
        se_diff = np.sqrt(se1**2 + se2**2)
        
        # TOST procedure
        t1 = (diff - (-margin)) / se_diff
        t2 = (diff - margin) / se_diff
        
        # Degrees of freedom
        df = len(scores1) + len(scores2) - 2
        
        # P-values for one-sided tests
        p1 = 1 - stats.t.cdf(t1, df)
        p2 = stats.t.cdf(t2, df)
        
        # Equivalence if both p-values < 0.05
        equivalent = max(p1, p2) < 0.05
        
        return {
            'equivalent': equivalent,
            'p_value': max(p1, p2),
            'difference': diff,
            'margin': margin,
            'confidence_interval': [diff - 1.96*se_diff, diff + 1.96*se_diff]
        }
    except Exception as e:
        print(f"Equivalence test failed: {e}")
        return None

def bayesian_rope_analysis(scores1, scores2, rope_margin=0.02):
    """Perform Bayesian ROPE analysis for practical equivalence."""
    try:
        # Simple Bayesian analysis using normal approximation
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        var1, var2 = np.var(scores1), np.var(scores2)
        n1, n2 = len(scores1), len(scores2)
        
        # Posterior distribution parameters
        post_mean = mean1 - mean2
        post_var = var1/n1 + var2/n2
        post_std = np.sqrt(post_var)
        
        # Calculate probability within ROPE
        prob_in_rope = stats.norm.cdf(rope_margin, post_mean, post_std) - \
                      stats.norm.cdf(-rope_margin, post_mean, post_std)
        
        return {
            'prob_in_rope': prob_in_rope,
            'posterior_mean': post_mean,
            'posterior_std': post_std,
            'rope_margin': rope_margin
        }
    except Exception as e:
        print(f"Bayesian ROPE analysis failed: {e}")
        return None

def create_analysis_plots(results_df, output_dir='analysis/plots'):
    """Create comprehensive analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance distribution plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=results_df, x='algorithm', y='score')
    plt.title('Performance Distribution by Algorithm')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean algorithm rank
    mean_ranks = results_df.groupby('algorithm')['score'].mean().rank(ascending=False)
    plt.figure(figsize=(10, 6))
    mean_ranks.plot(kind='bar')
    plt.title('Mean Algorithm Rank')
    plt.ylabel('Mean Rank')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mean_algorithm_rank.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis plots saved to {output_dir}/")

def plot_equivalence_analysis(scores1, scores2, algorithm1, algorithm2, output_dir='analysis/plots'):
    """Create equivalence analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot comparison
    data = [scores1, scores2]
    labels = [algorithm1, algorithm2]
    ax1.boxplot(data, labels=labels)
    ax1.set_title(f'Performance Comparison: {algorithm1} vs {algorithm2}')
    ax1.set_ylabel('Accuracy')
    
    # Distribution plot
    ax2.hist(scores1, alpha=0.7, label=algorithm1, bins=10)
    ax2.hist(scores2, alpha=0.7, label=algorithm2, bins=10)
    ax2.set_title(f'Score Distributions: {algorithm1} vs {algorithm2}')
    ax2.set_xlabel('Accuracy')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/equivalence_analysis_{algorithm1}_vs_{algorithm2}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("LPECO: Lion with Predictive Error Correction")
    print("Added comprehensive benchmarking framework")
