"""
LPECO: Lion with Predictive Error Correction Optimizer
A novel optimizer that enhances the Lion algorithm through principled error correction.

This module provides a comprehensive benchmark framework for comparing neural network
optimizers against gradient boosted decision trees on tabular data.

Author: Siavash Mobarhan, Sogand Tatlari
License: MIT
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
import argparse
from typing import Union, Dict, List, Any, Optional

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

def generate_meta_analysis_plot(results_df, output_dir='analysis/plots'):
    """Generate meta-analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate class balance for each dataset
    class_balance = []
    lpeco_advantages = []
    
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        
        # Calculate class balance (simplified)
        lpeco_score = dataset_data[dataset_data['algorithm'] == 'LPECO']['score'].mean()
        gbdt_scores = dataset_data[dataset_data['algorithm'].isin(['XGBoost', 'LightGBM'])]['score'].mean()
        
        advantage = lpeco_score - gbdt_scores
        lpeco_advantages.append(advantage)
        
        # Simplified class balance calculation
        balance = 0.5  # Placeholder - would need actual class distribution
        class_balance.append(balance)
    
    # Create meta-analysis plot
    plt.figure(figsize=(10, 6))
    plt.scatter(class_balance, lpeco_advantages)
    plt.xlabel('Class Balance')
    plt.ylabel('LPECO Advantage')
    plt.title('Meta-Analysis: LPECO Advantage vs Class Balance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/meta_analysis_class_balance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_convergence_plots(results_df, output_dir='analysis/plots'):
    """Create convergence analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # This would require storing epoch-by-epoch data
    # For now, create a placeholder plot
    plt.figure(figsize=(12, 8))
    plt.title('Training Convergence Analysis')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_speed_accuracy_plots(results_df, output_dir='analysis/plots'):
    """Create speed vs accuracy plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    for algorithm in results_df['algorithm'].unique():
        alg_data = results_df[results_df['algorithm'] == algorithm]
        plt.scatter(alg_data['training_time'], alg_data['score'], 
                   label=algorithm, alpha=0.7, s=50)
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Speed vs Accuracy Trade-off')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
    plt.savefig(f'{output_dir}/speed_vs_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

class OptimizerBenchmark:
    """Comprehensive benchmarking system for optimizers and GBDT models."""
    
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.results = []
        self.cache = load_cache() if use_cache else {}
        
    def run_full_benchmark(self, datasets=None, n_trials=128):
        """Run comprehensive benchmark across all datasets and algorithms."""
        if datasets is None:
            datasets = load_openml_tabular_datasets()
        
        algorithms = ['LPECO', 'Lion', 'AdamW', 'XGBoost', 'LightGBM']
        
        for dataset_name, dataset_data in datasets.items():
            print(f"Processing dataset: {dataset_name}")
            X, y = dataset_data['X'], dataset_data['y']
            
            for algorithm in algorithms:
                print(f"  Running {algorithm}...")
                
                # Check cache first
                cache_key = f"{algorithm}_{dataset_name}_neural_network_{generate_hyperparams_str({})}"
                if cache_key in self.cache:
                    result = self.cache[cache_key]
                    self.results.append({
                        'dataset': dataset_name,
                        'algorithm': algorithm,
                        'score': result['score'],
                        'training_time': result['training_time']
                    })
            continue
    
                # Run optimization
                if algorithm == 'LPECO':
                    score, time_taken = self._run_lpeco_optimization(X, y, n_trials)
                elif algorithm in ['Lion', 'AdamW']:
                    score, time_taken = self._run_optimizer_benchmark(algorithm, X, y)
                else:  # GBDT models
                    score, time_taken = self._run_gbdt_benchmark(algorithm, X, y)
                
                # Cache result
                if self.use_cache:
                    save_to_cache("", algorithm, dataset_name, "neural_network", score, time_taken)
                
                self.results.append({
                    'dataset': dataset_name,
                    'algorithm': algorithm,
                    'score': score,
                    'training_time': time_taken
                })
        
        return pd.DataFrame(self.results)
    
    def _run_lpeco_optimization(self, X, y, n_trials):
        """Run LPECO with hyperparameter optimization."""
        best_params, best_score = optimize_lpeco_hyperparams(X, y, n_trials)
        
        # Full evaluation with best parameters
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        times = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            X_train, y_train = preprocess_data(X_train, y_train)
            X_val, y_val = preprocess_data(X_val, y_val)
            
            model = create_simple_model(X_train.shape[1], len(np.unique(y_train)))
            model.compile(
                optimizer=LionPredictiveErrorCorrectionOptimizer(**best_params),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            start_time = time.time()
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                              epochs=100, verbose=0)
            end_time = time.time()
            
            scores.append(max(history.history['val_accuracy']))
            times.append(end_time - start_time)
        
        return np.mean(scores), np.mean(times)
    
    def _run_optimizer_benchmark(self, optimizer_name, X, y):
        """Run neural network optimizer benchmark."""
        if optimizer_name == 'Lion':
            optimizer = LionPredictiveErrorCorrectionOptimizer(
                learning_rate=0.001, error_decay=1.0, correction_strength=0.0
            )
        else:  # AdamW
            optimizer = keras.optimizers.AdamW(learning_rate=0.001)
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        times = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            X_train, y_train = preprocess_data(X_train, y_train)
            X_val, y_val = preprocess_data(X_val, y_val)
            
            model = create_simple_model(X_train.shape[1], len(np.unique(y_train)))
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            start_time = time.time()
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                              epochs=100, verbose=0)
            end_time = time.time()
            
            scores.append(max(history.history['val_accuracy']))
            times.append(end_time - start_time)
        
        return np.mean(scores), np.mean(times)
    
    def _run_gbdt_benchmark(self, algorithm, X, y):
        """Run GBDT model benchmark."""
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        times = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            X_train, y_train = preprocess_data(X_train, y_train)
            X_val, y_val = preprocess_data(X_val, y_val)
            
            start_time = time.time()
            if algorithm == 'XGBoost':
                acc, _ = train_xgboost(X_train, y_train, X_val, y_val)
            else:  # LightGBM
                acc, _ = train_lightgbm(X_train, y_train, X_val, y_val)
            end_time = time.time()
            
            scores.append(acc)
            times.append(end_time - start_time)
        
        return np.mean(scores), np.mean(times)

def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def configure_gpu(use_gpu: bool = True) -> bool:
    """Configure GPU usage for TensorFlow."""
    if use_gpu:
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return True
        except RuntimeError as e:
            print(f"GPU configuration failed: {e}")
    return False

def main(use_gpu: bool = True, use_cv: bool = False, use_cache: bool = True) -> OptimizerBenchmark:
    """Main function to run the complete benchmark.
    
    Args:
        use_gpu: Whether to use GPU acceleration
        use_cv: Whether to use cross-validation
        use_cache: Whether to use cached results
        
    Returns:
        OptimizerBenchmark: The benchmark instance with results
    """
    print("LPECO: Lion with Predictive Error Correction Optimizer")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Configure GPU
    gpu_available = configure_gpu(use_gpu)
    print(f"GPU available: {gpu_available}")
    
    # Initialize benchmark
    benchmark = OptimizerBenchmark(use_cache=use_cache)
    
    print("Running comprehensive benchmark...")
    print("This may take several hours depending on your hardware.")
    
    # Run benchmark
    results_df = benchmark.run_full_benchmark()
    
    print("\nGenerating analysis plots...")
    # Generate plots
    create_analysis_plots(results_df)
    generate_meta_analysis_plot(results_df)
    create_convergence_plots(results_df)
    create_speed_accuracy_plots(results_df)
    
    print("\nPerforming statistical analysis...")
    # Statistical analysis
    friedman_result = friedman_test(results_df)
    if friedman_result:
        print(f"Friedman test p-value: {friedman_result['p_value']:.4f}")
    
    print("\nBenchmark completed successfully!")
    print(f"Results saved to analysis/plots/")
    
    return benchmark

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LPECO Benchmark')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    parser.add_argument('--no-cache', action='store_true', help='Disable result caching')
    parser.add_argument('--cv', action='store_true', help='Use cross-validation')
    
    args = parser.parse_args()
    
    main(use_gpu=not args.no_gpu, use_cv=args.cv, use_cache=not args.no_cache)

# ==============================================================================
# HELPER FUNCTIONS FOR CACHING AND REPRODUCIBILITY
# ==============================================================================

def load_cache_helper() -> Dict[str, Any]:
    """Load cached benchmark results if available."""
    if 'load_cache' in globals() and callable(globals()['load_cache']):
        return globals()['load_cache']()
    return {}

def save_to_cache_helper(hyperparams_hash: str, name: str, dataset: str, 
                  model_type: str, score: float, training_time: float) -> None:
    """Save benchmark results to cache."""
    if 'save_to_cache' in globals() and callable(globals()['save_to_cache']):
        globals()['save_to_cache'](hyperparams_hash, name, dataset, model_type, score, training_time)

def set_random_seed_helper(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def generate_hyperparams_str_helper(param_dict: Dict[str, Any]) -> str:
    """Generate a stable hash string from hyperparameters."""
    sorted_params = sorted(param_dict.items())
    param_str = '_'.join([f"{k}_{v}" for k, v in sorted_params])
    return hashlib.md5(param_str.encode()).hexdigest()

# ==============================================================================
# CUSTOM CALLBACKS FOR PLOTTING EPOCH-BY-EPOCH CONVERGENCE
# ==============================================================================

class TimingCallback(keras.callbacks.Callback):
    """Keras callback to record the time of each epoch."""
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if not hasattr(self, 'epoch_times'):
            self.epoch_times = []
        self.epoch_times.append(time.time() - self.epoch_start_time)

class LGBMTimingAndAccuracyCallback:
    """LightGBM callback to record validation accuracy and time per boosting round."""
    
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        self.accuracies = []
        self.times = []
        self.start_time = None
    
    def __call__(self, env):
        if self.start_time is None:
            self.start_time = time.time()
        
        # Calculate accuracy
        y_pred = env.model.predict(self.X_val, num_iteration=env.iteration)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(self.y_val, y_pred_class)
        self.accuracies.append(accuracy)
        
        # Record time
        current_time = time.time()
        self.times.append(current_time - self.start_time)

class XGBoostTimingAndAccuracyCallback:
    """XGBoost callback to record validation accuracy and time per boosting round."""
    
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        self.accuracies = []
        self.times = []
        self.start_time = None
    
    def __call__(self, env):
        if self.start_time is None:
            self.start_time = time.time()
        
        # Calculate accuracy
        y_pred = env.model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        self.accuracies.append(accuracy)
        
        # Record time
        current_time = time.time()
        self.times.append(current_time - self.start_time)

# ==============================================================================
# ENHANCED DATASET LOADING FUNCTIONS
# ==============================================================================

def load_small_datasets() -> Dict[str, Any]:
    """Load small datasets for quick testing."""
    datasets = {}
    
    # Create synthetic datasets for testing
    from sklearn.datasets import make_classification
    
    for i, (n_samples, n_features, n_classes) in enumerate([(100, 10, 2), (150, 15, 3), (200, 20, 2)]):
        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                 n_classes=n_classes, random_state=42+i)
        datasets[f'synthetic_{i+1}'] = {'X': X, 'y': y}
    
    return datasets

def load_medium_datasets() -> Dict[str, Any]:
    """Load medium-sized datasets for comprehensive testing."""
    datasets = {}
    
    # Create synthetic datasets for testing
    from sklearn.datasets import make_classification
    
    for i, (n_samples, n_features, n_classes) in enumerate([(500, 25, 3), (750, 30, 4), (1000, 35, 2)]):
        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                 n_classes=n_classes, random_state=42+i)
        datasets[f'medium_synthetic_{i+1}'] = {'X': X, 'y': y}
    
    return datasets

def preprocess_data_properly(X_train, X_test, y_train, y_test):
    """Proper data preprocessing that handles train/test split correctly."""
    # Handle missing values
    if hasattr(X_train, 'fillna'):
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())  # Use train mean for test
    
    # Encode categorical variables
    le = LabelEncoder()
    if y_train.dtype == 'object':
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ==============================================================================
# ENHANCED STATISTICAL ANALYSIS FUNCTIONS
# ==============================================================================

def perform_comprehensive_statistical_analysis(results_df):
    """Perform comprehensive statistical analysis on benchmark results."""
    analysis_results = {}
    
    # Friedman test
    friedman_result = friedman_test(results_df)
    analysis_results['friedman'] = friedman_result
    
    # Pairwise equivalence tests
    algorithms = results_df['algorithm'].unique()
    equivalence_results = {}
    
    for i, alg1 in enumerate(algorithms):
        for alg2 in algorithms[i+1:]:
            scores1 = results_df[results_df['algorithm'] == alg1]['score'].values
            scores2 = results_df[results_df['algorithm'] == alg2]['score'].values
            
            # TOST equivalence test
            tost_result = equivalence_test(scores1, scores2)
            equivalence_results[f'{alg1}_vs_{alg2}'] = tost_result
            
            # Bayesian ROPE analysis
            rope_result = bayesian_rope_analysis(scores1, scores2)
            equivalence_results[f'{alg1}_vs_{alg2}_rope'] = rope_result
    
    analysis_results['equivalence'] = equivalence_results
    
    # Performance ranking
    mean_scores = results_df.groupby('algorithm')['score'].mean().sort_values(ascending=False)
    analysis_results['ranking'] = mean_scores.to_dict()
    
    return analysis_results

def create_detailed_convergence_analysis(dataset_name, X, y, algorithms=['LPECO', 'Lion', 'AdamW']):
    """Create detailed convergence analysis for a specific dataset."""
    convergence_data = {}
    
    for algorithm in algorithms:
        if algorithm == 'LPECO':
            # Use hyperparameter optimization
            best_params, _ = optimize_lpeco_hyperparams(X, y, n_trials=20)
            optimizer = LionPredictiveErrorCorrectionOptimizer(**best_params)
        elif algorithm == 'Lion':
            optimizer = LionPredictiveErrorCorrectionOptimizer(
                learning_rate=0.001, error_decay=1.0, correction_strength=0.0
            )
        else:  # AdamW
            optimizer = keras.optimizers.AdamW(learning_rate=0.001)
        
        # Train with callbacks
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = preprocess_data_properly(X_train, X_val, y_train, y_val)
        
        model = create_simple_model(X_train.shape[1], len(np.unique(y_train)))
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        timing_callback = TimingCallback()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            callbacks=[timing_callback],
            verbose=0
        )
        
        convergence_data[algorithm] = {
            'train_acc': history.history['accuracy'],
            'val_acc': history.history['val_accuracy'],
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'epoch_times': timing_callback.epoch_times
        }
    
    return convergence_data

def generate_comprehensive_plots(results_df, convergence_data=None, output_dir='analysis/plots'):
    """Generate comprehensive plots for the paper."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance distribution
    create_analysis_plots(results_df, output_dir)
    
    # 2. Meta-analysis
    generate_meta_analysis_plot(results_df, output_dir)
    
    # 3. Speed vs accuracy
    create_speed_accuracy_plots(results_df, output_dir)
    
    # 4. Convergence analysis
    if convergence_data:
        create_convergence_plots_from_data(convergence_data, output_dir)
    
    # 5. Statistical significance plots
    create_statistical_significance_plots(results_df, output_dir)
    
    print(f"All plots saved to {output_dir}/")

def create_convergence_plots_from_data(convergence_data, output_dir='analysis/plots'):
    """Create convergence plots from detailed convergence data."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    for algorithm, data in convergence_data.items():
        epochs = range(1, len(data['val_acc']) + 1)
        plt.plot(epochs, data['val_acc'], label=f'{algorithm} (Val)', linewidth=2)
        plt.plot(epochs, data['train_acc'], label=f'{algorithm} (Train)', 
                linestyle='--', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Convergence Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/convergence_epochs_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_significance_plots(results_df, output_dir='analysis/plots'):
    """Create plots showing statistical significance of results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a heatmap of p-values for pairwise comparisons
    algorithms = results_df['algorithm'].unique()
    p_value_matrix = np.ones((len(algorithms), len(algorithms)))
    
    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms):
            if i != j:
                scores1 = results_df[results_df['algorithm'] == alg1]['score'].values
                scores2 = results_df[results_df['algorithm'] == alg2]['score'].values
                
                # Perform t-test
                from scipy.stats import ttest_ind
                _, p_value = ttest_ind(scores1, scores2)
                p_value_matrix[i, j] = p_value
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_value_matrix, 
                xticklabels=algorithms, 
                yticklabels=algorithms,
                annot=True, 
                fmt='.3f',
                cmap='viridis')
    plt.title('Pairwise Comparison P-values')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/statistical_significance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# ENHANCED BENCHMARKING WITH DETAILED TRACKING
# ==============================================================================

class EnhancedOptimizerBenchmark(OptimizerBenchmark):
    """Enhanced benchmarking system with detailed convergence tracking."""
    
    def __init__(self, use_cache=True, track_convergence=True):
        super().__init__(use_cache)
        self.track_convergence = track_convergence
        self.convergence_data = {}
    
    def run_enhanced_benchmark(self, datasets=None, n_trials=128):
        """Run enhanced benchmark with detailed convergence tracking."""
        if datasets is None:
            datasets = load_openml_tabular_datasets()
        
        algorithms = ['LPECO', 'Lion', 'AdamW', 'XGBoost', 'LightGBM']
        
        for dataset_name, dataset_data in datasets.items():
            print(f"Processing dataset: {dataset_name}")
            X, y = dataset_data['X'], dataset_data['y']
            
            # Store convergence data for this dataset
            if self.track_convergence:
                self.convergence_data[dataset_name] = create_detailed_convergence_analysis(
                    dataset_name, X, y, ['LPECO', 'Lion', 'AdamW']
                )
            
            for algorithm in algorithms:
                print(f"  Running {algorithm}...")
                
                # Check cache first
                cache_key = f"{algorithm}_{dataset_name}_neural_network_{generate_hyperparams_str({})}"
                if cache_key in self.cache:
                    result = self.cache[cache_key]
                    self.results.append({
                        'dataset': dataset_name,
                        'algorithm': algorithm,
                        'score': result['score'],
                        'training_time': result['training_time']
                    })
                    continue
                
                # Run optimization
                if algorithm == 'LPECO':
                    score, time_taken = self._run_lpeco_optimization(X, y, n_trials)
                elif algorithm in ['Lion', 'AdamW']:
                    score, time_taken = self._run_optimizer_benchmark(algorithm, X, y)
                else:  # GBDT models
                    score, time_taken = self._run_gbdt_benchmark(algorithm, X, y)
                
                # Cache result
                if self.use_cache:
                    save_to_cache("", algorithm, dataset_name, "neural_network", score, time_taken)
                
                self.results.append({
                    'dataset': dataset_name,
                    'algorithm': algorithm,
                    'score': score,
                    'training_time': time_taken
                })
        
        return pd.DataFrame(self.results)
