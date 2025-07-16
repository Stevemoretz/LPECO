"""
Basic Lion Optimizer Implementation
Author: Siavash Mobarhan
"""

import tensorflow as tf
import keras
import numpy as np

class LionOptimizer(keras.optimizers.Optimizer):
    """Basic Lion optimizer implementation."""
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.99, name="Lion", **kwargs):
        super().__init__(name=name, **kwargs)
        self._learning_rate = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._get_hyper("learning_rate", var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        
        m = self.get_slot(var, "m")
        
        # Lion update rule
        u_t = beta_1_t * m + (1 - beta_1_t) * grad
        d_t = tf.sign(u_t)
        
        var_update = var - lr_t * d_t
        m_update = beta_2_t * m + (1 - beta_2_t) * grad
        
        return tf.group(
            var.assign(var_update),
            m.assign(m_update)
        )

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
        optimizer=LionOptimizer(learning_rate=0.001),
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

if __name__ == "__main__":
    print("Basic Lion Optimizer Implementation")
    print("Added neural network training capabilities")
