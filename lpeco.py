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

if __name__ == "__main__":
    print("Basic Lion Optimizer Implementation")
