#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
import numpy as np

NUM_PARAMS_NEURON = 4

class CMLPDense(keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        assert input_shape[-1] % 4 == 0

        self.w = self.add_weight(
            name='name',
            shape=(input_shape[-2], self.units, NUM_PARAMS_NEURON),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name='name',
            shape=(self.units, NUM_PARAMS_NEURON), initializer="zeros", trainable=True
        )

    def call(self, inputs: np.float64):

        def quaternion_hadamard_product(weights, inputs):
            r1, x1, y1, z1 = tf.unstack(weights, axis=-1)
            r2, x2, y2, z2 = tf.unstack(inputs, axis=-1)
            
            return tf.stack([tf.matmul(r2,r1) - tf.matmul(x2,x1) - tf.matmul(y2,y1) - tf.matmul(z2,z1),
                             tf.matmul(x2,r1) + tf.matmul(r2,x1) + tf.matmul(z2,y1) - tf.matmul(y2,z1),
                             tf.matmul(y2,r1) - tf.matmul(z2,x1) + tf.matmul(r2,y1) + tf.matmul(x2,z1),
                             tf.matmul(z2,r1) + tf.matmul(y2,x1) - tf.matmul(x2,y1) + tf.matmul(r2,z1)], axis=-1)
        
        hadamard_product = quaternion_hadamard_product(self.w, inputs)
        
        return hadamard_product + self.b
