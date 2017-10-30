# Copyright 2017 Bert Moons

# This file is part of QNN.

# QNN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# QNN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# The code for QNN is based on BinaryNet: https://github.com/MatthieuCourbariaux/BinaryNet

# You should have received a copy of the GNU General Public License
# along with QNN.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import sys
import time
import gc

import numpy as np
np.random.seed(1234) # for reproducibility?

import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import quantized_net

from collections import OrderedDict


def build_net(input,quantized,stochastic=False,H=1.0,W_LR_scale="Glorot",activation=quantized_net.quantized_hardtanh_unit,epsilon=1e-4,alpha=.1,patch_size=32,channels=3, window=3,nfA=64,nlA=1,nfB=64,nlB=1,nfC=64,nlC=1, hidden_layer_size=1024, classes=10):

    cnn = lasagne.layers.InputLayer(
            shape=(None, channels, patch_size, patch_size),
            input_var=input)

    # Block A
    for i in range(0,nlA):
        cnn = quantized_net.Conv2DLayer(
              cnn, 
              quantized=quantized,
              stochastic=stochastic,
              W_LR_scale=W_LR_scale,
              num_filters=nfA, 
              filter_size=(window, window),
              pad='same',
              nonlinearity=lasagne.nonlinearities.identity)

        cnn = lasagne.layers.BatchNormLayer(
              cnn,
              epsilon=epsilon, 
              alpha=alpha)

        cnn = lasagne.layers.NonlinearityLayer(
              cnn,
              nonlinearity=activation) 

        print(cnn.output_shape)


    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    print("After MaxPool: "+ str(cnn.output_shape))

    # Block B
    for i in range(0,nlB):
        cnn = quantized_net.Conv2DLayer(
              cnn, 
              quantized=quantized,
              stochastic=stochastic,
              W_LR_scale=W_LR_scale,
              num_filters=nfB, 
              filter_size=(window, window),
              pad='same',
              nonlinearity=lasagne.nonlinearities.identity)
         
        cnn = lasagne.layers.BatchNormLayer(
              cnn,
              epsilon=epsilon, 
              alpha=alpha)
                  
        cnn = lasagne.layers.NonlinearityLayer(
              cnn,
              nonlinearity=activation) 

        print(cnn.output_shape)


    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    print("After MaxPool: "+ str(cnn.output_shape))

    # Block C
    for i in range(0,nlC):
        cnn = quantized_net.Conv2DLayer(
              cnn, 
              quantized=quantized,
              stochastic=stochastic,
              W_LR_scale=W_LR_scale,
              num_filters=nfC, 
              filter_size=(window, window),
              pad='same',
              nonlinearity=lasagne.nonlinearities.identity)
         
        cnn = lasagne.layers.BatchNormLayer(
              cnn,
              epsilon=epsilon, 
              alpha=alpha)
                  
        cnn = lasagne.layers.NonlinearityLayer(
              cnn,
              nonlinearity=activation) 

        print(cnn.output_shape)


    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    print("After MaxPool: "+ str(cnn.output_shape))

    cnn = quantized_net.DenseLayer(
                cnn, 
                quantized=False,
                stochastic=stochastic,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=classes)

    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)

    return cnn
