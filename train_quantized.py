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
import argparse

import numpy as np
np.random.seed(1234) # for reproducibility?
import os
os.system("hostname")
import glob
listing = glob.glob('/usr/local/cuda*')
if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = '%s/lib64/'%(listing[0])
    print ('%s/lib64/'%(listing[0]))
if 'PYLEARN2_DATA_PATH' not in os.environ:
    os.environ['PYLEARN2_DATA_PATH'] = '/esat/leda1/users/bmoons/PYLEARN2'
    print ('/esat/leda1/users/bmoons/PYLEARN2')


#if listing:
#    os.environ["THEANO_FLAGS"] = "cuda.root=%s,device=gpu0,lib.cnmem=0.8,floatX=float32"%(listing[0])#


import lasagne
import theano.tensor as T
import theano

import quantized_net
import build_net
import load_dataset

from collections import OrderedDict

parser = argparse.ArgumentParser(description='train')
parser.add_argument('-ds','--dataset',help='dataset name, can be CIFAR-10, SVHN or MNIST in this repo', required=True)
parser.add_argument('-nb','--bits',help='number of bits throughout full network, both for weights and activations', required=True,type=int)
parser.add_argument('-q','--quantized',help='quantized (1) or floating point (0) flag', required=True,type=int)
parser.add_argument('-ne','--num_epochs',help='number of epochs for training', required=True,type=int)
parser.add_argument('-nfa','--num_filters_A',help='number of filters in block A', required=True,type=int)
parser.add_argument('-nla','--num_layers_A',help='number of layers in block A', required=True,type=int)
parser.add_argument('-nfb','--num_filters_B',help='number of filters in block B', required=True,type=int)
parser.add_argument('-nlb','--num_layers_B',help='number of layers in block B', required=True,type=int)
parser.add_argument('-nfc','--num_filters_C',help='number of filters in block C', required=True,type=int)
parser.add_argument('-nlc','--num_layers_C',help='number of layers in block C', required=True,type=int)
parser.add_argument('-ft','--finetune',help='Finetune flag, yes(1), no(0)', required=True,type=int)
parser.add_argument('-nl','--nonlinearity',help='Can be hardtanh or relu', required=True)
parser.add_argument('-sym','--symmetry',help='Symmetry flag, can be yes(1), no(0) \n If nl=relu, sym has to be zero', required=True,type=int)
parser.add_argument('-lr','--learning_rate',help='Learning rate as float', required=True,type=float)

args = parser.parse_args()

dataset = args.dataset
if dataset=='CIFAR-10':
	classes=10
	channels=3
	patch_size=32
elif dataset=='MNIST':
	classes=10
	channels=1
	patch_size=28

bits = args.bits
num_epochs = args.num_epochs
quantized = bool(args.quantized)
finetune = bool(args.finetune)
nonlinearity = args.nonlinearity
symmetry = bool(args.symmetry)
nfA = args.num_filters_A
nfB = args.num_filters_B
nfC = args.num_filters_C
nlA = args.num_layers_A
nlB = args.num_layers_B
nlC = args.num_layers_C
LR_start = args.learning_rate


if __name__ == "__main__":

    # nb_bits
    nb_bits = int(float(bits))
    quantized_net.set_global_nb_bits(nb_bits)
    print("nb_bits = "+ str(nb_bits))    

    # BN parameters
    batch_size = 64
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # binarizedOut
    #activation = lasagne.nonlinearities.tanh
    #print("activation = lasagne.nonlinearities.tanh")
    if(nonlinearity=='hardtanh'):
        if (quantized):
            activation = quantized_net.quantized_hardtanh_unit
            print("activation = quantized_net.quantized_hardtanh_unit")
        else:
            activation = quantized_net.float_hardtanh_unit
            print("activation = quantized_net.float_hardtanh_unit")
    else:
        symmetry = False
        if (quantized):
            activation = quantized_net.quantized_rectify_unit
            print("activation = quantized_net.quantized_rectify_unit")
        else:
            activation = lasagne.nonlinearities.rectify
            print("activation = lasagne.nonlinearities.rectify")
    quantized_net.set_global_symmetry(symmetry)
    
    # quantizedConnect
    # quantized = True
    print("quantized = "+str(quantized))
    stochastic = False
    print("stochastic = "+str(stochastic))
    H = 1.
    print("H = "+str(H))
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Training parameters
    print("num_epochs = "+str(num_epochs))
    
    # Decaying LR 
    LR_start = LR_start
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...


    if (quantized):
        save_path = "./models/%s_%s_%s_quantized_"%(nonlinearity,str(symmetry),dataset)+str(nlA)+"_"+str(nfA)+"_"+str(nlB)+"_"+str(nfB)+"_"+str(nlC)+"_"+str(nfC)+"_"+str(nb_bits)+"bits.npz"
    else:
        save_path = "./models/%s_%s_%s_float_"%(nonlinearity,str(symmetry),dataset)+str(nlA)+"_"+str(nfA)+"_"+str(nlB)+"_"+str(nfB)+"_"+str(nlC)+"_"+str(nfC)+".npz"
    print("save_path = "+str(save_path))
    if (quantized):
        load_path = "./models/%s_%s_%s_quantized_"%(nonlinearity,str(symmetry),dataset)+str(nlA)+"_"+str(nfA)+"_"+str(nlB)+"_"+str(nfB)+"_"+str(nlC)+"_"+str(nfC)+"_"+str(nb_bits)+"bits.npz"
    else:
        load_path = "./models/%s_%s_%s_float_"%(nonlinearity,str(symmetry),dataset)+str(nlA)+"_"+str(nfA)+"_"+str(nlB)+"_"+str(nfB)+"_"+str(nlC)+"_"+str(nfC)+".npz"
    print("load_path = "+str(load_path))

    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))

    print('Loading the Dataset...')   
	
    train_set, valid_set, test_set = load_dataset.load_dataset(dataset)

    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = build_net.build_net(\
        input,quantized=quantized,stochastic=stochastic,\
        H=H,W_LR_scale=W_LR_scale,activation=activation,\
        epsilon=epsilon,alpha=alpha,patch_size=patch_size,\
        channels=channels,window=3,\
        nfA=nfA,nlA=nlA,nfB=nfB,nlB=nlB,nfC=nfC,nlC=nlC,\
        hidden_layer_size=1024, classes=classes)


    train_output = lasagne.layers.get_output(cnn, deterministic=False, c_w_bits=nb_bits, c_f_bits=nb_bits)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    #all_layers = lasagne.layers.get_all_layers(cnn)
    #l2_p = lasagne.regularization.regularize_layer_params(all_layers,lasagne.regularization.l2)*0.0001
    #loss = loss + l2_p
    
    if (quantized):
        
        # W updates
        W = lasagne.layers.get_all_params(cnn, quantized=True)
        W_grads = quantized_net.compute_grads(loss,cnn)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = quantized_net.clipping_scaling(updates,cnn,nb_bits)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(cnn, trainable=True, quantized=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(cnn, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(cnn, deterministic=True, c_w_bits=nb_bits, c_f_bits=nb_bits)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    if(finetune):
        print('Load pretrained values...')
        with np.load(load_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(cnn, param_values)

    print('Training...')
    
    accuracy = quantized_net.train(
            train_fn,val_fn,
            cnn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            train_set.X,train_set.y,
            valid_set.X,valid_set.y,
            test_set.X,test_set.y,
	        save_path=save_path,
            shuffle_parts=shuffle_parts, data_augmentation=False)
