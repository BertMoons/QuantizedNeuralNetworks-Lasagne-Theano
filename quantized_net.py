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

import time

from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise



def set_global_nb_bits(nb_bits):
    global global_nb_bits
    global_nb_bits = nb_bits

def set_global_symmetry(symmetry):
    global global_symmetry
    global_symmetry = symmetry


# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 
        
round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

def rectify(x):
    return T.max(0,x)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1 
# during back propagation
def binarized_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.
    
def binarized_sigmoid_unit(x):
    return round3(hard_sigmoid(x))

def float_hardtanh_unit(x):
    return 2.*hard_sigmoid(x)-1

def quantized_hardtanh_unit(x):
    nb_bits = global_nb_bits
    if(global_symmetry): Xq = T.clip(2.*(round3(hard_sigmoid(x)*pow(2,nb_bits))/pow(2,nb_bits))-1.,-1,1)
    else: Xq = T.clip(2.*(round3(hard_sigmoid(x)*pow(2,nb_bits))/pow(2,nb_bits))-1.,-1,1-1.0/pow(2,nb_bits-1))
    return Xq 

def quantized_rectify_unit(x):
    # nonsymmetric by default, ex: 2-bit relu: [0,0.5] output can be represented using 2 bits. Weights, have to be non-symmetric as well!!!
    nb_bits = global_nb_bits
    Xq = T.clip(2.*(round3(hard_sigmoid(x)*pow(2,nb_bits))/pow(2,nb_bits))-1.,0,1-1.0/pow(2,nb_bits-1))
    return Xq 

# The weights' quantization function, 
def quantization(W,nb_bits,quantized=True,deterministic=False,stochastic=False,srng=None, symmetry=True):
    #nb_bits = nb_bits # this version is symmetric  
    maxW = T.max(abs(W))
    non_fractional = T.floor(T.log2(1./maxW)) #amount of bits extra to shift because of leading zeros after comma in all values
    non_sign_bits = nb_bits - 1

    # (deterministic == True) <-> test-time <-> inference-time
    if not quantized or (deterministic and stochastic):
        # print("not quantized")
        Wq = W
    elif quantized and nb_bits==0:
        # [-1,1] -> [0,1]
        Wq = hard_sigmoid(W/1)
        Wq = T.round(Wq)
        # 0 or 1 -> -1 or 1
        Wq = T.cast(T.switch(Wq,1,-1), theano.config.floatX)
    else:
        if deterministic and stochastic: 
            Wq = T.round(W*T.pow(2,non_sign_bits+non_fractional))
        else:
            if stochastic:
                Wq = W*T.pow(2,non_sign_bits+non_fractional)
                Wf = T.floor(Wq)
                Wq = Wq-Wf
                q = srng.uniform(size = T.shape(Wq))
                q = T.cast(q<Wq,theano.config.floatX)
                Wq = Wf+q
            else:
                Wq = T.round(W*T.pow(2,non_sign_bits+non_fractional))
        if (symmetry): lower_lim,upper_lim = (-T.pow(2,non_sign_bits),T.pow(2,non_sign_bits))
        else: lower_lim,upper_lim = (-T.pow(2,non_sign_bits),T.pow(2,non_sign_bits)-1)
        Wq = T.clip(Wq,lower_lim, upper_lim)
        Wq = Wq/T.pow(2,non_sign_bits+non_fractional)    

    return Wq

# This class extends the Lasagne DenseLayer to support binarizedConnect
class DenseLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, 
        quantized = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):
        
        self.quantized = quantized
        self.stochastic = stochastic
        self.symmetry = global_symmetry
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
            # print("H = "+str(self.H))
            
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        if self.quantized:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the quantized tag to weights
            self.params[self.W]=set(['quantized'])
            
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        
        self.Wb = quantization(self.W,kwargs.get("c_f_bits",4),self.quantized,deterministic,self.stochastic,self._srng, self.symmetry)
        Wr = self.W
        self.W = self.Wb
            
        rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
        
        self.W = Wr
        
        return rvalue

# This class extends the Lasagne Conv2DLayer to support binarizedConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size,
        quantized = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):
        
        self.quantized = quantized
        self.stochastic = stochastic
        self.symmetry = global_symmetry
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
            # print("H = "+str(self.H))
        
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
            # print("W_LR_scale = "+str(self.W_LR_scale))
            
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            
        if self.quantized:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the quantized tag to weights
            self.params[self.W]=set(['quantized'])
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)    
    
    def convolve(self, input, deterministic=False, **kwargs):
        
        self.Wb = quantization(self.W,kwargs.get("c_w_bits",4),self.quantized,deterministic,self.stochastic,self._srng, self.symmetry)
        Wr = self.W
        self.W = self.Wb
            
        rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        
        self.W = Wr
        
        return rvalue
        

# This function computes the gradient of the quantized weights
def compute_grads(loss,network):
        
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    
    for layer in layers:
    
        params = layer.get_params(quantized=True)
        if params:
            # print(params[0].name)
            grads.append(theano.grad(loss, wrt=layer.Wb))
                
    return grads

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network,nb_bits):
    
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(quantized=True)
        for param in params:
            print("W_LR_scale = "+str(layer.W_LR_scale))
            print("H = "+str(layer.H))
            updates[param] = param + layer.W_LR_scale*(updates[param] - param)   
            updates[param] = T.clip(updates[param], -1,1)

    return updates
        
# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn,val_fn,
            model,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,
            X_test,y_test,
            save_path=None,
            shuffle_parts=1,
            data_augmentation=False):
    ''' 
    datagen = ImageDataGenerator(
		rotation_range = 40,
		width_shift_range = 0.2,
		height_shift_range = 0.2,
		horizontal_flip = True)
    '''
    
    # A function which shuffles a dataset
    def shuffle(X,y):
        
        # print(len(X))
        
        chunk_size = len(X)/shuffle_parts
        shuffled_range = range(chunk_size)
        
        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])
        
        for k in range(shuffle_parts):
            
            np.random.shuffle(shuffled_range)

            for i in range(chunk_size):
                
                X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
                y_buffer[i] = y[k*chunk_size+shuffled_range[i]]
            
            X[k*chunk_size:(k+1)*chunk_size] = X_buffer
            y[k*chunk_size:(k+1)*chunk_size] = y_buffer
        
        return X,y

    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR):
        
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            if(data_augmentation):
                #for X_batch, Y_batch in datagen.flow(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size], batch_size=batch_size):
                #    loss += train_fn(np.asarray(X_batch,dtype=theano.config.floatX),Y_batch,LR)
                loss += train_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size],LR)
            else:
                loss += train_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size],LR)
        loss/=batches
        
        return loss
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y):
        
        err = 0
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss
        
        err = err / batches * 100
        loss /= batches

        return err, loss
    
    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    best_val_err = 100
    best_epoch = 1
    LR = LR_start
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        
        start_time = time.time()
        
        train_loss = train_epoch(X_train,y_train,LR)
        X_train,y_train = shuffle(X_train,y_train)
        
        val_err, val_loss = val_epoch(X_val,y_val)

        
        # test if validation error went down
        if val_err <= best_val_err:
            
            best_val_err = val_err
            best_epoch = epoch+1
            
            test_err, test_loss = val_epoch(X_test,y_test)
            
            if save_path is not None:
                np.savez(save_path, *lasagne.layers.get_all_param_values(model))
        
        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
		
       # if(epoch>25):
       #     if(10*train_loss<val_loss):
       #         break
       #     if(epoch-best_epoch > 15):
       #         break
        
        # decay the LR
        #if(epoch-best_epoch == 5):
        #    LR *= 0.5
        #elif(epoch-best_epoch == 10):
        #    LR *= 0.5
        #else:
        LR *= LR_decay

    return test_err

