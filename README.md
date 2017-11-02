# Training Quantized Neural Networks

## Introduction
Train your own __Quantized Neural Networks (QNN)__ - networks trained with quantized weights and activations - in __Lasagne / Theano__.
If you use this code, please cite "B.Moons et al. "Minimum Energy Quantized Neural Networks", Asilomar Conference on Signals, Systems and Computers, 2017". Take a look at our [presentation](https://www.researchgate.net/publication/320775013_Presentation_on_Minimum_Energy_Quantized_Neural_Networks_Asilomar_2017) or at the paper on [arxiv](https://arxiv.org/abs/1711.00215).

This code is based on a [lasagne/theano](https://github.com/MatthieuCourbariaux/BinaryNet) and a [Keras/Tensorflow](https://github.com/DingKe/BinaryNet) version of [BinaryNet](https://papers.nips.cc/paper/6573-binarized-neural-networks).

## Preliminaries
Running this code requires:
1. [Theano](http://deeplearning.net/software/theano/)
2. [Lasagne](https://lasagne.readthedocs.io/en/latest/)
3. [pylearn2](http://deeplearning.net/software/pylearn2/) + the correct PYLEARN2_DATA_PATH in your shell
3. A GPU with recent versions of [CUDA and CUDNN](https://developer.nvidia.com/cudnn)
4. Correct paths in the top of train_quantized.py

## Training your own QNN

This repo includes toy examples for CIFAR-10 and MNIST.
Training can be done directly from the command line:

python train_quantized.py <parameters>
  
The following parameters are crucial:
* -ds: dataset, either 'CIFAR-10' or 'MNIST'
* -nb: number of bits for weights and activations
* -q: if 1, the networks is quantized, otherwise it is floating point
* -ne: number of epochs, 100 should do
* -ft: if 1, a previous model is used to finetune, otherwise training is from scratch
* -nl: the type of nonlinearity required, can either be 'relu' or 'hardtanh'
* -sym: if nl=relu, should be 0, otherwise it should be 1
* -lr: learning rate, 0.01 or 0.001 should be fine
* nl<>: [the number of layers in block A, B, C](https://www.linkedin.com/in/bert-moons-41867143/)
* nf<>: [the number of filters in block A, B, C](https://www.linkedin.com/in/bert-moons-41867143/)

## Examples 
The included networks have parametrized sizes and are split into three blocks (A-B-C), each with a number of layers (nl) and a number of filters per layer (nf).

* This is how to train a 4-bit full qnn on CIFAR-10:

  python train_quantized.py -ds CIFAR-10 -nb 4 -q 1 -ne 100 -nla 1 -nfa 64 -nlb 1 -nfb 64 -nlc 1 -nfc 64 -ft 0 -nl relu -sym 0 -lr 0.001
  
* This is how to train a BinaryNet on CIFAR-10:

  python train_quantized.py -ds CIFAR-10 __-nb 0__ -q 1 -ne 100 -nla 1 -nfa 64 -nlb 1 -nfb 64 -nlc 1 -nfc 64 -ft 0 __-nl hardtanh -sym 1__ -lr 0.001
 
