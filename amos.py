from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy import misc
import argparse
#import csv 

class Conv(object): 
    def initalize(self, sess):
        pre_trained_weights = np.load(open(self.weight_path, "rb"), encoding="latin1").item()
        keys = sorted(pre_trained_weights.keys())
        for k in keys:
        #for k in list(filter(lambda x: 'conv' in x,keys)):
            with tf.variable_scope(k, reuse=True):
                temp = tf.get_variable('weights')
                sess.run(temp.assign(pre_trained_weights[k]['weights']))
            with tf.variable_scope(k, reuse=True):
                temp = tf.get_variable('biases')
                sess.run(temp.assign(pre_trained_weights[k]['biases']))
            
    def conv(self, input, filter_size, in_channels, out_channels, name, strides, padding, groups):
        with tf.variable_scope(name) as scope:
            filt = tf.get_variable('weights', shape=[filter_size, filter_size, int(in_channels/groups), out_channels])
            bias = tf.get_variable('biases',  shape=[out_channels])
        if groups == 1:
            return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, filt, strides=strides, padding=padding), bias))
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=input)
            filt_groups = tf.split(axis = 3, num_or_size_splits=groups, value=filt)
            output_groups = [ tf.nn.conv2d( i, k, strides = strides, padding = padding) for i,k in zip(input_groups, filt_groups)]

            conv = tf.concat(axis = 3, values = output_groups)
            return tf.nn.relu(tf.nn.bias_add(conv, bias))

    def fc(self, input, in_channels, out_channels, name, relu):
        input = tf.reshape(input , [-1, in_channels])
        with tf.variable_scope(name) as scope:
            filt = tf.get_variable('weights', shape=[in_channels , out_channels])
            bias = tf.get_variable('biases',  shape=[out_channels])
        if relu:
            return tf.nn.relu(tf.nn.bias_add(tf.matmul(input, filt), bias))
        else:
            return tf.nn.bias_add(tf.matmul(input, filt), bias)
        

    def pool(self, input, padding, name):
        return tf.nn.max_pool(input, ksize=[1,3,3,1], strides=[1,2,2,1], padding=padding, name= name)



    def model(self):    

        #placeholder for a random set of 20 images of fixed size -- 256,256
        self.input = tf.placeholder(tf.float32, shape = [None, 256, 256, 3])
        sliced_input = tf.slice(self.input, begin=[ 0, 14, 14, 0], size=[ -1, 227, 227, -1])
        

        # Conv-Layers
        net_layers={}
        net_layers['conv1'] = self.conv(sliced_input, 11, 3, 96, name= 'conv1', strides=[1,4,4,1] ,padding='VALID', groups=1)
        net_layers['pool1'] = self.pool(net_layers['conv1'], padding='VALID', name='pool1')
        net_layers['lrn1']  = tf.nn.lrn(net_layers['pool1'], depth_radius=2, alpha=2e-5, beta=0.75,name='norm1')

        net_layers['conv2'] = self.conv(net_layers['lrn1'], 5, 96, 256, name= 'conv2', strides=[1,1,1,1] ,padding='SAME', groups=2)
        net_layers['pool2'] = self.pool(net_layers['conv2'], padding='VALID', name='pool2')
        net_layers['lrn2']  = tf.nn.lrn(net_layers['pool2'], depth_radius=2, alpha=2e-5, beta=0.75,name='norm2')

        net_layers['conv3'] = self.conv(net_layers['lrn2'], 3, 256, 384, name='conv3', strides=[1,1,1,1] ,padding='SAME', groups=1)

        net_layers['conv4'] = self.conv(net_layers['conv3'], 3, 384, 384, name='conv4', strides=[1,1,1,1] ,padding='SAME', groups=2)

        net_layers['conv5'] = self.conv(net_layers['conv4'], 3, 384, 256, name='conv5', strides=[1,1,1,1] ,padding='SAME', groups=2)

        net_layers['conv6'] = self.conv(net_layers['conv5'], 3, 256, 256, name='conv6', strides=[1,1,1,1] ,padding='SAME', groups=2)
        net_layers['pool6'] = self.pool(net_layers['conv6'], padding='VALID', name='pool6')
        
        # FC layers
        net_layers['fc7'] = self.fc(net_layers['pool6'],  6*6*256, 4096, name='fc7_new', relu = 1)
        net_layers['fc8'] = self.fc(net_layers['fc7'], 4096, 2543, name='fc8_new', relu = 0)

        net_layers['prob'] = tf.nn.softmax(net_layers['fc8'])
        net_layers['pred'] = tf.argmax(tf.nn.softmax(net_layers['fc8']), axis = 1)

        self.net_layers = net_layers



    def __init__(self, layer, weight_path, batch_size, max_frames):
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.layer = layer
        self.weight_path = weight_path

        mean = [104, 114, 124]
        scale_size = (256,256)

        self.spec = [mean, scale_size]
        
        self.model()
        self.features = tf.reshape(self.net_layers[layer], [-1, self.max_frames, tf.reduce_prod(self.net_layers[layer].get_shape()[1:])])
        
