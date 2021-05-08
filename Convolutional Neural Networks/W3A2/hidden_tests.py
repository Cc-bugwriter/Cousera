import numpy as np
import solutions
from test_utils import single_test, multiple_test, summary
from termcolor import colored

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

import outputs

# Compare the two inputs
def comparator(learner, instructor):
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
    print(colored("All tests passed!", "green"))
            

# GRADED FUNCTION 1
def conv_block_test(target):
    input_size=(96, 128, 3)
    n_filters = 32
    inputs = Input(input_size)
    output_learner = target(inputs, n_filters * 1)
    model_learner = tf.keras.Model(inputs=inputs, outputs=output_learner)
    
    output_instructor = solutions.conv_block(inputs, n_filters * 1)
    model_instructor = tf.keras.Model(inputs=inputs, outputs=output_instructor)
    
    comparator(summary(model_instructor), summary(model_learner))
    
    inputs = Input(input_size)
    output_learner = target(inputs, n_filters * 32, dropout_prob=0.1, max_pooling=True)
    model_learner = tf.keras.Model(inputs=inputs, outputs=output_learner)
    
    output_instructor = solutions.conv_block(inputs, n_filters * 32, dropout_prob=0.1, max_pooling=True)
    model_instructor = tf.keras.Model(inputs=inputs, outputs=output_instructor)
    
    comparator(summary(model_instructor), summary(model_learner))

# GRADED FUNCTION 2
def upsampling_block_test(target):
    input_size1=(12, 16, 256)
    input_size2 = (24, 32, 128)
    n_filters = 32
    expansive_inputs = Input(input_size1)
    contractive_inputs =  Input(input_size2)
    learner_cblock = solutions.upsampling_block(expansive_inputs, contractive_inputs, n_filters * 1)
    model_instructor = tf.keras.Model(inputs=[expansive_inputs, contractive_inputs], outputs=learner_cblock)
    
    student_cblock = target(expansive_inputs, contractive_inputs, n_filters * 1)
    model_learner = tf.keras.Model(inputs=[expansive_inputs, contractive_inputs], outputs=student_cblock)


    comparator(summary(model_instructor), summary(model_learner))

    
# GRADED FUNCTION 3  
def unet_model_test(target):
    img_height = 96
    img_width = 128
    num_channels = 3

    model_instructor = solutions.unet_model((img_height, img_width, num_channels))
    model_learner = target((img_height, img_width, num_channels))
    comparator(summary(model_instructor), summary(model_learner))
    
    img_height = 96 * 2
    img_width = int(128 / 2)
    num_channels = 4
    n_classes=16

    model_instructor = solutions.unet_model((img_height, img_width, num_channels), n_filters= 64, n_classes=16)
    model_learner = target((img_height, img_width, num_channels), n_filters= 64, n_classes=16)
    comparator(summary(model_instructor), summary(model_learner))
