# Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

To complete this assignment, you should already be familiar with TensorFlow. If you are not, please refer back to the **TensorFlow Tutorial** of the third week of Course 2 ("**Improving deep neural networks**").

## Table of Contents

- [1 - Packages](#1)
    - [1.1 - Load the Data and Split the Data into Train/Test Sets](#1-1)
- [2 - Layers in TF Keras](#2)
- [3 - The Sequential API](#3)
    - [3.1 - Create the Sequential Model](#3-1)
        - [Exercise 1 - happyModel](#ex-1)
    - [3.2 - Train and Evaluate the Model](#3-2)
- [4 - The Functional API](#4)
    - [4.1 - Load the SIGNS Dataset](#4-1)
    - [4.2 - Split the Data into Train/Test Sets](#4-2)
    - [4.3 - Forward Propagation](#4-3)
        - [Exercise 2 - convolutional_model](#ex-2)
    - [4.4 - Train the Model](#4-4)
- [5 - History Object](#5)
- [6 - Bibliography](#6)

<a name='1'></a>
## 1 - Packages

As usual, begin by loading in the packages.


```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

<a name='1-1'></a>
### 1.1 - Load the Data and Split the Data into Train/Test Sets

You'll be using the Happy House dataset for this part of the assignment, which contains images of peoples' faces. Your task will be to build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!  


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)


You can display the images contained in the dataset. Images are **64x64** pixels in RGB format (3 channels).


```python
index = 124
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
```


![png](output_7_0.png)


<a name='2'></a>
## 2 - Layers in TF Keras 

In the previous assignment, you created layers manually in numpy. In TF Keras, you don't have to write code directly to create layers. Rather, TF Keras has pre-defined layers you can use. 

When you create a layer in TF Keras, you are creating a function that takes some input and transforms it into an output you can reuse later. Nice and easy! 

<a name='3'></a>
## 3 - The Sequential API

In the previous assignment, you built helper functions using `numpy` to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow, which allows for even more simplified and optimized model creation and training. 

For the first part of this assignment, you'll create a model using TF Keras' Sequential API, which allows you to build layer by layer, and is ideal for building models where each layer has **exactly one** input tensor and **one** output tensor. 

As you'll see, using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks. Later in this notebook you'll spend some time building with a more flexible, powerful alternative: the Functional API. 
 

<a name='3-1'></a>
### 3.1 - Create the Sequential Model

As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order. 

You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.

Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.  If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!

For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. Sequential models can be created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

<a name='ex-1'></a>
### Exercise 1 - happyModel

Implement the `happyModel` function below to build the following model: `ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 

Also, plug in the following parameters for all the steps:

 - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
 - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 1 neuron and a sigmoid activation. 
 
 
 **Hint:**
 
 Use **tfl** as shorthand for **tensorflow.keras.layers**


```python
# GRADED FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            
            ## Conv2D with 32 7x7 filters and stride of 1
            
            ## BatchNormalization for axis 3
            
            ## ReLU
            
            ## Max Pooling 2D with default parameters
            
            ## Flatten layer
            
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            
            # YOUR CODE STARTS HERE
            tfl.ZeroPadding2D(padding=(3, 3),input_shape=(64, 64, 3)),
            tfl.Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1)),
            tfl.BatchNormalization(axis=3),
            tfl.ReLU(),
            tfl.MaxPool2D(),
            tfl.Flatten(),
            tfl.Dense(units=1, activation='sigmoid'),
            # YOUR CODE ENDS HERE
        ])
    
    return model
```


```python
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)
    
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]
    
comparator(summary(happy_model), output)
```

    ['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
    ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
    ['BatchNormalization', (None, 64, 64, 32), 128]
    ['ReLU', (None, 64, 64, 32), 0]
    ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
    ['Flatten', (None, 32768), 0]
    ['Dense', (None, 1), 32769, 'sigmoid']
    [32mAll tests passed![0m


Now that your model is created, you can compile it for training with an optimizer and loss of your choice. When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).


```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

It's time to check your model's parameters with the `.summary()` method. This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer. 


```python
happy_model.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    zero_padding2d_1 (ZeroPaddin (None, 70, 70, 3)         0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 64, 64, 32)        4736      
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 64, 64, 32)        128       
    _________________________________________________________________
    re_lu_5 (ReLU)               (None, 64, 64, 32)        0         
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 32, 32, 32)        0         
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 32768)             0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 32769     
    =================================================================
    Total params: 37,633
    Trainable params: 37,569
    Non-trainable params: 64
    _________________________________________________________________


<a name='3-2'></a>
### 3.2 - Train and Evaluate the Model

After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents, you are now ready to build! 

Simply call `.fit()` to train. That's it! No need for mini-batching, saving, or complex backpropagation computations. That's all been done for you, as you're using a TensorFlow dataset with the batches specified already. You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).


```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```

    Epoch 1/10
    38/38 [==============================] - 4s 98ms/step - loss: 0.5970 - accuracy: 0.7983
    Epoch 2/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.3493 - accuracy: 0.8783
    Epoch 3/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1935 - accuracy: 0.9083
    Epoch 4/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1497 - accuracy: 0.9450
    Epoch 5/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1844 - accuracy: 0.9267
    Epoch 6/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1688 - accuracy: 0.9317
    Epoch 7/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1372 - accuracy: 0.9483
    Epoch 8/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1284 - accuracy: 0.9667
    Epoch 9/10
    38/38 [==============================] - 4s 92ms/step - loss: 0.1515 - accuracy: 0.9433
    Epoch 10/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0791 - accuracy: 0.9700





    <tensorflow.python.keras.callbacks.History at 0x7fe2b94c6e50>



After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.


```python
happy_model.evaluate(X_test, Y_test)
```

    5/5 [==============================] - 0s 25ms/step - loss: 0.4612 - accuracy: 0.8000





    [0.46122899651527405, 0.800000011920929]



Easy, right? But what if you need to build a model with shared layers, branches, or multiple inputs and outputs? This is where Sequential, with its beautifully simple yet limited functionality, won't be able to help you. 

Next up: Enter the Functional API, your slightly more complex, highly flexible friend.  

<a name='4'></a>
## 4 - The Functional API

Welcome to the second half of the assignment, where you'll use Keras' flexible [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that can differentiate between 6 sign language digits. 

The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect in many more ways than one. 

In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection, which is just one of the many ways a Functional model can be constructed. A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network. Don't worry, you'll be spending more time with skip connections very soon! 

<img src="images/seq_vs_func.png" style="width:350px;height:200px;">

<a name='4-1'></a>
### 4.1 - Load the SIGNS Dataset

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.


```python
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
```

<img src="images/SIGNS.png" style="width:800px;height:300px;">

The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 


```python
# Example of an image from the dataset
index = 9
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```

    y = 4



![png](output_28_1.png)


<a name='4-2'></a>
### 4.2 - Split the Data into Train/Test Sets

In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data. 


```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 1080
    number of test examples = 120
    X_train shape: (1080, 64, 64, 3)
    Y_train shape: (1080, 6)
    X_test shape: (120, 64, 64, 3)
    Y_test shape: (120, 6)


<a name='4-3'></a>
### 4.3 - Forward Propagation

In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:

- **input_img = tf.keras.Input(shape=input_shape):** 

Then, create a new node in the graph of layers by calling a layer on the `input_img` object: 

- **tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img):** Read the full documentation on [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- **tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'):** `MaxPool2D()` downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.  For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).

- **tf.keras.layers.ReLU():** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU).

- **tf.keras.layers.Flatten()**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.  

    * If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where $k=h \times w \times c$.  "k" equals the product of all the dimension sizes other than the first dimension.
    
    * For example, given a tensor with dimensions [100, 2, 3, 4], it flattens the tensor to be of shape [100, 24], where 24 = 2 * 3 * 4.  You can read the full documentation on [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten).

- **tf.keras.layers.Dense(units= ... , activation='softmax')(F):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

In the last function above (`tf.keras.layers.Dense()`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer): 

- **outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)**


#### Window, kernel, filter, pool

The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels. 

This is why the parameter `pool_size` refers to `kernel_size`, and you use `(f,f)` to refer to the filter size. 

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place. 

<a name='ex-2'></a>
### Exercise 2 - convolutional_model

Implement the `convolutional_model` function below to build the following model: `CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Use the functions above! 

Also, plug in the following parameters for all the steps:

 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
 - **ReLU**
 - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 6 neurons and a softmax activation. 


```python
# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    # Z1 = None
    ## RELU
    # A1 = None
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    # P1 = None
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    # Z2 = None
    ## RELU
    # A2 = None
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # P2 = None
    ## FLATTEN
    # F = None
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    # outputs = None
    # YOUR CODE STARTS HERE
    Z1 = tfl.Conv2D(filters= 8, kernel_size= (4, 4), strides=1, padding='same')(input_img)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(8, 8), strides=8, padding='same')(A1)
    Z2 = tfl.Conv2D(filters= 16, kernel_size= (2, 2), strides=1, padding='same')(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(4, 4), strides=4, padding='same')(A2)
    F = tfl.Flatten()(P2)
    outputs = tfl.Dense(6, activation='softmax')(F)
    
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```


```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
```

    Model: "functional_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         [(None, 64, 64, 3)]       0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 64, 64, 8)         392       
    _________________________________________________________________
    re_lu_8 (ReLU)               (None, 64, 64, 8)         0         
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 8, 8, 8)           0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 8, 8, 16)          528       
    _________________________________________________________________
    re_lu_9 (ReLU)               (None, 8, 8, 16)          0         
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 2, 2, 16)          0         
    _________________________________________________________________
    flatten_7 (Flatten)          (None, 64)                0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 6)                 390       
    =================================================================
    Total params: 1,310
    Trainable params: 1,310
    Non-trainable params: 0
    _________________________________________________________________
    [32mAll tests passed![0m


Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model! 

<a name='4-4'></a>
### 4.4 - Train the Model


```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```

    Epoch 1/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.8193 - accuracy: 0.1519 - val_loss: 1.7971 - val_accuracy: 0.1750
    Epoch 2/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.7921 - accuracy: 0.1759 - val_loss: 1.7923 - val_accuracy: 0.1583
    Epoch 3/100
    17/17 [==============================] - ETA: 0s - loss: 1.7861 - accuracy: 0.18 - 2s 111ms/step - loss: 1.7861 - accuracy: 0.1880 - val_loss: 1.7864 - val_accuracy: 0.1750
    Epoch 4/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7779 - accuracy: 0.2611 - val_loss: 1.7803 - val_accuracy: 0.1750
    Epoch 5/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7711 - accuracy: 0.2685 - val_loss: 1.7740 - val_accuracy: 0.2583
    Epoch 6/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7623 - accuracy: 0.2898 - val_loss: 1.7673 - val_accuracy: 0.2667
    Epoch 7/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7531 - accuracy: 0.3315 - val_loss: 1.7597 - val_accuracy: 0.3417
    Epoch 8/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7410 - accuracy: 0.3722 - val_loss: 1.7497 - val_accuracy: 0.3750
    Epoch 9/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7252 - accuracy: 0.3843 - val_loss: 1.7368 - val_accuracy: 0.3583
    Epoch 10/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.7060 - accuracy: 0.4259 - val_loss: 1.7226 - val_accuracy: 0.3917
    Epoch 11/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.6833 - accuracy: 0.4306 - val_loss: 1.7063 - val_accuracy: 0.3917
    Epoch 12/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.6593 - accuracy: 0.4444 - val_loss: 1.6892 - val_accuracy: 0.3917
    Epoch 13/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.6331 - accuracy: 0.4556 - val_loss: 1.6687 - val_accuracy: 0.3833
    Epoch 14/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.6039 - accuracy: 0.4648 - val_loss: 1.6477 - val_accuracy: 0.3917
    Epoch 15/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.5741 - accuracy: 0.4769 - val_loss: 1.6252 - val_accuracy: 0.4083
    Epoch 16/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.5429 - accuracy: 0.4824 - val_loss: 1.6010 - val_accuracy: 0.4250
    Epoch 17/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.5106 - accuracy: 0.4898 - val_loss: 1.5739 - val_accuracy: 0.4333
    Epoch 18/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.4773 - accuracy: 0.5148 - val_loss: 1.5457 - val_accuracy: 0.4417
    Epoch 19/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.4450 - accuracy: 0.5204 - val_loss: 1.5165 - val_accuracy: 0.4583
    Epoch 20/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.4129 - accuracy: 0.5352 - val_loss: 1.4865 - val_accuracy: 0.4667
    Epoch 21/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.3807 - accuracy: 0.5472 - val_loss: 1.4581 - val_accuracy: 0.5000
    Epoch 22/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.3485 - accuracy: 0.5565 - val_loss: 1.4273 - val_accuracy: 0.5000
    Epoch 23/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.3158 - accuracy: 0.5741 - val_loss: 1.3951 - val_accuracy: 0.5167
    Epoch 24/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.2855 - accuracy: 0.5843 - val_loss: 1.3656 - val_accuracy: 0.5417
    Epoch 25/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.2527 - accuracy: 0.5972 - val_loss: 1.3314 - val_accuracy: 0.5333
    Epoch 26/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.2230 - accuracy: 0.6037 - val_loss: 1.3008 - val_accuracy: 0.5500
    Epoch 27/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.1916 - accuracy: 0.6157 - val_loss: 1.2703 - val_accuracy: 0.5750
    Epoch 28/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.1632 - accuracy: 0.6204 - val_loss: 1.2433 - val_accuracy: 0.6083
    Epoch 29/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.1340 - accuracy: 0.6269 - val_loss: 1.2145 - val_accuracy: 0.5917
    Epoch 30/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.1078 - accuracy: 0.6324 - val_loss: 1.1861 - val_accuracy: 0.6000
    Epoch 31/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.0844 - accuracy: 0.6435 - val_loss: 1.1648 - val_accuracy: 0.6083
    Epoch 32/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.0570 - accuracy: 0.6435 - val_loss: 1.1440 - val_accuracy: 0.6333
    Epoch 33/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.0341 - accuracy: 0.6528 - val_loss: 1.1231 - val_accuracy: 0.6417
    Epoch 34/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.0122 - accuracy: 0.6611 - val_loss: 1.1049 - val_accuracy: 0.6083
    Epoch 35/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9910 - accuracy: 0.6731 - val_loss: 1.0851 - val_accuracy: 0.6167
    Epoch 36/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9695 - accuracy: 0.6824 - val_loss: 1.0668 - val_accuracy: 0.6167
    Epoch 37/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9522 - accuracy: 0.6861 - val_loss: 1.0535 - val_accuracy: 0.6333
    Epoch 38/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9330 - accuracy: 0.6917 - val_loss: 1.0345 - val_accuracy: 0.6250
    Epoch 39/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9173 - accuracy: 0.6963 - val_loss: 1.0235 - val_accuracy: 0.6333
    Epoch 40/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.9019 - accuracy: 0.7037 - val_loss: 1.0093 - val_accuracy: 0.6250
    Epoch 41/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.8868 - accuracy: 0.7046 - val_loss: 0.9978 - val_accuracy: 0.6167
    Epoch 42/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8733 - accuracy: 0.7083 - val_loss: 0.9827 - val_accuracy: 0.6333
    Epoch 43/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8609 - accuracy: 0.7083 - val_loss: 0.9742 - val_accuracy: 0.6250
    Epoch 44/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8480 - accuracy: 0.7139 - val_loss: 0.9594 - val_accuracy: 0.6333
    Epoch 45/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.8350 - accuracy: 0.7204 - val_loss: 0.9506 - val_accuracy: 0.6500
    Epoch 46/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8234 - accuracy: 0.7204 - val_loss: 0.9387 - val_accuracy: 0.6583
    Epoch 47/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8114 - accuracy: 0.7250 - val_loss: 0.9289 - val_accuracy: 0.6583
    Epoch 48/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8006 - accuracy: 0.7333 - val_loss: 0.9169 - val_accuracy: 0.6667
    Epoch 49/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.7905 - accuracy: 0.7380 - val_loss: 0.9095 - val_accuracy: 0.6667
    Epoch 50/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.7805 - accuracy: 0.7417 - val_loss: 0.9025 - val_accuracy: 0.6833
    Epoch 51/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7710 - accuracy: 0.7463 - val_loss: 0.8943 - val_accuracy: 0.6833
    Epoch 52/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.7624 - accuracy: 0.7472 - val_loss: 0.8860 - val_accuracy: 0.6917
    Epoch 53/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7535 - accuracy: 0.7500 - val_loss: 0.8770 - val_accuracy: 0.6833
    Epoch 54/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7450 - accuracy: 0.7519 - val_loss: 0.8672 - val_accuracy: 0.7000
    Epoch 55/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7369 - accuracy: 0.7546 - val_loss: 0.8608 - val_accuracy: 0.7083
    Epoch 56/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7297 - accuracy: 0.7583 - val_loss: 0.8558 - val_accuracy: 0.7167
    Epoch 57/100
    17/17 [==============================] - 2s 102ms/step - loss: 0.7226 - accuracy: 0.7593 - val_loss: 0.8494 - val_accuracy: 0.7167
    Epoch 58/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.7157 - accuracy: 0.7611 - val_loss: 0.8420 - val_accuracy: 0.7167
    Epoch 59/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7088 - accuracy: 0.7620 - val_loss: 0.8350 - val_accuracy: 0.7167
    Epoch 60/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7021 - accuracy: 0.7639 - val_loss: 0.8280 - val_accuracy: 0.7083
    Epoch 61/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6954 - accuracy: 0.7657 - val_loss: 0.8212 - val_accuracy: 0.7000
    Epoch 62/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.6890 - accuracy: 0.7704 - val_loss: 0.8148 - val_accuracy: 0.7000
    Epoch 63/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6826 - accuracy: 0.7685 - val_loss: 0.8086 - val_accuracy: 0.7083
    Epoch 64/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6759 - accuracy: 0.7750 - val_loss: 0.8036 - val_accuracy: 0.7167
    Epoch 65/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6698 - accuracy: 0.7769 - val_loss: 0.7991 - val_accuracy: 0.7083
    Epoch 66/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6634 - accuracy: 0.7778 - val_loss: 0.7882 - val_accuracy: 0.7167
    Epoch 67/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6616 - accuracy: 0.7759 - val_loss: 0.7881 - val_accuracy: 0.7083
    Epoch 68/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6535 - accuracy: 0.7769 - val_loss: 0.7803 - val_accuracy: 0.7083
    Epoch 69/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6461 - accuracy: 0.7870 - val_loss: 0.7726 - val_accuracy: 0.7083
    Epoch 70/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6398 - accuracy: 0.7861 - val_loss: 0.7659 - val_accuracy: 0.7167
    Epoch 71/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6328 - accuracy: 0.7889 - val_loss: 0.7591 - val_accuracy: 0.7167
    Epoch 72/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6264 - accuracy: 0.7926 - val_loss: 0.7538 - val_accuracy: 0.7333
    Epoch 73/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6204 - accuracy: 0.7944 - val_loss: 0.7482 - val_accuracy: 0.7333
    Epoch 74/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6146 - accuracy: 0.7954 - val_loss: 0.7437 - val_accuracy: 0.7333
    Epoch 75/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6092 - accuracy: 0.7963 - val_loss: 0.7389 - val_accuracy: 0.7250
    Epoch 76/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6038 - accuracy: 0.8000 - val_loss: 0.7344 - val_accuracy: 0.7333
    Epoch 77/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5989 - accuracy: 0.8009 - val_loss: 0.7305 - val_accuracy: 0.7333
    Epoch 78/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5939 - accuracy: 0.8037 - val_loss: 0.7263 - val_accuracy: 0.7417
    Epoch 79/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5890 - accuracy: 0.8046 - val_loss: 0.7224 - val_accuracy: 0.7417
    Epoch 80/100
    17/17 [==============================] - 2s 102ms/step - loss: 0.5840 - accuracy: 0.8056 - val_loss: 0.7169 - val_accuracy: 0.7500
    Epoch 81/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5791 - accuracy: 0.8102 - val_loss: 0.7140 - val_accuracy: 0.7500
    Epoch 82/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5745 - accuracy: 0.8120 - val_loss: 0.7086 - val_accuracy: 0.7583
    Epoch 83/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5697 - accuracy: 0.8167 - val_loss: 0.7069 - val_accuracy: 0.7667
    Epoch 84/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5654 - accuracy: 0.8139 - val_loss: 0.7009 - val_accuracy: 0.7583
    Epoch 85/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.5609 - accuracy: 0.8167 - val_loss: 0.6989 - val_accuracy: 0.7667
    Epoch 86/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.5562 - accuracy: 0.8213 - val_loss: 0.6942 - val_accuracy: 0.7583
    Epoch 87/100
    17/17 [==============================] - 2s 113ms/step - loss: 0.5520 - accuracy: 0.8222 - val_loss: 0.6914 - val_accuracy: 0.7667
    Epoch 88/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5477 - accuracy: 0.8241 - val_loss: 0.6870 - val_accuracy: 0.7667
    Epoch 89/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5436 - accuracy: 0.8241 - val_loss: 0.6857 - val_accuracy: 0.7583
    Epoch 90/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5397 - accuracy: 0.8250 - val_loss: 0.6811 - val_accuracy: 0.7583
    Epoch 91/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5356 - accuracy: 0.8259 - val_loss: 0.6780 - val_accuracy: 0.7583
    Epoch 92/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5316 - accuracy: 0.8259 - val_loss: 0.6743 - val_accuracy: 0.7583
    Epoch 93/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5276 - accuracy: 0.8296 - val_loss: 0.6710 - val_accuracy: 0.7583
    Epoch 94/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5239 - accuracy: 0.8315 - val_loss: 0.6687 - val_accuracy: 0.7583
    Epoch 95/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5201 - accuracy: 0.8324 - val_loss: 0.6667 - val_accuracy: 0.7583
    Epoch 96/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5165 - accuracy: 0.8333 - val_loss: 0.6653 - val_accuracy: 0.7583
    Epoch 97/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5128 - accuracy: 0.8333 - val_loss: 0.6619 - val_accuracy: 0.7583
    Epoch 98/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5090 - accuracy: 0.8361 - val_loss: 0.6592 - val_accuracy: 0.7667
    Epoch 99/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5054 - accuracy: 0.8380 - val_loss: 0.6571 - val_accuracy: 0.7667
    Epoch 100/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.5018 - accuracy: 0.8389 - val_loss: 0.6554 - val_accuracy: 0.7667


<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 


```python
history.history
```




    {'loss': [1.819347858428955,
      1.7920873165130615,
      1.7861220836639404,
      1.7778719663619995,
      1.771108865737915,
      1.7622989416122437,
      1.753074288368225,
      1.7409650087356567,
      1.72517728805542,
      1.706038236618042,
      1.6832777261734009,
      1.6592721939086914,
      1.6330815553665161,
      1.6039049625396729,
      1.574088215827942,
      1.542901635169983,
      1.5105674266815186,
      1.4772531986236572,
      1.4450256824493408,
      1.4129164218902588,
      1.380662441253662,
      1.3484872579574585,
      1.3157747983932495,
      1.2855350971221924,
      1.2526651620864868,
      1.222963809967041,
      1.191603183746338,
      1.163212537765503,
      1.1340484619140625,
      1.1077618598937988,
      1.084381341934204,
      1.0570499897003174,
      1.034116268157959,
      1.0121793746948242,
      0.9909574389457703,
      0.9694802761077881,
      0.9522454142570496,
      0.9330172538757324,
      0.9173436164855957,
      0.9019206762313843,
      0.8868355751037598,
      0.8732669353485107,
      0.8609331846237183,
      0.8480163812637329,
      0.8350375294685364,
      0.8233689069747925,
      0.811370313167572,
      0.800631046295166,
      0.7904532551765442,
      0.7804616093635559,
      0.7710326910018921,
      0.7624229788780212,
      0.7534827589988708,
      0.7449743151664734,
      0.7369307279586792,
      0.7296881079673767,
      0.7225639820098877,
      0.7156693339347839,
      0.708752453327179,
      0.7020761370658875,
      0.6954045295715332,
      0.6889812350273132,
      0.6826162934303284,
      0.6758984327316284,
      0.6698170900344849,
      0.6633802652359009,
      0.6615796089172363,
      0.6534919738769531,
      0.6460631489753723,
      0.6398306488990784,
      0.6328279376029968,
      0.6264326572418213,
      0.6204473376274109,
      0.6146408319473267,
      0.6092361211776733,
      0.6037893891334534,
      0.5989128947257996,
      0.5938661694526672,
      0.5889595150947571,
      0.5840051770210266,
      0.5790795683860779,
      0.5744581818580627,
      0.569685697555542,
      0.5653889775276184,
      0.5609118342399597,
      0.556227445602417,
      0.5520403981208801,
      0.5476565957069397,
      0.5435534715652466,
      0.5397315621376038,
      0.5355644226074219,
      0.5316352248191833,
      0.5276356339454651,
      0.5238587856292725,
      0.5201448798179626,
      0.5164870619773865,
      0.5128178000450134,
      0.5089683532714844,
      0.5053731203079224,
      0.501769483089447],
     'accuracy': [0.1518518477678299,
      0.17592592537403107,
      0.18796296417713165,
      0.2611111104488373,
      0.26851850748062134,
      0.2898148000240326,
      0.3314814865589142,
      0.3722222149372101,
      0.38425925374031067,
      0.42592594027519226,
      0.4305555522441864,
      0.4444444477558136,
      0.4555555582046509,
      0.46481481194496155,
      0.47685185074806213,
      0.48240742087364197,
      0.489814817905426,
      0.5148147940635681,
      0.520370364189148,
      0.5351851582527161,
      0.5472221970558167,
      0.5564814805984497,
      0.5740740895271301,
      0.5842592716217041,
      0.5972222089767456,
      0.6037036776542664,
      0.6157407164573669,
      0.6203703880310059,
      0.6268518567085266,
      0.6324074268341064,
      0.6435185074806213,
      0.6435185074806213,
      0.6527777910232544,
      0.6611111164093018,
      0.6731481552124023,
      0.6824073791503906,
      0.6861110925674438,
      0.6916666626930237,
      0.6962962746620178,
      0.7037037014961243,
      0.7046296000480652,
      0.7083333134651184,
      0.7083333134651184,
      0.7138888835906982,
      0.720370352268219,
      0.720370352268219,
      0.7250000238418579,
      0.7333333492279053,
      0.7379629611968994,
      0.7416666746139526,
      0.7462962865829468,
      0.7472222447395325,
      0.75,
      0.7518518567085266,
      0.7546296119689941,
      0.7583333253860474,
      0.7592592835426331,
      0.7611111402511597,
      0.7620370388031006,
      0.7638888955116272,
      0.7657407522201538,
      0.770370364189148,
      0.7685185074806213,
      0.7749999761581421,
      0.7768518328666687,
      0.7777777910232544,
      0.7759259343147278,
      0.7768518328666687,
      0.7870370149612427,
      0.7861111164093018,
      0.7888888716697693,
      0.7925925850868225,
      0.7944444417953491,
      0.7953703999519348,
      0.7962962985038757,
      0.800000011920929,
      0.8009259104728699,
      0.8037037253379822,
      0.8046296238899231,
      0.8055555820465088,
      0.8101851940155029,
      0.8120370507240295,
      0.8166666626930237,
      0.8138889074325562,
      0.8166666626930237,
      0.8212962746620178,
      0.8222222328186035,
      0.8240740895271301,
      0.8240740895271301,
      0.824999988079071,
      0.8259259462356567,
      0.8259259462356567,
      0.8296296000480652,
      0.8314814567565918,
      0.8324074149131775,
      0.8333333134651184,
      0.8333333134651184,
      0.8361111283302307,
      0.8379629850387573,
      0.8388888835906982],
     'val_loss': [1.7970744371414185,
      1.7923246622085571,
      1.7863564491271973,
      1.7803237438201904,
      1.77401864528656,
      1.7672897577285767,
      1.7597222328186035,
      1.7496953010559082,
      1.7368139028549194,
      1.7225733995437622,
      1.7062588930130005,
      1.68922758102417,
      1.6686882972717285,
      1.647699236869812,
      1.6251922845840454,
      1.600988745689392,
      1.5739328861236572,
      1.5456984043121338,
      1.516467571258545,
      1.486523151397705,
      1.4581080675125122,
      1.4273287057876587,
      1.3951166868209839,
      1.3656288385391235,
      1.3313772678375244,
      1.300817608833313,
      1.2702583074569702,
      1.2432886362075806,
      1.2144734859466553,
      1.1860830783843994,
      1.1647945642471313,
      1.1439989805221558,
      1.1231218576431274,
      1.104945421218872,
      1.0851153135299683,
      1.066766381263733,
      1.0535023212432861,
      1.0344812870025635,
      1.023495078086853,
      1.0093142986297607,
      0.9977531433105469,
      0.9827293753623962,
      0.9742087721824646,
      0.9593563079833984,
      0.9505912661552429,
      0.9387474060058594,
      0.9289032220840454,
      0.9169221520423889,
      0.9095070958137512,
      0.9025240540504456,
      0.8943018317222595,
      0.8860286474227905,
      0.8769620060920715,
      0.8671643137931824,
      0.86077880859375,
      0.8557845950126648,
      0.8494256138801575,
      0.8419667482376099,
      0.8349928259849548,
      0.8279778957366943,
      0.8211589455604553,
      0.8147865533828735,
      0.8085767030715942,
      0.8035898208618164,
      0.7991265058517456,
      0.7881837487220764,
      0.7881218194961548,
      0.7802972197532654,
      0.7725827097892761,
      0.7658604979515076,
      0.759064793586731,
      0.7538014650344849,
      0.7482097744941711,
      0.743740975856781,
      0.7389205098152161,
      0.7343528270721436,
      0.7305295467376709,
      0.7262809872627258,
      0.7223749160766602,
      0.716870129108429,
      0.7139986753463745,
      0.7086346745491028,
      0.7068785429000854,
      0.700948178768158,
      0.6988912224769592,
      0.6942367553710938,
      0.6914339661598206,
      0.6869889497756958,
      0.6857143640518188,
      0.6811069250106812,
      0.6780271530151367,
      0.6742923855781555,
      0.670995831489563,
      0.6687166094779968,
      0.666675329208374,
      0.6652816534042358,
      0.661935567855835,
      0.6591681838035583,
      0.6570838093757629,
      0.6554371118545532],
     'val_accuracy': [0.17499999701976776,
      0.15833333134651184,
      0.17499999701976776,
      0.17499999701976776,
      0.25833332538604736,
      0.2666666805744171,
      0.34166666865348816,
      0.375,
      0.3583333194255829,
      0.3916666805744171,
      0.3916666805744171,
      0.3916666805744171,
      0.38333332538604736,
      0.3916666805744171,
      0.40833333134651184,
      0.42500001192092896,
      0.4333333373069763,
      0.4416666626930237,
      0.4583333432674408,
      0.46666666865348816,
      0.5,
      0.5,
      0.5166666507720947,
      0.5416666865348816,
      0.5333333611488342,
      0.550000011920929,
      0.574999988079071,
      0.6083333492279053,
      0.5916666388511658,
      0.6000000238418579,
      0.6083333492279053,
      0.6333333253860474,
      0.6416666507720947,
      0.6083333492279053,
      0.6166666746139526,
      0.6166666746139526,
      0.6333333253860474,
      0.625,
      0.6333333253860474,
      0.625,
      0.6166666746139526,
      0.6333333253860474,
      0.625,
      0.6333333253860474,
      0.6499999761581421,
      0.6583333611488342,
      0.6583333611488342,
      0.6666666865348816,
      0.6666666865348816,
      0.6833333373069763,
      0.6833333373069763,
      0.6916666626930237,
      0.6833333373069763,
      0.699999988079071,
      0.7083333134651184,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7083333134651184,
      0.699999988079071,
      0.699999988079071,
      0.7083333134651184,
      0.7166666388511658,
      0.7083333134651184,
      0.7166666388511658,
      0.7083333134651184,
      0.7083333134651184,
      0.7083333134651184,
      0.7166666388511658,
      0.7166666388511658,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7250000238418579,
      0.7333333492279053,
      0.7333333492279053,
      0.7416666746139526,
      0.7416666746139526,
      0.75,
      0.75,
      0.7583333253860474,
      0.7666666507720947,
      0.7583333253860474,
      0.7666666507720947,
      0.7583333253860474,
      0.7666666507720947,
      0.7666666507720947,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7666666507720947,
      0.7666666507720947,
      0.7666666507720947]}



Now visualize the loss over time using `history.history`: 


```python
# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```




    [Text(0, 0.5, 'Accuracy'), Text(0.5, 0, 'Epoch')]




![png](output_41_1.png)



![png](output_41_2.png)


**Congratulations**! You've finished the assignment and built two models: One that recognizes  smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done! 

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets! 

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional
