import numpy as np
from numpy import array
from testCases import *
from test_utils import single_test, multiple_test

         
def update_parameters_with_gd_test(target):
    np.random.seed(1)
    learning_rate = 0.01
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    expected_output = {'W1': np.array([[ 1.63535156, -0.62320365, -0.53718766],
        [-1.07799357,  0.85639907, -2.29470142]]),
 'b1': np.array([[ 1.74604067],
        [-0.75184921]]),
 'W2': np.array([[ 0.32171798, -0.25467393,  1.46902454],
        [-2.05617317, -0.31554548, -0.3756023 ],
        [ 1.1404819 , -1.09976462, -0.1612551 ]]),
 'b2': np.array([[-0.88020257],
        [ 0.02561572],
        [ 0.57539477]])}

    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [params, grads, learning_rate],
            "expected": expected_output,
            "error":"Datatype mismatch."
        },
        {
            "name": "shape_check",
            "input": [params, grads, learning_rate],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [params, grads, learning_rate],
            "expected": expected_output,
            "error": "Wrong output"
        }
        
    ]
    
    multiple_test(test_cases, target)
    
            
        
def random_mini_batches_test(target):
    np.random.seed(1)
    mini_batch_size = 2
    X = np.random.randn(7, 3)
    Y = np.random.randn(1, 3) < 0.5
    expected_output = [(np.array([[-0.52817175, -0.61175641],
         [-2.3015387 ,  0.86540763],
         [ 0.3190391 , -0.7612069 ],
         [-2.06014071,  1.46210794],
         [ 1.13376944, -0.38405435],
         [-0.87785842, -0.17242821],
         [-1.10061918,  0.58281521]]),
  np.array([[False, False]])),
 (np.array([[ 1.62434536],
         [-1.07296862],
         [ 1.74481176],
         [-0.24937038],
         [-0.3224172 ],
         [-1.09989127],
         [ 0.04221375]]),
  np.array([[False]]))]
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)

def initialize_velocity_test(target):
    parameters = initialize_velocity_test_case()
    expected_output = {'dW1': np.array([[0., 0., 0.],
        [0., 0., 0.]]),
 'db1': np.array([[0.],
        [0.]]),
 'dW2': np.array([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]),
 'db2': np.array([[0.],
        [0.],
        [0.]])}
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)

def update_parameters_with_momentum_test(target):
    parameters, grads, v = update_parameters_with_momentum_test_case()
    beta = 0.9
    learning_rate = 0.01
    expected_parameters = {'W1': np.array([[ 1.62544598, -0.61290114, -0.52907334],
        [-1.07347112,  0.86450677, -2.30085497]]),
 'b1': np.array([[ 1.74493465],
        [-0.76027113]]),
 'W2': np.array([[ 0.31930698, -0.24990073,  1.4627996 ],
        [-2.05974396, -0.32173003, -0.38320915],
        [ 1.13444069, -1.0998786 , -0.1713109 ]]),
 'b2': np.array([[-0.87809283],
        [ 0.04055394],
        [ 0.58207317]])}
    expected_v = {'dW1': np.array([[-0.11006192,  0.11447237,  0.09015907],
        [ 0.05024943,  0.09008559, -0.06837279]]),
 'dW2': np.array([[-0.02678881,  0.05303555, -0.06916608],
        [-0.03967535, -0.06871727, -0.08452056],
        [-0.06712461, -0.00126646, -0.11173103]]),
 'db1': np.array([[-0.01228902],
        [-0.09357694]]),
 'db2': np.array([[0.02344157],
        [0.16598022],
        [0.07420442]])}
    expected_output = (expected_parameters, expected_v)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)    
        
def initialize_adam_test(target):
    parameters = initialize_adam_test_case()
    expected_v = {'dW1': np.array([[0., 0., 0.],
        [0., 0., 0.]]),
 'db1': np.array([[0.],
        [0.]]),
 'dW2': np.array([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]),
 'db2': np.array([[0.],
        [0.],
        [0.]])}
    expected_s = {'dW1': np.array([[0., 0., 0.],
        [0., 0., 0.]]),
 'db1': np.array([[0.],
        [0.]]),
 'dW2': np.array([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]),
 'db2': np.array([[0.],
        [0.],
        [0.]])}
    expected_output = (expected_v, expected_s)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters],
            "expected": expected_output,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)

def update_parameters_with_adam_test(target):
    parameters, grads, v, s = update_parameters_with_adam_test_case()
    t = 2
    learning_rate = 0.02
    beta1 = 0.8
    beta2 = 0.888
    epsilon = 1e-2
    
    expected_parameters = {'W1': np.array([[ 1.63949493, -0.62691477, -0.54326465], 
                                 [-1.08769515,  0.85031501, -2.28657079]]),
                            'b1': np.array([[1.7549895], [-0.7461017]]),
                            'W2': np.array([[ 0.33262355, -0.26414959,  1.47708248], 
                                 [-2.0457142 , -0.30744639, -0.36898502], 
                                 [ 1.14872646, -1.09849003, -0.15727519]]),
                            'b2': np.array([[-0.89102966], [0.02699863], [0.56780324]])}
            
    expected_v = {'v["dW1"]': np.array([[-0.22012384,  0.22894474,  0.18031814], 
                                       [ 0.10049887,  0.18017119, -0.13674557]]),
                            'v["dW2"]': np.array([[-0.05357762,  0.10607109, -0.13833215], 
                                       [-0.07935071, -0.13743454, -0.16904113], 
                                       [-0.13424923, -0.00253292, -0.22346207]]),
                            'v["db1"]': np.array([[-0.02457805], [-0.18715389]]),
                            'v["db2"]': np.array([[0.04688314], [0.33196044], [0.14840883]])}
    
    expected_s = {'s["dW1"]': np.array([[0.13567261, 0.14676395, 0.09104097], 
                                       [0.02828006, 0.09089264, 0.05235818]]),
                            's["dW2"]': np.array([[0.00803757, 0.03150302, 0.05358019], 
                                       [0.0176303 , 0.05288711, 0.08000973], 
                                       [5.04639932e-02, 1.79639114e-05, 1.39818830e-01]]),
                            's["db1"]': np.array([[0.00169142], [0.09807442]]),
                            's["db2"]': np.array([[0.00615448], [0.30855365], [0.06167051]])}
    
    expected_output = (expected_parameters, expected_v, expected_s)

    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)
def update_lr_test(target):
    learning_rate = 1
    epoch_num = 2
    decay_rate = 1
    expected_output = 0.3333333333333333
    test_cases = [
        {
            "name": "shape_check",
            "input": [learning_rate, epoch_num, decay_rate],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [learning_rate, epoch_num, decay_rate],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)
def schedule_lr_decay_test(target):
    learning_rate = 1
    epoch_num_1 = 100
    epoch_num_2 = 10
    decay_rate = 1
    time_interval = 100
    expected_output_1 = 0.5
    expected_output_2 = 1
    
    test_cases = [
        {
            "name": "shape_check",
            "input": [learning_rate, epoch_num_1, decay_rate, time_interval],
            "expected": expected_output_1,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [learning_rate, epoch_num_1, decay_rate, time_interval],
            "expected": expected_output_1,
            "error": "Wrong output"
        },
        {
            "name": "shape_check",
            "input": [learning_rate, epoch_num_2, decay_rate, time_interval],
            "expected": expected_output_2,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [learning_rate, epoch_num_2, decay_rate, time_interval],
            "expected": expected_output_2,
            "error": "Wrong output"
        }
    ]
    multiple_test(test_cases, target)
    