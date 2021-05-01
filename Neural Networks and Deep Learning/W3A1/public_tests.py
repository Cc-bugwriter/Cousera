import numpy as np
from test_utils import single_test, multiple_test

         
def layer_sizes_test(target):
    np.random.seed(1)
    X = np.random.randn(5, 3)
    Y = np.random.randn(2, 3)
    expected_output = (5, 4, 2)
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X, Y],
            "expected": expected_output,
            "error":"Datatype mismatch."
        },
        {
            "name": "equation_output_check",
            "input": [X, Y],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)
    
            
        
def initialize_parameters_test(target):
    n_x, n_h, n_y = 2, 4, 1
    expected_W1 = np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]])
    expected_b1 = np.array([[0.],
                            [0.],
                            [0.],
                            [0.]])
    expected_W2 = np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]])
    expected_b2 = np.array([[0.]])
    expected_output = {"W1": expected_W1,
                  "b1": expected_b1,
                  "W2": expected_W2,
                  "b2": expected_b2}
    test_cases = [
        {
            "name":"datatype_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [n_x, n_h, n_y],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)

def forward_propagation_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    b1 = np.random.randn(4,1)
    b2 = np.array([[ -1.3]])

    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': b1,
     'b2': b2}
    expected_A1 = np.array([[ 0.9400694 ,  0.94101876,  0.94118266],
                     [-0.67151964, -0.62547205, -0.65709025],
                     [ 0.29034152,  0.31196971,  0.33449821],
                     [-0.22397799, -0.25730819, -0.2197236 ]])
    expected_A2 = np.array([[0.21292656, 0.21274673, 0.21295976]])

    expected_Z1 = np.array([[ 1.7386459 ,  1.74687437,  1.74830797],
                        [-0.81350569, -0.73394355, -0.78767559],
                        [ 0.29893918,  0.32272601,  0.34788465],
                        [-0.2278403 , -0.2632236 , -0.22336567]])

    expected_Z2 = np.array([[-1.30737426, -1.30844761, -1.30717618]])
    expected_cache = {"Z1": expected_Z1,
             "A1": expected_A1,
             "Z2": expected_Z2,
             "A2": expected_A2}
    expected_output = (expected_A2, expected_cache)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, parameters],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)

def compute_cost_test(target):
    np.random.seed(1)
    Y = (np.random.randn(1, 3) > 0)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'b2': np.array([[ 0.]])}

    A2 = (np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]))

    expected_output = 0.6930587610394646
    test_cases = [
        {
            "name":"datatype_check",
            "input": [A2, Y],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "equation_output_check",
            "input": [A2, Y],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)    
        
def backward_propagation_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    Y = (np.random.randn(1, 3) > 0)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
     'b2': np.array([[ 0.]])}

    cache = {'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619],
         [-0.05225116,  0.02725659, -0.02646251],
         [-0.02009721,  0.0036869 ,  0.02883756],
         [ 0.02152675, -0.01385234,  0.02599885]]),
  'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]),
  'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ],
         [-0.05229879,  0.02726335, -0.02646869],
         [-0.02009991,  0.00368692,  0.02884556],
         [ 0.02153007, -0.01385322,  0.02600471]]),
  'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])}
    
    expected_dW1 = np.array([[ 0.00301023, -0.00747267],
        [ 0.00257968, -0.00641288],
        [-0.00156892,  0.003893  ],
        [-0.00652037,  0.01618243]])
    expected_dW2 = np.array([[ 0.00078841,  0.01765429, -0.00084166, -0.01022527]])
    expected_db1 = np.array([[ 0.00176201],
            [ 0.00150995],
            [-0.00091736],
            [-0.00381422]])
    expected_db2 = np.array([[-0.16655712]])
    expected_output = {"dW1": expected_dW1,
             "db1": expected_db1,
             "dW2": expected_dW2,
             "db2": expected_db2}
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, cache, X, Y],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)

def update_parameters_test(target):
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
 'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
 'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
 'b2': np.array([[  9.14954378e-05]])}

    grads = {'dW1': np.array([[ 0.00023322, -0.00205423],
        [ 0.00082222, -0.00700776],
        [-0.00031831,  0.0028636 ],
        [-0.00092857,  0.00809933]]),
 'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03,
          -2.55715317e-03]]),
 'db1': np.array([[  1.05570087e-07],
        [ -3.81814487e-06],
        [ -1.90155145e-07],
        [  5.46467802e-07]]),
 'db2': np.array([[ -1.08923140e-05]])}
    
    expected_W1 = np.array([[-0.00643025,  0.01936718],
        [-0.02410458,  0.03978052],
        [-0.01653973, -0.02096177],
        [ 0.01046864, -0.05990141]])
    expected_b1 = np.array([[-1.02420756e-06],
            [ 1.27373948e-05],
            [ 8.32996807e-07],
            [-3.20136836e-06]])
    expected_W2 = np.array([[-0.01041081, -0.04463285,  0.01758031,  0.04747113]])
    expected_b2 = np.array([[0.00010457]])
    
    expected_output = {"W1": expected_W1,
                  "b1": expected_b1,
                  "W2": expected_W2,
                  "b2": expected_b2}

    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, grads],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)
    
def nn_model_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    Y = (np.random.randn(1, 3) > 0)
    n_h = 4
    expected_W1 = np.array([[-0.65848169,  1.21866811],
        [-0.76204273,  1.39377573],
        [ 0.5792005 , -1.10397703],
        [ 0.76773391, -1.41477129]])
    expected_W2 = np.array([[-2.45566237, -3.27042274,  2.00784958,  3.36773273]])
    expected_b1 = np.array([[ 0.287592  ],
            [ 0.3511264 ],
            [-0.2431246 ],
            [-0.35772805]])
    expected_b2 = np.array([[0.20459656]])
    expected_output = {"W1": expected_W1,
                  "b1": expected_b1,
                  "W2": expected_W2,
                  "b2": expected_b2}

    test_cases = [
        {
            "name":"datatype_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, n_h],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)
    
def predict_test(target):
    np.random.seed(1)
    X = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
     'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
     'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
     'b2': np.array([[  9.14954378e-05]])}
    expected_output = np.array([[True, False, True]])

    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error":"Data type mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, X],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    single_test(test_cases, target)

    
