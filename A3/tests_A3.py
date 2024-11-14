import os
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal


from activations import Activation
from layers import FullyConnectedLayer
from loss import CrossEntropyLoss


###### TESTING RELU ##########
def test_activation_relu_forward():
    test_data = np.load(os.path.join(os.path.dirname(__file__), 'tests', 'test_files', 'TestReLU.npz'))
    act = Activation("relu")

    test_result = {"Passed Tests": [], "Failed Tests": []}
    desired, output = {}, {}
    start_test_num, end_test_num = 0, 6

    for i in range(start_test_num, end_test_num):
        input_data = test_data[str(i) + "input"]
        output_data = test_data[str(i) + "output"]
        act_output = act.forward(*input_data)
        try:
            assert_almost_equal(output_data, act_output, decimal=4)
            test_result["Passed Tests"].append(i)
        except:
            test_result["Failed Tests"].append(i)
            output[i] = act_output
            desired[i] = output_data

    test_result_print(test_result, end_test_num, start_test_num, output, desired)


def test_activation_relu_backward():
    test_data = np.load(os.path.join(os.path.dirname(__file__), 'tests', 'test_files', 'TestReLU.npz'))
    act = Activation("relu")

    test_result = {"Passed Tests": [], "Failed Tests": []}
    desired, output = {}, {}
    start_test_num, end_test_num = 0, 6

    for i in range(start_test_num, end_test_num):
        input_data = test_data[str(i) + "input"]
        upstream_grad = test_data[str(i) + "upstream_grad"]
        backward_data = test_data[str(i) + "backward"]
        backward_output = act.backward(*input_data, dY=upstream_grad)
        try:
            assert_almost_equal(backward_data, backward_output, decimal=4)
            test_result["Passed Tests"].append(i)
        except:
            test_result["Failed Tests"].append(i)
            output[i] = backward_output
            desired[i] = backward_data
    
    test_result_print(test_result, end_test_num, start_test_num, output, desired)



###### TESTING FULLY CONNECTED LAYER ##########
def test_fully_connected_forward_relu():
    test_data = np.load(os.path.join(os.path.dirname(__file__), 'tests', 'test_files', 'TestFullyConnected.npz'))
    
    test_result = {"Passed Tests": [], "Failed Tests": []}
    desired, output = {}, {}
    start_test_num, end_test_num = 0, 6

    for i in range(start_test_num, end_test_num):
        input_data = test_data[str(i) + "input"]
        output_data = test_data[str(i) + "output"]
        layer = FullyConnectedLayer(output_data.shape[1], "relu")
        
        layer.forward(*input_data)
        layer.parameters["W"] = test_data[str(i) + "W"]
        layer.parameters["b"] = test_data[str(i) + "b"]   
        layer_output = layer.forward(*input_data)
        try:
            assert_almost_equal(output_data, layer_output, decimal=4)
            test_result["Passed Tests"].append(i)
        except:
            test_result["Failed Tests"].append(i)
            output[i] = layer_output
            desired[i] = output_data

    test_result_print(test_result, end_test_num, start_test_num, output, desired)


def test_fully_connected_backward_relu():
    test_data = np.load(os.path.join(os.path.dirname(__file__), 'tests', 'test_files', 'TestFullyConnected.npz'))

    test_result = {"Passed Tests": [], "Failed Tests": []}
    desired, output = {}, {}
    start_test_num, end_test_num = 0, 6

    for i in range(start_test_num, end_test_num):
        input_data = test_data[str(i) + "input"]
        output_data = test_data[str(i) + "output"]
        upstream_grad = test_data[str(i) + "upstream_grad"]
        backward_data = test_data[str(i) + "backward"]
        layer = FullyConnectedLayer(output_data.shape[1], "relu")
        
        layer.forward(*input_data)
        layer.parameters["W"] = test_data[str(i) + "W"]
        layer.parameters["b"] = test_data[str(i) + "b"]   
        layer_output = layer.forward(*input_data)
        backward_output = layer.backward(upstream_grad)   
        try:
            assert_almost_equal(backward_data, backward_output, decimal=4)
            test_result["Passed Tests"].append(i)
        except:
            test_result["Failed Tests"].append(i)
            output[i] = backward_output
            desired[i] = backward_data             
        
    test_result_print(test_result, end_test_num, start_test_num, output, desired)


def test_fully_connected_forward_tanh():
    test_data = np.load(os.path.join(os.path.dirname(__file__), 'tests', 'test_files', 'TestFullyConnected.npz'))

    test_result = {"Passed Tests": [], "Failed Tests": []}
    desired, output = {}, {}
    start_test_num, end_test_num = 7, 18

    for i in range(start_test_num, end_test_num):
        input_data = test_data[str(i) + "input"]
        output_data = test_data[str(i) + "output"]
        layer = FullyConnectedLayer(output_data.shape[1], "tanh")
        
        layer.forward(*input_data)
        layer.parameters["W"] = test_data[str(i) + "W"]
        layer.parameters["b"] = test_data[str(i) + "b"]   
        layer_output = layer.forward(*input_data)
        try:
            assert_almost_equal(output_data, layer_output, decimal=4)
            test_result["Passed Tests"].append(i)
        except:
            test_result["Failed Tests"].append(i)
            output[i] = layer_output
            desired[i] = output_data
        
    test_result_print(test_result, end_test_num, start_test_num, output, desired)


def test_fully_connected_backward_tanh():
    test_data = np.load(os.path.join(os.path.dirname(__file__), 'tests', 'test_files', 'TestFullyConnected.npz'))

    test_result = {"Passed Tests": [], "Failed Tests": []}
    desired, output = {}, {}
    start_test_num, end_test_num = 7, 18

    for i in range(start_test_num, end_test_num):
        input_data = test_data[str(i) + "input"]
        output_data = test_data[str(i) + "output"]
        upstream_grad = test_data[str(i) + "upstream_grad"]
        backward_data = test_data[str(i) + "backward"]
        layer = FullyConnectedLayer(output_data.shape[1], "tanh")
        
        layer.forward(*input_data)
        layer.parameters["W"] = test_data[str(i) + "W"]
        layer.parameters["b"] = test_data[str(i) + "b"]   
        layer_output = layer.forward(*input_data)
        backward_output = layer.backward(upstream_grad)
        try:
            assert_almost_equal(backward_data, backward_output, decimal=4)
            test_result["Passed Tests"].append(i)
        except:
            test_result["Failed Tests"].append(i)
            output[i] = backward_output
            desired[i] = backward_data             
        
    test_result_print(test_result, end_test_num, start_test_num, output, desired)


###### TESTING SOFTMAX ##########
def test_activation_softmax_forward():
    test_data = np.load(os.path.join(os.path.dirname(__file__), 'tests', 'test_files', 'TestSoftMax.npz'))
    act = Activation("softmax")
    test_result = {"Passed Tests": [], "Failed Tests": []}
    desired, output = {}, {}
    start_test_num, end_test_num = 0, 6
    for i in range(start_test_num, end_test_num):
        input_data = test_data[str(i) + "input"]
        output_data = test_data[str(i) + "output"]
        act_output = act.forward(*input_data)
        try:
            assert_almost_equal(output_data, act_output, decimal=4)
            test_result["Passed Tests"].append(i)
        except:
            test_result["Failed Tests"].append(i)
            output[i] = act_output
            desired[i] = output_data
    test_result_print(test_result, end_test_num, start_test_num, output, desired)


def test_activation_softmax_backward():
    test_data = np.load(os.path.join(os.path.dirname(__file__), 'tests', 'test_files', 'TestSoftMax.npz'))
    act = Activation("softmax")

    test_result = {"Passed Tests": [], "Failed Tests": []}
    desired, output = {}, {}
    start_test_num, end_test_num = 0, 6

    for i in range(start_test_num, end_test_num):
        input_data = test_data[str(i) + "input"]
        upstream_grad = test_data[str(i) + "upstream_grad"]
        backward_data = test_data[str(i) + "backward"]
        backward_output = act.backward(*input_data, dY=upstream_grad)
        try:
            assert_almost_equal(backward_data, backward_output, decimal=4)
            test_result["Passed Tests"].append(i)
        except:
            test_result["Failed Tests"].append(i)
            output[i] = backward_output
            desired[i] = backward_data
        
    test_result_print(test_result, end_test_num, start_test_num, output, desired)


###### TESTING CROSS ENTROPY LOSS ##########
def test_loss_cross_entropy_forward():
    test_data = np.load(os.path.join(os.path.dirname(__file__), 'tests', 'test_files', 'TestCrossEntropy.npz'))
    loss = CrossEntropyLoss()
    
    test_result = {"Passed Tests": [], "Failed Tests": []}
    desired, output = {}, {}
    start_test_num, end_test_num = 0, 6

    for i in range(start_test_num, end_test_num):
        input_data = test_data[str(i) + "input"]
        output_data = test_data[str(i) + "output"]
        act_output = loss.forward(*input_data)
        try:
            assert_almost_equal(output_data, act_output, decimal=4)
            test_result["Passed Tests"].append(i)
        except:
            test_result["Failed Tests"].append(i)
            output[i] = act_output
            desired[i] = output_data
        
    test_result_print(test_result, end_test_num, start_test_num, output, desired)

def test_loss_cross_entropy_backward():
    test_data = np.load(os.path.join(os.path.dirname(__file__), 'tests', 'test_files', 'TestCrossEntropy.npz'))
    loss = CrossEntropyLoss()
    
    test_result = {"Passed Tests": [], "Failed Tests": []}
    desired, output = {}, {}
    start_test_num, end_test_num = 0, 6

    for i in range(start_test_num, end_test_num):
        input_data = test_data[str(i) + "input"]
        backward_data = test_data[str(i) + "backward"]
        backward_output = loss.backward(*input_data)
        try:
            assert_almost_equal(backward_data, backward_output, decimal=4)
            test_result["Passed Tests"].append(i)
        except:
            test_result["Failed Tests"].append(i)
            output[i] = backward_output
            desired[i] = backward_data
        
    test_result_print(test_result, end_test_num, start_test_num, output, desired)


###########################################################
def test_gradients_activation():

    failed = 0
    X = np.random.randn(2, 3)
    dLdY = np.random.randn(2, 3)
    print(f"Relative error for each test case is expected to be less than 1e-8")
    for activation_name in ["relu", "softmax"]:
        # initialize a fully connected layer
        # and perform a forward and backward pass
        activation = Activation(activation_name)
        _ = activation.forward(X)
        grad = activation.backward(X, dLdY)

        # check the gradients w.r.t. each parameter
        relative_error = check_gradients(
                                         fn=activation.forward,  # the function we are checking
                                         grad=grad,  # the analytically computed gradient
                                         x=X,        # the variable w.r.t. which we are taking the gradient
                                         dLdf=dLdY,  # gradient at previous layer
                                        )
        if relative_error > 1e-8:
            failed += 1
        print(f"Relative error for {activation_name} activation:", relative_error)
    print(f"Test Result: {2-failed}/2")

def test_gradients_optional_sigmoid():
    failed = 0
    X = np.random.randn(2, 3)
    dLdY = np.random.randn(2, 3)
    print(f"Relative error for each test case is expected to be less than 1e-8")
    for activation_name in ["sigmoid"]:
        # initialize a fully connected layer
        # and perform a forward and backward pass
        activation = Activation(activation_name)
        _ = activation.forward(X)
        grad = activation.backward(X, dLdY)

        # check the gradients w.r.t. each parameter
        relative_error = check_gradients(
                                         fn=activation.forward,  # the function we are checking
                                         grad=grad,  # the analytically computed gradient
                                         x=X,        # the variable w.r.t. which we are taking the gradient
                                         dLdf=dLdY,  # gradient at previous layer
                                        )
        if relative_error > 1e-8:
            failed += 1
        print(f"Relative error for {activation_name} activation:", relative_error)
    print(f"* Test Result for Sigmoid Gradients (Optional, no extra credit):  {1-failed}/1")


def test_gradients_fully_connected_layer():

    failed = 0
    print(f"Relative error for each test case is expected to be less than 1e-8")
    X = np.random.randn(2, 3)
    dLdY = np.random.randn(2, 4)

    # initialize a fully connected layer
    # and perform a forward and backward pass
    fc_layer = FullyConnectedLayer(n_out=4, activation="linear")
    _ = fc_layer.forward(X)
    _ = fc_layer.backward(dLdY)

    # check the gradients w.r.t. each parameter
    for param in fc_layer.parameters:
        relative_error = check_gradients(
                fn=fc_layer.forward_with_param(param, X),  # the function we are checking
                grad=fc_layer.gradients[param],  # the analytically computed gradient
                x=fc_layer.parameters[param],  # the variable w.r.t. which we are taking the gradient
                dLdf=dLdY,                     # gradient at previous layer
            )
        if relative_error > 1e-8:
            failed += 1
        print(f"Relative error for {param}:", relative_error)
    print(f"Test Result: {2-failed}/2")



############# HELPER FUNCTIONS ##############
def test_result_print(test_result, end_test_num, start_test_num, output, desired):
    print(f"Test Result: {len(test_result['Passed Tests'])}/{end_test_num-start_test_num}")
    if len(test_result['Passed Tests']) != (end_test_num-start_test_num):
        print(f"Passed Tests: {test_result['Passed Tests']}")
        print(f"Failed Tests: {test_result['Failed Tests']}")
        for i in test_result["Failed Tests"]:
            print(f"Test {i}:\nReturned Output:\n{output[i]}\nDesired Output:\n{desired[i]}\n")


def check_gradients(fn, grad, x, dLdf, h=1e-6):
    """
    Performs numerical gradient checking by numerically approximating
    the gradient using a two-sided finite difference.

    For each position in `x`, this function computes the numerical gradient as:
                  fn(x + h) - fn(x - h)
        numgrad = ---------------------
                            2h
    Then, we use chain rule to find numerical gradients.
    """
    # x must be a float vector
    if x.dtype != np.float32 and x.dtype != np.float64:
        raise TypeError(f"`x` must be a float vector but was {x.dtype}")

    # initialize the numerical gradient variable
    numgrad = np.zeros_like(x)

    # compute the numerical gradient for each position in x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        pos = fn(x).copy()
        x[ix] = oldval - h
        neg = fn(x).copy()
        x[ix] = oldval

        # compute the derivative, also apply the chain rule
        numgrad[ix] = np.sum((pos - neg) * dLdf) / (2 * h)
        it.iternext()

    return np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

if __name__ == '__main__':
    # Get all attributes defined in this module
    for attr_name in dir():
        # Get the actual attribute
        tests_to_skip = [
            # optional test
            "test_gradients_optional_sigmoid", 
            # util functions
            "test_result_print",
            "check_gradients"
        ]
        attr = globals()[attr_name]
        # Check if it's a function and starts with "test"
        if callable(attr) and attr_name.startswith('test'):
            # Call the test function
            if attr_name in tests_to_skip:
                print(f"Skipping {attr_name}")
            else:
                print(f"Running {attr_name}")
                attr()
    # test_gradients_activation()
    # test_gradients_optional_sigmoid()
    # test_gradients_fully_connected_layer()
