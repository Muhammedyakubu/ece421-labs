import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    """
    This function computes the parameters w of a linear plane which best fits
    the training dataset.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes a Real value.
    
    Returns
    -------
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of the line computed by the pocket
        algorithm that best separates the two classes of training data points.
        The dimensions of this vector is (d+1) as the offset term is accounted
        in the computation.
    """

    # Add a column of ones to X_train for the bias term
    X = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
    # Compute X^T X
    XTX = np.matmul(X.transpose(), X)
    
    # Compute the pseudo-inverse of X^T X
    XTX_pseudo_inv = np.linalg.pinv(XTX)
    
    # Compute X^T y
    XTy = np.matmul(X.transpose(), y_train)
    
    # Compute the weights
    w = np.matmul(XTX_pseudo_inv, XTy)
    
    return w


def mse(X_train, y_train, w):
    """
    This function finds the mean squared error introduced by the linear plane
    defined by w.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes a Real value.
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of a linear plane.

    Returns
    -------
    avgError: float
        he mean squared error introduced by the linear plane defined by w.
    """

    # Add a column of ones to X for the bias term
    X_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
    # Compute predictions
    y_pred = np.matmul(X_with_bias, w)
    
    # Compute the difference between predictions and true values
    diff = y_pred - y_train
    
    # Compute the mean squared error
    mse = np.mean(np.square(diff))
    
    return mse


def pred(x_i, w):
    """
    This function finds finds the prediction by the classifier defined by w.

    Parameters
    ----------
    x_i: numpy.ndarray with shape (d+1,)
        Represents the feature vector of (d+1) dimensions of the ith test
        datapoint.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.
    
    Returns
    -------
    pred_i: int
        The predicted class.
    """

    pred_i = np.dot(x_i, w)
    return pred_i


def test_SciKit(X_train, X_test, y_train, y_test):
    """
    This function will output the mean squared error on the test set, which is
    obtained from the mean_squared_error function imported from sklearn.metrics
    library to report the performance of the model fitted using the 
    LinearRegression model available in the sklearn.linear_model library.
    
    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        Represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    X_test: numpy.ndarray with shape (M,d)
        Represents the matrix of input features where M is the total number of
        testing samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents output observed in the training set for the
        ith row in X_train matrix which corresponds to the ith input feature.
    y_test: numpy.ndarray with shape (M,)
        The ith component represents output observed in the test set for the
        ith row in X_test matrix which corresponds to the ith input feature.
    
    Returns
    -------
    error: float
        The mean squared error on the test set.
    """

    # initiate an object of the LinearRegression type. 
    linreg_model = LinearRegression()
    
    # run the fit function to train the model. 
    linreg_model.fit(X_train, y_train) 
    
    # use the predict function to perform predictions using the trained
    # model. 
    predictions = linreg_model.predict(X_test)
    
    # use the mean_squared_error function to find the mean squared error
    # on the test set. Don't forget to return the mean squared error.
    return mean_squared_error(y_test, predictions)


def subtestFn():
    """
    This function tests if your solution is robust against singular matrix.
    X_train has two perfectly correlated features.
    """

    X_train = np.asarray([[1, 2],
                          [2, 4],
                          [3, 6],
                          [4, 8]])
    y_train = np.asarray([1, 2, 3, 4])
    
    try:
      w = fit_LinRegr(X_train, y_train)
      print("weights: ", w)
      print("NO ERROR")
    except:
      print("ERROR")


def testFn_Part2():
    """
    This function loads diabetes dataset and splits it into train and test set.
    Then it finds and prints the mean squared error from your linear regression
    model and the one from the scikit library.
    """

    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    
    w = fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e = mse(X_test, y_test, w)
    
    #Testing Part 2b
    scikit = test_SciKit(X_train, X_test, y_train, y_test)
    
    print(f"Mean squared error from Part 2a is {e}")
    print(f"Mean squared error from Part 2b is {scikit}")


if __name__ == "__main__":
    print (f"{12*'-'}subtestFn{12*'-'}")
    subtestFn()

    print (f"{12*'-'}testFn_Part2{12*'-'}")
    testFn_Part2()

# The mean squared error from the above test is almost identical,
# with the error varying beginning at the 10th decimal place. 
# The performance of our implementation is effectively the same as the scikit-learn implementation
