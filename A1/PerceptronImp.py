import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris


def fit_perceptron(X_train, y_train, max_epochs=5000):
    """
    This function computes the parameters w of a linear plane which separates
    the input features from the training set into two classes specified by the
    training dataset.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes the value +1 or −1 to represent
        the first class and second class respectively.
    
    Returns
    -------
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of the line computed by the pocket
        algorithm that best separates the two classes of training data points.
        The dimensions of this vector is (d+1) as the offset term is accounted
        in the computation.
    """

    row, col = X_train.shape
    Ein_best = 1000

    # initialize the weight vector w
    w = np.zeros(col+1)
    w_best = w

    # Augment X_train so that it would have an additional column of 1's.
    X_train_aug = np.hstack((np.ones(shape=(row, 1)), X_train))

    # go over the entire dataset for max_epochs number of times. 
    # In each epoch, we examin all datapoints one by one and update w if neccessary. 
    # If after update you have a new best E_in, then save the updated weight in
    # your pocket to return it at the end
    for i in range(max_epochs):
        for j in range(row):
          	# If point is misclassified, update w as defined by PLA
            if misclassifiedPoint(y_train[j], X_train_aug[j], w):
                w += y_train[j]*X_train_aug[j]
            currError = errorPer(X_train_aug, y_train, w)
            # Return current w if Ein(w) = 0
            if currError == 0:
                return w
            # Update best w if Ein(w) is smaller than the current smallest error
            if currError < Ein_best:
                Ein_best = currError
                w_best = w
    return w_best

def errorPer(X_train, y_train, w):
    """
    This function finds the average number of points that are misclassified by
    the plane defined by w.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d+1)
        Represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
        Note the additional dimension which is for the additional column of ones
        added to the front of the original input.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.
    
    Returns
    -------
    avgError: float
        The average number of points that are misclassified by the plane
        defined by w.
    """
    N = X_train.shape[0]
    incorrectCount = 0

    for i in range(0, N):
        if misclassifiedPoint(y_train[i], X_train[i], w):
            incorrectCount += 1

    return incorrectCount / N

def misclassifiedPoint(y, x, w):
    """
    This function determines if a given input is misclassified by the prediction.

    Parameters
    ----------
    x: numpy.ndarray with shape (d+1,)
        Represents an input feature where d is the dimension.
        Note the additional dimension which is for the additional column of ones
        added to the front of the original input.
    y: float
        The correct label for the given input x
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.
    
    Returns
    -------
    boolean
        True if point is misclassified, False otherwise.
    """
    h_x = pred(x, w)
    # If prediction and true label are not equal or the point lies on the hyperplane, return True
    return y != h_x or np.dot(x, w) == 0

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

    pred_i = -1 if np.dot(x_i, w) < 0 else 1
    return pred_i

  
def confMatrix(X_train, y_train, w):
    """
    This function populates the confusion matrix. 

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        Represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.
    
    Returns
    -------
    conf_mat: numpy.ndarray with shape (2,2), composed of integer values
        - conf_mat[0, 0]: True Negative
            number of points correctly classified to be class −1.
        - conf_mat[0, 1]: False Positive
            number of points that are in class −1 but are classified to be class
            +1 by the classifier.
        - conf_mat[1, 0]: False Negative
            number of points that are in class +1 but are classified to be class
            −1 by the classifier.
        - conf_mat[1, 1]: True Positive
            number of points correctly classified to be class +1.
    """

    conf_mat = np.zeros((2,2))
    N = X_train.shape[0]
    X_train_aug = np.hstack((np.ones(shape=(N, 1)), X_train))
    
    for i in range(N):
      f_x = y_train[i]
      h_x = pred(X_train_aug[i], w)
      if f_x == -1:
          if h_x == -1:
              # This is a true negative
              conf_mat[0, 0] += 1
          else:
              # This is a false positive
              conf_mat[0, 1] += 1
            
      else:
          if h_x == -1:
              # This is a false negative
              conf_mat[1, 0] += 1
          else:
              # This is a true positive
              conf_mat[1, 1] += 1
              
    return conf_mat
 

def test_SciKit(X_train, X_test, y_train, y_test):
    """
    This function uses Perceptron imported from sklearn.linear_model to fit the
    linear classifer using the Perceptron learning algorithm. Then it returns
    the result obtained from the confusion_matrix function imported from
    sklearn.metrics to report the performance of the fitted model.
    
    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        Represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    X_test: numpy.ndarray with shape (M,d)
        Represents the matrix of input features where M is the total number of
        testing samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature.
    y_test: numpy.ndarray with shape (M,)
        The ith component represents the output observed in the test set
        for the ith row in X_test matrix.
    
    Returns
    -------
    conf_mat: numpy.ndarray with shape (2,2), composed of integer values
        - conf_mat[0, 0]: True Negative
            number of points correctly classified to be class −1.
        - conf_mat[0, 1]: False Positive
            number of points that are in class −1 but are classified to be class
            +1 by the classifier.
        - conf_mat[1, 0]: False Negative
            number of points that are in class +1 but are classified to be class
            −1 by the classifier.
        - conf_mat[1, 1]: True Positive
            number of points correctly classified to be class +1.
    """

    # initiate an object of the Perceptron type. 
    perceptron = Perceptron()
    
    # run the fit function to train the classifier. 
    perceptron.fit(X_train, y_train)
    
    # use the predict function to perform predictions using the trained
    # algorithm. 
    y_pred = perceptron.predict(X_test)
    
    # Use the confusion_matrix function to find the confusion matrix.
    conf_mat = confusion_matrix(y_test, y_pred)
    return conf_mat


def test_Part1():
    """
    This is the main routine function. It loads IRIS dataset, picks its last
    100 datapoints and split them into train and test set. Then finds and prints
    the confusion matrix from part 1a and 1b.
    """

    # Loading and splitting IRIS dataset into train and test set
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],
                                                        y_train[50:],
                                                        test_size=0.2,
                                                        random_state=42)

    # Set the labels to +1 and -1. 
    # The original labels in the IRIS dataset are 1 and 2. We change label 2 to -1. 
    y_train[y_train != 1] = -1
    y_test[y_test != 1] = -1

    # Pocket algorithm using Numpy
    w = fit_perceptron(X_train, y_train)
    my_conf_mat = confMatrix(X_test, y_test, w)

    # Pocket algorithm using scikit-learn
    scikit_conf_mat = test_SciKit(X_train, X_test, y_train, y_test)
    
    # Print the result
    print(f"{12*'-'}Test Result{12*'-'}")
    print("Confusion Matrix from Part 1a is: \n", my_conf_mat)
    print("\nConfusion Matrix from Part 1b is: \n", scikit_conf_mat)
    

if __name__ == "__main__":
    test_Part1()
