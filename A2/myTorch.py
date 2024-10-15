import numpy as np
import random
import collections
import util

# YOU ARE NOT ALLOWED TO USE sklearn or Pytorch in this assignment


class Optimizer:

    def __init__(
        self, name, lr=0.001, gama=0.9, beta_m=0.9, beta_v=0.999, epsilon=1e-8
    ):
        # self.lr will be set as the learning rate that we use upon creating the object, i.e., lr
        # e.g., by creating an object with Optimizer("sgd", lr=0.0001), the self.lr will be set as 0.0001
        self.lr = lr

        # Based on the name used for creating an Optimizer object,
        # we set the self.optimize to be the desiarable method.
        if name == "sgd":
            self.optimize = self.sgd
        elif name == "heavyball_momentum":
            # setting the gamma parameter and initializing the momentum
            self.gama = gama
            self.v = 0
            self.optimize = self.heavyball_momentum
        elif name == "nestrov_momentum":
            # setting the gamma parameter and initializing the momentum
            self.gama = gama
            self.v = 0
            self.optimize = self.nestrov_momentum
        elif name == "adam":
            # setting beta_m, beta_v, and epsilon
            # (read the handout to see what these parametrs are)
            self.beta_m = beta_m
            self.beta_v = beta_v
            self.epsilon = epsilon

            # setting the initial first momentum of the gradient
            # (read the handout for more info)
            self.v = 0

            # setting the initial second momentum of the gradient
            # (read the handout for more info)
            self.m = 0

            # initializing the iteration number
            self.t = 1

            self.optimize = self.adam

    def sgd(self, gradient):
        return -1*self.lr*gradient

    def heavyball_momentum(self, gradient):
        self.v = -1*self.lr*gradient + self.gama*self.v
        return self.v

    def nestrov_momentum(self, gradient):
        return self.heavyball_momentum(gradient)

    def adam(self, gradient):
        self.m = (1 - self.beta_m) * gradient  + self.beta_m * self.m
        
        self.v = (1 - self.beta_v) * (gradient ** 2) + self.beta_v * self.v
        
        m_hat = self.m / (1 - self.beta_m ** self.t)
        
        v_hat = self.v / (1 - self.beta_v ** self.t)
        
        # Increment the iteration counter
        self.t += 1

        # Compute update
        update = -(self.lr * m_hat) / (np.sqrt(v_hat) + self.epsilon)
        
        # return the update vector
        return np.array(update, dtype=np.float64)


class MultiClassLogisticRegression:
    def __init__(self, n_iter=10000, thres=1e-3):
        self.n_iter = n_iter
        self.thres = thres

    def fit(
        self,
        X,
        y,
        batch_size=64,
        lr=0.001,
        gama=0.9,
        beta_m=0.9,
        beta_v=0.999,
        epsilon=1e-8,
        rand_seed=4,
        verbose=False,
        optimizer="sgd",
    ):
        # setting the random state for consistency.
        np.random.seed(rand_seed)

        # find all classes in the train dataset.
        self.classes = self.unique_classes_(y)

        # assigning an integer value to each class, from 0 to (len(self.classes)-1)
        self.class_labels = self.class_labels_(self.classes)

        # one-hot-encode the labels.
        self.y_one_hot_encoded = self.one_hot(y)

        # add a column of 1 to the leftmost column.
        X = self.add_bias(X)

        # initialize the E_in list to keep track of E_in after each iteration.
        self.loss = []

        # initialize the weight parameters with a matrix of all zeros.
        # each row of self.weights contains the weights for one of the classes.
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))

        # create an instance of optimizer
        opt = Optimizer(
            optimizer, lr=lr, gama=gama, beta_m=beta_m, beta_v=beta_v, epsilon=epsilon
        )

        i, update = 0, 0
        while i < self.n_iter:
            self.loss.append(
                self.cross_entropy(self.y_one_hot_encoded, self.predict_with_X_aug_(X))
            )
            "*** YOUR CODE STARTS HERE ***"
            # TODO: sample a batch of data, X_batch and y_batch, with batch_size number of datapoint uniformly at random

            # TODO: find the gradient that should be inputed the optimization function.
            # NOTE: for nestrov_momentum, the gradient is derived at a point different from self.weights
            # See the assignments handout or the lecture note for more information.

            # TODO: find the update vector by using the optimization method and update self.weights, accordingly.

            # TODO: stopping criterion. check if norm infinity of the update vector is smaller than self.thres.
            # if so, break the while loop.

            "*** YOUR CODE ENDS HERE ***"
            if i % 1000 == 0 and verbose:
                print(
                    " Training Accuray at {} iterations is {}".format(
                        i, self.evaluate_(X, self.y_one_hot_encoded)
                    )
                )
            i += 1
        return self

    def add_bias(self, X):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def unique_classes_(self, y):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def class_labels_(self, classes):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def one_hot(self, y):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def softmax(self, z):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def predict_with_X_aug_(self, X_aug):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def predict(self, X):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def predict_classes(self, X):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def score(self, X, y):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def evaluate_(self, X_aug, y_one_hot_encoded):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def cross_entropy(self, y_one_hot_encoded, probs):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"

    def compute_grad(self, X_aug, y_one_hot_encoded, w):
        # TODO: add your implementation here
        "*** YOUR CODE STARTS HERE ***"
        pass
        "*** YOUR CODE ENDS HERE ***"


def kmeans(examples, K, maxIters):
    """
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """

    # TODO: add your implementation here
    "*** YOUR CODE STARTS HERE ***"
    pass
    "*** YOUR CODE ENDS HERE ***"
