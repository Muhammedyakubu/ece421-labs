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
        gradient = np.array(gradient, dtype=np.float64)
        self.m = (1 - self.beta_m) * gradient + self.beta_m * self.m
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
        # add a column of 1 to the leftmost column.
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def unique_classes_(self, y):
        return np.unique(y)

    def class_labels_(self, classes):
        return {c: i for i, c in enumerate(classes)}

    def one_hot(self, y):
        y_indices = np.array([self.class_labels[label] for label in y])
        
        # Create the one-hot encoded matrix
        one_hot_matrix = np.zeros((len(y), len(self.classes)))
        one_hot_matrix[np.arange(len(y)), y_indices] = 1
        
        return one_hot_matrix

    def softmax(self, z):
        # Compute the exponential of the values
        exp_scores = np.exp(z)
        
        # Compute the sum of exponentials for each row
        sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
        
        # Divide each exponential by the sum to get probabilities
        softmax_probs = exp_scores / sum_exp_scores
        
        return softmax_probs
    
    def predict_with_X_aug_(self, X_aug):
        z = np.dot(X_aug, self.weights.T)
        return self.softmax(z)

    def predict(self, X):
        X_aug = self.add_bias(X)
        return self.predict_with_X_aug_(X_aug)

    def predict_classes(self, X):
        probs = self.predict(X)
        # Get the predicted class indices
        predicted_indices = np.argmax(probs, axis=1)
        
        # Convert the indices back to original class labels using self.classes
        return self.classes[predicted_indices]
    
    def score(self, X, y):
        y_pred = self.predict_classes(X)
        return np.mean(y_pred == y)

    def evaluate_(self, X_aug, y_one_hot_encoded):
        probs = self.predict_with_X_aug_(X_aug)
        return np.mean(np.argmax(probs, axis=1) == np.argmax(y_one_hot_encoded, axis=1))


    def cross_entropy(self, y_one_hot_encoded, probs):
        return -np.sum(y_one_hot_encoded * np.log(probs))/y_one_hot_encoded.shape[0]

    def compute_grad(self, X_aug, y_one_hot_encoded, w):
        """
        Compute the gradient of the cross-entropy loss with respect to the weights.
        
        Parameters:
        X_aug: Augmented input data, shape (N, D+1) where N is number of samples and D is number of features
        y_one_hot_encoded: One-hot encoded true labels, shape (N, C) where C is number of classes
        w: Current weights, shape (C, D+1)
        
        Returns:
        gradient: Computed gradient, shape (C, D+1)
        """
        # Compute the predicted probabilities using the provided weights w
        z = np.dot(X_aug, w.T)
        probs = self.softmax(z)
        
        # Compute the difference between predicted probabilities and true labels
        diff = probs - y_one_hot_encoded
        
        # Compute the gradient using the formula: (probs - y_one_hot_encoded).T * X_aug
        # and scale it by the number of samples
        N = X_aug.shape[0]  # number of samples
        gradient = np.dot(diff.T, X_aug) / N
        
        return gradient


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

