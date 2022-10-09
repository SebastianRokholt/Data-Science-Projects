import math
import random
import numpy as np


class SGD:
    """
    An implementation of the stochastic gradient descent algorithm for binomial logistic regression.

    I used the algorithm description in 'Speech and Language Processing' (Jurafsky & Martin, 2022), page 93,
    as a guideline for this implementation.

    The main difference between this algorithm and the one outlined
    by Jurafsky & Martin is that this algorithm doesn't take the loss function 'L(f(x; theta), y)'
    and the function parametrized by theta 'f(x, theta)' as input. Since this implementation
    is for binomial logistic regression, I've added L and f into the class methods.
    The cross-entropy loss function L (sigmoid(w.dot(x) + b) - y) * x[i]) is part of the gradient() method,
    while the function parametrized by theta is the sigmoid function implemented in the sigmoid() method.
    """

    def __init__(self, random_state=42):
        self.theta = None
        self.b = 0
        self.w = None
        self.random_state = random_state
        self.epoch = 0

    @classmethod
    def sigmoid(cls, z):
        """
        The logistic function.
        Maps a real value to the range [0, 1].
        Used to calculate the gradient during gradient descent.
        :param z: A floating point (real) value
        :return: A floating point value in the range [0, 1].
        """
        return 1 / (1 + math.exp(-z))

    @classmethod
    def gradient(cls, x, y, theta):
        """
        Calculates the loss and the gradient of the loss function.
        :param x: A numpy array. Represents a single training sample.
        :param y: A scalar. Represents the label for the training sample (0 or 1).
        :param theta: A numpy array. Represents the current values for the weights.
        :return: A numpy array of size len(x) + 1. Represents the gradient of the training sample and the bias.
        """
        # Extracts the weights from the parameter vector
        w = np.delete(theta, -1)
        # Extracts the bias from the parameter vector
        b = theta[-1]
        # Calculates the loss for each weight and then calculates the derivative
        delta_w = np.array([(cls.sigmoid(w.dot(x) + b) - y) * x[i] for i in range(len(x))])
        # Calculates the loss for the bias and then calculates the derivative
        delta_b = cls.sigmoid(w.dot(x) + b) - y
        # Extends the derivative of the bias with the derivative of the weights
        delta = np.append(delta_w, delta_b)
        # Returns the derivative (gradient) of the (entire) loss function
        return delta

    def train(self, X_train, y_train, learning_rate=0.01, max_iter=50, tolerance=0.0001):
        """
        Runs the gradient descent algorithm on the training data.
        :param X_train: A numpy array of arrays, where each subarray contains the feature values
                        for a single training sample.
        :param y_train: A numpy array containing the values for the label of each training sample.
        :param learning_rate: The value of the learning rate hyperparameter,
                              a.k.a. the "step size" for gradient descent.
        :param max_iter: The maximum number of epochs (passes/iterations) to performed during gradient descent.
        :param tolerance: The minimum
        :return: Theta, i.e. the optimal parameters for the weights w and the bias b.
        """

        # Basic error handling for user input
        if len(X_train) == 0:
            raise ValueError("X_train must have at least 1 sample. ")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must be of the same length.")
        if learning_rate <= 0 or max_iter <= 0 or tolerance <= 0:
            raise ValueError("The hyperparameters 'learning_rate', 'max_iter' and 'tolerance' must be greater than 0.")

        # Initializes the weights and the bias to 0
        self.theta = np.array([0 for i in range(len(X_train[0]))] + [self.b])
        print(f"Number of parameters to tune: {len(self.theta)}")

        # Randomly selects a set of indices of training samples to use for SGD algorithm
        random.seed(self.random_state)
        random_indices = random.sample([i for i in range(len(X_train))], len(X_train))

        # Performs gradient descent with max_iter number of steps towards the optimum
        for epoch in range(max_iter + 1):
            self.epoch += 1
            # Randomly selects a sample
            for i in random_indices:
                # Calculates the gradient
                delta = self.gradient(X_train[i], y_train[i], self.theta)
                # Checks whether the change in weights will be greater than the tolerance
                if np.all(np.abs(learning_rate * delta) <= tolerance):
                    # If it isn't, stop the gradient descent and return the parameters tuned after the previous epoch
                    print(f"The optimal parameters after {self.epoch} epochs are: {self.theta}")
                    return self.theta
                # Updates the weights
                else:
                    self.theta = self.theta - (learning_rate * delta)
            if epoch == 0 or (epoch + 1)% 10 == 0:
                print(f"Parameters after epoch {epoch + 1}: {self.theta}")

        print(f"The optimal parameters after {self.epoch} epochs are: {self.theta}")
        return self.theta
