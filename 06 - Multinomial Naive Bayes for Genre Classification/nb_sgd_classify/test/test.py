from nb_sgd_classify.sgd import SGD
from nb_sgd_classify.mn_naivebayes import MultinomialNaiveBayes
import numpy as np

# Testing the Multinomial Naive Bayes class
X_train = [
    ["fun", "couple", "love", "love"],
    ["fast", "furious", "shoot"],
    ["couple", "fly", "fast", "fun", "fun"],
    ["furious", "shoot", "shoot", "fun"],
    ["fly", "fast", "shoot", "love"]
]
y_train = ["comedy", "action", "comedy", "action", "action"]
X_test = ["fast", "couple", "shoot", "fly"]
# X_test = ["fast", "couple", "love", "shoot"]

nb = MultinomialNaiveBayes()
nb.train(X_train, y_train, k=2)
y_pred = nb.test(X_test)
print(y_pred)


# Testing the Stochastic Gradient Descent class
# Training data
X_train = np.array([[3, 2], ])  # A single training example
y_train = np.array([1, ])  # The label for our training example (1 = "positive sentiment")

# Instantiating the Stochastic Gradient Descent algorithm
sgd = SGD(random_state=42)
theta = sgd.train(X_train, y_train, learning_rate=0.01, max_iter=1000, tolerance=0.001)
