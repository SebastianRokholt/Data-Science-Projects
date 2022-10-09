import math
from collections import defaultdict


class MultinomialNaiveBayes:
    """
    A class for training and testing a Multinomial Naive Bayes classifier
    with optional add-k smoothing.
    """

    def __init__(self):
        self.classes = []  # The classification labels
        self.class_feature_freq = defaultdict(dict)  # The number of times each feature (word) occurs for each class
        self.class_total_feature_freq = {}  # The total number of features per class
        self.total_doc_count = 0  # The total number of documents in the training data
        self.priorlogs = {}  # The a priori probabilities by class
        self.feature_ll = defaultdict(dict)  # The log likelihoods for each feature
        self.sum_ll = {}  # The test document's log likelihood ratios for each class

    def set_feature_freq_per_class(self, X_train: list, y_train: list, k: int = 1):
        """
        Extracts the class for each word in the training data, counts their frequency
        and sets the class_feature_freq attribute accordingly.
        :param X_train: The training data (documents). A list of lists.
        :param y_train: The training data classification labels.
        :param k: The add-k smoothing parameter. Default is 1.
        """

        # Loop over the training examples and update the frequency of each word for each class
        for doc, label in zip(X_train, y_train):
            for word in doc:
                if word in self.class_feature_freq[label].keys():
                    self.class_feature_freq[label][word] += 1
                else:
                    # If it is a new word for the current class, add it to the frequency dict with an initial value
                    self.class_feature_freq[label][word] = 1 + k

    def train(self, X_train: list, y_train: list, k: int = 1):
        """
        Trains the Multinomial Naive Bayes model on the training data.

        An implementation inspired by the Multinomial Naive Bayes training algorithm presented in
        Speech and Language Processing (Jurafsky & Martin, 2022), page 62, Figure 4.2.
        :param X_train: The training data features. A list of lists containing the tokenized and lemmatized documents.
        :param y_train: The training data labels.
        :param k: The parameter for add-k smoothing (optional).
        """
        # Retrieve unique classes in the training data
        self.classes = list(set(y_train))
        print("Unique classes: ", self.classes)

        # Calculates the total number of documents
        self.total_doc_count = len(X_train)
        print(f"Total number of documents: ", self.total_doc_count)

        # Calculates the frequency of each word per class in the training data
        self.set_feature_freq_per_class(X_train, y_train, k)
        print("Training sample counts per class: ", self.class_feature_freq.items())

        for label in self.classes:
            # Calculates the total number of features per class
            self.class_total_feature_freq[label] = sum(self.class_feature_freq[label].values())
            print(f"{label} total feature counts:", self.class_total_feature_freq[label])
            # Calculates the log a priori probabilities by class
            self.priorlogs[label] = math.log(self.class_total_feature_freq[label] / self.total_doc_count)
            print(f"Log prior for {label}: ", self.priorlogs[label])

            # Calculates the log likelihood of each feature (word) for each class.
            for word, count in self.class_feature_freq[label].items():
                self.feature_ll[label][word] = math.log(count
                                                        / (self.class_total_feature_freq[label] - count))
        print(f"Log likelihood ratio of each feature for each class: \n", self.feature_ll)

    def test(self, tokenized_document: list):
        """
        Classifies a single text document by summing the log likelihoods for each class
        for each word in the test document, with probability smoothing for words that occur in multiple classes,
        and returning the class with the largest (highest) sum.
        An implementation inspired by the Multinomial Naive Bayes test algorithm presented in
        Speech and Language Processing (Jurafsky & Martin, 2022), page 62, Figure 4.2.
        :param tokenized_document: Test document on the form of a single list of lemmatized tokens.
        :return: A string with the predicted classification
        """
        smoothing_values = {}
        # Retrieves the priors for the genres and saves them to a new variable
        for label in self.classes:
            self.sum_ll[label] = self.priorlogs[label]

            # Values for smoothing the probabilities when words in the test document aren't found in the current genre,
            # but are found in a different one
            smoothing_values[label] = math.log(1 / (self.class_total_feature_freq[label] + self.total_doc_count))

        # Loops over the feature classes and the words in the input document
        # and calculates the sums of the log likelihoods for each class
        for current_class in self.classes:
            for word in tokenized_document:
                # If the word has occurred in training data for current class
                if word in self.feature_ll[current_class]:
                    # Add its log likelihood to the sum of likelihoods for the current class
                    self.sum_ll[current_class] += self.feature_ll[current_class][word]
                # Else, check if the word has occurred in the training data for a different class
                else:
                    other_classes = [label for label in self.classes if label != current_class]
                    for other_class in other_classes:
                        # If it does, smooth the probability for the current class
                        if word in self.feature_ll[other_class]:
                            self.sum_ll[current_class] += smoothing_values[current_class]
        print(f"Sum of log likelihoods for D = {tokenized_document}: ", self.sum_ll)

        # Determines the test document class by selecting the class with the maximum log likelihood
        return max(self.sum_ll, key=self.sum_ll.get)


# Training data
X_train = [
    ["fun", "couple", "love", "love"],
    ["fast", "furious", "shoot"],
    ["couple", "fly", "fast", "fun", "fun"],
    ["furious", "shoot", "shoot", "fun"],
    ["fly", "fast", "shoot", "love"],
]
y_train = ["comedy", "action", "comedy", "action", "action"]

# Building the classifier
nb = MultinomialNaiveBayes()
nb.train(X_train, y_train, k=2)

# Test data (two different samples)
X_test_1 = ["fast", "couple", "shoot", "fly"]
X_test_2 = ["fast", "couple", "love", "furious"]

# Running predictions on two different test samples
y_pred_1 = nb.test(X_test_1)
y_pred_2 = nb.test(X_test_2)
print(y_pred_1)
print(y_pred_2)
