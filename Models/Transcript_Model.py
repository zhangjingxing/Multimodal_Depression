import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

# Suppress specific warnings from NumPy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Load the dataset
file_path = 'drive/MyDrive/ML data/practice/Transcript_Feature.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Preprocess Data
feature_columns = data.columns.drop(['PHQ8', 'Case'])
X = data[feature_columns].values
y = data['PHQ8'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom Random Forest
class CustomDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Stopping criteria
        if len(set(y)) == 1 or depth >= self.max_depth:
            return np.mean(y)

        # Find the best split
        best_feature, best_value, best_split = None, None, None
        min_error = float('inf')
        for feature in range(X.shape[1]):
            for value in set(X[:, feature]):
                left = y[X[:, feature] <= value]
                right = y[X[:, feature] > value]
                error = len(left) * np.var(left) + len(right) * np.var(right)
                if error < min_error:
                    min_error = error
                    best_feature, best_value, best_split = feature, value, (left, right)

        # Create subtree
        if best_split:
            left_tree = self._build_tree(X[X[:, best_feature] <= best_value], best_split[0], depth + 1)
            right_tree = self._build_tree(X[X[:, best_feature] > best_value], best_split[1], depth + 1)
            return (best_feature, best_value, left_tree, right_tree)
        return np.mean(y)

    def predict(self, X):
        def traverse_tree(x, tree):
            if not isinstance(tree, tuple):
                return tree
            feature, value, left, right = tree
            return traverse_tree(x, left if x[feature] <= value else right)

        return np.array([traverse_tree(x, self.tree) for x in X])

class CustomRandomForest:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.feature_importances_ = None

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        for _ in range(self.n_trees):
            sample_indices = np.random.choice(len(X), len(X))
            sample_X, sample_y = X[sample_indices], y[sample_indices]
            tree = CustomDecisionTree(max_depth=self.max_depth)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)
            # Update feature importance (count of splits on each feature)
            self._update_feature_importance(tree, sample_X)

    def _update_feature_importance(self, tree, X):
        def traverse_and_collect(tree):
            if not isinstance(tree, tuple):  # Leaf node
                return
            feature, _, left, right = tree
            self.feature_importances_[feature] += 1
            traverse_and_collect(left)
            traverse_and_collect(right)

        traverse_and_collect(tree.tree)

    def predict(self, X):
        all_predictions = []
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            if len(tree_predictions) == 0:  # Handle empty predictions
                continue
            all_predictions.append(tree_predictions)
        if len(all_predictions) == 0:  # Handle case where no valid predictions are made
            raise ValueError("No valid predictions made by the forest.")
        return np.round(np.mean(all_predictions, axis=0))

    def get_feature_importance(self):
        # Normalize feature importance
        if self.feature_importances_ is not None:
            total_splits = np.sum(self.feature_importances_)
            return self.feature_importances_ / total_splits
        return None



# Custom SVM (Linear Kernel)
class CustomSVM:
    def __init__(self, lr=0.001, epochs=1000, C=1):
        self.lr = lr  # Learning rate
        self.epochs = epochs  # Number of iterations
        self.C = C  # Regularization parameter

    def fit(self, X, y):
        # Transform labels to -1 and 1
        y = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(len(y)):
                # Compute SVM condition
                condition = y[i] * (np.dot(X[i], self.weights) - self.bias) >= 1
                if condition:
                    # Correct classification, only apply regularization
                    self.weights -= self.lr * (2 * self.C * self.weights)
                else:
                    # Misclassification, update weights and bias
                    self.weights -= self.lr * (2 * self.C * self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.lr * y[i]

    def predict(self, X):
        # Compute predictions
        predictions = np.dot(X, self.weights) - self.bias
        # Map predictions back to 0 and 1
        return np.where(predictions >= 0, 1, 0)


class CustomNaiveBayes:
    def __init__(self):
        self.class_prior = {}
        self.feature_stats = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.class_prior[c] = len(X_c) / len(y)
            self.feature_stats[c] = {
                "mean": np.mean(X_c, axis=0),
                "var": np.var(X_c, axis=0) + 1e-6  # Add small value to prevent division by zero
            }

    def _calculate_likelihood(self, x, mean, var):
        # Gaussian likelihood
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _calculate_posterior(self, x):
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.class_prior[c])
            likelihoods = np.sum(
                np.log(self._calculate_likelihood(x, self.feature_stats[c]["mean"], self.feature_stats[c]["var"]))
            )
            posteriors[c] = prior + likelihoods
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._calculate_posterior(x) for x in X])