import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Definition and operation of triangular fuzzy numbers
class TriangularFuzzyNumber:
    def __init__(self, l, m, r):
        self.l = l  # left endpoint
        self.m = m  # vertex
        self.r = r  # right endpoint

    # addition
    def __add__(self, other):
        return TriangularFuzzyNumber(self.l + other.l, self.m + other.m, self.r + other.r)

    # Subtraction
    def __sub__(self, other):
        return TriangularFuzzyNumber(self.l - other.l, self.m - other.m, self.r - other.r)

    # multiplication
    def __mul__(self, other):
        l = min(self.l * other.l, self.l * other.r, self.r * other.l, self.r * other.r)
        m = self.m * other.m
        r = max(self.l * other.l, self.l * other.r, self.r * other.l, self.r * other.r)
        return TriangularFuzzyNumber(l, m, r)

    # division
    def __truediv__(self, other):
        l = min(self.l / other.l, self.l / other.r, self.r / other.l, self.r / other.r)
        m = self.m / other.m
        r = max(self.l / other.l, self.l / other.r, self.r / other.l, self.r / other.r)
        return TriangularFuzzyNumber(l, m, r)

    # Fuzzify a binary label
    @staticmethod
    def fuzzyize_binary(y, m, l, r, I_L, I_U):
        U = np.random.uniform(I_L, I_U)
        if y == 0:
            return TriangularFuzzyNumber(0 - m * l * U, 0, 0 + m * r * U)
        elif y == 1:
            return TriangularFuzzyNumber(1 - m * l * U, 1, 1 + m * r * U)
        else:
            raise ValueError("Binary label must be either 0 or 1.")


# fuzzy logistic classifier
class FuzzyLogicClassifier:
    def __init__(self, m, l, r, I_L, I_U, learning_rate=0.01, epochs=1000):
        # fuzzification parameters
        self.m = m  # fuzzification degree
        self.l = l  # Left endpoint symmetry
        self.r = r  # Right endpoint symmetry
        self.I_L = I_L  # Lower limit of random number interval
        self.I_U = I_U  # Upper limit of random number interval

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.l_params = None
        self.beta_params = None
        self.r_params = None

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y

    def preprocess_data(self, X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Fuzzification of true value
    def fuzzyize_labels(self, y):
        fuzzy_labels = []
        for label in y:
            fuzzy_labels.append(
                TriangularFuzzyNumber.fuzzyize_binary(label, self.m, self.l, self.r, self.I_L, self.I_U))
        return np.array(fuzzy_labels)

    # fuzzy prediction
    def fuzzy_predict(self, X, l, beta, r):
        pi_l = 1 / (1 + np.exp(-(np.dot(l, X))))  # Left fuzzy probability
        pi = 1 / (1 + np.exp(-(np.dot(beta, X))))  # Center fuzzy probability
        pi_r = 1 / (1 + np.exp(-(np.dot(r, X))))  # Right fuzzy probability
        return pi_l, pi, pi_r

    # Centre-of-Gravity defuzzification
    def cfcs_defuzzification(self, fuzzy_number):
        y_vals = np.linspace(fuzzy_number.l, fuzzy_number.r, 1000)  # 离散化区间
        mu_vals = np.zeros_like(y_vals)
        for i, y in enumerate(y_vals):
            if y <= fuzzy_number.l or y >= fuzzy_number.r:
                mu_vals[i] = 0
            elif y <= fuzzy_number.m:
                mu_vals[i] = (y - fuzzy_number.l) / (fuzzy_number.m - fuzzy_number.l)
            else:
                mu_vals[i] = (fuzzy_number.r - y) / (fuzzy_number.r - fuzzy_number.m)
        numerator = np.sum(y_vals * mu_vals)
        denominator = np.sum(mu_vals)
        if denominator != 0:
            x_j_crisp = numerator / denominator
        else:
            x_j_crisp = (fuzzy_number.l + fuzzy_number.m + fuzzy_number.r) / 3

        return x_j_crisp

    # Loss function calculation
    def compute_loss(self, X, y_fuzzy, l, beta, r):
        loss = 0
        for i in range(len(X)):
            xi = X[i]
            yi_fuzzy = y_fuzzy[i]
            pi_l, pi, pi_r = self.fuzzy_predict(xi, l, beta, r)
            loss += (yi_fuzzy.l - pi_l) ** 2 + (yi_fuzzy.m - pi) ** 2 + (yi_fuzzy.r - pi_r) ** 2
        return loss / len(X)

    def train(self, X_train, y_train):
        n_features = X_train.shape[1]
        # Initialize fuzzy number parameters
        self.l_params = np.random.randn(n_features)
        self.beta_params = np.random.randn(n_features)
        self.r_params = np.random.randn(n_features)

        for epoch in range(self.epochs):
            for i in range(len(X_train)):
                xi = X_train[i]
                yi = y_train[i]
                pi_l, pi, pi_r = self.fuzzy_predict(xi, self.l_params, self.beta_params, self.r_params)
                # Calculate gradient
                loss_gradient_l = 2 * (pi_l - yi.l) * xi
                loss_gradient_beta = 2 * (pi - yi.m) * xi
                loss_gradient_r = 2 * (pi_r - yi.r) * xi
                # Update parameters
                self.l_params -= self.learning_rate * loss_gradient_l
                self.beta_params -= self.learning_rate * loss_gradient_beta
                self.r_params -= self.learning_rate * loss_gradient_r
        self.l_params = np.minimum(self.l_params, self.beta_params)
        self.r_params = np.maximum(self.r_params, self.beta_params)

    # Model evaluation
    def evaluate(self, X_test, y_test):
        y_fuzzy = self.fuzzyize_labels(y_test)
        pi_test = []

        # Make a prediction for each sample
        for xi in X_test:
            pi_l, pi, pi_r = self.fuzzy_predict(xi, self.l_params, self.beta_params, self.r_params)
            pi_test.append((pi_l, pi, pi_r))

        # Defuzzification: Defuzzify the fuzzy prediction results of each sample
        pi_test_crisp = []
        for pi_l, pi, pi_r in pi_test:
            # Pass the fuzzy prediction result to the defuzzification method
            fuzzy_number = TriangularFuzzyNumber(pi_l, pi, pi_r)
            pi_test_crisp.append(self.cfcs_defuzzification(fuzzy_number))

        assert len(pi_test_crisp) == len(
            y_test), f"Error: Length mismatch between predicted and true labels: {len(pi_test_crisp)} vs {len(y_test)}"

        # Calculate evaluation metrics using crisp prediction probabilities
        auc = roc_auc_score(y_test, pi_test_crisp)
        cm = confusion_matrix(y_test, np.array(pi_test_crisp) > 0.6)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, np.array(pi_test_crisp) > 0.6)
        mcc = matthews_corrcoef(y_test, np.array(pi_test_crisp) > 0.6)

        return auc, sensitivity, specificity, f1, mcc

# Training and evaluating the model
def main(file_path):
    m, l, r, I_L, I_U = 0.2, 0.5, 0.5, 0.01, 1  # fuzzification parameters
    clf = FuzzyLogicClassifier(m, l, r, I_L, I_U)
    X, y = clf.load_data(file_path)
    X_train, X_test, y_train, y_test = clf.preprocess_data(X, y)
    y_fuzzy_train = clf.fuzzyize_labels(y_train)
    clf.train(X_train, y_fuzzy_train)
    auc, sensitivity, specificity, f1, mcc = clf.evaluate(X_test, y_test)
    # Print evaluation results
    print(f"AUC: {auc}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"F1-Score: {f1}")
    print(f"MCC: {mcc}")

# main function
file_path = r"data.csv"
main(file_path)

