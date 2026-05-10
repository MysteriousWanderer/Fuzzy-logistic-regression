import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import random

class TriangularFuzzyNumber:
    def __init__(self, l, m, r):
        self.l = l
        self.m = m
        self.r = r

    def __repr__(self):
        return f"TriangularFuzzyNumber(l={self.l}, m={self.m}, r={self.r})"

    @staticmethod
    def fuzzyize_binary(y, m, l, r, I_L, I_U):
        U = np.random.uniform(I_L, I_U)
        if y == 0:
            return TriangularFuzzyNumber(0 - m * l * U, 0, 0 + m * r * U)
        elif y == 1:
            return TriangularFuzzyNumber(1 - m * l * U, 1, 1 + m * r * U)
        else:
            raise ValueError("Binary label must be either 0 or 1.")

class FuzzyLogicClassifier:
    def __init__(self, m, l, r, I_L, I_U, learning_rate=0.01, epochs=1000):
        self.m = m
        self.l = l
        self.r = r
        self.I_L = I_L
        self.I_U = I_U
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
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5,
                                                        random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def fuzzyize_labels(self, y):
        fuzzy_labels = []
        for label in y:
            fuzzy_labels.append(
                TriangularFuzzyNumber.fuzzyize_binary(label, self.m, self.l, self.r, self.I_L, self.I_U))
        return np.array(fuzzy_labels)

    def fuzzy_predict(self, X, l, beta, r):
        pi_l = 1 / (1 + np.exp(-(np.dot(X, l))))
        pi = 1 / (1 + np.exp(-(np.dot(X, beta))))
        pi_r = 1 / (1 + np.exp(-(np.dot(X, r))))
        return pi_l, pi, pi_r

    def cfcs_defuzzification(self, fuzzy_number):
        y_vals = np.linspace(fuzzy_number.l, fuzzy_number.r, 1000)
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

    def compute_loss(self, X, y_fuzzy, l, beta, r):
        loss = 0
        for i in range(len(X)):
            xi = X[i]
            yi_fuzzy = y_fuzzy[i]
            pi_l, pi, pi_r = self.fuzzy_predict(xi, l, beta, r)
            loss += (yi_fuzzy.l - pi_l) ** 2 + (yi_fuzzy.m - pi) ** 2 + (yi_fuzzy.r - pi_r) ** 2
        return loss / len(X)

    def train(self, X_train, y_train, random_state=None):
        if random_state is None:
            random_state = random.randint(0, 1000)
        np.random.seed(random_state)

        n_features = X_train.shape[1]
        self.l_params = np.random.randn(n_features)
        self.beta_params = np.random.randn(n_features)
        self.r_params = np.random.randn(n_features)

        for epoch in range(self.epochs):
            for i in range(len(X_train)):
                xi = X_train[i]
                yi = y_train[i]
                pi_l, pi, pi_r = self.fuzzy_predict(xi, self.l_params, self.beta_params, self.r_params)
                loss_gradient_l = 2 * (pi_l - yi.l) * xi
                loss_gradient_beta = 2 * (pi - yi.m) * xi
                loss_gradient_r = 2 * (pi_r - yi.r) * xi
                self.l_params -= self.learning_rate * loss_gradient_l
                self.beta_params -= self.learning_rate * loss_gradient_beta
                self.r_params -= self.learning_rate * loss_gradient_r
        self.l_params = np.minimum(self.l_params, self.beta_params)
        self.r_params = np.maximum(self.r_params, self.beta_params)

    def evaluate(self, X_test, y_test):
        y_fuzzy = self.fuzzyize_labels(y_test)
        pi_test = []

        for xi in X_test:
            pi_l, pi, pi_r = self.fuzzy_predict(xi, self.l_params, self.beta_params, self.r_params)
            pi_test.append((pi_l, pi, pi_r))

        pi_test_crisp = []
        for pi_l, pi, pi_r in pi_test:
            fuzzy_number = TriangularFuzzyNumber(pi_l, pi, pi_r)
            pi_test_crisp.append(self.cfcs_defuzzification(fuzzy_number))

        auc = roc_auc_score(y_test, pi_test_crisp)
        cm = confusion_matrix(y_test, np.array(pi_test_crisp) > 0.6)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, np.array(pi_test_crisp) > 0.6)
        mcc = matthews_corrcoef(y_test, np.array(pi_test_crisp) > 0.6)

        return auc, sensitivity, specificity, f1, mcc

def smote_dsr_sampling(X, y, base_classifiers_count):
    N_min = np.sum(y == 1)
    N_maj = np.sum(y == 0)
    D = N_maj - N_min

    resampled_datasets = []
    resampled_datasets.append((X, y))
    for m in range(1, base_classifiers_count):
        SR_m = (m / (base_classifiers_count - 1))
        NS_m = N_min + SR_m * D
        sampling_strategy = NS_m / N_maj
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random.randint(0, 100))
        X_resampled, y_resampled = smote.fit_resample(X, y)
        resampled_datasets.append((X_resampled, y_resampled))

    return resampled_datasets

def heterogeneous_fuzzy_weighting(pi_test, sensitivities, specificities):
    weighted_predictions_l = []
    weighted_predictions_m = []
    weighted_predictions_r = []

    for i in range(len(pi_test)):
        weighted_sum_l = 0
        weighted_sum_m = 0
        weighted_sum_r = 0
        weight_sum = 0

        for j in range(len(pi_test[i])):
            pi_l, pi_m, pi_r = pi_test[i][j]
            weight_m = sensitivities[j] if pi_m >= 0.001 else (1 - specificities[j])
            weighted_sum_l += weight_m * pi_l
            weighted_sum_m += weight_m * pi_m
            weighted_sum_r += weight_m * pi_r
            weight_sum += weight_m

        final_prediction_l = weighted_sum_l / weight_sum if weight_sum != 0 else 0.5
        final_prediction_m = weighted_sum_m / weight_sum if weight_sum != 0 else 0.5
        final_prediction_r = weighted_sum_r / weight_sum if weight_sum != 0 else 0.5

        weighted_predictions_l.append(final_prediction_l)
        weighted_predictions_m.append(final_prediction_m)
        weighted_predictions_r.append(final_prediction_r)

    final_predictions = [TriangularFuzzyNumber(l, m, r) for l, m, r in
                         zip(weighted_predictions_l, weighted_predictions_m, weighted_predictions_r)]
    return final_predictions

def main(file_path, base_classifiers_count=6):
    m, l, r, I_L, I_U = 0.2, 0.5, 0.5, 0.01, 1
    clf = FuzzyLogicClassifier(m, l, r, I_L, I_U)
    X, y = clf.load_data(file_path)
    X_train, X_val, X_test, y_train, y_val, y_test = clf.preprocess_data(X, y)
    resampled_datasets = smote_dsr_sampling(X_train, y_train, base_classifiers_count)
    models = []
    for X_resampled, y_resampled in resampled_datasets:
        y_fuzzy_train = clf.fuzzyize_labels(y_resampled)
        clf.train(X_resampled, y_fuzzy_train)
        models.append(clf)
    sensitivity_scores = []
    specificity_scores = []
    for model in models:
        auc, sensitivity, specificity, f1, mcc = model.evaluate(X_val, y_val)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
    pi_test = []
    for i in range(len(X_test)):
        sample_predictions = []
        for model in models:
            pi_l, pi, pi_r = model.fuzzy_predict(X_test[i], model.l_params, model.beta_params, model.r_params)
            sample_predictions.append((pi_l, pi, pi_r))
        pi_test.append(sample_predictions)

    weighted_predictions = heterogeneous_fuzzy_weighting(pi_test, sensitivity_scores, specificity_scores)

    final_predictions = []
    for fuzzy_pred in weighted_predictions:
        final_predictions.append(model.cfcs_defuzzification(fuzzy_pred))

    auc_weighted = roc_auc_score(y_test, final_predictions)
    cm = confusion_matrix(y_test, np.array(final_predictions) > 0.6)
    tn, fp, fn, tp = cm.ravel()

    sensitivity_weighted = tp / (tp + fn)
    specificity_weighted = tn / (tn + fp)
    f1_weighted = f1_score(y_test, np.array(final_predictions) > 0.6)
    mcc_weighted = matthews_corrcoef(y_test, np.array(final_predictions) > 0.6)

    print(f"AUC (Weighted): {auc_weighted}")
    print(f"Sensitivity (Weighted): {sensitivity_weighted}")
    print(f"Specificity (Weighted): {specificity_weighted}")
    print(f"F1-Score (Weighted): {f1_weighted}")
    print(f"MCC (Weighted): {mcc_weighted}")

file_path = r"data.csv"
main(file_path)

