"""Spam Email Detection using a manually implemented Multinomial Naive Bayes classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "emails.csv"
TRAIN_SIZE = 4500
TEST_SIZE = 672
RANDOM_SEED = 42


def load_and_validate_dataset(path: Path):
    """Load the CSV, validate its shape and label structure, and return X and y."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    print("Dataset Information")
    print("-------------------")
    print(f"Dataset path: {path}")
    print(f"Total emails: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")

    expected_rows = 5172
    expected_cols = 3002
    if df.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, but found {df.shape[0]}. "
            "The dataset format is invalid for this assignment."
        )
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns, but found {df.shape[1]}. "
            "The dataset format is invalid for this assignment."
        )

    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values. Please verify the CSV before training.")

    # First column is the email ID/name, last column is label, middle columns are word counts.
    identifier_column = df.columns[0]
    target_column = df.columns[-1]
    if identifier_column == target_column:
        raise ValueError("The first and last columns appear to be the same. Dataset structure is invalid.")

    X_df = df.iloc[:, 1:-1]
    y_series = df.iloc[:, -1]

    if X_df.empty:
        raise ValueError("Feature columns are missing. The CSV is missing the word-count matrix.")

    # Ensure all feature values are numeric.
    try:
        X = X_df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    except Exception as exc:
        raise ValueError("Feature columns contain non-numeric values. Expected word counts.") from exc

    # Ensure the target is binary 0/1.
    try:
        y = y_series.astype(int).to_numpy()
    except Exception as exc:
        raise ValueError("Target label column could not be converted to integers.") from exc

    unique_labels = np.unique(y)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(f"Target labels must be 0 or 1 only. Found unique labels: {unique_labels}")

    print(f"First column (identifier): {identifier_column}")
    print(f"Last column (label): {target_column}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def split_train_test(X: np.ndarray, y: np.ndarray):
    """Split the data into 4500 training rows and 672 testing rows using a fixed random seed."""
    n_samples = X.shape[0]
    if n_samples != 5172:
        raise ValueError(f"This split function expects 5172 rows, but found {n_samples}.")

    rng = np.random.RandomState(RANDOM_SEED)
    permutation = rng.permutation(n_samples)
    train_idx = permutation[:TRAIN_SIZE]
    test_idx = permutation[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

    if len(train_idx) != TRAIN_SIZE:
        raise ValueError(f"Training split length mismatch: expected {TRAIN_SIZE}, found {len(train_idx)}")
    if len(test_idx) != TEST_SIZE:
        raise ValueError(f"Testing split length mismatch: expected {TEST_SIZE}, found {len(test_idx)}")

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print("\nTrain/Test Split")
    print("----------------")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Spam training emails: {int(np.sum(y_train == 1))}")
    print(f"Non-spam training emails: {int(np.sum(y_train == 0))}")

    return X_train, y_train, X_test, y_test


class NaiveBayesClassifier:
    """Manual multinomial Naive Bayes classifier for word-count features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.classes_ = None
        self.class_priors_ = {}
        self.feature_log_prob_ = {}
        self.log_class_priors_ = {}
        self.vocabulary_size_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Estimate class priors and word likelihoods with Laplace smoothing."""
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.int64)

        self.classes_ = np.unique(y)
        self.vocabulary_size_ = X.shape[1]

        total_samples = y.shape[0]
        for class_value in self.classes_:
            class_mask = (y == class_value)
            class_count = int(np.sum(class_mask))
            class_word_total = float(np.sum(X[class_mask]))

            # P(class) = count(class) / total samples
            prior = class_count / total_samples
            self.class_priors_[class_value] = prior
            self.log_class_priors_[class_value] = np.log(prior)

            # For each feature j: P(word_j | class) = (count(word_j, class) + alpha) / (total words in class + alpha * vocabulary_size)
            feature_counts = X[class_mask].sum(axis=0)
            likelihood = (feature_counts + self.alpha) / (class_word_total + self.alpha * self.vocabulary_size_)
            self.feature_log_prob_[class_value] = np.log(likelihood)

    def _compute_log_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute log score for each class and each sample: log P(class) + sum_j x_j * log P(word_j | class)."""
        X = np.asarray(X, dtype=np.float64)
        log_scores = []
        for class_value in self.classes_:
            class_log_prior = self.log_class_priors_[class_value]
            feature_log_prob = self.feature_log_prob_[class_value]
            score = class_log_prior + X.dot(feature_log_prob)
            log_scores.append(score)
        return np.column_stack(log_scores)

    def predict_proba(self, X: np.ndarray):
        """Return class probabilities by normalizing the class log-scores."""
        log_scores = self._compute_log_scores(X)
        max_log_scores = np.max(log_scores, axis=1, keepdims=True)
        shifted = np.exp(log_scores - max_log_scores)
        normalizer = np.sum(shifted, axis=1, keepdims=True)
        probabilities = shifted / normalizer
        return probabilities

    def predict(self, X: np.ndarray):
        """Predict the class with the highest log score."""
        log_scores = self._compute_log_scores(X)
        predicted_class_indices = np.argmax(log_scores, axis=1)
        classes_array = np.asarray(list(self.classes_))
        return classes_array[predicted_class_indices]


def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute confusion matrix with layout [[TN, FP], [FN, TP]]."""
    tn = fp = fn = tp = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            tp += 1
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = np.sum(y_true == y_pred)
    total = y_true.shape[0]
    return correct / total if total > 0 else 0.0


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def print_evaluation_results(y_true: np.ndarray, y_pred: np.ndarray):
    """Print and return the confusion matrix and evaluation metrics."""
    matrix = confusion_matrix_manual(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nModel Information")
    print("-----------------")
    print("Algorithm: Multinomial Naive Bayes")
    print("Smoothing: Laplace smoothing")
    print("Alpha: 1.0")
    print("Probability calculation: Log probabilities")

    print("\nEvaluation Results")
    print("------------------")
    print("Confusion Matrix:")
    print("              Predicted")
    print("              0       1")
    print(f"Actual 0      {matrix[0, 0]}      {matrix[0, 1]}")
    print(f"Actual 1      {matrix[1, 0]}      {matrix[1, 1]}")
    print(f"Accuracy:  {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall:    {rec:.6f}")
    print(f"F1 Score:  {f1:.6f}")

    return matrix, acc, prec, rec, f1


def save_results_file(text: str, output_path: Path):
    """Save the final evaluation output to results.txt."""
    output_path.write_text(text, encoding="utf-8")


def main():
    """Run the full spam-email detection pipeline."""
    X, y = load_and_validate_dataset(DATA_PATH)
    X_train, y_train, X_test, y_test = split_train_test(X, y)

    classifier = NaiveBayesClassifier(alpha=1.0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    matrix, accuracy, precision, recall, f1 = print_evaluation_results(y_test, y_pred)

    output_lines = [
        "Dataset Information",
        "-------------------",
        f"Total emails: {X.shape[0]}",
        f"Number of features: {X.shape[1]}",
        f"Training samples: {TRAIN_SIZE}",
        f"Testing samples: {TEST_SIZE}",
        f"Number of spam training emails: {int(np.sum(y_train == 1))}",
        f"Number of non-spam training emails: {int(np.sum(y_train == 0))}",
        "",
        "Model Information",
        "-----------------",
        "Algorithm: Multinomial Naive Bayes",
        "Smoothing: Laplace smoothing",
        "Alpha: 1.0",
        "Probability calculation: Log probabilities",
        "",
        "Evaluation Results",
        "------------------",
        "Confusion Matrix:",
        "              Predicted",
        "              0       1",
        f"Actual 0      {matrix[0, 0]}      {matrix[0, 1]}",
        f"Actual 1      {matrix[1, 0]}      {matrix[1, 1]}",
        f"Accuracy:  {accuracy:.6f}",
        f"Precision: {precision:.6f}",
        f"Recall:    {recall:.6f}",
        f"F1 Score:  {f1:.6f}",
    ]
    output_text = "\n".join(output_lines) + "\n"
    save_results_file(output_text, Path(__file__).resolve().parent / "results.txt")
    print("\nSaved final results to results.txt")


if __name__ == "__main__":
    main()
