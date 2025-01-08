import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Weighted Voting Ensemble Learning')
parser.add_argument('--dataset', type=str, required=True, help='Path to the CSV dataset')
parser.add_argument('--weights', type=str, required=True, help='Comma-separated list of weights for the models')
parser.add_argument('--output', type=str, default='evaluation_metrics.csv', help='Output CSV filename for evaluation metrics')
args = parser.parse_args()

# Load dataset
data = pd.read_csv(args.dataset)
X = data.iloc[:, :-1].values  # Assuming the last column is the target
y = data.iloc[:, -1].values

# Determine the number of classes
n_classes = len(np.unique(y))

# Convert weights to a 2D numpy array
weights = np.array([float(w) for w in args.weights.split(',')]).reshape(-1, n_classes)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize models
models = [
    LogisticRegression(random_state=42, max_iter=200, multi_class='multinomial'),  # MLR
    DecisionTreeClassifier(random_state=42, criterion='entropy'),  # J48
    DecisionTreeClassifier(random_state=42, max_depth=3),  # JRIP
    DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5),  # REPTree
    MLPClassifier(random_state=42, max_iter=1000),  # MLP
    SVC(random_state=42),  # SVM
    GaussianNB(),  # GNB
    KNeighborsClassifier(n_neighbors=3)  # IBk
]

# Ensure the shape of weights matches the number of models and classes
if weights.shape != (len(models), n_classes):
    raise ValueError("Shape of weights must be (number of models, number of classes)")

# Stratified K-Folds cross-validator
skf = StratifiedKFold(n_splits=1, shuffle=True, random_state=42,test_size=0.2)



# Storage for predictions
all_predictions = []

# Train models and collect predictions
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    fold_predictions = []
    for model in models:
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)
        fold_predictions.append(preds)  # Collect probabilities per model

    all_predictions.append((val_index, np.hstack(fold_predictions)))  # Stack predictions horizontally

# Aggregate predictions
predictions = np.zeros((len(y), len(models), n_classes))
for val_index, fold_predictions in all_predictions:
    for i, model_preds in enumerate(np.split(fold_predictions, len(models), axis=1)):
        predictions[val_index, i, :] = model_preds

# Perform weighted voting
weighted_sum = np.tensordot(predictions, weights, axes=([1, 2], [0, 1]))

# Determine final predictions based on problem type
if n_classes == 2:  # Binary classification
    final_predictions = (weighted_sum > 0.5).astype(int)
else:  # Multi-class classification
    final_predictions = np.ceil(np.round(weighted_sum))

# Calculate accuracy
accuracy = accuracy_score(y, final_predictions)

# Calculate macro-averaged precision, recall, and F1-score
precision_macro = precision_score(y, final_predictions, average='macro', zero_division=0)
recall_macro = recall_score(y, final_predictions, average='macro', zero_division=0)
f1_macro = f1_score(y, final_predictions, average='macro', zero_division=0)

# Calculate balanced accuracy
balanced_acc = balanced_accuracy_score(y, final_predictions)

# Format the metrics to two decimal places
accuracy = format(accuracy, '.2f')
precision_macro = format(precision_macro, '.2f')
recall_macro = format(recall_macro, '.2f')
f1_macro = format(f1_macro, '.2f')
balanced_acc = format(balanced_acc, '.2f')

# Define a dictionary to store the formatted evaluation metrics
metrics_dict = {
    'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-score', 'Balanced Accuracy'],
    'Value': [accuracy, precision_macro, recall_macro, f1_macro, balanced_acc]
}

# Convert the dictionary to a DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# Save the DataFrame to a CSV file
metrics_df.to_csv(args.output, index=False)

# Print a message indicating where the CSV file is saved
print(f'Evaluation metrics saved to: {args.output}')
