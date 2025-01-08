import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def main(csv_file, output_file):
    # Step 2: Load the dataset from a CSV file
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1].values  # Assuming the last column is the target
    y = data.iloc[:, -1].values

    # Step 3: Initialize multiple classifiers
    classifiers = {
        'MLR': LogisticRegression(random_state=42, max_iter=200, multi_class='multinomial'),  # MLR
        'J48': DecisionTreeClassifier(random_state=42, criterion='entropy'),  # J48
        'JRIP': DecisionTreeClassifier(random_state=42, max_depth=3),  # JRIP (approximation)
        'REPTree': DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5),  # REPTree
        'MLP': MLPClassifier(random_state=42, max_iter=1000),  # MLP
        'SVM': SVC(random_state=42),  # SVM
        'GNB': GaussianNB(),  # GNB
        'IBk': KNeighborsClassifier(n_neighbors=3)  # IBk
    }


    # Step 4: Perform stratified k-fold cross-validation for each classifier
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42, test_size=0.2)
    

    results = {}

    for clf_name, clf in classifiers.items():
        per_class_accuracies = []
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
            per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
            per_class_accuracies.append(per_class_accuracy)

        # Aggregate the per-class accuracy matrix
        per_class_accuracies = np.array(per_class_accuracies)
        average_per_class_accuracy = np.mean(per_class_accuracies, axis=0)

        # Round accuracies to 2 decimal places
        average_per_class_accuracy = np.round(average_per_class_accuracy, 2)

        # Calculate the average accuracy across all classes
        overall_average_accuracy = np.round(np.mean(average_per_class_accuracy), 2)

        # Store the results (excluding the first and last column)
        results[clf_name] = list(average_per_class_accuracy) + [overall_average_accuracy]

    # Create a DataFrame to store the results
    class_labels = list(np.unique(y)) + ['Overall Average']
    accuracy_df = pd.DataFrame(results, index=class_labels).T

    # Sort the DataFrame by the 'Overall Average' column in descending order
    accuracy_df = accuracy_df.sort_values(by='Overall Average', ascending=False)

    # Step 6: Save the per-class accuracies (including all columns) of all classifiers to a CSV file
    accuracy_df.to_csv(output_file)
    print(f'Results saved to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute per-class accuracy using stratified k-fold cross-validation.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing the dataset')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file to save the accuracies')
    args = parser.parse_args()
    main(args.csv_file, args.output_file)
