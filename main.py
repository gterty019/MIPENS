import pandas as pd
import numpy as np
import time
import os
import gurobipy as gp


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from datetime import datetime
from scipy.optimize import differential_evolution
from scipy.optimize import LinearConstraint
from gurobipy import GRB
from joblib import dump, load
from itertools import combinations


# -- Examined weighting schemes --
WEIGHTING_SCHEMES = [
    "UW-PC",  # Uniform weights per classifier
    "UW-PCC",  # Uniform weights per classifier and class
    "WA-PC",  # Weighted average based on normalized accuracy per classifier
    "WA-PCC",  # Weighted average based on normalized accuracy per classifier and class
    "DE",  # Differential evolution (per classifier)
    "BMA",  # Bayesian model averaging (per classifier and class)
    "MIP"  # Mixed integer programming (per classifier and class -- proposed approach incorporating elastic net regularization)
]


# -- Base classifiers --
# Base classifiers can be changed to the desired ones
classifiers = {
    "MLR": LogisticRegression(random_state=42, max_iter=500),
    "J48": DecisionTreeClassifier(random_state=42, criterion='entropy'),
    "JRIP": DecisionTreeClassifier(random_state=42, max_depth=3),  # JRIP (approximation)
    "REPTree": DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=5),
    "MLP": MLPClassifier(random_state=42, max_iter=1000),
    "SVM": SVC(random_state=42, probability=True),
    "GNB": GaussianNB(),
    "IBk": KNeighborsClassifier(n_neighbors=3)
}


# -- Examined ensemble sizes --
ENSEMBLE_SIZES = [2, 3, 4, 5, 6, 7, 8]


# Objective function for Differential Evolution weighting scheme
def de_objective(classifier_weights, overall_acc):
    weighted_avg = np.dot(classifier_weights, overall_acc)

    return -weighted_avg


def weighting_scheme(scheme_name, n_classifiers, n_classes, classes, accuracies_df, ensemble_size, timestamp, dataset_base_name, combination):
    # Initialize weight matrix per classifier and class
    weights = np.zeros((n_classifiers, n_classes))

    # Extract accuracies
    class_accuracy_cols = [f"Class_{c}_MeanAccuracy" for c in classes]
    class_accuracies = accuracies_df[class_accuracy_cols].values  # Mean validation accuracy of each classifier per class
    overall_acc = accuracies_df["OverallMeanAccuracy"].values  # Overall mean validation accuracy of each classifier across all classes

    # Create a mask to ignore classifiers not in the given combination
    active_classifiers = np.ones(n_classifiers, dtype=bool) if combination is None else np.zeros(n_classifiers, dtype=bool)
    if combination is not None:
        for idx in combination:
            active_classifiers[idx] = True

    # UW-PC:
    if scheme_name == "UW-PC":
        weights[active_classifiers, :] = 1.0 / np.sum(active_classifiers)

    # UW-PCC:
    elif scheme_name == "UW-PCC":
        weights[active_classifiers, :] = 1.0 / (np.sum(active_classifiers) * n_classes)

    # WA-PC:
    elif scheme_name == "WA-PC":
        # Normalize accuracy per classifier
        total_acc = np.sum(overall_acc)
        if total_acc == 0:
            # Assign uniform weights if total_acc is zero
            uniform_weight = 1.0 / np.sum(active_classifiers)
            weights[active_classifiers, :] = uniform_weight
        else:
            normalized_acc = overall_acc / total_acc
            weights[active_classifiers, :] = normalized_acc[active_classifiers, None]

    # WA-PCC:
    elif scheme_name == "WA-PCC":
        # Normalize weights per class
        for c in range(n_classes):
            sum_acc_for_class = np.sum(class_accuracies[:, c])
            if sum_acc_for_class == 0:
                # Assign uniform weights if sum_acc_for_class is zero
                uniform_weight = 1.0 / np.sum(active_classifiers)
                weights[active_classifiers, c] = uniform_weight
            else:
                weights[active_classifiers, c] = class_accuracies[active_classifiers, c] / sum_acc_for_class

    # DE:
    elif scheme_name == "DE":
        # Bounds of weights per classifier
        bounds = [(0.0, 1.0)] * np.sum(active_classifiers)

        # Constraint: all classifier weights should sum to 1
        weights_constraint = LinearConstraint(np.ones(np.sum(active_classifiers)), 1, 1)

        result = differential_evolution(de_objective, bounds, args=(overall_acc[active_classifiers],), constraints=weights_constraint, seed=42, polish=False)  # No local optimization required as the objective function is linear

        best_solution = result.x

        # Assign weights so that each active classifier has the same weight across all classes
        act_idx = np.where(active_classifiers)[0]
        for local_i, global_i in enumerate(act_idx):
            weights[global_i, :] = best_solution[local_i]

    # BMA:
    elif scheme_name == "BMA":
        act_idx = np.where(active_classifiers)[0]
        active_class_acc = class_accuracies[act_idx, :]

        # Exponentiate accuracies (mean validation accuracies are used as likelihoods)
        exp_active_acc = np.exp(active_class_acc)

        # Normalize per class
        bma_weights = exp_active_acc / np.sum(exp_active_acc, axis=0)

        # Assign each active row back into the weights matrix
        for local_i, global_i in enumerate(act_idx):
            weights[global_i, :] = bma_weights[local_i, :]

    # MIP:
    elif scheme_name == "MIP":
        print("Gurobi Start")

        #   Generate model for Gurobi solver
        m = gp.Model("MIP")

        # Decision variables: x corresponding to classifiers, w corresponding to weights per classifier-class
        x = m.addVars(n_classifiers, vtype=GRB.BINARY, name="x")  # binary variables x

        w = m.addVars(n_classifiers, n_classes, vtype=GRB.CONTINUOUS, lb=0.0, name="w")  # non-negative continuous variables w

        # Constraints:
        # Ensemble size constraint:
        m.addConstr(gp.quicksum(x[i] for i in range(n_classifiers)) == ensemble_size, name="ensemble_size_constraint")

        # Weight constraints per class
        for j in range(n_classes):
            m.addConstr(gp.quicksum(w[i, j] for i in range(n_classifiers)) == 1, name=f"weight_constraint_per_class_{j}")

        # Weight constraints per classifier -- upper bound
        for i in range(n_classifiers):
            m.addConstr(gp.quicksum(w[i, j] for j in range(n_classes)) <= n_classes * x[i], name=f"weight_constraint_per_classifier_upper_bound_{i}")

        # Weight constraints per classifier -- lower bound
        M = 1000  # Large positive constant
        EPSILON = 0.000001  # Small positive constant

        for i in range(n_classifiers):
            m.addConstr(gp.quicksum(w[i, j] for j in range(n_classes)) + M * (1 - x[i]) >= EPSILON, name=f"weight_constraint_per_classifier_lower_bound_{i}")

        # Accuracy constraints -- per class
        for j in range(n_classes):
            weighted_average_accuracy_per_class = gp.quicksum(w[i, j] * class_accuracies[i, j] for i in range(n_classifiers))

            average_accuracy_per_class = (1.0 / n_classifiers) * gp.quicksum(class_accuracies[i, j] for i in range(n_classifiers))

            m.addConstr(weighted_average_accuracy_per_class >= average_accuracy_per_class + EPSILON, name=f"accuracy_constraint_per_class_{j}")

        # Accuracy constraints -- overall
        weighted_average_accuracy_overall = (1.0 / n_classes) * gp.quicksum(w[i, j] * class_accuracies[i, j] for i in range(n_classifiers) for j in range(n_classes))

        average_accuracy_overall = (1.0 / (n_classifiers * n_classes)) * gp.quicksum(class_accuracies[i, j] for i in range(n_classifiers) for j in range(n_classes))

        m.addConstr(weighted_average_accuracy_overall >= average_accuracy_overall + EPSILON, name="overall_accuracy_constraint")

        # Objective function:
        LAMBDA = 0.81  # Regularization strength parameter for elastic net -- can be changed
        ALPHA = 0.92  # Balancing parameter for the L1 and L2 penalties in elastic net -- can be changed

        l1 = gp.quicksum(w[i, j] for i in range(n_classifiers) for j in range(n_classes))  # L1 (lasso) penalty
        l2 = gp.quicksum(w[i, j] * w[i, j] for i in range(n_classifiers) for j in range(n_classes))  # L2 (ridge) penalty

        elastic_net_regularization = LAMBDA * (ALPHA * l1 + ((1 - ALPHA)/2.0) * l2)

        objective = weighted_average_accuracy_overall - elastic_net_regularization

        m.setObjective(objective, GRB.MAXIMIZE)

        # Set Gurobi log file path
        log_file = f"{dataset_base_name}_gurobi_log_{timestamp}.log"
        m.Params.LogFile = log_file
        m.Params.LogToConsole = 0

        # Optimize the model
        m.optimize()

        # Check optimization status and save results
        if m.status == GRB.OPTIMAL:
            print("Optimal solution found.")
            runtime = m.Runtime
            optimality_gap = m.MIPGap
            optimal_solution = m.ObjVal
            print(f"Gurobi Runtime: {runtime:.2f} seconds")
            print(f"Optimality Gap: {optimality_gap * 100:.2f}%")
            print(f"Optimal Solution Value: {optimal_solution:.6f}")

            # Collect model statistics
            total_variables = m.NumVars
            num_binary_variables = m.NumBinVars
            num_integer_nonbinary_variables = m.NumIntVars - m.NumBinVars
            num_continuous_variables = total_variables - (num_binary_variables + num_integer_nonbinary_variables)
            total_constraints = m.NumConstrs
            num_quadratic_constraints = m.NumQConstrs
            num_linear_constraints = total_constraints - num_quadratic_constraints

            # Extract the calculated weights
            for i in range(n_classifiers):
                for j in range(n_classes):
                    weights[i, j] = w[i, j].x

            # Save Gurobi solution to CSV
            solution_file = f"{dataset_base_name}_gurobi_results.csv"
            date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            solution_df = pd.DataFrame([[
                dataset_base_name, timestamp, date_time, ensemble_size, runtime, optimality_gap * 100,
                optimal_solution, LAMBDA, ALPHA, total_variables, num_binary_variables, num_integer_nonbinary_variables, num_continuous_variables,
                total_constraints, num_linear_constraints, num_quadratic_constraints
            ]], columns=[
                "Dataset", "RunID", "Timestamp", "EnsembleSize", "Runtime(s)", "OptimalityGap(%)",
                "OptimalSolutionValue", "Lambda", "Alpha", "TotalNumberOfVariables", "NumberOfBinaryVariables", "NumberOfIntegerNonBinaryVariables",
                "NumberOfContinuousVariables", "TotalNumberOfConstraints",
                "NumberOfLinearConstraints", "NumberOfQuadraticConstraints"
            ])

            if not os.path.exists(solution_file):
                # Write with headers if file does not exist
                solution_df.to_csv(solution_file, index=False, mode='w')
            else:
                # Append without headers if file exists
                solution_df.to_csv(solution_file, index=False, mode='a', header=False)

        elif m.status == GRB.INFEASIBLE:
            print("No feasible solution found.")
        elif m.status == GRB.UNBOUNDED:
            print("The model is unbounded.")
        else:
            print(f"Optimization ended with status {m.status}.")

        print("Gurobi End")

    return weights


def main():
    # -- Preprocessing --
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Used as RunID

    # Read the dataset CSV file (first row should be the column headers)
    file_path = input("Please enter the path (including the filename) of the dataset CSV file: ").strip()
    rep_time = datetime.now()
    data = pd.read_csv(file_path)
    print(f"---- PREPROCESSING ----")
    print(f"Started preprocessing at {rep_time.strftime('%Y-%m-%d %H:%M:%S')} \nRead dataset in {file_path}")

    # Save the number of instances per class
    class_counts = data.iloc[:, -1].value_counts()  # Count instances per class
    class_counts_df = class_counts.reset_index()
    class_counts_df.columns = ["Class", "InstanceCount"]

    # Extract the base name from the dataset file path
    dataset_base_name = os.path.splitext(os.path.basename(file_path))[0]  # Get dataset filename
    output_file_name = f"{dataset_base_name}_class_counts_{timestamp}.csv"
    class_counts_df.to_csv(output_file_name, index=False)
    print(f"Class counts saved to {output_file_name}")

    # Separate features and labels
    X = data.iloc[:, :-1].values  # features
    y = data.iloc[:, -1].values   # class labels (last column)

    # Encode categorical class labels (if any)
    if y.dtype == "object" or isinstance(y[0], str):
        print("Class labels are categorical. They will be encoded.")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Ensure consistent class mapping across all splits
    classes = np.unique(y)
    n_classes = len(classes)

    # Handle missing values (if any)
    if pd.DataFrame(X).isnull().values.any():
        X = pd.DataFrame(X).fillna(X.mean()).values

    # Convert categorical features (if any)
    categorical_columns = data.iloc[:, :-1].select_dtypes(include=["object"]).columns
    if len(categorical_columns) > 0:
        print(f"Categorical features found: {categorical_columns}. They will be encoded.")
        X = pd.get_dummies(data.iloc[:, :-1], columns=categorical_columns).values
    else:
        X = data.iloc[:, :-1].values  # Retain original features if no categorical columns

    if X.shape[1] == 0:
        raise ValueError("No valid features found in the dataset after preprocessing.")

    # Normalize/Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training/validation and test sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # 80% training/validation set and 20% test set
    train_idx, test_idx = next(sss.split(X, y))
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    # Stratified 5-fold cross-validation
    # Calculate minimum number of instances per class
    counts = dict(Counter(y))
    min_class_count = min(counts.values())

    # Dynamically set n_splits for StratifiedKFold
    n_splits = min(5, min_class_count)  # Use up to 5 splits, limited by the smallest class size
    skf = StratifiedKFold(n_splits=n_splits)

    rep_time = datetime.now()
    print(f"Finished preprocessing at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # -- Training/Validation (Phase 1) --
    rep_time = datetime.now()
    print(f"---- PHASE 1 - TRAINING/VALIDATION ----\nStarted training/validation at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")
    n_classifiers = len(classifiers)

    # Ask user if base classifiers are pre-trained
    pretrained_option = input("Are the base classifiers pre-trained for this dataset? (yes/no): ").strip().lower()

    if pretrained_option == "no":
        print("Started training/validation of base classifiers...")

        start_training_time = time.time()
        fold_models = []
        class_wise_accuracies = {name: {c: [] for c in np.unique(y)} for name in classifiers.keys()}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
            X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

            fold_model_set = {}
            for name, clf in classifiers.items():
                clf.fit(X_train, y_train)
                fold_model_set[name] = clf

                # Calculate class-wise accuracies
                y_pred = clf.predict(X_val)
                for c in np.unique(y_val):
                    class_mask = y_val == c
                    class_acc = np.sum(y_pred[class_mask] == y_val[class_mask]) / np.sum(class_mask)
                    class_wise_accuracies[name][c].append(class_acc)

            # Save fold models and fold index
            fold_models.append({"fold_idx": fold_idx, "models": fold_model_set})

        # Save fold models to disk
        model_dir = f"Trained_Fold_Models_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"fold_models.joblib")
        dump(fold_models, model_path)
        print(f"Trained models per fold saved to: {model_path}")

        # Calculate mean class-wise accuracies and overall mean accuracy
        accuracies_matrix = []
        for name in classifiers.keys():
            row = [name]  # Classifier name
            # Mean accuracy for each class
            class_means = [np.mean(class_wise_accuracies[name][c]) for c in classes]
            # Overall accuracy (mean of class-wise means)
            overall_mean = np.mean(class_means)
            row.extend(class_means)
            row.append(overall_mean)  # Append overall mean accuracy
            accuracies_matrix.append(row)

        # Create a DataFrame for the accuracies matrix
        columns = ["Classifier"] + [f"Class_{c}_MeanAccuracy" for c in classes] + ["OverallMeanAccuracy"]
        accuracies_df = pd.DataFrame(accuracies_matrix, columns=columns)

        # Save the accuracies matrix to a CSV file
        output_file_name = f"{dataset_base_name}_accuracies_matrix_{timestamp}.csv"
        accuracies_df.to_csv(output_file_name, index=False)
        print(f"Accuracies matrix saved to {output_file_name}")

        # Calculate training time
        end_training_time = time.time()
        training_time = end_training_time - start_training_time

    elif pretrained_option == "yes":
        # Load class-wise accuracies
        # Prompt user for CSV file path containing pre-trained accuracies
        csv_path = input("Please enter the path (including the filename) of the accuracies CSV file: ").strip()

        # Check if file exists and validate its structure
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"The file '{csv_path}' does not exist.")

        accuracies_df = pd.read_csv(csv_path)

        # Validate structure of the loaded accuracies_df
        expected_columns = ["Classifier"] + [f"Class_{c}_MeanAccuracy" for c in classes] + ["OverallMeanAccuracy"]
        if list(accuracies_df.columns) != expected_columns:
            raise ValueError(f"Invalid CSV format. Expected columns: {expected_columns}")

        if accuracies_df.shape != (n_classifiers, n_classes + 2):
            raise ValueError(f"Invalid CSV structure. Expected {n_classifiers + 1} rows and {len(expected_columns)} columns, including {n_classifiers} classifiers and {n_classes} classes.")

        # Load pre-trained models per fold
        # Prompt user for the directory containing the pre-trained models
        model_dir = input("Please enter the directory containing the pre-trained models per fold: ").strip()

        model_path = os.path.join(model_dir, f"fold_models.joblib")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"The file '{model_path}' does not exist.")
        fold_models = load(model_path)

        training_time = 0
        print(f"Finished loading pre-trained classifiers and accuracies.")

    else:
        raise ValueError("Invalid option.")

    rep_time = datetime.now()
    print(f"Finished training/validation at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Loop over all ensemble sizes
    for ensemble_size in ENSEMBLE_SIZES:
        print(f"---------------- ENSEMBLE SIZE K = {ensemble_size} ----------------")

        # Loop over all weighting schemes
        for scheme_name in WEIGHTING_SCHEMES:
            # In all weighting schemes (except MIP) we need to check all combinations of classifiers for the specific ensemble size
            if scheme_name != "MIP":
                # Generate all combinations of classifiers for the specific ensemble size
                classifier_combinations = list(combinations(range(n_classifiers), ensemble_size))
                total_combinations = len(classifier_combinations)

                for idx, combination in enumerate(classifier_combinations):
                    combination_number = f"{idx + 1} of {total_combinations}"
                    combination_name = "|".join([list(classifiers.keys())[i] for i in combination])

                    # -- Weight Calculation (Phase 2) --
                    rep_time = datetime.now()
                    print(f"---- {scheme_name} - CLASSIFIER COMBINATION {combination_number} ({combination_name}): PHASE 2 - WEIGHT CALCULATION ----\nStarted weight calculation at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")

                    # Create a new accuracies dataframe where all classifiers except those in combination have their accuracies set to 0
                    modified_accuracies_df = accuracies_df.copy()

                    for i, classifier in enumerate(accuracies_df["Classifier"]):
                        if i not in combination:
                            # Set class-wise accuracies and overall accuracy to 0
                            class_accuracy_cols = [f"Class_{c}_MeanAccuracy" for c in classes]
                            modified_accuracies_df.loc[
                                modified_accuracies_df["Classifier"] == classifier, class_accuracy_cols] = 0
                            modified_accuracies_df.loc[
                                modified_accuracies_df["Classifier"] == classifier, "OverallMeanAccuracy"] = 0

                    # Weighting scheme
                    start_weighting_time = time.time()

                    weights = weighting_scheme(scheme_name, n_classifiers, n_classes, classes, modified_accuracies_df, ensemble_size, timestamp, dataset_base_name, combination)

                    end_weighting_time = time.time()
                    weighting_time = end_weighting_time - start_weighting_time

                    # Save the weights matrix to a CSV file
                    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    weights_file = f"{dataset_base_name}_weight_matrices.csv"
                    weights_df = pd.DataFrame(weights, columns=[f"Class_{c}" for c in classes],
                                              index=list(classifiers.keys()))

                    # Add additional columns for metadata
                    weights_df.reset_index(inplace=True)
                    weights_df.rename(columns={"index": "Classifier"}, inplace=True)
                    weights_df["Dataset"] = dataset_base_name
                    weights_df["RunID"] = timestamp
                    weights_df["Timestamp"] = date_time
                    weights_df["EnsembleSize"] = ensemble_size
                    weights_df["WeightingScheme"] = scheme_name
                    weights_df["ClassifierCombinationNumber"] = combination_number
                    weights_df["ClassifierCombination"] = combination_name

                    # Reorder the columns
                    ordered_columns = ["Dataset", "RunID", "Timestamp", "EnsembleSize", "WeightingScheme", "ClassifierCombinationNumber", "ClassifierCombination", "Classifier"] + [f"Class_{c}" for c in classes]
                    weights_df = weights_df[ordered_columns]

                    # Write or append to the weights matrix file
                    if not os.path.exists(weights_file):  # If the file does not exist, write with headers
                        weights_df.to_csv(weights_file, index=False, mode='w')
                    else:  # If the file exists, append without headers
                        weights_df.to_csv(weights_file, index=False, mode='a', header=False)

                    rep_time = datetime.now()
                    print(f"Finished weight calculation at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")

                    # -- Test/Evaluation (Phase 3) --
                    rep_time = datetime.now()
                    print(f"---- {scheme_name} - CLASSIFIER COMBINATION {combination_number} ({combination_name}): PHASE 3 - TEST/EVALUATION ----\nStarted test/evaluation at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")

                    start_test_time = time.time()

                    averaged_probabilities = {name: np.zeros((X_test.shape[0], n_classes)) for name in classifiers.keys()}

                    for fold_model_set in fold_models:
                        for name, clf in fold_model_set["models"].items():
                            if hasattr(clf, "predict_proba"):
                                proba = clf.predict_proba(X_test)
                            else:
                                y_pred = clf.predict(X_test)
                                proba = np.zeros((X_test.shape[0], n_classes))
                                for i, pred_class in enumerate(y_pred):
                                    proba[i, classes == pred_class] = 1.0

                            # Align probabilities to class order
                            aligned_proba = np.zeros((X_test.shape[0], n_classes))
                            for c_idx, c in enumerate(classes):
                                if c in clf.classes_:
                                    local_index = np.argwhere(clf.classes_ == c)[0][0]
                                    aligned_proba[:, c_idx] = proba[:, local_index]

                            # Accumulate probabilities for averaging
                            averaged_probabilities[name] += aligned_proba

                    # Average probabilities over folds
                    for name in averaged_probabilities.keys():
                        averaged_probabilities[name] /= len(fold_models)

                    ensemble_scores = np.zeros((X_test.shape[0], n_classes))

                    for j, name in enumerate(classifiers.keys()):
                        ensemble_scores += averaged_probabilities[name] * weights[j]

                    # Final predictions
                    ensemble_indices = np.argmax(ensemble_scores, axis=1)
                    ensemble_preds = classes[ensemble_indices]

                    end_test_time = time.time()
                    test_time = end_test_time - start_test_time

                    # Metrics
                    balanced_acc = balanced_accuracy_score(y_test, ensemble_preds)
                    precision = precision_score(y_test, ensemble_preds, average='macro', zero_division=0)
                    recall = recall_score(y_test, ensemble_preds, average='macro', zero_division=0)
                    f1 = f1_score(y_test, ensemble_preds, average='macro', zero_division=0)

                    # Save results to CSV file
                    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    results_file = f"{dataset_base_name}_results.csv"
                    results_df = pd.DataFrame([[dataset_base_name, timestamp, date_time, ensemble_size, scheme_name, combination_number, combination_name,
                                                round(balanced_acc, 12), round(precision, 12), round(recall, 12), round(f1, 12),
                                                round(training_time, 6), round(weighting_time, 6), round(test_time, 6)]],
                                              columns=["Dataset", "RunID", "Timestamp", "EnsembleSize", "WeightingScheme", "ClassifierCombinationNumber", "ClassifierCombination",
                                                       "BalancedAccuracy", "MacroAvgPrecision", "MacroAvgRecall", "MacroAvgF1Score",
                                                       "TrainingTime(s)", "WeightingTime(s)", "TestTime(s)"])

                    if not os.path.exists(results_file):  # If the file does not exist, write with headers
                        results_df.to_csv(results_file, index=False, mode='w')
                    else:  # If the file exists, append without headers
                        results_df.to_csv(results_file, index=False, mode='a', header=False)

                    rep_time = datetime.now()
                    print(f"Finished test/evaluation at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # MIP seamlessly selects the specified number of classifiers while assigning their weights
            elif scheme_name == "MIP":
                # -- Weight Calculation (Phase 2) --
                rep_time = datetime.now()
                print(f"---- {scheme_name}: PHASE 2 - WEIGHT CALCULATION ----\nStarted weight calculation at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Weighting scheme
                start_weighting_time = time.time()

                weights = weighting_scheme(scheme_name, n_classifiers, n_classes, classes, accuracies_df, ensemble_size, timestamp, dataset_base_name, None)

                end_weighting_time = time.time()
                weighting_time = end_weighting_time - start_weighting_time

                # Save the weights matrix to a CSV file
                date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                weights_file = f"{dataset_base_name}_weight_matrices.csv"
                weights_df = pd.DataFrame(weights, columns=[f"Class_{c}" for c in classes], index=list(classifiers.keys()))

                # Add additional columns for metadata
                weights_df.reset_index(inplace=True)
                weights_df.rename(columns={"index": "Classifier"}, inplace=True)
                weights_df["Dataset"] = dataset_base_name
                weights_df["RunID"] = timestamp
                weights_df["Timestamp"] = date_time
                weights_df["EnsembleSize"] = ensemble_size
                weights_df["WeightingScheme"] = scheme_name
                weights_df["ClassifierCombinationNumber"] = None
                weights_df["ClassifierCombination"] = None

                # Reorder the columns
                ordered_columns = ["Dataset", "RunID", "Timestamp", "EnsembleSize", "WeightingScheme", "ClassifierCombinationNumber", "ClassifierCombination", "Classifier"] + [f"Class_{c}" for c in classes]
                weights_df = weights_df[ordered_columns]

                # Write or append to the weights matrix file
                if not os.path.exists(weights_file):  # If the file does not exist, write with headers
                    weights_df.to_csv(weights_file, index=False, mode='w')
                else:  # If the file exists, append without headers
                    weights_df.to_csv(weights_file, index=False, mode='a', header=False)

                rep_time = datetime.now()
                print(f"Finished weight calculation at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")

                # -- Test/Evaluation (Phase 3) --
                rep_time = datetime.now()
                print(f"---- {scheme_name}: PHASE 3 - TEST/EVALUATION ----\nStarted test/evaluation at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")

                start_test_time = time.time()

                averaged_probabilities = {name: np.zeros((X_test.shape[0], n_classes)) for name in classifiers.keys()}

                for fold_model_set in fold_models:
                    for name, clf in fold_model_set["models"].items():
                        if hasattr(clf, "predict_proba"):
                            proba = clf.predict_proba(X_test)
                        else:
                            y_pred = clf.predict(X_test)
                            proba = np.zeros((X_test.shape[0], n_classes))
                            for i, pred_class in enumerate(y_pred):
                                proba[i, classes == pred_class] = 1.0

                        # Align probabilities to class order
                        aligned_proba = np.zeros((X_test.shape[0], n_classes))
                        for c_idx, c in enumerate(classes):
                            if c in clf.classes_:
                                local_index = np.argwhere(clf.classes_ == c)[0][0]
                                aligned_proba[:, c_idx] = proba[:, local_index]

                        # Accumulate probabilities for averaging
                        averaged_probabilities[name] += aligned_proba

                # Average probabilities over folds
                for name in averaged_probabilities.keys():
                    averaged_probabilities[name] /= len(fold_models)

                ensemble_scores = np.zeros((X_test.shape[0], n_classes))

                for j, name in enumerate(classifiers.keys()):
                    ensemble_scores += averaged_probabilities[name] * weights[j]

                # Final predictions
                ensemble_indices = np.argmax(ensemble_scores, axis=1)
                ensemble_preds = classes[ensemble_indices]

                end_test_time = time.time()
                test_time = end_test_time - start_test_time

                # Metrics
                balanced_acc = balanced_accuracy_score(y_test, ensemble_preds)
                precision = precision_score(y_test, ensemble_preds, average='macro', zero_division=0)
                recall = recall_score(y_test, ensemble_preds, average='macro', zero_division=0)
                f1 = f1_score(y_test, ensemble_preds, average='macro', zero_division=0)

                # Save results to CSV file
                date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                results_file = f"{dataset_base_name}_results.csv"
                results_df = pd.DataFrame([[dataset_base_name, timestamp, date_time, ensemble_size, scheme_name, None, None,
                                            round(balanced_acc, 12), round(precision, 12), round(recall, 12), round(f1, 12),
                                            round(training_time, 6), round(weighting_time, 6), round(test_time, 6)]],
                                          columns=["Dataset", "RunID", "Timestamp", "EnsembleSize", "WeightingScheme", "ClassifierCombinationNumber", "ClassifierCombination",
                                                   "BalancedAccuracy", "MacroAvgPrecision", "MacroAvgRecall", "MacroAvgF1Score",
                                                   "TrainingTime(s)", "WeightingTime(s)", "TestTime(s)"])

                if not os.path.exists(results_file):  # If the file does not exist, write with headers
                    results_df.to_csv(results_file, index=False, mode='w')
                else:  # If the file exists, append without headers
                    results_df.to_csv(results_file, index=False, mode='a', header=False)

                rep_time = datetime.now()
                print(f"Finished test/evaluation at {rep_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"Weight matrices saved to {weights_file}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
