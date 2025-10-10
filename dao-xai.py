import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sympy import symbols, Eq, Or, And, simplify_logic, Symbol, Not
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

SEED = 33


def format_formula(formula):
    return (
        formula.replace("&", "AND")
        .replace("|", "OR")
        .replace("~", "NOT ")
        .replace("_[M*]", "")
    )


def feature_selection(train, holdout, k, method):
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = holdout.iloc[:, :-1]
    y_test = holdout.iloc[:, -1]

    # Remove constant features
    X_train = X_train.loc[:, (X_train != X_train.iloc[0]).any()]
    X_test = X_test[X_train.columns]

    selector = SelectKBest(k=k, score_func=method)
    selector.fit(X_train, y_train)

    cols_idxs = selector.get_support(indices=True)
    X_train = X_train.iloc[:, cols_idxs]
    X_test = X_test.iloc[:, cols_idxs]

    # print("columns selected: ", X_train.columns.tolist())

    train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
    holdout = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)

    return train, holdout


# The core DAO-XAI algorithm
# Gets the optimal classifier formula for the given training set
# Returns both a string representation of the formula as well as the classifier as a dict
def get_optimal_classifier_formula(train, median_values):
    unique_multisets = train.iloc[:, :-1].drop_duplicates().values.tolist()
    X_train = train.iloc[:, :-1].values.tolist()
    y_train = train.iloc[:, -1].values.tolist()

    one_predictors = []
    for u_multiset in unique_multisets:
        all_predictions = 0
        one_predictions = 0
        for i in range(len(X_train)):
            if u_multiset == X_train[i]:
                all_predictions += 1
                if y_train[i] == 1:
                    one_predictions += 1
            else:
                continue
        if all_predictions != 0 and one_predictions / all_predictions > 0.5:
            one_predictors.append(u_multiset)

    # Create Sympy symbols for each feature column (excluding the target)
    sym_vars = {col: Symbol(col) for col in train.columns[:-1]}

    clauses = []
    for predictor in one_predictors:
        conj = []
        for i, val in enumerate(predictor):
            if train.columns[i].endswith("[M*]"):
                symbol = Symbol(
                    f"{sym_vars[train.columns[i]]} >= {median_values[train.columns[i]]}"
                )
                conj.append(symbol)
            else:
                if val == 1:
                    conj.append(Symbol(f"{sym_vars[train.columns[i]]}"))
                else:
                    conj.append(Not(Symbol(f"{sym_vars[train.columns[i]]}")))
        clauses.append(And(*conj))

    if clauses:
        formula = Or(*clauses)
        formula = str(simplify_logic(formula, force=True))
    else:
        formula = "False"

    # Also return a version that is more easily processed
    classifier = {}
    for predictor in one_predictors:
        for i, val in enumerate(predictor):
            classifier[train.columns[i]] = {"value": val, "type": "boolean"}
    support, confidence = support_and_confidence(classifier, train)
    return formula, (classifier, support, confidence)


# Calculate support (number of true positives / total number of instances)
# and confidence (number of true positives / number of positive predictions)
def support_and_confidence(classifier, data):
    X = data.iloc[:, :-1].values.tolist()
    y = data.iloc[:, -1].values.tolist()

    true_positives = 0
    false_positives = 0
    total_positives = 0

    for i in range(len(y)):
        if y[i] == 1:
            total_positives += 1

        prediction = 1
        for feature, condition in classifier.items():
            col_index = data.columns.get_loc(feature)
            if X[i][col_index] != condition["value"]:
                prediction = 0
                break

        if prediction == 1 and y[i] == 1:
            true_positives += 1
        elif prediction == 1 and y[i] == 0:
            false_positives += 1

    support = true_positives / len(y) if len(y) > 0 else 0
    confidence = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )

    return support, confidence


# Returns predictions for the given test set with the given DAO-XAI classifier dict
def predict_with_classifier_formula(classifier, test):
    X_test = test.iloc[:, :-1].values.tolist()
    y_test = test.iloc[:, -1].values.tolist()

    predictions = []
    for i in range(len(y_test)):
        if classifier == {}:
            predictions.append(0)
            continue

        prediction = 1
        for feature, condition in classifier.items():
            col_index = test.columns.get_loc(feature)
            if X_test[i][col_index] != condition["value"]:
                prediction = 0
                break
        predictions.append(prediction)

    return predictions


# The whole DAO-XAI pipeline
# Performs model selection (feature selection method) and evaluation
# Returns the model formula, classifier dict and predictions for the test set
def dao_xai(boolean_train, boolean_test, median_values, max_features):
    # Split train data into training and validation sets
    train_train, train_validation = train_test_split(
        boolean_train, test_size=0.3, random_state=SEED
    )

    print("Predicting feature: ", boolean_train.columns[-1])

    amount_of_features = len(boolean_train.columns) - 1

    if max_features > amount_of_features:
        print(
            f"Note: dataset has {amount_of_features} features, which is smaller than the desired maximum amount {max_features}."
        )
        max_features = amount_of_features

    best_accuracy = 0
    best_method = None
    best_n_features = 0

    # Model selection

    for method in [f_classif, chi2]:
        print("using method: ", method)
        print("------")
        for i in range(min(max_features, amount_of_features)):
            local_train, local_validation = train_train.copy(), train_validation.copy()
            local_train, local_validation = feature_selection(
                local_train, local_validation, i + 1, method
            )
            formula, (classifier, support, confidence) = get_optimal_classifier_formula(
                local_train, median_values
            )
            predictions = predict_with_classifier_formula(classifier, local_validation)
            accuracy = accuracy_score(local_validation.iloc[:, -1], predictions)
            print(f"Accuracy for length {i + 1}: {accuracy:3f}\n")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_method = method
                best_n_features = i + 1

    # Model evaluation

    print("------")
    print("Evaluating model on test set")
    print("------")
    print("Using method: ", best_method)

    local_train_validation, local_test = boolean_train.copy(), boolean_test.copy()
    local_train_validation, local_test = feature_selection(
        local_train_validation, local_test, best_n_features, best_method
    )
    formula, (classifier, support, confidence) = get_optimal_classifier_formula(
        local_train_validation, median_values
    )
    predictions = predict_with_classifier_formula(classifier, local_test)

    return formula, (classifier, support, confidence), predictions


# Runs DAO-XAI multiclass classification with the given dict of classifiers
# Each classifier corresponds to one y-label
# Returns the final predictions for the test set
def dao_xai_multiclass(boolean_test, classifiers):
    classifier_predictions = {}
    for disease in classifiers.keys():
        classifier, support, confidence = classifiers[disease]
        classifier_predictions[disease] = predict_with_classifier_formula(
            classifier, boolean_test
        )

    # Break ties (where multiple classifiers predict positive) by choosing the classifier with highest confidence
    final_predictions = []
    for i in range(len(boolean_test.iloc[:, -1])):
        predicted = 0
        highest_confidence = 0

        for disease, predictions in classifier_predictions.items():
            if predictions[i] == 1:
                _, _, confidence = classifiers[disease]
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    predicted = disease

        if predicted == 0:
            predicted = "Menieres_disease_vertigo"  # Default prediction if no classifier predicts positive
        final_predictions.append(predicted)

    return final_predictions

    # Create and plot confusion matrix
    cm = confusion_matrix(boolean_test.iloc[:, -1], final_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(boolean_test.iloc[:, -1]),
        yticklabels=np.unique(boolean_test.iloc[:, -1]),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return final_predictions


def classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0, average="macro")
    sensitivity = recall_score(y_true, y_pred, zero_division=0, average="macro")
    return accuracy, f1, sensitivity


# Run DAO-XAI binary classification on the given dataset
# Returns classification metrics
def predict_dataset(
    train_index,
    test_index,
    boolean_data,
    median_values,
    args,
    verbose=False,
    results_file=None,
):

    results = {}

    boolean_train, boolean_test = (
        boolean_data.iloc[train_index],
        boolean_data.iloc[test_index],
    )

    formula, (classifier, support, confidence), predictions = dao_xai(
        boolean_train, boolean_test, median_values, args.max_features
    )

    accuracy, f1, sensitivity = classification_metrics(
        boolean_test.iloc[:, -1], predictions
    )
    results["dao-xai"] = [accuracy, f1, sensitivity]

    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Test F1-score: {f1:.3f}")
    print(f"Test sensitivity: {sensitivity:.3f}\n")

    if verbose and results_file:
        results_file.write(
            f"Example formula: {format_formula(formula)} (support: {support:.3f}, confidence: {confidence:.3f}) (accuracy: {accuracy:.3f}, F1: {f1:.3f}, sensitivity: {sensitivity:.3f})\n"
        )

    return results


# Return classification metrics for the following classifiers:
# 1. Dummy classifier that always predicts the most frequent class
# 2. Random Forest
# 3. XGBoost
def other_classifiers(numeric_train, numeric_test):

    results = {}

    # Compare against baseline and random forest
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(numeric_train.iloc[:, :-1], numeric_train.iloc[:, -1])
    dummy_predictions = dummy.predict(numeric_test.iloc[:, :-1])
    results["baseline"] = classification_metrics(
        numeric_test.iloc[:, -1], dummy_predictions
    )

    print(
        f"Baseline test metrics - accuracy: {results['baseline'][0]:.3f}, F1: {results['baseline'][1]:.3f}, sensitivity: {results['baseline'][2]:.3f}"
    )

    rf = RandomForestClassifier()
    rf.fit(numeric_train.iloc[:, :-1], numeric_train.iloc[:, -1])
    rf_predictions = rf.predict(numeric_test.iloc[:, :-1])
    rf_accuracy, rf_f1, rf_sens = classification_metrics(
        numeric_test.iloc[:, -1], rf_predictions
    )
    results["rf"] = [rf_accuracy, rf_f1, rf_sens]

    # XGBoost requires integer labels for some reason
    le = LabelEncoder()
    int_y_train = le.fit_transform(numeric_train.iloc[:, -1])
    int_y_test = le.transform(numeric_test.iloc[:, -1])

    bst = XGBClassifier()
    bst.fit(numeric_train.iloc[:, :-1], int_y_train)
    bst_predictions = bst.predict(numeric_test.iloc[:, :-1])
    bst_accuracy, bst_f1, bst_sens = classification_metrics(int_y_test, bst_predictions)
    results["xgboost"] = [bst_accuracy, bst_f1, bst_sens]

    print(
        f"Random Forest test metrics - accuracy: {rf_accuracy:.3f}, F1: {rf_f1:.3f}, sensitivity: {rf_sens:.3f}"
    )
    print(
        f"XGBoost test metrics - accuracy: {bst_accuracy:.3f}, F1: {bst_f1:.3f}, sensitivity: {bst_sens:.3f}"
    )

    return results


def main():

    parser = argparse.ArgumentParser(description="Load binary matrix")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        action="store",
        help="Path to dataset OR a .txt file with a list of datasets.",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--max-features",
        type=int,
        action="store",
        help="Maximum features to select.",
        default=5,
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        action="store",
        help="Number of iterations of 10-CV to run (1-10, default 1).",
        default=1,
        choices=range(1, 11),
    )
    parser.add_argument(
        "-m",
        "--multiclass",
        action="store_true",
        help="Report multiclass classification results instead of binary classification for each disease separately.",
        default=False,
    )
    args = parser.parse_args()

    # Load data
    if args.path.endswith(".txt"):
        with open(args.path, "r") as f:
            datasets = [line.strip() for line in f if line.strip()]
    else:
        datasets = [args.path]

    median_values = {}
    with open(f"huimausdata_median_values.txt", "r") as f:
        for line in f:
            col, med_val = line.strip().split(":")
            median_values[col.strip() + "_[M*]"] = med_val.strip()

    # Load full datasets
    full_boolean_data = pd.read_csv("huimausdata_boolean.csv")
    full_numeric_data = pd.read_csv("huimausdata.csv")

    # Open results file
    if args.multiclass:
        results_filename = "results.txt"
    else:
        results_filename = "results_binary.txt"
    results_file = open(results_filename, "w")

    # Write current time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    results_file.write(
        f"{current_time}\nMulticlass: {args.multiclass}\nMax features: {args.max_features}\nIterations: {args.iterations}\n"
    )

    # Store metrics across all iterations and folds
    all_metrics = {
        "dao-xai": {"accuracy": [], "f1": [], "sensitivity": []},
        "baseline": {"accuracy": [], "f1": [], "sensitivity": []},
        "rf": {"accuracy": [], "f1": [], "sensitivity": []},
        "xgboost": {"accuracy": [], "f1": [], "sensitivity": []},
    }

    if args.multiclass:
        classes = ["all"]
    else:
        classes = datasets

    for disease in classes:
        if not args.multiclass:
            boolean_data = pd.read_csv("boolean_datasets/boolean_" + disease + ".csv")
            results_file.write(f"\n{disease}\n")
        for iteration in range(1, args.iterations + 1):
            print("\n\nIteration ", iteration)
            # Store metrics for current iteration
            iter_metrics = {
                "dao-xai": {"accuracy": [], "f1": [], "sensitivity": []},
                "baseline": {"accuracy": [], "f1": [], "sensitivity": []},
                "rf": {"accuracy": [], "f1": [], "sensitivity": []},
                "xgboost": {"accuracy": [], "f1": [], "sensitivity": []},
            }

            for fold in range(1, 11):
                print(f"\nFold {fold}")
                verbose = fold == 1

                # Get indices for train and test sets based on CV column
                test_index = full_numeric_data[
                    full_numeric_data[f"CV{iteration}"] == fold
                ].index
                train_index = full_numeric_data[
                    full_numeric_data[f"CV{iteration}"] != fold
                ].index

                results = {}

                classifiers = {}
                # For multiclass prediction, find classifiers for each disease and store them
                if args.multiclass:
                    for disease in datasets:
                        print(disease)
                        boolean_data = pd.read_csv(
                            "boolean_datasets/boolean_" + disease + ".csv"
                        )

                        if args.multiclass:
                            formula, classifier, predictions = dao_xai(
                                boolean_data.iloc[train_index],
                                boolean_data.iloc[test_index],
                                median_values,
                                args.max_features,
                            )
                            classifiers[disease] = classifier
                # For binary classification, just report the results for the given disease
                else:
                    results = predict_dataset(
                        train_index,
                        test_index,
                        boolean_data,
                        median_values,
                        args,
                        verbose,
                        results_file,
                    )

                    numeric_data = pd.read_csv("datasets/" + disease + ".csv")
                    other_results = other_classifiers(
                        numeric_data.iloc[train_index],
                        numeric_data.iloc[test_index],
                    )
                    results.update(other_results)
                    for method in ["dao-xai", "baseline", "rf", "xgboost"]:
                        accuracy, f1, sensitivity = results[method]
                        iter_metrics[method]["accuracy"].append(accuracy)
                        iter_metrics[method]["f1"].append(f1)
                        iter_metrics[method]["sensitivity"].append(sensitivity)
                        all_metrics[method]["accuracy"].append(accuracy)
                        all_metrics[method]["f1"].append(f1)
                        all_metrics[method]["sensitivity"].append(sensitivity)
                    continue

                # Predict using dao-xai multiclass
                predictions = dao_xai_multiclass(
                    full_boolean_data.iloc[test_index], classifiers
                )
                results["dao-xai"] = classification_metrics(
                    full_numeric_data.iloc[test_index].iloc[:, -1], predictions
                )

                other_results = other_classifiers(
                    full_numeric_data.iloc[train_index],
                    full_numeric_data.iloc[test_index],
                )
                results.update(other_results)

                # Store metrics for each method
                for method in ["dao-xai", "baseline", "rf", "xgboost"]:
                    accuracy, f1, sensitivity = results[method]
                    iter_metrics[method]["accuracy"].append(accuracy)
                    iter_metrics[method]["f1"].append(f1)
                    iter_metrics[method]["sensitivity"].append(sensitivity)
                    all_metrics[method]["accuracy"].append(accuracy)
                    all_metrics[method]["f1"].append(f1)
                    all_metrics[method]["sensitivity"].append(sensitivity)

            print("\n---10CV---")
            for method in ["dao-xai", "baseline", "rf", "xgboost"]:
                method_name = "DAOXAI" if method == "dao-xai" else method.capitalize()
                print(
                    f"{method_name}: {np.mean(iter_metrics[method]['accuracy']):.3f} "
                    + f"(F1: {np.mean(iter_metrics[method]['f1']):.3f}, "
                    + f"sens: {np.mean(iter_metrics[method]['sensitivity']):.3f})"
                )

        print("-----------\n")
        print("\n---Overall---")

        # Print and write overall results
        for method in ["dao-xai", "baseline", "rf", "xgboost"]:
            method_name = "DAOXAI" if method == "dao-xai" else method.capitalize()
            avg_acc = np.mean(all_metrics[method]["accuracy"])
            avg_f1 = np.mean(all_metrics[method]["f1"])
            avg_sens = np.mean(all_metrics[method]["sensitivity"])

            result_line = (
                f"{method_name} {args.iterations * 10} folds:\t\t{avg_acc:.3f} "
                + f"(F1: {avg_f1:.3f}, sens: {avg_sens:.3f})"
            )

            print(result_line)
            results_file.write(result_line + "\n")


if __name__ == "__main__":
    main()
