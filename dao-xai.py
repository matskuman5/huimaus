import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sympy import symbols, Eq, Or, And, simplify_logic, Symbol, Not
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

SEED = 33


def format_formula(formula):
    return (
        formula.replace("&", "AND")
        .replace("|", "OR")
        .replace("~", "NOT ")
        .replace("_[M*]", "")
    )


def load_binary_matrix(path):
    data = pd.read_csv(path + ".csv")

    # Split data into training and testing sets
    train, holdout = train_test_split(data, test_size=0.3, random_state=SEED)
    train_train, train_validation = train_test_split(
        train, test_size=0.3, random_state=SEED
    )

    return train_train, train_validation, holdout


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


def predict(train, holdout, median_values):

    unique_multisets = train.iloc[:, :-1].drop_duplicates().values.tolist()
    X_train = train.iloc[:, :-1].values.tolist()
    y_train = train.iloc[:, -1].values.tolist()

    X_test = holdout.iloc[:, :-1].values.tolist()
    y_test = holdout.iloc[:, -1].values.tolist()

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

    predictions = []
    for i in range(len(y_test)):
        prediction = 0
        if X_test[i] in one_predictors:
            prediction = 1
        predictions.append(prediction)

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

    print("Formula:", format_formula(formula))

    return formula, predictions


def classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    return accuracy, f1, sensitivity


def predict_dataset(
    train_index,
    test_index,
    boolean_data,
    numeric_data,
    args,
    verbose=False,
    results_file=None,
):

    boolean_train, boolean_test = (
        boolean_data.iloc[train_index],
        boolean_data.iloc[test_index],
    )
    numeric_train, numeric_test = (
        numeric_data.iloc[train_index],
        numeric_data.iloc[test_index],
    )

    # Split train data into training and validation sets
    train_train, train_validation = train_test_split(
        boolean_train, test_size=0.3, random_state=SEED
    )

    print("Predicting feature: ", boolean_train.columns[-1])

    median_values = {}
    with open(f"huimausdata_median_values.txt", "r") as f:
        for line in f:
            col, med_val = line.strip().split(":")
            median_values[col.strip() + "_[M*]"] = med_val.strip()

    amount_of_features = len(boolean_train.columns) - 1

    if args.iteration > amount_of_features:
        print(
            f"Note: dataset has {amount_of_features} features, which is smaller than the desired maximum amount {args.iteration}."
        )
        args.iteration = amount_of_features

    best_accuracy = 0
    best_method = None
    best_n_features = 0

    # Model selection

    for method in [mutual_info_classif, f_classif, chi2]:
        print("using method: ", method)
        print("------")
        for i in range(min(args.iteration, amount_of_features)):
            local_train, local_validation = train_train.copy(), train_validation.copy()
            local_train, local_validation = feature_selection(
                local_train, local_validation, i + 1, method
            )
            formula, predictions = predict(local_train, local_validation, median_values)
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

    results = {}

    local_train_validation, local_test = boolean_train.copy(), boolean_test.copy()
    local_train_validation, local_test = feature_selection(
        local_train_validation, local_test, best_n_features, best_method
    )
    formula, predictions = predict(local_train_validation, local_test, median_values)

    accuracy, f1, sensitivity = classification_metrics(
        local_test.iloc[:, -1], predictions
    )
    results["dao-xai"] = [accuracy, f1, sensitivity]

    print(f"Test accuracy: {accuracy:.3f}")
    print(f"Test F1-score: {f1:.3f}")
    print(f"Test sensitivity: {sensitivity:.3f}\n")

    if verbose and results_file:
        results_file.write(
            f"Example formula: {format_formula(formula)} (accuracy: {accuracy:.3f}, F1: {f1:.3f}, sensitivity: {sensitivity:.3f})\n"
        )

    # Create and plot confusion matrix
    # cm = confusion_matrix(local_test.iloc[:, -1], predictions)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(
    #     cm,
    #     annot=True,
    #     fmt="d",
    #     cmap="Blues",
    #     xticklabels=np.unique(local_test.iloc[:, -1]),
    #     yticklabels=np.unique(local_test.iloc[:, -1]),
    # )
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title("Confusion Matrix")
    # plt.tight_layout()
    # plt.show()

    # Compare against baseline and random forest
    bot_accuracy = accuracy_score(boolean_test.iloc[:, -1], [0] * len(boolean_test))
    top_accuracy = accuracy_score(boolean_test.iloc[:, -1], [1] * len(boolean_test))
    baseline_predictions = [1 if top_accuracy > bot_accuracy else 0] * len(boolean_test)
    baseline_accuracy, baseline_f1, baseline_sensitivity = classification_metrics(
        local_test.iloc[:, -1], baseline_predictions
    )
    results["baseline"] = [baseline_accuracy, baseline_f1, baseline_sensitivity]

    print(
        f"Baseline test metrics - accuracy: {baseline_accuracy:.3f}, F1: {baseline_f1:.3f}, sensitivity: {baseline_sensitivity:.3f}"
    )

    rf = RandomForestClassifier()
    bst = XGBClassifier()
    rf.fit(numeric_train.iloc[:, :-1], numeric_train.iloc[:, -1])
    bst.fit(numeric_train.iloc[:, :-1], numeric_train.iloc[:, -1])

    rf_predictions = rf.predict(numeric_test.iloc[:, :-1])
    bst_predictions = bst.predict(numeric_test.iloc[:, :-1])

    rf_accuracy, rf_f1, rf_sens = classification_metrics(
        numeric_test.iloc[:, -1], rf_predictions
    )
    results["rf"] = [rf_accuracy, rf_f1, rf_sens]

    bst_accuracy, bst_f1, bst_sens = classification_metrics(
        numeric_test.iloc[:, -1], bst_predictions
    )
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
        "-i",
        "--iteration",
        type=int,
        action="store",
        help="Maximum features to select.",
        default=5,
    )
    parser.add_argument(
        "-cv",
        "--cross-validation",
        action="store_true",
        help="Use 10-fold cross-validation.",
        default=False,
    )
    args = parser.parse_args()

    # Load data
    if args.path.endswith(".txt"):
        with open(args.path, "r") as f:
            datasets = [line.strip() for line in f if line.strip()]
    else:
        datasets = [args.path]

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        boolean_data = pd.read_csv("boolean_datasets/boolean_" + dataset + ".csv")
        numeric_data = pd.read_csv("datasets/" + dataset + ".csv")

        # Open results.txt
        results_file = open("results.txt", "a")
        results_file.write(f"\nResults for dataset: {dataset}\n")

        # 10-CV
        if args.cross_validation:
            # Store metrics across all iterations and folds
            all_metrics = {
                "dao-xai": {"accuracy": [], "f1": [], "sensitivity": []},
                "baseline": {"accuracy": [], "f1": [], "sensitivity": []},
                "rf": {"accuracy": [], "f1": [], "sensitivity": []},
                "xgboost": {"accuracy": [], "f1": [], "sensitivity": []},
            }

            for iteration in range(1, 11):
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
                    test_index = numeric_data[
                        numeric_data[f"CV{iteration}"] == fold
                    ].index
                    train_index = numeric_data[
                        numeric_data[f"CV{iteration}"] != fold
                    ].index

                    results = predict_dataset(
                        train_index,
                        test_index,
                        boolean_data,
                        numeric_data,
                        args,
                        verbose=verbose,
                        results_file=results_file,
                    )

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
                    method_name = (
                        "DAOXAI" if method == "dao-xai" else method.capitalize()
                    )
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
                    f"{method_name} 100 folds:\t\t{avg_acc:.3f} "
                    + f"(F1: {avg_f1:.3f}, sens: {avg_sens:.3f})"
                )

                print(result_line)
                results_file.write(result_line + "\n")


if __name__ == "__main__":
    main()
