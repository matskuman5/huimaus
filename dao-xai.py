import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sympy import symbols, Eq, Or, And, simplify_logic, Symbol, Not
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_binary_matrix(path):
    data = pd.read_csv(path + ".csv")

    # Split data into training and testing sets
    train, holdout = train_test_split(data, test_size=0.3, random_state=42)
    train_train, train_validation = train_test_split(
        train, test_size=0.3, random_state=1
    )

    return train_train, train_validation, holdout


# def load_10_cv_fold(base_path, fold):
#     train = pd.read_csv(base_path + f"_000_train_split_{fold}.pp", header=None, sep=" ")
#     validation = pd.read_csv(
#         base_path + f"_000_val_split_{fold}.pp", header=None, sep=" "
#     )
#     test = pd.read_csv(base_path + f"_000_test_split_{fold}.pp", header=None, sep=" ")

#     with open(base_path + "_000_variables.txt", "r") as f:
#         column_names = f.readline().strip().split()
#     train.rename(
#         columns={i: column_names[i] for i in range(len(column_names))}, inplace=True
#     )
#     validation.rename(
#         columns={i: column_names[i] for i in range(len(column_names))}, inplace=True
#     )
#     test.rename(
#         columns={i: column_names[i] for i in range(len(column_names))}, inplace=True
#     )
#     return train, validation, test


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
    # j = 1
    for u_multiset in unique_multisets:
        all_predictions = 0
        one_predictions = 0
        # print(f"{j}/{len(unique_multisets)}")
        # j += 1
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

    accuracy = accuracy_score(y_test, predictions)

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

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(y_test)):
        if y_test[i] == 1 and predictions[i] == 1:
            true_positives += 1
        elif y_test[i] == 0 and predictions[i] == 0:
            true_negatives += 1
        elif y_test[i] == 1 and predictions[i] == 0:
            false_negatives += 1
        elif y_test[i] == 0 and predictions[i] == 1:
            false_positives += 1

    # print("True positives / false positives:", true_positives, "/", false_positives)
    # print("True negatives / false negatives:", true_negatives, "/", false_negatives)

    print(
        "Formula:",
        formula.replace("&", "AND")
        .replace("|", "OR")
        .replace("~", "NOT ")
        .replace("_[M*]", ""),
    )

    return formula, accuracy


def predict_dataset(train, validation, test, path, args):

    print("Predicting feature: ", train.columns[-1])

    median_values = {}
    with open(f"huimausdata_median_values.txt", "r") as f:
        for line in f:
            col, med_val = line.strip().split(":")
            median_values[col.strip() + "_[M*]"] = med_val.strip()

    amount_of_features = len(train.columns) - 1

    if args.iteration > amount_of_features:
        print(
            f"Note: dataset has {amount_of_features} features, which is smaller than the desired maximum amount {args.iteration}."
        )
        args.iteration = amount_of_features

    best_formula = []
    best_accuracy = 0
    best_method = None
    best_n_features = 0

    # Model selection

    # mutual info method disabled for now since it's quite slow
    for method in [f_classif, chi2]:
        print("using method: ", method)
        print("------")
        for i in range(min(args.iteration, amount_of_features)):
            local_train, local_validation = train.copy(), validation.copy()
            local_train, local_validation = feature_selection(
                train, validation, i + 1, method
            )
            formula, accuracy = predict(local_train, local_validation, median_values)
            print(f"Accuracy for length {i + 1}: {accuracy}\n")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_formula = formula
                best_method = method
                best_n_features = i + 1

    # Model evaluation

    print("------")
    print("Evaluating model on test set")
    print("------")
    print("Using method: ", best_method)
    print("Best formula: ", best_formula.replace("&", "AND"))

    train_validation = pd.concat([train, validation], ignore_index=True)

    local_train_validation, local_test = train_validation.copy(), test.copy()
    local_train_validation, local_test = feature_selection(
        train_validation, test, best_n_features, best_method
    )
    formula, accuracy = predict(local_train_validation, local_test, median_values)
    print(f"Test accuracy: {accuracy}\n")

    # Compare against baseline and random forest

    print("\n------\n")
    bot_accuracy = accuracy_score(test.iloc[:, -1], [0] * len(test))
    top_accuracy = accuracy_score(test.iloc[:, -1], [1] * len(test))
    print(
        "Baseline test accuracy (true/false for every input):",
        max(bot_accuracy, top_accuracy),
    )
    print(
        "Best formula: ",
        best_formula.replace("&", "AND")
        .replace("|", "OR")
        .replace("~", "NOT ")
        .replace("_[M*]", ""),
    )

    rf = RandomForestClassifier()
    rf.fit(train_validation.iloc[:, :-1], train_validation.iloc[:, -1])
    predictions = rf.predict(test.iloc[:, :-1])
    rf_accuracy = accuracy_score(test.iloc[:, -1], predictions)

    print("Random Forest test accuracy: ", rf_accuracy)

    return best_accuracy, max(bot_accuracy, top_accuracy), rf_accuracy


def main():

    parser = argparse.ArgumentParser(description="Load binary matrix")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        action="store",
        help="Base path for the data folder.",
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

    train_train, train_validation, holdout = load_binary_matrix(args.path)
    predict_dataset(train_train, train_validation, holdout, "huimausdata", args)


if __name__ == "__main__":
    main()
