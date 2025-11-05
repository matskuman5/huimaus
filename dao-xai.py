import pandas as pd
import argparse
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import chi2
import numpy as np
from datetime import datetime
from utils import SEED, classification_metrics
from other_classifiers import other_classifiers
from optimal_rule_list import OptimalRuleList
from ideal_dnf import IdealDNF


def standardize_data(train: pd.DataFrame, test: pd.DataFrame):
    """
    Standardizes data by scaling it. Has no effect on tree-based models;
    this is simply to ensure that the pipeline is the same for all models.
    """
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames and combine with target
    X_train_scaled = pd.DataFrame(
        X_train_scaled, index=X_train.index, columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, index=X_test.index, columns=X_test.columns
    )

    scaled_train = pd.concat([X_train_scaled, y_train], axis=1)
    scaled_test = pd.concat([X_test_scaled, y_test], axis=1)

    return scaled_train, scaled_test


def get_and_write_results(boolean_data, numeric_data, args, results_filename):
    all_results = []
    for iteration in range(1, args.iterations + 1):
        iter_results = []
        for fold in range(1, 10):
            print(f"\nIteration {iteration}, Fold {fold}")
            # Get indices for train and test sets based on CV column
            test_index = numeric_data[numeric_data[f"cv{iteration}"] == fold].index
            train_index = numeric_data[numeric_data[f"cv{iteration}"] != fold].index

            # drop cv columns
            numeric_data_cleaned = numeric_data.drop(
                columns=[col for col in numeric_data.columns if col.startswith("cv")]
            )
            boolean_data_cleaned = boolean_data.drop(
                columns=[col for col in boolean_data.columns if col.startswith("cv")]
            )

            orl = OptimalRuleList(
                max_literals=args.max_literals,
                timeout=5,
            )
            icm = IdealDNF(
                k=args.max_features,
                method=chi2,
            )
            orl.fit(
                boolean_data_cleaned.iloc[train_index].iloc[:, :-1],
                boolean_data_cleaned.iloc[train_index].iloc[:, -1],
            )
            icm.fit(
                boolean_data_cleaned.iloc[train_index].iloc[:, :-1],
                boolean_data_cleaned.iloc[train_index].iloc[:, -1],
            )
            orl_predictions = orl.predict(
                boolean_data_cleaned.iloc[test_index].iloc[:, :-1]
            )
            icm_predictions = icm.predict(
                boolean_data_cleaned.iloc[test_index].iloc[:, :-1]
            )

            fucking_shit = {}
            fucking_shit["orl"] = classification_metrics(
                boolean_data_cleaned.iloc[test_index].iloc[:, -1], orl_predictions
            )
            fucking_shit["icm"] = classification_metrics(
                boolean_data_cleaned.iloc[test_index].iloc[:, -1], icm_predictions
            )

            print(
                f"\nORL Formula: {orl.get_formula()}\n accuracy: {fucking_shit['orl'][0]:.3f}, F1: {fucking_shit['orl'][1]:.3f}, sensitivity: {fucking_shit['orl'][2]:.3f}"
            )
            print(
                f"\nICM Formula: {icm.get_formula()}\n accuracy: {fucking_shit['icm'][0]:.3f}, F1: {fucking_shit['icm'][1]:.3f}, sensitivity: {fucking_shit['icm'][2]:.3f}"
            )

            if fold == 1:
                orl_formula = orl.get_formula()
                icm_formula = icm.get_formula()

            full_scaled_numeric_train, full_scaled_numeric_test = standardize_data(
                numeric_data_cleaned.iloc[train_index],
                numeric_data_cleaned.iloc[test_index],
            )
            other_results = other_classifiers(
                full_scaled_numeric_train,
                full_scaled_numeric_test,
                args.optimize,
            )

            iter_results.append(other_results)
            iter_results.append(fucking_shit)
            all_results.append(other_results)
            all_results.append(fucking_shit)

        print("\n---10CV---")
        # Aggregate results across all folds for this iteration
        method_results = {}
        for fold_results in iter_results:
            for method, metrics in fold_results.items():
                if method not in method_results:
                    method_results[method] = {
                        "accuracy": [],
                        "f1": [],
                        "sensitivity": [],
                    }
                method_results[method]["accuracy"].append(metrics[0])
                method_results[method]["f1"].append(metrics[1])
                method_results[method]["sensitivity"].append(metrics[2])

        # Print averages for this iteration
        for method in method_results.keys():
            accuracy = np.mean(method_results[method]["accuracy"])
            f1 = np.mean(method_results[method]["f1"])
            sensitivity = np.mean(method_results[method]["sensitivity"])
            print(
                f"{method}: {accuracy:.3f} "
                + f"(F1: {f1:.3f}, "
                + f"sens: {sensitivity:.3f})"
            )

    print("\n---ALL ITERATIONS---")
    with open(results_filename, "a") as results_file:
        # Aggregate results across all iterations
        overall_method_results = {}
        for iter_results in all_results:
            for method, metrics in iter_results.items():
                if method not in overall_method_results:
                    overall_method_results[method] = {
                        "accuracy": [],
                        "f1": [],
                        "sensitivity": [],
                    }
                overall_method_results[method]["accuracy"].append(metrics[0])
                overall_method_results[method]["f1"].append(metrics[1])
                overall_method_results[method]["sensitivity"].append(metrics[2])

        # Print averages across all iterations
        for method in overall_method_results.keys():
            accuracy = np.mean(overall_method_results[method]["accuracy"])
            f1 = np.mean(overall_method_results[method]["f1"])
            sensitivity = np.mean(overall_method_results[method]["sensitivity"])
            result_line = (
                f"{method}: {accuracy:.3f} "
                + f"(F1: {f1:.3f}, "
                + f"sens: {sensitivity:.3f})\n"
            )
            print(result_line)
            results_file.write(result_line)

        results_file.write("\nORL Formula:\n")
        results_file.write(orl_formula)

        results_file.write("\n\nICM Formula:\n")
        results_file.write(icm_formula)


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
        help="Maximum features to select for ICM.",
        default=5,
    )
    parser.add_argument(
        "-l",
        "--max-literals",
        type=int,
        action="store",
        help="Maximum number of literals for ORL.",
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
    parser.add_argument(
        "-o",
        "--optimize",
        action="store_true",
        help="Optimize hyperparameters for Random Forest and XGBoost classifiers with Optuna.",
        default=False,
    )
    args = parser.parse_args()

    # Load data
    if args.path.endswith(".txt"):
        with open(args.path, "r") as f:
            datasets = [line.strip() for line in f if line.strip()]
    else:
        datasets = [args.path]

    # Record current time
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")

    if args.multiclass:
        with open("results.txt", "w", encoding="utf-8") as results_file:
            results_file.write(
                f"{current_time}\nMulticlass: {args.multiclass}\nMax features: {args.max_features}\nIterations: {args.iterations}\n"
            )

        full_boolean_data = pd.read_csv("huimaus_clean.csv")
        full_numeric_data = pd.read_csv("huimausdata.csv")
        get_and_write_results(full_boolean_data, full_numeric_data, args, "results.txt")

    else:
        for disease in datasets:
            print(f"\n=== Dataset: {disease} ===")
            # Load disease-specific datasets
            boolean_data = pd.read_csv(f"boolean_datasets/boolean_{disease}.csv")
            numeric_data = pd.read_csv(f"datasets/{disease}.csv")

            results_filename = f"results_{disease}.txt"

            with open(results_filename, "w", encoding="utf-8") as results_file:
                results_file.write(
                    f"{current_time}\nMulticlass: {args.multiclass}\nMax features: {args.max_features}\nIterations: {args.iterations}\n"
                )

            get_and_write_results(boolean_data, numeric_data, args, results_filename)


if __name__ == "__main__":
    main()
