import pandas as pd
import argparse
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import chi2
import numpy as np
from datetime import datetime
from utils import SEED, classification_metrics
from other_classifiers import other_classifiers

# from optimal_rule_list import OptimalRuleList
# from ideal_dnf import IdealDNF

# Disease class labels and their acronyms
DISEASE_ACRONYMS = {
    "Acoustic_neurinoma": "ANE",
    "Benign_positional_vertigo": "BPV",
    "Menieres_disease_vertigo": "MEN",
    "Sudden_deafness": "SUD",
    "Traumatic_vertigo": "TRA",
    "Vestibular_neuritis": "VNE",
    "Benign_recurrent_vertigo": "BRV",
    "Vestibulopatia": "VES",
    "Central_lesion": "CL",
}

# Ordered list matching manuscript Table 3/6 columns
DISEASE_ORDER = [
    "Acoustic_neurinoma",
    "Benign_positional_vertigo",
    "Menieres_disease_vertigo",
    "Sudden_deafness",
    "Traumatic_vertigo",
    "Vestibular_neuritis",
    "Benign_recurrent_vertigo",
    "Vestibulopatia",
    "Central_lesion",
]


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


def get_and_write_results(
    boolean_data, numeric_data, args, results_filename, class_labels=None
):
    """
    Run cross-validation and report results matching manuscript Tables 3/6.

    Results format: per-class sensitivities, accuracy, F1_macro
    """
    all_results = []

    # Get class labels from data if not provided
    if class_labels is None:
        target_col = numeric_data.columns[-1]
        if target_col.startswith("CV"):
            # Find the actual target column (last non-CV column)
            non_cv_cols = [c for c in numeric_data.columns if not c.startswith("CV")]
            target_col = non_cv_cols[-1]
        class_labels = np.array(sorted(numeric_data[target_col].unique()))

    for iteration in range(1, args.iterations + 1):
        iter_results = []
        for fold in range(1, 11):
            print(f"\nIteration {iteration}, Fold {fold}")
            # Get indices for train and test sets based on CV column
            test_index = numeric_data[numeric_data[f"CV{iteration}"] == fold].index
            train_index = numeric_data[numeric_data[f"CV{iteration}"] != fold].index

            # drop cv columns
            numeric_data_cleaned = numeric_data.drop(
                columns=[col for col in numeric_data.columns if col.startswith("CV")]
            )
            boolean_data_cleaned = boolean_data.drop(
                columns=[col for col in boolean_data.columns if col.startswith("CV")]
            )

            full_scaled_numeric_train, full_scaled_numeric_test = standardize_data(
                numeric_data_cleaned.iloc[train_index],
                numeric_data_cleaned.iloc[test_index],
            )
            other_results = other_classifiers(
                full_scaled_numeric_train,
                full_scaled_numeric_test,
                args.optimize,
                class_labels=class_labels,
            )

            iter_results.append(other_results)
            all_results.append(other_results)

        print("\n---10CV---")
        _print_aggregated_results(iter_results, class_labels)

    print("\n---ALL ITERATIONS---")
    _print_and_write_final_results(all_results, class_labels, results_filename)


def _print_aggregated_results(results_list, class_labels):
    """Print aggregated results for a set of folds/iterations."""
    method_results = _aggregate_results(results_list, class_labels)

    for method, metrics in method_results.items():
        acc = np.mean(metrics["accuracy"])
        f1 = np.mean(metrics["f1"])
        sens_str = " | ".join(
            f"{DISEASE_ACRONYMS.get(label, label[:3])}: {np.mean(metrics['sens'][label])*100:.1f}"
            for label in class_labels
            if label in metrics["sens"]
        )
        print(f"{method}: ACC={acc*100:.1f}% F1={f1*100:.1f}% | {sens_str}")


def _print_and_write_final_results(all_results, class_labels, results_filename):
    """Print and write final aggregated results."""
    method_results = _aggregate_results(all_results, class_labels)

    # Build header matching manuscript format
    header_parts = ["Method"]
    for label in class_labels:
        header_parts.append(DISEASE_ACRONYMS.get(label, label[:3]))
    header_parts.extend(["ACC", "F1_macro"])
    header = "\t".join(header_parts)

    print(header)
    with open(results_filename, "a") as results_file:
        results_file.write("\n" + header + "\n")

        for method, metrics in method_results.items():
            acc = np.mean(metrics["accuracy"]) * 100
            f1 = np.mean(metrics["f1"]) * 100

            row_parts = [method]
            for label in class_labels:
                if label in metrics["sens"]:
                    sens = np.mean(metrics["sens"][label]) * 100
                    row_parts.append(f"{sens:.1f}")
                else:
                    row_parts.append("N/A")
            row_parts.extend([f"{acc:.1f}", f"{f1:.1f}"])

            row = "\t".join(row_parts)
            print(row)
            results_file.write(row + "\n")


def _aggregate_results(results_list, class_labels):
    """
    Aggregate results from multiple folds/iterations.

    Returns dict: {method: {"accuracy": [...], "f1": [...], "sens": {label: [...]}}}
    """
    method_results = {}

    for fold_results in results_list:
        for method, metrics in fold_results.items():
            accuracy, f1, sens_dict = metrics

            if method not in method_results:
                method_results[method] = {
                    "accuracy": [],
                    "f1": [],
                    "sens": {label: [] for label in class_labels},
                }

            method_results[method]["accuracy"].append(accuracy)
            method_results[method]["f1"].append(f1)

            for label in class_labels:
                if label in sens_dict:
                    method_results[method]["sens"][label].append(sens_dict[label])

    return method_results


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

        # Use ordered class labels
        class_labels = np.array(DISEASE_ORDER)
        get_and_write_results(
            full_boolean_data,
            full_numeric_data,
            args,
            "results.txt",
            class_labels=class_labels,
        )

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
