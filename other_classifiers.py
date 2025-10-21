import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from utils import SEED, classification_metrics


# Return classification metrics for the following classifiers:
# 1. Dummy classifier that always predicts the most frequent class
# 2. Random Forest with hyperparameter tuning
# 3. XGBoost with hyperparameter tuning
def other_classifiers(numeric_train, numeric_test, optimize):
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

    # Split training data for hyperparameter optimization
    X_train, X_val, y_train, y_val = train_test_split(
        numeric_train.iloc[:, :-1],
        numeric_train.iloc[:, -1],
        test_size=0.3,
        random_state=SEED,
    )

    # Optimize Random Forest hyperparameters
    def objective_rf(trial):
        max_depth_options = [None, 2, 3, 4]
        max_features_options = [
            "sqrt",
            "log2",
            None,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ]

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 3000, log=True),
            "max_depth": trial.suggest_categorical("max_depth", max_depth_options),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_features": trial.suggest_categorical(
                "max_features", max_features_options
            ),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 3]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "min_impurity_decrease": trial.suggest_categorical(
                "min_impurity_decrease", [0.0, 0.01, 0.02, 0.05]
            ),
            "random_state": SEED,
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)
        return accuracy

    # Optimize XGBoost hyperparameters
    def objective_xgb(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 11),
            "n_estimators": trial.suggest_int("n_estimators", 100, 5900, step=200),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 100.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 0.7, log=True),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 7.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 4.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "random_state": SEED,
        }

        # XGBoost requires integer labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_val_encoded = le.transform(y_val)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train_encoded)
        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val_encoded, preds)
        return accuracy

    if optimize:

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run hyperparameter optimization for Random Forest
        print("Optimizing Random Forest hyperparameters...")
        study_rf = optuna.create_study(direction="maximize")
        study_rf.optimize(
            objective_rf, n_trials=100, show_progress_bar=True
        )  # Reduced trials for faster execution
        best_rf_params = study_rf.best_params
        print(f"Best Random Forest parameters: {best_rf_params}")

        # Train Random Forest with best parameters on full training data
        rf = RandomForestClassifier(**best_rf_params, random_state=SEED)
        rf.fit(numeric_train.iloc[:, :-1], numeric_train.iloc[:, -1])
        rf_predictions = rf.predict(numeric_test.iloc[:, :-1])
        rf_accuracy, rf_f1, rf_sens = classification_metrics(
            numeric_test.iloc[:, -1], rf_predictions
        )
        results["rf"] = [rf_accuracy, rf_f1, rf_sens]

        # Run hyperparameter optimization for XGBoost
        print("Optimizing XGBoost hyperparameters...")
        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(
            objective_xgb, n_trials=100, show_progress_bar=True
        )  # Reduced trials for faster execution
        best_xgb_params = study_xgb.best_params
        print(f"Best XGBoost parameters: {best_xgb_params}")

        # XGBoost requires integer labels for some reason
        le = LabelEncoder()
        int_y_train = le.fit_transform(numeric_train.iloc[:, -1])
        int_y_test = le.transform(numeric_test.iloc[:, -1])

        # Train XGBoost with best parameters on full training data
        bst = XGBClassifier(**best_xgb_params, random_state=SEED)
        bst.fit(numeric_train.iloc[:, :-1], int_y_train)
        bst_predictions = bst.predict(numeric_test.iloc[:, :-1])
        bst_accuracy, bst_f1, bst_sens = classification_metrics(
            int_y_test, bst_predictions
        )
        results["xgboost"] = [bst_accuracy, bst_f1, bst_sens]
    else:
        # Train Random Forest with default parameters
        rf = RandomForestClassifier(random_state=SEED)
        rf.fit(numeric_train.iloc[:, :-1], numeric_train.iloc[:, -1])
        rf_predictions = rf.predict(numeric_test.iloc[:, :-1])
        rf_accuracy, rf_f1, rf_sens = classification_metrics(
            numeric_test.iloc[:, -1], rf_predictions
        )
        results["rf"] = [rf_accuracy, rf_f1, rf_sens]

        # Train XGBoost with default parameters
        le = LabelEncoder()
        int_y_train = le.fit_transform(numeric_train.iloc[:, -1])
        int_y_test = le.transform(numeric_test.iloc[:, -1])

        bst = XGBClassifier(random_state=SEED)
        bst.fit(numeric_train.iloc[:, :-1], int_y_train)
        bst_predictions = bst.predict(numeric_test.iloc[:, :-1])
        bst_accuracy, bst_f1, bst_sens = classification_metrics(
            int_y_test, bst_predictions
        )
        results["xgboost"] = [bst_accuracy, bst_f1, bst_sens]

    print(
        f"Random Forest test metrics - accuracy: {rf_accuracy:.3f}, F1: {rf_f1:.3f}, sensitivity: {rf_sens:.3f}"
    )
    print(
        f"XGBoost test metrics - accuracy: {bst_accuracy:.3f}, F1: {bst_f1:.3f}, sensitivity: {bst_sens:.3f}"
    )

    return results
