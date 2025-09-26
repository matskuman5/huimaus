import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


def analyze_data(csv_path, test_size=0.3, random_state=1):
    # Load the CSV file
    data = pd.read_csv(csv_path)

    # Separate features and target
    X = data.iloc[:, :-1]  # All columns except the last
    y = data.iloc[:, -1]  # Only the last column

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(random_state=random_state)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Feature importance
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": rf_model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # Create and plot confusion matrix
    import matplotlib.pyplot as plt

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y),
        yticklabels=np.unique(y),
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Replace 'your_data.csv' with your actual CSV file path
    csv_file = "huimausdata.csv"
    model, importance = analyze_data(csv_file)
