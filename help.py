import pandas as pd
import os
import sys

import scipy.io


def to_csv(mat_file_path, output_csv=None):
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    print(mat_data["diseases"])

    # Extract headers, diseases, and data
    headers = [h[0] for h in mat_data["cvcolheaders"][0]]
    diseases = [d[0][0] for d in mat_data["diseases"]]
    data = mat_data["cvdata_imputed"]

    # Create a DataFrame with the data
    df = pd.DataFrame(data, columns=headers)

    # "1.0" -> "1", "0.0" -> "0"
    for col in df.select_dtypes(include=["float64"]).columns:
        if all(df[col].dropna().apply(lambda x: x.is_integer())):
            df[col] = df[col].astype("Int64")

    # replace the FINAL_DIAG column with the actual disease names
    df["FINAL_DIAG"] = df["FINAL_DIAG"].apply(
        lambda x: diseases[int(x - 1)] if 0 < x <= len(diseases) else "Unknown"
    )

    # move final_diag to the end
    final_diag = df.pop("FINAL_DIAG")
    df["FINAL_DIAG"] = final_diag

    # drop ID column
    df = df.drop(columns=["ID"])

    # Print summary
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"Features: {', '.join(headers[:5])}... (total: {len(headers)})")
    print(f"Diseases: {', '.join(diseases[:5])}... (total: {len(diseases)})")

    # Save to CSV if output path is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")
    else:
        # Default output name based on input file
        output_name = os.path.splitext(os.path.basename(mat_file_path))[0] + ".csv"
        df.to_csv(output_name, index=False)
        print(f"Data saved to {output_name}")

    # Calculate and save median values for feature columns
    median_values = df.drop(columns=["FINAL_DIAG"]).median()

    with open("huimausdata_median_values.txt", "w") as f:
        for column, value in median_values.items():
            f.write(f"{column}: {value}\n")

    print("Median values saved to huimausdata_median_values.txt")

    # Create folders for disease-specific datasets
    disease_folder = "datasets"
    boolean_folder = "boolean_datasets"
    if not os.path.exists(disease_folder):
        os.makedirs(disease_folder)
        print(f"Created folder {disease_folder}")
    if not os.path.exists(boolean_folder):
        os.makedirs(boolean_folder)
        print(f"Created folder {boolean_folder}")

    # Get unique diseases
    unique_diseases = df["FINAL_DIAG"].unique()

    # For each disease, create a binary dataset
    for disease in unique_diseases:
        df_disease = df.copy()

        # Create binary FINAL_DIAG column (1 for current disease, 0 for others)
        df_disease["FINAL_DIAG"] = (df_disease["FINAL_DIAG"] == disease).astype(int)

        df_boolean = df_disease.copy()

        # Loop through all columns except FINAL_DIAG
        for column in df_boolean.drop(columns=["FINAL_DIAG"]):
            # Check if column is non-categorical (more than 2 unique values)
            if df_boolean[column].nunique() > 2:
                # Booleanize values based on median
                median = median_values[column]
                df_boolean[column] = (df_boolean[column] > median).astype(int)

        # Save the boolean version
        safe_disease_name = disease.replace(" ", "_").replace("/", "_")
        disease_output = os.path.join(disease_folder, f"{safe_disease_name}.csv")
        df_disease.to_csv(disease_output, index=False)
        boolean_output = os.path.join(
            boolean_folder, f"boolean_{safe_disease_name}.csv"
        )
        df_boolean.to_csv(boolean_output, index=False)

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python help.py input.mat [output.csv]")
    elif len(sys.argv) == 2:
        to_csv(sys.argv[1])
    else:
        to_csv(sys.argv[1], sys.argv[2])
