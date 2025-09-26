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

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python help.py input.mat [output.csv]")
    elif len(sys.argv) == 2:
        to_csv(sys.argv[1])
    else:
        to_csv(sys.argv[1], sys.argv[2])
