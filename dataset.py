import kagglehub
import os

# Define the dataset and the specific file
dataset_slug = "mlg-ulb/creditcardfraud"
file_name = "creditcard.csv"

# Download the dataset (kagglehub downloads the whole archive)
# and get the path to the specific file within the cache.
try:
    file_path = kagglehub.dataset_download(dataset_slug, path=file_name)

    print(f"Path to {file_name}: {file_path}")

    # Optional: Verify the file exists
    if os.path.exists(file_path):
        print(f"Successfully located {file_name}.")
    else:
        print(f"Error: {file_name} not found at the expected path.")

except Exception as e:
    print(f"An error occurred: {e}")
    print(
        "Please ensure you have the Kaggle API token configured"
        " (e.g., in ~/.kaggle/kaggle.json or environment variables)"
        " and the dataset/file name is correct."
    )

