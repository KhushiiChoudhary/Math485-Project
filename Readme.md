# Automated Feature Engineering & Model Selection for Tabular Data

## Description

This project implements a basic AutoML pipeline in Python designed to automate parts of the machine learning workflow for tabular datasets. It focuses on:

*   Automated feature selection/dimensionality reduction using techniques like RFE and PCA.
*   Automated training and evaluation of different machine learning models (Decision Trees, SVM, Neural Networks).
*   Comparing the performance of different preprocessing/model combinations across various datasets.

The goal is to streamline the process of finding effective preprocessing strategies and models for structured data.

## Features

*   Loads CSV datasets.
*   Performs basic preprocessing (imputation, scaling, encoding).
*   Applies optional feature selection (RFE) or dimensionality reduction (PCA).
*   Trains and evaluates Decision Tree, SVM, and a simple Neural Network.
*   Compares results across different preprocessing steps and models.
*   Outputs results to a CSV file.

## Technologies Used

*   Python 3.x
*   Pandas
*   NumPy
*   Scikit-learn
*   TensorFlow (Keras)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Create a Python environment (recommended):**
    *   Using Conda:
        ```bash
        conda create -n automl_env python=3.9 # Or another version
        conda activate automl_env
        conda install pandas numpy scikit-learn tensorflow matplotlib seaborn -c conda-forge
        ```
    *   Using pip and venv:
        ```bash
        python -m venv venv
        source venv/bin/activate # On Windows use `venv\Scripts\activate`
        pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
        ```

## Usage

1.  **Configure Dataset Paths:**
    *   Open the main script (`automl_pipeline.py` or your chosen filename).
    *   Locate the `CONFIG` dictionary near the top.
    *   **IMPORTANT:** Update the `"path"` values within `CONFIG["datasets"]` to point to the actual locations of your dataset CSV files (e.g., Titanic, Credit Card Fraud).

2.  **Run the Pipeline:**
    ```bash
    python automl_pipeline.py
    ```

3.  **Check Results:**
    *   The script will print progress to the console.
    *   A summary of the results will be saved to `automl_pipeline_results.csv` (or the filename specified in `CONFIG["results_file"]`).

## Team Members

*   Khushi Choudhary
*   Snehitha Gorantla
*   Zakir Elaskar

