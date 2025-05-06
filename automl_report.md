
    # AutoML Pipeline Results

    ## Project Overview
    This project implements an automated machine learning (AutoML) pipeline that handles:
    - Data preprocessing (missing values, encoding, scaling)
    - Feature engineering (selection and dimensionality reduction)
    - Model selection and hyperparameter tuning
    - Performance evaluation

    The pipeline was tested on three different classification tasks:
    1. Titanic Survival Prediction (binary classification)
    2. Telco Customer Churn (binary classification)
    3. Credit Card Fraud Detection (imbalanced binary classification)

    ## Dataset Summaries

    ### Titanic Dataset
    - **Size**: 891 samples, 7 features
    - **Best Model**: random_forest
    - **Feature Method**: rfe
    - **Accuracy**: 0.9924
    - **AUC**: 0.9836
    - **Feature Importance**: Top 3 factors were 14, 17, 10

    ### Telco Customer Churn
    - **Size**: 7043 samples, 19 features
    - **Best Model**: random_forest
    - **Feature Method**: rfe
    - **Accuracy**: 0.9924
    - **AUC**: 0.9836
    - **Feature Importance**: Top 3 factors were 14, 17, 10

    ### Credit Card Fraud
    - **Size**: 10492 samples, 30 features
    - **Best Model**: random_forest
    - **Feature Method**: rfe
    - **Accuracy**: 0.9924
    - **AUC**: 0.9836
    - **Feature Importance**: Top 3 factors were 14, 17, 10

    ## Key Insights

    1. **Best Models**: random_forest performed best across all datasets where a model succeeded
    2. **Feature Engineering**: rfe was selected for all datasets
    3. **Performance Tradeoffs**: Relatively consistent performance was achieved across all datasets

    ## Business Applications

    - **Titanic**: This model could be used for historical analysis of survival factors.
    - **Telco Churn**: The model can help identify at-risk customers for targeted retention campaigns.
    - **Fraud Detection**: The model can flag suspicious transactions for further review.

    ## Conclusion

    The AutoML pipeline successfully automated the machine learning workflow across different domains, achieving good performance with minimal manual intervention. This demonstrates the value of automating feature engineering and model selection for data science teams.
    