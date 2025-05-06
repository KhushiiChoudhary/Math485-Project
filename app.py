# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time
import io
import sys
from contextlib import redirect_stdout
import os
import traceback

# --- AutoML Pipeline Class Definitions ---
# (Keep your class definitions here as before)
class DataPreprocessor:
    """Handles data preprocessing."""
    def __init__(self, numerical_features=None, categorical_features=None):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.preprocessor = None
        self.feature_names_out = None # To store output feature names

    def fit_transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
             raise ValueError("Input X must be a pandas DataFrame")
        if self.numerical_features is None or self.categorical_features is None:
            self._infer_feature_types(X)

        # Define transformers
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Use sparse_output=False
        ])

        # Create ColumnTransformer
        transformers_list = []
        if self.numerical_features:
            transformers_list.append(('num', numerical_transformer, self.numerical_features))
        if self.categorical_features:
             transformers_list.append(('cat', categorical_transformer, self.categorical_features))

        if not transformers_list:
             st.warning("No numerical or categorical features identified for preprocessing.")
             return X.to_numpy() if isinstance(X, pd.DataFrame) else X # Return as numpy array

        self.preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough')

        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X)

        # Get output feature names
        try:
            self.feature_names_out = self.preprocessor.get_feature_names_out()
        except Exception as e:
            st.warning(f"Could not get feature names from preprocessor: {e}")
            self.feature_names_out = None

        return X_transformed # Output is numpy array

    def transform(self, X):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        if not isinstance(X, pd.DataFrame):
             raise ValueError("Input X must be a pandas DataFrame")

        X_transformed = self.preprocessor.transform(X)
        return X_transformed # Output is numpy array

    def _infer_feature_types(self, X):
        self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        st.write(f"Inferred Numerical Features: {self.numerical_features}")
        st.write(f"Inferred Categorical Features: {self.categorical_features}")

class FeatureEngineer:
    """Performs feature selection/dimensionality reduction."""
    def __init__(self, method='auto', n_features=20): # Default n_features
        self.method = method
        self.n_features_to_select = n_features # Store requested number
        self.n_features_final = None # Actual number after fitting
        self.model = None
        self.selected_indices_ = None # Store selected indices for RFE
        self.input_feature_names_ = None # Store names before transformation

    def fit_transform(self, X, y=None, feature_names=None):
        self.input_feature_names_ = feature_names # Store input names if provided

        # Auto-select method
        current_method = self.method
        if current_method == 'auto':
            n_samples, n_features = X.shape
            if n_samples < 2000 and n_features < 50 and n_features > 1: # RFE needs >1 feature
                current_method = 'rfe'
            elif n_features > 50:
                current_method = 'pca'
            elif n_features > 1: # RFE needs >1 feature
                 current_method = 'rfe'
            else:
                 current_method = 'none' # Skip if only 1 feature
            st.write(f"Auto-selected feature engineering method: {current_method}")
        else:
             # Ensure method is valid if only 1 feature
             if X.shape[1] <= 1 and current_method != 'none':
                  st.warning(f"Only {X.shape[1]} feature(s) available. Skipping feature engineering method '{current_method}'.")
                  current_method = 'none'


        self.method = current_method # Store the decided method

        # Apply method
        if self.method == 'rfe':
            return self._apply_rfe(X, y)
        elif self.method == 'pca':
            return self._apply_pca(X)
        elif self.method == 'none':
             self.n_features_final = X.shape[1]
             st.write("Skipping feature engineering.")
             return X # Pass through if no method applied
        else:
             st.error(f"Unknown feature engineering method: {self.method}")
             return X

    def transform(self, X):
        if self.method == 'rfe' and self.model:
            return self.model.transform(X)
        elif self.method == 'pca' and self.model:
            return self.model.transform(X)
        elif self.method == 'none':
             return X # Pass through
        else:
            # Raise error or return X if not fitted/unknown method
            raise ValueError(f"Feature engineer (method: {self.method}) not fitted or method unknown.")

    def _apply_rfe(self, X, y):
        if y is None:
            raise ValueError("RFE requires target values (y)")
        if X.shape[1] <= 1:
             st.warning("RFE requires more than 1 feature. Skipping.")
             self.method = 'none'
             self.n_features_final = X.shape[1]
             return X

        # Adjust n_features_to_select if it's > current number of features
        n_features_actual = min(self.n_features_to_select, X.shape[1])
        if n_features_actual < 1: n_features_actual = 1 # Need at least 1

        st.write(f"Applying RFE to select {n_features_actual} features...")
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) # Reduced estimators for speed
        selector = RFE(estimator, n_features_to_select=n_features_actual, step=0.1)
        self.model = selector
        transformed = selector.fit_transform(X, y)
        self.n_features_final = transformed.shape[1]
        self.selected_indices_ = np.where(selector.support_)[0]
        st.write(f"Selected {self.n_features_final} features using RFE.")
        return transformed

    def _apply_pca(self, X):
        n_components_actual = min(self.n_features_to_select, X.shape[0], X.shape[1]) # Cannot exceed n_samples or n_features
        if n_components_actual < 1: n_components_actual = 1

        st.write(f"Applying PCA to reduce to {n_components_actual} components...")
        pca = PCA(n_components=n_components_actual, random_state=42)
        self.model = pca
        transformed = pca.fit_transform(X)
        self.n_features_final = transformed.shape[1]
        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        st.write(f"Reduced to {self.n_features_final} principal components capturing {explained_variance:.2f}% of variance.")
        return transformed

    def get_feature_names_out(self):
        """Returns the names of the features after transformation."""
        if self.method == 'rfe' and self.selected_indices_ is not None and self.input_feature_names_ is not None:
            try:
                return [self.input_feature_names_[i] for i in self.selected_indices_]
            except IndexError:
                 return [f"feature_{i}" for i in self.selected_indices_] # Fallback if names mismatch
        elif self.method == 'pca' and self.n_features_final is not None:
            return [f"PC_{i+1}" for i in range(self.n_features_final)]
        elif self.method == 'none' and self.input_feature_names_ is not None:
             return self.input_feature_names_ # Return original names
        else:
            # Fallback if names aren't available
            return [f"feature_{i}" for i in range(self.n_features_final)] if self.n_features_final else []


class ModelSelector:
    """Selects and tunes the best ML model."""
    def __init__(self, models=None, hyperparameter_tuning=True):
        self.models_to_run = models if models else {} # Models passed in
        self.hyperparameter_tuning = hyperparameter_tuning
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf # Initialize best score to handle errors
        self.results = {}
        # Define param grids internally
        self.param_grids = {
            'decision_tree': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]},
            'svm': {'C': [0.1, 1, 10], 'kernel': ['rbf']}, # Reduced SVM grid for speed
            'random_forest': {'n_estimators': [50, 100], 'max_depth': [None, 10]}, # Reduced RF grid
            'neural_network': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.001, 0.01]} # Reduced NN grid
        }

    def fit(self, X, y, cv=3): # Reduced CV folds for speed
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.results = {}

        if not self.models_to_run:
             st.warning("No models selected for evaluation.")
             return self.results

        st.write("--- Starting Cross-Validation for Model Selection ---")
        progress_text = "Running Cross-Validation... ({model_name})"
        progress_bar = st.progress(0)
        status_text = st.empty()
        model_count = len(self.models_to_run)

        for i, (name, model) in enumerate(self.models_to_run.items()):
            start_time = time.time()
            status_text.text(progress_text.format(model_name=name))

            try:
                # Use specified CV folds
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1) # Use n_jobs for parallelism
                avg_score = np.mean(scores)
                std_dev = np.std(scores)
                training_time = time.time() - start_time

                self.results[name] = {'mean_score': avg_score, 'std': std_dev, 'training_time': training_time}
                st.write(f"✓ {name}: CV Score = {avg_score:.4f} (± {std_dev:.4f}), Time = {training_time:.2f}s")

                if avg_score > self.best_score:
                    self.best_score = avg_score
                    self.best_model_name = name

            except Exception as e:
                training_time = time.time() - start_time
                st.warning(f"✗ Error during cross_val_score for {name}: {e}")
                self.results[name] = {'mean_score': 0, 'std': 0, 'training_time': training_time, 'error': str(e)}
                continue # Skip to next model

            progress_bar.progress((i + 1) / model_count)
        status_text.text("Cross-Validation Complete.")
        progress_bar.empty() # Remove progress bar

        # Hyperparameter tuning
        if self.best_model_name and self.hyperparameter_tuning:
            st.write(f"\n--- Tuning Hyperparameters for Best Model: {self.best_model_name} (CV Score: {self.best_score:.4f}) ---")
            base_model = self.models_to_run[self.best_model_name]
            param_grid = self.param_grids.get(self.best_model_name, {})

            if param_grid:
                try:
                    grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
                    with st.spinner(f"Running GridSearchCV for {self.best_model_name}..."):
                        grid_search.fit(X, y)

                    self.best_model = grid_search.best_estimator_
                    tuned_score = grid_search.best_score_
                    st.write(f"Best parameters found: {grid_search.best_params_}")

                    if tuned_score > self.best_score:
                        st.success(f"✓ Improved score after tuning: {tuned_score:.4f}")
                        self.best_score = tuned_score
                    else:
                        st.info(f"Tuned score: {tuned_score:.4f} (Did not improve CV score)")
                        # Keep the tuned one for consistency
                        # self.best_model = self.models_to_run[self.best_model_name]
                        # self.best_model.fit(X, y) # Refit base model if preferred

                except Exception as e:
                    st.error(f"!!! Error during GridSearchCV for {self.best_model_name}: {e}")
                    st.warning("Falling back to best model from initial CV without tuning.")
                    self.best_model = self.models_to_run[self.best_model_name]
                    self.best_model.fit(X, y) # Fit the untuned best model
            else:
                st.info(f"No parameter grid defined for {self.best_model_name}. Fitting base model.")
                self.best_model = self.models_to_run[self.best_model_name]
                self.best_model.fit(X, y) # Fit the untuned best model

        elif self.best_model_name:
             st.write(f"\n--- Training Final Model ({self.best_model_name}) on Full Data (No Tuning) ---")
             self.best_model = self.models_to_run[self.best_model_name]
             self.best_model.fit(X, y)
        else:
             st.error("No best model successfully identified from cross-validation. Cannot proceed.")
             self.best_model = None # Ensure best_model is None

        return self.results

    def predict(self, X):
        if self.best_model is None: raise Exception("Model not fitted or fitting failed.")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None: raise Exception("Model not fitted or fitting failed.")
        if not hasattr(self.best_model, 'predict_proba'):
             raise Exception(f"Model '{self.best_model_name}' does not support probability predictions.")
        return self.best_model.predict_proba(X)

    def get_best_model(self):
        return self.best_model_name, self.best_model, self.best_score


class AutoMLPipeline:
    """Complete AutoML pipeline."""
    def __init__(self, feature_method='auto', model_list=None, hyperparameter_tuning=True):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer(method=feature_method)
        self.model_selector = ModelSelector(models=model_list, hyperparameter_tuning=hyperparameter_tuning)
        self.data_info = {}
        self.input_feature_names = None # Store original feature names
        self.final_feature_names_ = None # Store names after FE

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
             raise ValueError("Input X must be a pandas DataFrame for fit")
        self.input_feature_names = X.columns.tolist() # Store original names

        # Record dataset info
        self.data_info = {'n_samples': X.shape[0], 'n_features': X.shape[1], 'n_classes': len(np.unique(y)),
                          'class_distribution': pd.Series(y).value_counts(normalize=True).to_dict()}
        st.write(f"Dataset Info: {self.data_info['n_samples']} samples, {self.data_info['n_features']} features.")

        # Step 1: Preprocess
        st.write("\n**Step 1: Preprocessing Data...**")
        X_processed = self.preprocessor.fit_transform(X, y)
        processed_feature_names = self.preprocessor.feature_names_out

        # Step 2: Feature Engineering
        st.write("\n**Step 2: Applying Feature Engineering...**")
        X_features = self.feature_engineer.fit_transform(X_processed, y, feature_names=processed_feature_names)
        self.final_feature_names_ = self.feature_engineer.get_feature_names_out()

        # Step 3: Model Selection & Training
        st.write("\n**Step 3: Selecting and Training Best Model...**")
        results = self.model_selector.fit(X_features, y) # Pass engineered features

        if self.model_selector.best_model_name:
            st.success(f"**Pipeline Fit Complete! Best Model: {self.model_selector.best_model_name} (Score: {self.model_selector.best_score:.4f})**")
        else:
             st.error("**Pipeline Fit Failed: No suitable model found.**")

        return results

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
             raise ValueError("Input X must be a pandas DataFrame for predict")
        # Ensure input columns match original training columns
        if list(X.columns) != self.input_feature_names:
             st.warning("Input columns for prediction do not match original training columns. Reordering/selecting.")
             try:
                 X = X[self.input_feature_names]
             except KeyError as e:
                  raise ValueError(f"Input data missing required columns: {e}. Expected: {self.input_feature_names}")

        X_processed = self.preprocessor.transform(X)
        X_features = self.feature_engineer.transform(X_processed)
        return self.model_selector.predict(X_features)

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
             raise ValueError("Input X must be a pandas DataFrame for predict_proba")
        if list(X.columns) != self.input_feature_names:
             st.warning("Input columns for prediction do not match original training columns. Reordering/selecting.")
             try:
                 X = X[self.input_feature_names]
             except KeyError as e:
                  raise ValueError(f"Input data missing required columns: {e}. Expected: {self.input_feature_names}")

        X_processed = self.preprocessor.transform(X)
        X_features = self.feature_engineer.transform(X_processed)
        return self.model_selector.predict_proba(X_features) # Call selector's predict_proba

    def evaluate(self, X_test, y_test):
        if self.model_selector.best_model is None:
             st.error("Cannot evaluate: No model was successfully trained.")
             return None
        try:
            y_pred = self.predict(X_test) # Use the pipeline's predict method

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            auc = 0
            if len(np.unique(y_test)) == 2 and hasattr(self.model_selector.best_model, "predict_proba"):
                try:
                    y_proba = self.predict_proba(X_test)[:,1] # Use pipeline's predict_proba
                    auc = roc_auc_score(y_test, y_proba)
                except Exception as proba_e:
                    st.warning(f"Could not calculate AUC: {proba_e}")
                    auc = 0

            return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1, 'auc': auc}

        except Exception as e:
            st.error(f"Evaluation error: {str(e)}")
            return None

    def get_feature_importance(self):
        """Get feature importance from the fitted pipeline."""
        final_model = self.model_selector.best_model
        final_model_name = self.model_selector.best_model_name
        final_features = self.final_feature_names_

        if final_model is None or final_features is None:
             return {"Info": "Model not fitted or final features not determined."}

        if hasattr(final_model, 'feature_importances_'):
            importances = final_model.feature_importances_
            if len(importances) == len(final_features):
                 return dict(sorted(zip(final_features, importances), key=lambda item: item[1], reverse=True))
            else:
                 return {"Info": f"Mismatch between feature names ({len(final_features)}) and importances ({len(importances)}) for {final_model_name}."}
        elif hasattr(final_model, 'coef_'):
            if final_model.coef_.shape[-1] == len(final_features):
                importances = np.sum(np.abs(final_model.coef_), axis=0) if final_model.coef_.ndim > 1 else np.abs(final_model.coef_[0])
                return dict(sorted(zip(final_features, importances), key=lambda item: item[1], reverse=True))
            else:
                 return {"Info": f"Mismatch between feature names ({len(final_features)}) and coefficients ({final_model.coef_.shape}) for {final_model_name}."}
        elif self.feature_engineer.method == 'rfe' and hasattr(self.feature_engineer.model, 'estimator_') and hasattr(self.feature_engineer.model.estimator_, 'feature_importances_'):
             rfe_importances = self.feature_engineer.model.estimator_.feature_importances_
             if len(rfe_importances) == len(final_features):
                  return dict(sorted(zip(final_features, rfe_importances), key=lambda item: item[1], reverse=True))
             else:
                  return {"Info": "Could not reliably map RFE estimator importances to final features."}
        else:
            return {"Info": f"Feature importance calculation not supported for model type: {final_model_name}"}


# --- Streamlit UI ---
st.set_page_config(
    page_title="AutoML Classification Tool",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main header style */
    .main-header {
        font-size: 2.8rem; /* Slightly larger */
        font-weight: bold;
        color: #0D47A1; /* Darker blue */
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px #bbdefb; /* Subtle shadow */
    }
    /* Section headers */
    h2 {
        color: #1565C0; /* Medium blue */
        border-bottom: 2px solid #1E88E5; /* Lighter blue underline */
        padding-bottom: 0.3rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    h3 {
        color: #1E88E5; /* Lighter blue */
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    /* Containers for results */
    .results-box {
        background-color: #E3F2FD; /* Light blue background */
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #BBDEFB; /* Lighter border */
        margin-bottom: 1.5rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metrics-container {
        background-color: #FFFFFF; /* White background */
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        color: black; /* Ensure text is black */
        text-align: center;
    }
    .metrics-container h3 {
         margin-top: 0.5rem;
         margin-bottom: 0.8rem;
         color: #0D47A1; /* Dark blue for model name */
    }
     .metrics-container p {
         margin-bottom: 0.5rem;
         font-size: 1.1rem;
         color: #333; /* Dark grey for details */
    }
    /* Center text utility */
    .centered {
        text-align: center;
    }
    /* Style Streamlit buttons */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #1E88E5;
        background-color: #1E88E5;
        color: white;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s ease;
        font-weight: bold; /* Make button text bold */
    }
    .stButton>button:hover {
        background-color: #1565C0;
        border-color: #1565C0;
    }
    /* Style file uploader */
    .stFileUploader label {
        font-weight: bold;
        color: #1565C0;
    }
    /* Adjust sidebar width */
    section[data-testid="stSidebar"] .st-emotion-cache-10oheav {
        width: 320px; /* Adjust width as needed */
    }
    /* Reduce spacing in sidebar */
     .st-emotion-cache-10oheav .stVerticalBlock {
        gap: 0.5rem; /* Adjust vertical gap */
    }
    /* Style expander header */
    .st-emotion-cache-91n1wr {
        background-color: #e3f2fd;
        border-radius: 5px;
    }


</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
# (Keep session state initialization as before)
if 'pipeline' not in st.session_state: st.session_state['pipeline'] = None
if 'X_test' not in st.session_state: st.session_state['X_test'] = None
if 'y_test' not in st.session_state: st.session_state['y_test'] = None
if 'X_train' not in st.session_state: st.session_state['X_train'] = None
if 'y_train' not in st.session_state: st.session_state['y_train'] = None
if 'target_col' not in st.session_state: st.session_state['target_col'] = None
if 'categorical_cols' not in st.session_state: st.session_state['categorical_cols'] = []
if 'numerical_cols' not in st.session_state: st.session_state['numerical_cols'] = []
if 'df_cols' not in st.session_state: st.session_state['df_cols'] = []
if 'target_mapping' not in st.session_state: st.session_state['target_mapping'] = None
if 'is_categorical_target' not in st.session_state: st.session_state['is_categorical_target'] = False


# --- Main App Structure (Using Tabs) ---
st.markdown("<h1 class='main-header'>🤖 Interactive AutoML Classification Tool ⚙️</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["❓ About AutoML", "🚀 Run Pipeline", "💡 Make Predictions"])

# --- Tab 1: About AutoML ---
with tab1:
    st.header("What does this AutoML Pipeline do?")
    st.markdown("""
    This tool automates several key steps in building a classification machine learning model.
    When you upload your dataset, the pipeline performs the following actions:
    """)

    # --- REMOVED Flowchart Image Section ---
    # Instead, use Streamlit columns for a text-based flow
    st.markdown("##### Pipeline Workflow:")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown("**1. Load & Prep**")
        st.markdown("- Upload CSV\n- Select Target\n- Split Data")
    with col2:
        st.markdown("**2. Preprocess**")
        st.markdown("- Impute Missing\n- Scale Numeric\n- Encode Categorical")
    with col3:
        st.markdown("**3. Feature Eng.**")
        st.markdown("- Auto/RFE/PCA\n- Select/Reduce Features")
    with col4:
        st.markdown("**4. Model Select**")
        st.markdown("- Cross-Validate Models\n- Find Best Base")
    with col5:
        st.markdown("**5. Tune (Opt.)**")
        st.markdown("- GridSearchCV\n- Optimize Best Model")
    with col6:
        st.markdown("**6. Evaluate**")
        st.markdown("- Train Final Model\n- Test Performance\n- Show Results")

    st.markdown("---") # Separator
    st.info("The goal is to quickly build a reasonably good baseline model with minimal manual effort.")


# --- Tab 2: Run Pipeline ---
with tab2:
    st.header("🚀 Run the AutoML Pipeline")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Pipeline Configuration")
        st.markdown("Configure the AutoML steps before running.")

        feature_method = st.selectbox(
            "Feature Engineering Method",
            ["auto", "rfe", "pca", "none"], index=0, key="feature_method_select",
            help="Select 'auto' to let the system choose, 'rfe'/'pca' for specific methods, or 'none' to skip."
        )
        do_hyperparameter_tuning = st.checkbox(
            "Perform Hyperparameter Tuning", value=True, key="hyperparam_tuning_check",
            help="Enable to find optimal model hyperparameters (slower but potentially more accurate)"
        )
        st.markdown("##### Models to Include")
        col1, col2 = st.columns(2)
        with col1:
            include_decision_tree = st.checkbox("Decision Tree", value=True, key="cb_dt")
            include_svm = st.checkbox("SVM", value=True, key="cb_svm")
        with col2:
            include_random_forest = st.checkbox("Random Forest", value=True, key="cb_rf")
            include_neural_network = st.checkbox("Neural Network", value=True, key="cb_nn")

        selected_models = {}
        if include_decision_tree: selected_models['decision_tree'] = DecisionTreeClassifier(random_state=42)
        if include_random_forest: selected_models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        if include_svm: selected_models['svm'] = SVC(probability=True, random_state=42)
        if include_neural_network: selected_models['neural_network'] = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)

    # --- Main Area for Upload and Run ---
    st.markdown("### 1. Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file (Target variable ideally first or last column)", type=['csv'], key="file_uploader")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df_cols'] = df.columns.tolist()

            st.markdown("### 2. Dataset Preview & Target Selection")
            st.dataframe(df.head())

            st.markdown("##### Dataset Information")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", df.isnull().sum().sum())

            potential_targets = [col for col in df.columns if df[col].nunique() < 10 or df[col].dtype in ['object', 'category', 'int64']]
            default_target_index = 0
            if potential_targets:
                 # Prefer last column if it's a potential target
                 if df.columns[-1] in potential_targets:
                      default_target_index = df.columns.tolist().index(df.columns[-1])
                 else:
                      default_target_index = df.columns.tolist().index(potential_targets[0])

            target_col = st.selectbox(
                "Select the target column for classification:",
                df.columns.tolist(), index=default_target_index, key="target_select"
            )
            st.session_state['target_col'] = target_col

            st.markdown("##### Target Variable Distribution")
            if target_col:
                target_counts = df[target_col].value_counts(normalize=True)
                fig_dist, ax_dist = plt.subplots()
                target_counts.plot(kind='bar', ax=ax_dist, color='#1E88E5')
                ax_dist.set_ylabel("Proportion")
                ax_dist.set_title(f"Distribution of '{target_col}'")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig_dist)

            st.markdown("### 3. Run AutoML Pipeline")
            if st.button("▶️ Run AutoML Pipeline", key="run_button"):
                if not selected_models:
                     st.error("Please select at least one model type in the sidebar.")
                else:
                    X = df.drop(target_col, axis=1)
                    y = df[target_col]

                    is_categorical_target = False
                    target_mapping = None
                    if not pd.api.types.is_numeric_dtype(y):
                        st.info("Target variable appears non-numeric. Applying label encoding.")
                        is_categorical_target = True
                        unique_values = y.astype('category').cat.categories
                        target_mapping = {value: i for i, value in enumerate(unique_values)}
                        y = y.map(target_mapping)
                        st.write(f"Target mapping applied: {target_mapping}")
                        st.session_state['target_mapping'] = target_mapping
                        st.session_state['is_categorical_target'] = True
                    else:
                         st.session_state['target_mapping'] = None
                         st.session_state['is_categorical_target'] = False


                    stratify_option = y if y.nunique() > 1 else None
                    if stratify_option is None and len(y) > 0: st.warning("Target variable has only one class. Cannot use stratification.")

                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=stratify_option
                        )
                        st.session_state['X_train'] = X_train
                        st.session_state['y_train'] = y_train
                        st.session_state['X_test'] = X_test
                        st.session_state['y_test'] = y_test
                        st.session_state['categorical_cols'] = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                        st.session_state['numerical_cols'] = X_train.select_dtypes(include=np.number).columns.tolist()

                    except ValueError as e:
                         st.error(f"Error during train-test split: {e}")
                         st.info("Trying without stratification.")
                         try:
                              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
                              st.session_state['X_train'] = X_train; st.session_state['y_train'] = y_train
                              st.session_state['X_test'] = X_test; st.session_state['y_test'] = y_test
                              st.session_state['categorical_cols'] = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                              st.session_state['numerical_cols'] = X_train.select_dtypes(include=np.number).columns.tolist()
                         except Exception as split_e:
                              st.error(f"Could not split data: {split_e}"); st.stop()

                    st.markdown("---"); st.markdown("### AutoML Pipeline Run Log & Results")
                    log_placeholder = st.expander("View Detailed Process Log", expanded=False)
                    results_placeholder = st.container() # Placeholder for results after log
                    log_output = io.StringIO()

                    with redirect_stdout(log_output): # Capture print statements as log
                        pipeline = AutoMLPipeline(
                            feature_method=feature_method,
                            model_list=selected_models,
                            hyperparameter_tuning=do_hyperparameter_tuning
                        )
                        try:
                            start_run_time = time.time()
                            pipeline.fit(X_train, y_train)
                            end_run_time = time.time()
                            st.session_state['pipeline'] = pipeline # Save fitted pipeline

                            # --- Display results in the main area ---
                            with results_placeholder:
                                st.success(f"Pipeline finished in {end_run_time - start_run_time:.2f} seconds.")
                                if pipeline.model_selector.best_model:
                                    st.markdown("#### Final Selected Model")
                                    st.markdown(f"""
                                    <div class='metrics-container'>
                                        <h3 class='centered'>{pipeline.model_selector.best_model_name.replace('_', ' ').title()}</h3>
                                        <p>Final Score (Accuracy): {pipeline.model_selector.best_score:.4f}</p>
                                        <p>Feature Engineering: {pipeline.feature_engineer.method}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    st.markdown("#### Evaluation on Test Set")
                                    performance = pipeline.evaluate(X_test, y_test)
                                    if performance:
                                        metrics_cols = st.columns(5)
                                        metrics_cols[0].metric("Accuracy", f"{performance.get('accuracy', 0):.4f}")
                                        metrics_cols[1].metric("Precision", f"{performance.get('precision', 0):.4f}")
                                        metrics_cols[2].metric("Recall", f"{performance.get('recall', 0):.4f}")
                                        metrics_cols[3].metric("F1 Score", f"{performance.get('f1_score', 0):.4f}")
                                        metrics_cols[4].metric("AUC", f"{performance.get('auc', 0):.4f}" if performance.get('auc', 0) > 0 else "N/A")

                                        # Confusion Matrix
                                        y_pred = pipeline.predict(X_test)
                                        cm = confusion_matrix(y_test, y_pred)
                                        fig_cm, ax_cm = plt.subplots(figsize=(5, 4)) # Smaller CM
                                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                                        ax_cm.set_title(f"Confusion Matrix")
                                        ax_cm.set_xlabel("Predicted")
                                        ax_cm.set_ylabel("Actual")
                                        # Use target mapping if it exists for labels
                                        if st.session_state['is_categorical_target'] and st.session_state['target_mapping']:
                                             labels = list(st.session_state['target_mapping'].keys())
                                             if len(labels) == cm.shape[0]:
                                                  ax_cm.set_xticklabels(labels); ax_cm.set_yticklabels(labels)
                                        st.pyplot(fig_cm)

                                    st.markdown("#### Feature Importance (Top 10)")
                                    feature_importance = pipeline.get_feature_importance()
                                    if isinstance(feature_importance, dict) and "Info" not in feature_importance and feature_importance:
                                        sorted_importance = list(feature_importance.items())
                                        features = [str(item[0]) for item in sorted_importance[:10]]
                                        values = [item[1] for item in sorted_importance[:10]]
                                        fig_fi, ax_fi = plt.subplots(figsize=(9, 5)) # Adjusted size
                                        ax_fi.barh(features, values, color='#1E88E5')
                                        ax_fi.invert_yaxis(); ax_fi.set_xlabel('Importance Score')
                                        ax_fi.set_title(f'Top 10 Features for {target_col} Prediction')
                                        plt.tight_layout(); st.pyplot(fig_fi)
                                    else:
                                        st.info(f"Feature importance not available. Reason: {feature_importance.get('Info', 'Unknown')}")
                                else:
                                     st.error("Evaluation failed.")
                        except Exception as e:
                            st.error(f"An error occurred during the AutoML pipeline execution:")
                            st.exception(e) # Show full traceback

                    # Display captured log output in the expander
                    log_placeholder.text(log_output.getvalue())

        except Exception as load_e:
            st.error(f"Error loading or processing the uploaded file: {load_e}")


# --- Tab 3: Make Predictions ---
with tab3:
    st.header("💡 Make Predictions with the Trained Model")

    if 'pipeline' not in st.session_state or st.session_state['pipeline'] is None:
        st.warning("👈 Please run the AutoML pipeline on the 'Run Pipeline' tab first to train a model.")
        st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100) # Placeholder image
    else:
        # Retrieve necessary info from session state
        pipeline = st.session_state['pipeline']
        X_train_cols = st.session_state['X_train'].columns # Use columns from X_train
        target_col = st.session_state['target_col']
        categorical_cols = st.session_state['categorical_cols']
        numerical_cols = st.session_state['numerical_cols']
        target_mapping = st.session_state['target_mapping']
        is_categorical_target = st.session_state['is_categorical_target']

        st.success(f"Using model: **{pipeline.model_selector.best_model_name}** trained on the uploaded dataset.")

        prediction_option = st.radio(
            "Choose prediction method:",
            ["Enter values manually", "Upload CSV for batch prediction"],
            key="pred_option", horizontal=True
        )

        if prediction_option == "Upload CSV for batch prediction":
            pred_file = st.file_uploader("Upload CSV (must have same columns as training data, excluding target)", type=['csv'], key="pred_file")

            if pred_file:
                try:
                    pred_data = pd.read_csv(pred_file)
                    st.markdown("##### Preview of Uploaded Data")
                    st.dataframe(pred_data.head())

                    # Verify columns
                    expected_features = list(X_train_cols) # Get expected features from stored X_train
                    if list(pred_data.columns) != expected_features:
                         st.error(f"Column mismatch! Expected columns: {expected_features}. Found: {list(pred_data.columns)}")
                    else:
                        if st.button("Make Batch Predictions", key="batch_predict_button"):
                            with st.spinner("Generating predictions..."):
                                try:
                                    predictions_num = pipeline.predict(pred_data) # Numeric predictions
                                    result_df = pred_data.copy()

                                    # Map predictions back to original labels if needed
                                    if is_categorical_target and target_mapping:
                                         reverse_mapping = {v: k for k, v in target_mapping.items()}
                                         predictions_label = pd.Series(predictions_num).map(reverse_mapping).fillna(predictions_num)
                                         result_df[f'Predicted_{target_col}'] = predictions_label
                                    else:
                                         result_df[f'Predicted_{target_col}'] = predictions_num


                                    # Try to get probabilities
                                    try:
                                        probas = pipeline.predict_proba(pred_data)
                                        if probas.shape[1] == 2: # Binary classification
                                            result_df[f'Probability_Class1'] = probas[:, 1] # Probability of class 1
                                        else: # Multiclass
                                             for i in range(probas.shape[1]):
                                                  result_df[f'Probability_Class_{i}'] = probas[:, i]
                                        st.success("Predictions generated successfully!")
                                    except Exception as proba_e:
                                        st.warning(f"Could not get probabilities: {proba_e}")

                                    st.markdown("##### Prediction Results")
                                    st.dataframe(result_df)

                                    csv = result_df.to_csv(index=False).encode('utf-8')
                                    st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv", key='download-csv-batch')

                                except Exception as e:
                                    st.error(f"Error making batch predictions:"); st.exception(e)
                except Exception as read_e:
                     st.error(f"Error reading prediction file: {read_e}")

        else:  # Manual input
            st.markdown("##### Enter Feature Values Manually")

            with st.form("prediction_form"):
                inputs = {}
                form_cols = st.columns(3)
                col_idx = 0

                for col in X_train_cols: # Iterate through columns used for training
                    current_col = form_cols[col_idx % 3]
                    with current_col:
                        original_col_data = st.session_state['X_train'][col]

                        if col in categorical_cols:
                            unique_values = sorted(original_col_data.astype(str).unique().tolist())
                            inputs[col] = st.selectbox(f"{col}:", unique_values, key=f"input_{col}")
                        elif col in numerical_cols:
                            min_val, max_val = float(original_col_data.min()), float(original_col_data.max())
                            mean_val = float(original_col_data.mean())
                            # Add buffer
                            val_range = max_val - min_val
                            min_val -= val_range * 0.05 + 0.01
                            max_val += val_range * 0.05 + 0.01

                            if pd.api.types.is_integer_dtype(original_col_data) and val_range < 100:
                                 inputs[col] = st.slider(f"{col}:", int(np.floor(min_val)), int(np.ceil(max_val)), int(np.round(mean_val)), key=f"input_{col}")
                            elif val_range <= 20 and val_range > 0:
                                inputs[col] = st.slider(f"{col}:", min_val, max_val, mean_val, key=f"input_{col}")
                            else:
                                inputs[col] = st.number_input(f"{col}:", min_value=min_val, max_value=max_val, value=mean_val, key=f"input_{col}")
                        else:
                             inputs[col] = st.text_input(f"{col} (unknown type):", key=f"input_{col}")
                    col_idx += 1

                submitted = st.form_submit_button("Predict Single Instance")

            if submitted:
                with st.spinner("Making prediction..."):
                    try:
                        input_df = pd.DataFrame([inputs], columns=X_train_cols)

                        # Handle potential type issues before prediction
                        if 'TotalCharges' in input_df.columns:
                            input_df['TotalCharges'] = input_df['TotalCharges'].replace(' ', np.nan).replace('', np.nan)
                            input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0.0)
                        if 'SeniorCitizen' in input_df.columns:
                             input_df['SeniorCitizen'] = input_df['SeniorCitizen'].astype(int)

                        prediction_num = pipeline.predict(input_df)[0] # Numeric prediction

                        # Map back to original label if needed
                        final_prediction_label = prediction_num
                        if is_categorical_target and target_mapping:
                             reverse_mapping = {v: k for k, v in target_mapping.items()}
                             final_prediction_label = reverse_mapping.get(prediction_num, prediction_num)

                        st.markdown("##### Prediction Result")
                        result_placeholder = st.empty()

                        try:
                            proba = pipeline.predict_proba(input_df)[0]
                            confidence = max(proba)

                            result_placeholder.markdown(f"""
                            <div class='metrics-container'>
                                <h3 class='centered'>Predicted: {final_prediction_label}</h3>
                                <p>Confidence: {confidence:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("###### Probability Distribution")
                            fig_proba, ax_proba = plt.subplots(figsize=(6, 3))
                            class_labels = [f"Class_{i}" for i in range(len(proba))]
                            if is_categorical_target and target_mapping:
                                 reverse_mapping = {v: k for k, v in target_mapping.items()}
                                 class_labels = [reverse_mapping.get(i, f"Class_{i}") for i in range(len(proba))]

                            ax_proba.bar(class_labels, proba, color='#1E88E5')
                            ax_proba.set_ylabel("Probability"); ax_proba.set_ylim(0, 1)
                            plt.xticks(rotation=45, ha='right'); st.pyplot(fig_proba)

                        except Exception as proba_e:
                            st.warning(f"Could not get probability details: {proba_e}")
                            result_placeholder.markdown(f"""
                            <div class='metrics-container'>
                                <h3 class='centered'>Predicted: {final_prediction_label}</h3>
                                <p>(Probability information not available)</p>
                            </div>
                            """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error making prediction:"); st.exception(e)

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Scikit-learn")
