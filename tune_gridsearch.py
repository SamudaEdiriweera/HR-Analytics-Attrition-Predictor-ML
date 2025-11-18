# tune_gridsearch.py

import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src import config
from src.train import get_model_selection
from src.data_preprocessing import create_preprocessor, load_and_preprocess_data


def run_grid_search(model_name):
    """Performs GridSearchCV for a specified model."""
    
    if model_name not in config.GRID_SEARCH_GRIDS:
        raise ValueError(f"Grid search parameters for {model_name} not defined in config.")
        
    # --- 1. Setup ---
    print(f"▶️ Starting GridSearchCV for {model_name}...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    preprocessor = create_preprocessor()
    
    # Get model info from config
    model_info = config.MODELS[model_name]
    model_class = get_model_selection(model_info['class'])
    # model_params = model_info['params']
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_class(random_state=config.RANDOM_STATE))
    ])
    
    # --- 2. Run GridSearchCV ---
    print("▶️ Starting GridSearchCV...")
    mlflow.set_experiment("HR_Attrition_Hyperparameter_Tuning")
    with mlflow.start_run(run_name=f"GridSearch_{model_name}"):
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config.GRID_SEARCH_GRIDS[model_name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        # --- 3. Log Results ---
        print("✅ GridSearchCV finished. Logging results...")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_metric("best_roc_auc_cv", grid_search.best_score_)
        
        best_params = {f"best_{k}": v for k, v in grid_search.best_params_.items()}
        mlflow.log_params(best_params)
        
        # Log each trial as a nested run for detailed tracking
        for i, params in enumerate(grid_search.cv_results_['params']):
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_test_score", grid_search.cv_results_['mean_test_score'][i])
                mlflow.log_metric("std_test_score", grid_search.cv_results_['std_test_score'][i])

        print(f"\n🏆 Best Score (ROC AUC): {grid_search.best_score_}")
        print(f"📋 Best Parameters found: {grid_search.best_params_}")

    print("🏁 GridSearch process complete.")

if __name__ == "__main__":
    for model_name in config.MODELS.keys():
        run_grid_search(model_name)

