# tune_optuna.py

import optuna
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from src.data_preprocessing import create_preprocessor, load_and_preprocess_data
from src.train import get_model_selection # Use the correct function name
from src import config

def run_optuna_study():
    """
    Loops through all models defined in config.py, runs an Optuna study for each,
    and logs the results to MLflow.
    """
    # --- 1. Load Data Once ---
    # It's more efficient to load the data outside the loop
    print("▶️ Loading data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    preprocessor = create_preprocessor()
    
    # --- 2. Loop Through Each Model ---
    for model_name in config.MODELS.keys():
        print(f"\n--- Starting Optuna Study for {model_name} ---")

        # --- 3. Define the Objective Function INSIDE the loop ---
        # This is crucial so the objective function knows which model it's working with.
        def objective(trial):
            """The objective function for Optuna to optimize for a specific model."""
            
            # 3a. Get the model class
            model_class = get_model_selection(config.MODELS[model_name]['class'])
            
            # 3b. Define the hyperparameter search space for the CURRENT model
            if model_name == 'LogisticRegression':
                params = {
                    'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'max_iter': 1000,
                    'random_state': config.RANDOM_STATE
                }
            elif model_name == 'RandomForestClassifier':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'random_state': config.RANDOM_STATE
                }
            elif model_name == 'XGBClassifier':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': config.RANDOM_STATE
                }
            else:
                params = {} # Default for any other model
            
            # 3c. Create the full pipeline with the suggested parameters
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model_class(**params))
            ])
            
            # 3d. Evaluate the model using cross-validation
            score = cross_val_score(pipeline, X_train, y_train, n_jobs=-1, cv=5, scoring='roc_auc').mean()
            
            return score

        # --- 4. Run the Optuna Study for the Current Model ---
        mlflow.set_experiment("HR_Attrition_Hyperparameter_Tuning")
        
        # Start a parent run for each model's tuning study
        with mlflow.start_run(run_name=f"Optuna_{model_name}") as parent_run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("tuning_method", "Optuna")

            study = optuna.create_study(direction='maximize')
            
            # Pass a lambda function to `optimize` so `objective` can run without arguments
            study.optimize(objective, n_trials=config.OPTUNA_TRIALS, show_progress_bar=True)
            
            # --- 5. Log Best Results to the Parent Run ---
            print("✅ Optuna study finished. Logging best results to parent run...")
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_roc_auc_cv", study.best_value)

            print(f"🏆 Best Score (ROC AUC) for {model_name}: {study.best_value}")
            print(f"📋 Best Parameters: {study.best_params}")

    print("\n🏁 All Optuna tuning processes complete.")

if __name__ == "__main__":
    run_optuna_study()