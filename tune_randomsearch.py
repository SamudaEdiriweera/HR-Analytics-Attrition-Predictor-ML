import mlflow
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src import config
from src.train import get_model_selection
from src.data_preprocessing import create_preprocessor, load_and_preprocess_data

def run_random_search(model_name):
    """Performs RandomSearchCV for a specified model."""
    
    if model_name not in config.RANDOM_SEARCH_GRIDS:
        raise ValueError(f"Random search parameters for {model_name} not defined in config.")
    
    # --- 1. Setup ---
    print(f"▶️ Starting RandomSearchCV for {model_name}...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    preprocessor = create_preprocessor()
    
    # Get model info from config
    model_info = config.MODELS[model_name]
    model_class = get_model_selection(model_info['class'])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_class(random_state=config.RANDOM_STATE))
    ])
    
    # --- 2. Run RandomSerachCV ---
    print("▶️ Starting RandomSearchCV...")
    mlflow.set_experiment("HR_Attrition_Hyperparameter_Tuning")
    with mlflow.start_run(run_name=f"RandomSearch_{model_name}"):
        
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config.RANDOM_SEARCH_GRIDS[model_name],
            n_iter=config.N_ITER_RANDOM_SEARCH,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
            verbose=2
        )
        
        random_search.fit(X_train, y_train)
        
        # --- 3. Log Results ---
        print("✅ RandomizedSearchCV finished. Logging results...")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_metric("best_roc_auc_cv", random_search.best_score_)
        
        best_params = {f"best_{k}": v for k, v in random_search.best_params_.items()}
        mlflow.log_params(best_params)
        
        # Log each trial as a nested run for detailed tracking
        for i, params in enumerate(random_search.cv_results_['params']):
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])
                mlflow.log_metric("std_test_score", random_search.cv_results_['std_test_score'][i])
        
        print(f"\n🏆 Best Score (ROC AUC): {random_search.best_score_}")
        print(f"📋 Best Parameters found: {random_search.best_params_}")
        
    print("🏁 RandomizedSearchCV process complete.")
    
if __name__ == "__main__":
    for model_name in config.MODELS.keys():
        run_random_search(model_name)

        