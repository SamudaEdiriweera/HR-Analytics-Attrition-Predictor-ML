from src.data_preprocessing import load_and_preprocess_data, create_preprocessor
from src.train import train_model
from src.model_evaluation import evaluate_model
import mlflow
import mlflow.sklearn
from src import config
import joblib

def main():
    print("▶️ Starting the HR Analytics Pipeline... ")
    
    # Set the experiment name. If it doesn't exist, it will be created
    mlflow.set_experiment("HR_Analytics_Attrition_Prediction")
    
    # Start an MLflow run
    with  mlflow.start_run():
        mlflow.log_param("test_size", config.TEST_SIZE)
        mlflow.log_param("random_state", config.RANDOM_STATE)
        mlflow.log_params(config.LOGISTIC_REGRESSION_PARAMS)
        print("✅ Logged parameters to MLflow.")
        
        # 1. Load and preprocess data
        print("🔄 Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        print("✅ Data loaded and preprocessed.")
        
        # 2. Create preprocessor
        print("🔄 Creating preprocessor pipeline...")
        preprocessor = create_preprocessor()
        # Save the preprocessor to be used later
        joblib.dump(preprocessor, 'preprocessor.joblib')
        mlflow.log_artifact('preprocessor.joblib') # Log preprocessor as artifact
        print("✅ Preprocessor created and logged.")
        
        # 3. Train model
        print("🔄 Training the model...")
        model = train_model(X_train, y_train, preprocessor)
        print("✅ Model training completed.")
        
        # 4. Evaluate model
        print("🔄 Evaluating the model...")
        accuracy, roc_auc, conf_matrix, class_report = evaluate_model(model, X_test, y_test)
        print("✅ Model evaluation completed.")
        
        # 5. Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        print("✅ Logged metrics to MLflow.")
        
        # 6. Log the trained model to Mlflow
        print("🔄 Logging the trained model to MLflow...")
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        print("✅ Model logged to MLflow.")
        
        print("🎉 HR Analytics Pipeline completed successfully!")
        
if __name__ == "__main__":
    main()