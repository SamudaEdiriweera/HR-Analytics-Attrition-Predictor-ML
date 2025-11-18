from sklearn.pipeline import Pipeline
from src import config
import importlib

def get_model_selection(class_path):
    """ Dynamically imports a class from a string path"""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def train_model(X_train, y_train, preprocessor, model_name):
    """
    Creates and trains a pipeline with the specified preprocessor and model.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        preprocessor: The ColumnTransformer for preprocessing.
        model_name (str): The name of the model to use (e.g., 'LogisticRegression').
        
    Returns:
        A trained scikit-learn pipeline.
    """
    if model_name not in config.MODELS:
        raise ValueError(f"Model '{model_name}' is not defined in config.py")
    
    # Get model info from config
    model_info = config.MODELS[model_name]
    model_class = get_model_selection(model_info['class'])
    model_params = model_info['params']
    
    
    classifier = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_class(**model_params))
    ])
    
    print(f"Fitting pipeline with model: {model_name}")
    classifier.fit(X_train, y_train)

    return classifier

""" 

*   **Key Changes:**
    *   The function now takes a `model_name` argument.
    *   It uses a helper function `get_model_class` to dynamically import the correct model class (e.g., `sklearn.linear_model.LogisticRegression`) from the string path defined in `config.py`.
    *   It looks up the corresponding hyperparameters from the config.
    *   It builds and fits the pipeline with the selected model.


"""
