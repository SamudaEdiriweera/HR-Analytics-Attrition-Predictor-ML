from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import XGBClassifier
from src import config

def train_model(X_train, y_train, preprocessor):
    classifier = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS))
    ])    
    classifier.fit(X_train, y_train)
    
    return classifier