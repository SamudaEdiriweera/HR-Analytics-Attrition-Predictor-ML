import os


DIR = os.getcwd()
DATA_PATH = os.path.join(DIR, "data", "HR_Data.csv")

TARGET_COLUMN = "Attrition"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Columns to drop immediately based on Phase 1 & 2 analysis
DROP_COLS = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
    "Gender",
    "JobLevel",
    "YearsAtCompany",
    "PerformanceRating",
]

# Numerical cols based on EDA except drop columns
NUMERICAL_COLS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

ORDINAL_COLS = [
    "BusinessTravel",
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance"
]

ORDINAL_CATEGORIES = [
    ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
    [1, 2, 3, 4, 5],             # Assuming 1='Below College' to 5='Doctor'
    [1, 2, 3, 4],            # Assuming 1='Low' to 4='Very High'
    [1, 2, 3, 4],                     # Assuming 1='Low' to 4='Very High'
    [1, 2, 3, 4],                    # Assuming 1='Low' to 4='Very High'
    [1, 2, 3, 4],           # Assuming 1='Low' to 4='Very High'
    [0, 1, 2, 3],                   # 0='None' to 3='High Level'
    [1, 2, 3, 4]                     # Assuming 1='Bad' to 4='Best'
]

NOMINAL_COLS = [
    "MaritalStatus",
    "Department",
    "JobRole",
    "EducationField",
    "OverTime"
]

# --- Model Parameters ---
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
}

XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': RANDOM_STATE,
}

# --- Model Selection ---
# Defines the models we want to compare in our baseline run.

MODELS = {
    'LogisticRegression': {
        'class': 'sklearn.linear_model.LogisticRegression',
        'params': LOGISTIC_REGRESSION_PARAMS
    },
    'RandomForestClassifier': {
        'class': 'sklearn.ensemble.RandomForestClassifier',
        'params': RANDOM_FOREST_PARAMS
    },
    'XGBClassifier': {
        'class': 'xgboost.XGBClassifier',
        'params': XGBOOST_PARAMS
    }
}


# --- Hyperparameter Tuning Spaces ---
from scipy.stats import randint, uniform, loguniform

# Number of iterations/trials for stochastic methods
N_ITER_RANDOM_SEARCH = 50
OPTUNA_TRIALS = 50

# 1. Grid for GridSearchCV (Specific values)
GRID_SEARCH_GRIDS = {
    'LogisticRegression': {
        'model__penalty': ['l1', 'l2'],
        'model__C': [0.01, 0.1, 1, 10],
        'model__solver': ['liblinear', 'saga']
    },
    'RandomForestClassifier': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20]
    },
    'XGBClassifier': {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.1, 0.2]
    }
}

# 2. Distributions for RandomizedSearchCV
RANDOM_SEARCH_GRIDS = {
    'LogisticRegression': {
        'model__penalty': ['l1', 'l2'],
        'model__C': loguniform(1e-3, 1e2),
        'model__solver': ['liblinear', 'saga']
    },
    'RandomForestClassifier' : {
        'model__n_estimators': randint(100, 500),
        'model__max_depth': [5, 10, 15, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'XGBClassifier': {
        'model__n_estimators': randint(100, 500),
        'model__max_depth': [3, 5, 7, 10],
        'model__learning_rate': uniform(0.01, 0.3),
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
}

# 3. Optuna: Defines the search space for the objective function.
OPTUNA_PARAMS = {
    'n_trials': 10 # Number of trials to run
}

# --- Now completed the hyperparameter tune ---
# --- FINAL MODEL CONFIGURATION ---
# After hyperparameter tuning, Logistic Regression was chosen as the best model.
FINAL_MODEL_PARAMS = {
    'C': 10,
    'solver': 'liblinear',
    'penalty': 'l1',
    'max_iter': 1000,
    'random_state': RANDOM_STATE
}

FINAL_MODEL = {
    'LogisticRegression': {
        'class': 'sklearn.linear_model.LogisticRegression',
        'params': FINAL_MODEL_PARAMS
    },
}