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