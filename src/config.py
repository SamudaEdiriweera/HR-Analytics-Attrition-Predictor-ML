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
    "Joblevel", 
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

ORDINAL_CATEGORIES = {
    'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
    'Education': [1, 2, 3, 4, 5],                       # Assuming 1='Below College' to 5='Doctor'
    'EnvironmentSatisfaction': [1, 2, 3, 4],            # Assuming 1='Low' to 4='Very High'
    'JobInvolvement': [1, 2, 3, 4],                     # Assuming 1='Low' to 4='Very High'
    'JobSatisfaction': [1, 2, 3, 4],                    # Assuming 1='Low' to 4='Very High'
    'RelationshipSatisfaction': [1, 2, 3, 4],           # Assuming 1='Low' to 4='Very High'
    'StockOptionLevel': [0, 1, 2, 3],                   # 0='None' to 3='High Level' 
    'WorkLifeBalance': [1, 2, 3, 4]                     # Assuming 1='Bad' to 4='Best'
    }

NOMINAL_COLS = [
    "MaritalStatus", 
    "Department", 
    "JobRole", 
    "EducationField", 
    "OverTime"
    ]

# --- Model Parameters ---
LOGISTIC_REGRESSION_PARAMS = {
    "max_tier": 1000,
    "random_state": RANDOM_STATE,
}