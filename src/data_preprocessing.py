import pandas as pd
from sklearn.model_selection import train_test_split
from src import config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess_data():
    """ Loads dara, perofrms initial cleaning, and split into train/test"""
    df = pd.read_csv(config.DATA_PATH)
    
    # 2. Initail cleaning (from phase 1)
    df = df.drop(columns=config.DROP_COLS)
    
    # 3. Encode Taget Varaible
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)
    
    independent_feature_cols = config.NUMERICAL_COLS + config.ORDINAL_COLS + config.NOMINAL_COLS
    
    X = df[independent_feature_cols]
    y = df[config.TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
                                X, y, 
                                test_size=config.TEST_SIZE, 
                                random_state=config.RANDOM_STATE, 
                                stratify=y
                            )
    
    return X_train, X_test, y_train, y_test
    
def create_preprocessor():
    
    numeric_transformer = Pipeline([
        ('impute', SimpleImputer(strategy="median")),
        ('scale', StandardScaler())
    ])
    
    ordinal_transformer = Pipeline([
        ('impute', SimpleImputer(strategy="most_frequent")),
        ('encode', OrdinalEncoder(categories=config.ORDINAL_CATEGORIES))
    ])
    
    nominal_transformer = Pipeline([
        ('impute', SimpleImputer(strategy="most_frequent")),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('numeric', numeric_transformer, config.NUMERICAL_COLS),
        ('ordinal', ordinal_transformer, config.ORDINAL_COLS),
        ('nominal', nominal_transformer, config.NOMINAL_COLS)
    ])
    
    return preprocessor