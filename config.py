# ====   PATHS ===================

PATH_TO_DATASET = "titanic.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'


# ======= PARAMETERS ===============

# imputation parameters
IMPUTATION_DICT = {'age': 28.0, 'fare': 14.4542}


# encoding parameters
FREQUENT_LABELS = {
        'sex':['female', 'male'],
        'cabin':['C', 'Missing'],
        'embarked':['C', 'Q', 'S'],
        'title':['Miss','Mr','Mrs']
        }


DUMMY_VARIABLES = {
        'sex':{'female':0, 'male':1}, 
        'cabin':{'C':1, 'Missing':1}, 
        'embarked':{'C':0, 'Q':1, 'S': 2}, 
        'title': {'Miss':0, 'Mrs':1, 'Mr':2}
        }


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_TO_IMPUTE = ['age', 'fare'] 