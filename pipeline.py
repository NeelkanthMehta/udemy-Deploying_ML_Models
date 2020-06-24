from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    [('categorical_impute', pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
    ('missing_indicate', pp.MissingIndicator(variables=config.NUMERICAL_VARS)),
    ('numerical_impute', pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
    ('cabin_initials', pp.ExtractFirstLetter(variables=config.CABIN)),
    ('club_rare_labels', pp.RareLabelCategoricalEncoder(tol=0.05, variables=config.CATEGORICAL_VARS)),
    ('category_encode', pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
    ('stardardize', StandardScaler()),
    ('model_instance', LogisticRegression(penalty='l1',C=0.0005, random_state=0, solver='saga'))
    ])