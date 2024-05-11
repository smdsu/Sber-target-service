import datetime

import dill
import numpy as np
import pandas as pd
import pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings


warnings.filterwarnings("ignore")


# Load the data
print('Loading data...')
df = pd.read_csv('data/updpipeline.csv')

df.drop_duplicates(inplace=True)

x = df.drop("is_target", axis=1)
y = df['is_target']

# Define the preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessing_functions = Pipeline(steps=[
    ("clearing", FunctionTransformer(pipeline.clear_df)),
    ('featuring', FunctionTransformer(pipeline.create_new_features))
])

preprocessor = ColumnTransformer(transformers=[
    ('numeric_transformer', numeric_transformer, make_column_selector(dtype_include=['float64', 'int64'])),
    ('categorical_transformer', categorical_transformer, make_column_selector(dtype_include=object))
])

# Define the models
models = [
    LogisticRegression(),
    RandomForestClassifier(),
    # SVC()
]

param_grids = [
    {
        'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'classifier__C': np.logspace(-10, 10, 10),
        'classifier__solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
        'classifier__max_iter': [100, 1000, 2500, 5000]
    },
    {
        'classifier__n_estimators': [int(x) for x in np.linspace(start=10, stop=500, num=10)],
        'classifier__max_features': ['auto', 'sqrt'],
        'classifier__max_depth': [2, 4],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__bootstrap': [True, False]
    }
]

best_models = []
best_scores = []

print('Iterating over models to find the best hyperparameters')
# Iterate over models to find the best hyperparameters
for model, param_grid in zip(models, param_grids):
    pipe = Pipeline(steps=[
        ('functions', preprocessing_functions),
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=4,
        scoring='roc_auc',
        verbose=10,
        n_jobs=-1
    )

    grid_search.fit(x, y)

    print(f"Best parameters for {type(model).__name__}: {grid_search.best_params_}")

    best_models.append(grid_search.best_estimator_)
    best_scores.append(grid_search.best_score_)

best_index = best_scores.index(max(best_scores))
best_model = best_models[best_index]
best_score = best_scores[best_index]

print(f"Best score for best_model {type(best_model).__name__}: {best_score}")

filename = 'models/target_action_model_best_params.pkl'
with open(filename, 'wb') as f:
    dill.dump({
        'model': best_model,
        'metadata': {
            'name': 'target_action_model_best_params',
            'author': 'smdsu',
            'version': 1.0,
            'creationDate': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': type(best_model.named_steps["classifier"]).__name__,
            'roc_auc': best_score
        }
    }, f)
