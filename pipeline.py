import datetime

import dill
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")


def clear_df(data):
    df_new = data.copy()
    columns_to_drop = [
        'session_id',
        'client_id',
        'utm_source',
        'utm_campaign',
        'utm_adcontent',
        'utm_keyword',
        'device_screen_resolution',
        'device_os',
        'device_model',
        'geo_city'
    ]
    return df_new.drop(columns_to_drop, axis=1)


def create_new_features(data):
    df_new = data.copy()
    columns_to_drop = [
        'visit_time',
        'visit_date',
    ]

    df_new.visit_time = pd.to_datetime(df_new.visit_time, format='%H:%M:%S')
    df_new['visit_hour'] = df_new['visit_time'].dt.hour

    df_new.visit_date = pd.to_datetime(df_new.visit_date, format='ISO8601')
    df_new['visit_day_of_week'] = df_new.visit_date.dt.dayofweek
    df_new['visit_month'] = df_new.visit_date.dt.month

    return df_new.drop(columns_to_drop, axis=1)


def pipeline():
    print('Loading data...')
    df = pd.read_csv('data/updpipeline.csv')

    df.drop_duplicates(inplace=True)

    x = df.drop("is_target", axis=1)
    y = df['is_target']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessing_functions = Pipeline(steps=[
        ("clearing", FunctionTransformer(clear_df)),
        ('featuring', FunctionTransformer(create_new_features))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numeric_transformer', numeric_transformer, make_column_selector(dtype_include=['float64', 'int64'])),
        ('categorical_transformer', categorical_transformer, make_column_selector(dtype_include=object))
    ])

    models = [
        LogisticRegression(
            solver='saga',
            C=37.92690190732238,
            max_iter=1000,
            penalty='none'
        ),
        RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            min_samples_split=2,
            max_features='sqrt',
            max_depth=4,
            bootstrap=True
        ),
        # SVC()
    ]

    best_score = .0
    best_pipe = None
    print("Time to models:")
    for model in models:
        pipe = Pipeline(steps=[
            ('functions', preprocessing_functions),
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, x, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc: {score.mean():.4f}, std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')

    best_pipe.fit(x, y)
    filename = 'models/target_action_model.pkl'

    with open(filename, 'wb') as f:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'target_action_model',
                'author': 'smdsu',
                'version': 1.0,
                'creationDate': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'roc_auc': best_score
            }
        }, f)


if __name__ == '__main__':
    pipeline()
