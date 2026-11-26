"""Public pilot utilities adapted from exploratory dairy lameness experiments."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import classification_report, cohen_kappa_score


def combine_records(paths):
    """Combine locally authorised spreadsheets; no data is bundled."""
    return pd.concat([pd.read_excel(p) for p in paths], ignore_index=True)


def build_pipeline(X, model, imputation='median', oversample=False):
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    imputer = KNNImputer() if imputation == 'knn' else SimpleImputer(strategy='median')
    pre = ColumnTransformer([
        ('numeric', Pipeline([('impute', imputer), ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ])
    models = {
        'knn': (KNeighborsClassifier(), {'model__n_neighbors': [3, 5, 7]}),
        'tree': (DecisionTreeClassifier(random_state=42), {'model__max_depth': [3, 6, None]}),
        'forest': (RandomForestClassifier(random_state=42), {'model__max_depth': [4, 8, None]}),
        'mlp': (MLPClassifier(random_state=42, max_iter=500), {'model__hidden_layer_sizes': [(32,), (32, 64)]}),
    }
    estimator, grid = models[model]
    steps = [('preprocess', pre)]
    if oversample:
        from imblearn.pipeline import Pipeline as SamplingPipeline
        from imblearn.over_sampling import RandomOverSampler
        return SamplingPipeline(steps + [('sample', RandomOverSampler(random_state=42)), ('model', estimator)]), grid
    return Pipeline(steps + [('model', estimator)]), grid


def run_experiment(frame, *, features, target, animal_id, model='forest', imputation='median', oversample=False):
    """Group by animal; callers must select only features available at prediction time."""
    if target in features or animal_id in features or target == animal_id:
        raise ValueError('Target and animal identifier must not be predictors.')
    df = frame.dropna(subset=[target, animal_id])
    X, y, groups = df[features], df[target], df[animal_id]
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
    pipe, grid = build_pipeline(X.iloc[train], model, imputation, oversample)
    n_groups = groups.iloc[train].nunique()
    if n_groups < 3:
        raise ValueError('At least three training animal groups are required.')
    search = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=min(5, n_groups)), scoring='f1_macro', error_score='raise')
    search.fit(X.iloc[train], y.iloc[train], groups=groups.iloc[train])
    pred = search.predict(X.iloc[test])
    return {'classification_report': classification_report(y.iloc[test], pred, output_dict=True, zero_division=0),
            'quadratic_weighted_kappa': cohen_kappa_score(y.iloc[test], pred, weights='quadratic')}
