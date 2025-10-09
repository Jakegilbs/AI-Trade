import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def build_model(random_state=42) -> Pipeline:
    return Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

def fit_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model.fit(X_train, y_train)
    return model
