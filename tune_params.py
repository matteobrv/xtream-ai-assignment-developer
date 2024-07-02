import pandas as pd
from optuna.trial import Trial
from sklearn.metrics import mean_absolute_error # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from xgboost import XGBRegressor


def xgb_reg_obj(
    trial: Trial, x_train: pd.DataFrame, y_train: pd.Series
) -> float:
    """An objective function, returning a `mean_absolute_error` score,
    that Optuna tries to minimize to get the best hyperparameters for
    an `XGBRegressor` model instance. See `train_models.train_xgboost_regression`.
    """
    x_train_v, x_val, y_train_v, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )

    params = {
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", [0.3, 0.4, 0.5, 0.7]
        ),
        "subsample": trial.suggest_categorical(
            "subsample", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "random_state": 42,
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "enable_categorical": True,
    }

    model = XGBRegressor(**params)
    model.fit(x_train_v, y_train_v)

    preds = model.predict(x_val)
    mae: float = mean_absolute_error(y_val, preds)

    return mae
