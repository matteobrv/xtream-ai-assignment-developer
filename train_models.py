from datetime import datetime
from functools import partial
from pathlib import Path
from pickle import dump

import numpy as np
import optuna
import pandas as pd
import pytz
import typer
from dvclive import Live  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error, r2_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from xgboost import XGBRegressor

from prepare_data import DiamondData
from tune_params import xgb_reg_obj

# directory storing pickled model files
MODELS_DIR = Path(__file__).parent / "models"


def train_linear_regression(
    data: pd.DataFrame, t_size: float, model_name: str, time_stamp: str
) -> None:
    """Train a `LinearRegression` instance, track the corresponding
    R2 and MAE metrics and pickle the fitted model.

    Args:
        `data` (pd.DataFrame): a validated diamond dataset.
        `t_size` (float): amount of data reserved for testing.
        `model_name` (str): an identifier for the model (e.g. `lin_reg`).
        `time_stamp` (str): current date and time `[dd_mm_YYYY]_[hour_min_sec]`.
    """
    x, y = data.drop(columns="price"), data.price
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=t_size, random_state=42
    )

    with Live(dir=f"./dvclive/{model_name}", dvcyaml=False) as live:
        reg = LinearRegression()
        reg.fit(x_train, np.log(y_train))

        pred_log = reg.predict(x_test)
        pred = np.exp(pred_log)

        r2 = round(r2_score(y_test, pred), 4)
        mae = round(mean_absolute_error(y_test, pred), 2)

        live.log_metric(f"{model_name}_{time_stamp}/R2", r2, plot=False)
        live.log_metric(f"{model_name}_{time_stamp}/MAE", mae, plot=False)

    # create sub-directory storing pickled lin. reg. model files
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(exist_ok=True, parents=True)

    with open(f"{model_dir}/{model_name}_{time_stamp}.pkl", "wb") as f:
        dump(reg, f, protocol=5)


def train_xgboost_regression(
    data: pd.DataFrame, t_size: float, model_name: str, time_stamp: str
) -> None:
    """Train an `XGBRegressor` instance, track the corresponding
    R2 and MAE metrics and pickle the fitted model.

    Args:
        `data` (pd.DataFrame): a validated diamond dataset.
        `t_size` (float): amount of data reserved for testing.
        `model_name` (str): an identifier for the model (e.g. `xgb_reg`).
        `time_stamp` (str): current date and time `[dd_mm_YYYY]_[hour_min_sec]`.
    """
    x, y = data.drop(columns="price"), data.price
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=t_size, random_state=42
    )

    # tune XGBRegressor hyperparameters using Optuna
    objective = partial(xgb_reg_obj, x_train=x_train, y_train=y_train)
    study = optuna.create_study(direction="minimize", study_name="xgb_reg")
    study.optimize(objective, n_trials=5)

    with Live(dir=f"./dvclive/{model_name}", dvcyaml=False) as live:
        xgb_opt = XGBRegressor(
            **study.best_params, enable_categorical=True, random_state=42
        )
        xgb_opt.fit(x_train, y_train)
        xgb_opt_pred = xgb_opt.predict(x_test)

        r2 = round(r2_score(y_test, xgb_opt_pred), 4)
        mae = round(mean_absolute_error(y_test, xgb_opt_pred), 2)

        live.log_metric(f"{model_name}_{time_stamp}/R2", r2, plot=False)
        live.log_metric(f"{model_name}_{time_stamp}/MAE", mae, plot=False)

    # create sub-directory storing pickled xgb. reg. model files
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(exist_ok=True, parents=True)

    with open(f"{model_dir}/{model_name}_{time_stamp}.pkl", "wb") as f:
        dump(xgb_opt, f, protocol=5)


def main(
    model_name: str, data_source: str, t_size: float, drop_rows: bool
) -> None:

    data = DiamondData(data_source, drop_rows)

    # set logging level to WARNING to reduce verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # generate a model time-stamp: [dd_mm_YYYY]_[hour_min_sec]
    time_stamp = datetime.now(tz=pytz.timezone("Europe/Rome"))
    time_stamp_str = time_stamp.strftime("[%d_%m_%Y]_[%H_%M_%S]")

    match model_name:
        case "lin_reg":
            lin_reg_df = data.lin_reg_data
            train_linear_regression(
                lin_reg_df, t_size, model_name, time_stamp_str
            )
        case "xgb_reg":
            xgb_reg_df = data.xgboost_data
            train_xgboost_regression(
                xgb_reg_df, t_size, model_name, time_stamp_str
            )
        case _:
            print(f"Unsupported model: {model_name}.")


if __name__ == "__main__":
    typer.run(main)
