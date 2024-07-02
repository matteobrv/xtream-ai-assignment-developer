from pathlib import Path

import pandas as pd
import typer


def collect_metrics(dvc_metrics: str, models_dir: str) -> None:
    """Create a csv file storing the metrics for a given model
    and update it for each subsequent model run.

    Args:
        `dvc_metrics` (str): path to the json metrics file generated
            by dvc for the current model.
        `models_dir` (str): path to the current model's sub-directory
            in the models folder (e.g. `models/lin_reg`), where the
            corresponding pickles and `metrics.csv` files are stored.
    """
    metrics_csv = Path(f"{models_dir}/metrics.csv")

    current_metrics = pd.read_json(dvc_metrics).transpose()
    current_metrics.reset_index(inplace=True)
    current_metrics.rename(columns={"index": "model"}, inplace=True)

    if metrics_csv.exists():
        metrics = pd.read_csv(metrics_csv)
        metrics = pd.concat([metrics, current_metrics], axis=0)
    else:
        metrics = current_metrics

    metrics.to_csv(metrics_csv, mode="w", index=False)



if __name__ == "__main__":
    typer.run(collect_metrics)
