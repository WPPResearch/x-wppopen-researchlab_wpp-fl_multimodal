import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.settings import settings


def load_synthetic_data(
    partition_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load synthetic data for either server or a client partition.

    :param partition_id: If None, load server data. Otherwise, load client partition data.
    :return:
        - If server data: X, y, None, None
        - If client data: X_train, y_train, X_valid, y_valid
    """
    version = settings.general.data_version
    text_embedding_task = settings.general.text_embedding_task
    data_split = settings.general.data_split
    alpha = settings.general.alpha
    num_clients = settings.client.num_clients
    target_column = settings.general.target

    # Determine subfolder based on split type
    match data_split:
        case "homogeneous":
            subfolder = f"split={data_split}_clients={num_clients}"
        case "heterogeneous":
            subfolder = f"split={data_split}_clients={num_clients}_alpha={alpha}"
        case _:
            raise ValueError(f"Invalid split {data_split}")

    # Determine file path
    if partition_id is None:
        file_path = f"data/{version}/{subfolder}/server/encoded/server_test_emb_pdfm_{text_embedding_task}.parquet"
    else:
        file_path = (
            f"data/{version}/{subfolder}/client/encoded/"
            f"client_{partition_id}_train_emb_pdfm_{text_embedding_task}.parquet"
        )

    # Load and shuffle data
    df = (
        pd.read_parquet(file_path, engine="fastparquet")
        .dropna()
        .sample(frac=1.0, random_state=settings.general.random_seed)
    )

    # One-hot encode specified modalities
    for mod in settings.general.one_hot_modalities:
        df.drop([col for col in df.columns if f"{mod}_emb" in col], axis=1, inplace=True)
        one_hot = pd.get_dummies(df[mod])
        one_hot.columns = [f"{mod}_emb_{i:03d}" for i in range(len(one_hot.columns))]
        df = pd.concat([df, one_hot], axis=1)

    # Select feature columns
    feature_cols = [c for c in df.columns if any(m in c for m in settings.general.modalities)]

    if partition_id is None:
        # Server data: return X, y only
        X = df[feature_cols]
        y = df[target_column]
        return X, y, None, None
    else:
        # Client data: split into train/valid sets
        df_train, df_valid = train_test_split(
            df, test_size=0.1, random_state=settings.general.random_seed, stratify=df[target_column]
        )
        X_train = df_train[feature_cols]
        y_train = df_train[target_column]
        X_valid = df_valid[feature_cols]
        y_valid = df_valid[target_column]
        return X_train, y_train, X_valid, y_valid


def create_run_dir() -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(settings.config_path, save_path)

    return save_path, run_dir
