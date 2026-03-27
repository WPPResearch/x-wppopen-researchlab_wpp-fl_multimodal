import argparse
import json
import os
import re
import subprocess
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import torch
import yaml
from loguru import logger
from skorch import NeuralNetClassifier

from src.models import MODELS
from src.settings import settings
from src.utils import create_pipeline

# Silence all Python warnings
warnings.filterwarnings("ignore")

ACC_REGEX = re.compile(r"model_state_fed_acc_([0-9.]+)_round_\d+\.pth")
API_BASE_URL = os.getenv("EVAL_BASE_URL", None)
API_KEY = os.getenv("EVAL_API_KEY", None)
IDENTITY_TOKEN = os.getenv("EVAL_IDENTITY_TOKEN", None)

if API_KEY is None:
    raise ValueError("API Key not found. Please set EVAL_API_KEY in your .env file.")
if IDENTITY_TOKEN is None:
    raise ValueError("Identity Token not found. Please set EVAL_IDENTITY_TOKEN in your .env file.")


def find_latest_date_folder(base_dir: Path) -> Path:
    date_folders = [d for d in base_dir.iterdir() if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)]
    if not date_folders:
        raise FileNotFoundError("No date folders found")

    return max(date_folders, key=lambda d: datetime.strptime(d.name, "%Y-%m-%d"))


def find_latest_time_folder(date_dir: Path) -> Path:
    time_folders = [d for d in date_dir.iterdir() if d.is_dir() and re.match(r"\d{2}-\d{2}-\d{2}", d.name)]
    if not time_folders:
        raise FileNotFoundError(f"No time folders found in {date_dir}")

    return max(time_folders)


def load_run_config(run_path: Path) -> dict:
    config_path = run_path / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.yaml not found in {run_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_best_model(model_dir: Path) -> Path:
    best_acc = -1.0
    best_model = None

    for file in model_dir.glob("model_state_fed_acc_*.pth"):
        match = ACC_REGEX.match(file.name)
        if not match:
            continue

        acc = float(match.group(1))
        if acc > best_acc:
            best_acc = acc
            best_model = file

    if best_model is None:
        raise FileNotFoundError("No valid model files found")

    return best_model


def set_test_dataset():
    version = settings.general.data_version
    text_embedding_task = settings.general.text_embedding_task
    data_split = settings.general.data_split
    alpha = settings.general.alpha
    num_clients = settings.client.num_clients

    # Determine subfolder based on split type
    match data_split:
        case "homogeneous" | "synthetic":
            subfolder = f"split={data_split}_clients={num_clients}"
        case "heterogeneous":
            subfolder = f"split={data_split}_clients={num_clients}_alpha={alpha}"
        case _:
            raise ValueError(f"Invalid split {data_split}")

    train_file_path = (
        f"data/{version}/{subfolder}/client/encoded/" f"client_{0}_train_emb_pdfm_{text_embedding_task}.parquet"
    )
    # Load and shuffle train data
    df_train = (
        pd.read_parquet(train_file_path, engine="fastparquet")
        .dropna()
        .sample(frac=1.0, random_state=settings.general.random_seed)
    )

    test_file_path = f"data/{version}/{subfolder}/server/encoded/" f"server_test_emb_pdfm_classification.parquet"
    df_infer = pd.read_parquet(test_file_path, engine="fastparquet")
    feature_cols = [c for c in df_train.columns if any(m in c for m in settings.general.modalities)]
    df_infer = df_infer[feature_cols]

    for mod in settings.general.one_hot_modalities:
        df_infer.drop([col for col in df_train.columns if f"{mod}_emb" in col], axis=1, inplace=True)
        one_hot = pd.get_dummies(df_infer[mod])
        one_hot.columns = [f"{mod}_emb_{i:03d}" for i in range(len(one_hot.columns))]
        df_infer = pd.concat([df_infer, one_hot], axis=1)

    return df_infer


def load_predictions(file: str) -> str:
    df = pd.read_csv(file)
    # Use a temporary file
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w") as tmp:
        tmp.write("\n".join(df["0"].astype("str").tolist()))
    return path


def run_test(
    file: str,
    model_name: str,
    model_version: str,
    version: str,
    commit: str,
    branch: str,
    api_url: str = API_BASE_URL,
    team: str = "lab",
    allow_override: bool = False,
):
    """Runs the test against the local API."""
    print("--- Testing Prediction Evaluation API ---")
    print(f"Endpoint: {api_url}/evaluate")
    print(f"Dataset Version: {version}")

    # Create a prediction file that matches the ground truth count (10)
    predictions_path = load_predictions(file)
    try:
        with open(predictions_path, "rb") as f:
            # Prepare the multipart/form-data payload
            files = {"predictions_file": ("predictions.txt", f, "text/plain")}
            data = {
                "dataset_version": version,
                "team": team,
                "model_nickname": model_name,
                "model_version": model_version,
                "repo_commit_sha": commit,
                "repo_branch": branch,
                "allow_override": allow_override,
            }
            headers = {"Authorization": f"Bearer {IDENTITY_TOKEN}", "X-API-Key": API_KEY}

            print("\nSending request...")
            response = requests.post(f"{API_BASE_URL}/evaluate", headers=headers, data=data, files=files)

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                print("Evaluation successful!")
                results = response.json()
                print("\n--- Results ---")
                print(json.dumps(results, indent=2))
            else:
                print("Evaluation failed!")
                try:
                    print(f"Error details: {response.json()}")
                except json.JSONDecodeError:
                    print(f"Raw error response: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"\nConnection Error: Could not connect to the API at {API_BASE_URL}. Is the server running?")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


def get_git_info() -> tuple[str, str]:
    """Get branch name and last commit of the current file."""
    branch = subprocess.check_output(["git", "branch", "--show-current"]).strip().decode()
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()

    return branch, commit_hash


def infer(run_folder: str | None = None, run_test_flag: bool = False, allow_override: bool = False):
    """
    Run inference using the best available trained FL model and evaluate results.

    Args:
        run_folder (str | None): Path to the specific run folder. If None, find the latest.
    """

    logger.info("🚀 Starting inference")

    outputs_dir = Path("./outputs")
    model_name = settings.model.name

    # ------------------------------------------------------------------
    # Validate model configuration
    # ------------------------------------------------------------------
    if model_name not in MODELS:
        logger.error(f"Invalid model name: {model_name}")
        raise ValueError(f"Invalid model name: {model_name}")

    logger.info(f"Using model: {model_name}")

    # ------------------------------------------------------------------
    # Validate outputs directory
    # ------------------------------------------------------------------
    if not outputs_dir.exists():
        logger.error(f"outputs_dir does not exist: {outputs_dir.resolve()}")
        raise FileNotFoundError(f"outputs_dir does not exist: {outputs_dir.resolve()}")

    if not outputs_dir.is_dir():
        logger.error(f"outputs_dir is not a directory: {outputs_dir.resolve()}")
        raise NotADirectoryError(f"outputs_dir is not a directory: {outputs_dir.resolve()}")

    logger.info(f"Searching trained models in: {outputs_dir.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------------------------------------------------
    # Locate latest run and best-performing model
    # ------------------------------------------------------------------
    data_version = settings.general.data_version
    if run_folder:
        run_path = outputs_dir / run_folder
        if not run_path.is_dir():
            raise NotADirectoryError(f"Provided run folder does not exist or is not a directory: {run_path}")
        best_model_path = find_best_model(run_path)
        logger.info(f"Using provided run folder: {run_folder}")
        logger.info(f"Best FL model selection → ({best_model_path.name})")

        run_config = load_run_config(run_path)
        data_version = run_config.get("general", {}).get("data_version", data_version)
        settings.general.data_version = data_version
        settings.general.data_split = run_config.get("general", {}).get("data_split", settings.general.data_split)
        settings.client.num_clients = run_config.get("client", {}).get("num_clients", settings.client.num_clients)
        logger.info(f"Using data_version from run config: {data_version}")

    else:
        latest_date = find_latest_date_folder(outputs_dir)
        latest_time = find_latest_time_folder(latest_date)
        best_model_path = find_best_model(latest_time)
        logger.info(f"Best FL model selection → {latest_date.name}/{latest_time.name} " f"({best_model_path.name})")

        run_config = load_run_config(latest_time)
        data_version = run_config.get("general", {}).get("data_version", data_version)
        settings.general.data_version = data_version
        settings.general.data_split = run_config.get("general", {}).get("data_split", settings.general.data_split)
        settings.client.num_clients = run_config.get("client", {}).get("num_clients", settings.client.num_clients)
        logger.info(f"Using data_version from latest run config: {data_version}")

    # ------------------------------------------------------------------
    # Load model weights
    # ------------------------------------------------------------------
    logger.info("Loading model state dict")
    state_dict = torch.load(best_model_path, map_location=device)

    model = MODELS[model_name].model
    model.load_state_dict(state_dict, strict=True)
    logger.info("Model weights loaded successfully")

    # ------------------------------------------------------------------
    # Build inference pipeline
    # ------------------------------------------------------------------
    pipeline = create_pipeline(model, eval=True)
    net: NeuralNetClassifier = pipeline.named_steps["classifier"]

    # Ensure strict inference mode
    net.module_.eval()
    net.train = False

    # ------------------------------------------------------------------
    # Load inference dataset and predict
    # ------------------------------------------------------------------
    logger.info("Loading test dataset")
    df_infer = set_test_dataset()

    logger.info(f"Running inference on {len(df_infer)} samples")
    y_infer = pipeline.predict(df_infer)
    logger.info(f"Number of predictions generated: {len(y_infer)}")

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    model_name = f"Masker_optimiser_{settings.general.text_embedding_task}_avg"
    who = f"-FL-{settings.general.data_split}-{settings.client.num_clients}clients"
    model_version = 1
    model_nickname = model_name + who
    filename = (
        f"pred_{data_version}_{settings.general.text_embedding_task}_"
        f"split={settings.general.data_split}_clients={settings.client.num_clients}_"
        f"model_version={model_version}.csv"
    )
    save_scoring_path = Path(f"datasets/{data_version}/scoring/{filename}")
    save_scoring_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(y_infer).to_csv(save_scoring_path, index=False)
    logger.info(f"Predictions saved to: {save_scoring_path.resolve()}")

    # ------------------------------------------------------------------
    # Prepare metadata for evaluation
    # ------------------------------------------------------------------
    branch, commit_hash = get_git_info()
    repo_commit_sha = commit_hash

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    if run_test_flag:
        run_test(
            save_scoring_path,
            model_nickname,
            model_version,
            data_version,
            repo_commit_sha,
            "masker_optimiser_eval",
            allow_override=allow_override,
        )
    else:
        logger.info("Skipping evaluation (run_test_flag is False)")

    logger.info("✅ Inference and evaluation completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained FL model.")
    parser.add_argument(
        "--run-folder",
        type=str,
        help="Path to the specific run folder containing the model.",
        default=None,
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="If true, run the run_test function after inference. Default is False.",
    )
    parser.add_argument(
        "--allow-override",
        action="store_true",
        help="If true, allow overriding existing evaluation results. Default is False.",
    )
    args = parser.parse_args()
    try:
        infer(run_folder=args.run_folder, run_test_flag=args.run_test, allow_override=args.allow_override)
    except (FileNotFoundError, NotADirectoryError) as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
