import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import click
import numpy as np
import pandas as pd
import polars as pl
from google.api_core import exceptions as google_exceptions
from tqdm import tqdm
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from src.llm import get_vertexai_client

ZCTA_PATH = Path("data/zcta_embeddings.csv")
if not ZCTA_PATH.exists():
    raise ValueError(f"Required file missing: {ZCTA_PATH.resolve()}. Please ensure it exists in the data directory.")

PDFM_EMB = pd.read_csv(ZCTA_PATH)
BATCH_SIZE = 50000
EMBED_COLS = ["audience", "brand", "creative", "platform", "geo"]
GEO_PDFM = True

TEXT_EMBEDDING_MODEL = "text-embedding-005"
TEXT_EMBEDDING_DIMENSIONALITY = 512
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#task_type
# TEXT_EMBEDDING_TASK = "CLUSTERING"
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#task_type
TEXT_EMBEDDING_TASK = "CLASSIFICATION"
GEN_MODEL = "gemini-2.5-flash-lite-preview-06-17"
MAX_RETRIES = 2
GEMINI_WORKERS = 1
NUM_MAPPED_AUDIENCES = 4


# read data
def read_dataset(file_path: Path, input_name: str):
    if ".parquet" in file_path.suffix:
        df = pd.read_parquet(str(file_path))
    elif ".csv" in file_path.suffix:
        df = pd.read_csv(str(file_path))
    else:
        with open(file_path, "r") as f:
            content = f.read()

        # Robustly parse JSON objects (handles standard JSON, JSONL, and concatenated JSON)
        data = []
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            # Skip leading whitespace
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break

            try:
                obj, next_pos = decoder.raw_decode(content, pos)
                if isinstance(obj, list):
                    data.extend(obj)
                else:
                    data.append(obj)
                pos = next_pos
            except json.JSONDecodeError as e:
                # If it's a small file or the first object fails, it might be a standard error
                if not data:
                    raise e
                # Otherwise, try to skip a line or character if possible, or just break
                print(f"Warning: Stopped parsing {file_path} due to error at char {pos}: {e}")
                break

        df = pd.DataFrame(data)

    # normalize data
    df = df.rename(columns={"par_audience": "audience", "par_brand": "brand", "par_creative": "creative"})
    if "label" in df.columns:
        df["label"] = df["label"].map({"neg": 0, "avg": 1, "pos": 2}).astype(int)

    list_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()]
    for c in list_cols:
        # Prevent TypeError by checking if x is a list before calling len()
        if df[c].apply(lambda x: len(x) if isinstance(x, list) else 0).max() == 1:
            df[c] = df[c].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) == 1 else (np.nan if isinstance(x, list) else x)
            )

    # drop rows
    df.dropna(inplace=True)

    for col in EMBED_COLS:
        if col in df.columns and col != "geo":
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)

    df.to_csv(f"data/{input_name}_temp.csv", index=False)
    return df


def pdfm_encode(zipcodes, emb_df):
    if type(zipcodes) is str:
        zipcodes = [zipcodes]
    elif type(zipcodes) is not list:
        return None

    filtered_embs = (
        emb_df.loc[emb_df["place"].isin(zipcodes), emb_df.columns[emb_df.columns.str.startswith("feature")]]
        .mean()
        .to_list()
    )

    if len(filtered_embs) != 330:
        print("Length mismatch")
        return None

    return filtered_embs


def embed_texts(
    texts, model_obj, dimensionality=TEXT_EMBEDDING_DIMENSIONALITY, task=TEXT_EMBEDDING_TASK
) -> list[list[float]]:
    """Embeds texts with the text-embedding-005 model and the vertexai SDK. Maximum of 250 texts per request.
    Returns:
        A list of lists containing the embedding vectors for each input text
    """
    inputs = [TextEmbeddingInput(text, task) for text in texts]

    # Vertex AI has rate limits, adding retries with exponential backoff
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            embeddings = model_obj.get_embeddings(inputs, output_dimensionality=dimensionality)
            return [embedding.values for embedding in embeddings]
        except (
            google_exceptions.TooManyRequests,
            google_exceptions.ServiceUnavailable,
            google_exceptions.InternalServerError,
        ) as e:
            if attempt == max_attempts - 1:
                raise e
            wait_time = (2**attempt) + random.uniform(0, 1)
            print(
                f"\nTransient error or rate limit ({type(e).__name__}). "
                f"Retrying in {wait_time:.2f}s... (Attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(wait_time)
        except Exception as e:
            print(f"\nAn unexpected error occurred during embedding: {e}")
            raise e

    raise RuntimeError("Unexpected fall-through in embed_texts retry loop")


def embed_texts_batch(texts, batch_size=200, max_workers=2):
    _ = get_vertexai_client()
    model_obj = TextEmbeddingModel.from_pretrained(TEXT_EMBEDDING_MODEL)
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
        for batch_result in tqdm(
            executor.map(lambda b: embed_texts(b, model_obj=model_obj), batches),
            total=len(batches),
            desc="Embedding batches",
        ):
            results.extend(batch_result)
    return results


def encode(df, version, partition_case, type: Literal["client", "server"], input_name: str, client_id=None):
    # Determine output path first to check for existence
    match type:
        case "client":
            output_path = Path(
                f"data/{version}/{partition_case}/client/encoded/"
                f"{input_name}_emb_pdfm_{TEXT_EMBEDDING_TASK.lower()}.parquet"
            )
        case "server":
            output_path = Path(
                f"data/{version}/{partition_case}/server/encoded/"
                f"{input_name}_emb_pdfm_{TEXT_EMBEDDING_TASK.lower()}.parquet"
            )
        case _:
            raise ValueError(f"Unknown type {type}. Must be either 'client' or 'server'")

    if output_path.exists():
        print(f"Skipping {input_name}: Output file {output_path} already exists.")
        return

    encoded_cols = []

    # embed texts per modality
    for col in EMBED_COLS:
        print(f"Embedding {col}")
        # keep unique values and remove nulls
        col_df = df[col].dropna().drop_duplicates().reset_index(drop=True).tolist()

        encoded_cols.append(col)
        emb_vals = []
        if col == "geo" and GEO_PDFM:
            emb_vals = [pdfm_encode(v, PDFM_EMB) for v in col_df]
        else:
            for batch in range(0, len(col_df), BATCH_SIZE):
                emb_vals.extend(embed_texts_batch(col_df[batch : batch + BATCH_SIZE]))
                time.sleep(2)

        # set column names and save
        emb_df = pd.DataFrame(emb_vals, columns=[f"{col}_emb_{str(i).rjust(3, '0')}" for i in range(len(emb_vals[0]))])
        emb_df[col] = col_df
        emb_df.to_csv(f"data/{input_name}_{col}_embs.csv", index=False)

    del df, emb_df, emb_vals

    # read dataset again
    df = pl.read_csv(f"data/{input_name}_temp.csv")

    # read encoded values
    all_dfs = {col: pl.read_csv(f"data/{input_name}_{col}_embs.csv") for col in encoded_cols}
    # merge df with encoded values
    for col, col_df in all_dfs.items():
        df = df.join(col_df, on=col, how="left")

    # save final dataset
    df.write_parquet(output_path)

    # delete every temporal file
    for pattern in (f"{input_name}*embs.csv", f"{input_name}*temp.csv"):
        for file in Path("data").glob(pattern):
            file.unlink()


@click.command()
@click.option("--version", required=True, help='Subfolder under datasets/, e.g. "V16"')
@click.option("--num_clients", default=10, show_default=True)
@click.option("--split", default="homogeneous", show_default=True, help="homogeneous|heterogeneous")
@click.option("--alpha", default=1.0, show_default=True, help="Dirichlet alpha (heterogeneous only)")
def main(version: str, num_clients: int, split: str, alpha: float):
    match split:
        case "homogeneous" | "synthetic":
            partition_case = f"split={split}_clients={num_clients}"
        case "heterogeneous":
            partition_case = f"split={split}_clients={num_clients}_alpha={alpha}"
        case "_":
            raise ValueError(f"Unknown split {split}. Must be either 'homogeneous' or 'heterogeneous'")

    # Create (if not exists) encoded folders for clients and server
    for type in ["server", "client"]:
        encoded_path = Path(f"data/{version}/{partition_case}/{type}/encoded")
        encoded_path.mkdir(parents=True, exist_ok=True)

    # Create the encoded client datasets
    for client_file_path in sorted(Path(f"data/{version}/{partition_case}/client/raw").glob("client_*_train.json*")):
        client_id_match = re.search(r"client_(\d+)", client_file_path.name)
        client_id = int(client_id_match.group(1)) if client_id_match else None
        print(f"Encoding id: client {client_id}")
        df = read_dataset(client_file_path, client_file_path.stem)
        encode(df, version, partition_case, "client", client_file_path.stem, client_id)

    # Create the encoded server datasets
    for server_file_path in sorted(Path(f"data/{version}/{partition_case}/server/raw").glob("server_*.json*")):
        print(f"Encoding server: {server_file_path.name}")
        df = read_dataset(server_file_path, server_file_path.stem)
        encode(df, version, partition_case, "server", server_file_path.stem)


if __name__ == "__main__":
    main()
