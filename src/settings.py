import base64
import json
import os
from pathlib import Path
from typing import Optional

import torch
from google import genai
from google.genai.types import HttpOptions
from google.oauth2 import service_account
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    computed_field,
    field_validator,
)
from yaml import safe_load


class Server(BaseModel):
    strategy: str
    fraction_fit: float
    fraction_eval: float
    num_rounds: int
    server_dataset_included: bool

    @field_validator("fraction_fit", "fraction_eval")
    def validate_percentages(cls, value, info):
        """
        Validate individual percentage fields are between 0.0 and 1.0.
        :param value: Field validator
        :param info: Instance of Server class
        :return: Validated fields are between 0.0 and 1.0 or raises exception.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Under server configuration: {info.field_name} must be between 0.0 and 1.0. Got {value}")
        return value

    @field_validator("num_rounds")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Server class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under server configuration: {info.field_name} must be positive. Got {value}")
        return value

    @field_validator("strategy")
    def validate_server_strategy(cls, value: str, info: ValidationInfo):
        """
        Validate strategy type is Mean (can be added more in the future)
        :param value: Field validator
        :param info: Instance of Strategy type
        :return: Validated attack type or raises exception.
        """
        strategy_types = ["Mean"]
        if value not in strategy_types:
            raise ValueError(f"Under server configuration: {info.field_name} must be in {strategy_types}. Got {value}")
        return value


class Client(BaseModel):
    num_clients: int

    @field_validator("num_clients")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Client class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under client configuration: {info.field_name} must be positive. Got {value}")
        return value


class Model(BaseModel):
    name: str
    hidden_embed_dim: int
    embed_dim: int
    hidden_dim: int
    dropout_prob: float
    modality_dropout_prob: float
    local_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    optuna_n_trials: int = 1

    @field_validator(
        "hidden_embed_dim",
        "embed_dim",
        "hidden_dim",
        "dropout_prob",
        "modality_dropout_prob",
        "local_epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "optuna_n_trials",
    )
    def validate_non_negative(cls, value, info):
        """
        Validate individual fields are non-negative.
        :param value: Field validator
        :param info: Instance of Model class
        :return: Validated fields are non-negative values or raises exception.
        """
        if value < 0:
            raise ValueError(f"Under model configuration: {info.field_name} must be non-negative. Got {value}")
        return value


class General(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    use_wandb: bool
    random_seed: int
    text_embedding_task: str
    genai_client: genai.Client
    data_split: str
    alpha: float = 1.0
    one_hot_modalities: Optional[list[str]] = None
    data_version: str
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target: str = "label"
    modalities: list[str] = ["audience", "brand", "creative", "platform", "geo"]
    dims: list[int] = [512, 512, 512, 512, 330]

    @computed_field
    @property
    def model_dir(self) -> str:
        model_name = f"Masker_optimiser_{self.text_embedding_task}_avg"
        model_version = 1
        return f"{model_name}_v{model_version}_d{self.data_version}"

    @property
    def genai_client(self) -> genai.Client:
        if self.genai_client is not None:
            return self.genai_client

        gcp_sa_key_base64_str = os.getenv("GCP_SA_BASE64_ENCODED_KEY")
        if not gcp_sa_key_base64_str:
            raise RuntimeError("GCP_SA_BASE64_ENCODED_KEY is not set")

        gcp_sa_key_decoded_json = json.loads(base64.b64decode(gcp_sa_key_base64_str).decode("utf-8"))

        credentials = service_account.Credentials.from_service_account_info(
            gcp_sa_key_decoded_json,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        self.genai_client = genai.Client(
            credentials=credentials,
            http_options=HttpOptions(api_version="v1"),
        )

        return self.genai_client

    @genai_client.setter
    def genai_client(self, value):
        self.genai_client = value

    @field_validator("text_embedding_task")
    def validate_text_embedding_task(cls, value: str, info: ValidationInfo):
        """
        Validate text embedding task type is either classification or clustering
        :param value: Field validator
        :param info: Instance of text embedding task type
        :return: Validated text embedding task type or raises exception.
        """
        text_embedding_tasks = ["classification", "clustering"]
        if value not in text_embedding_tasks:
            raise ValueError(
                f"Under server configuration: {info.field_name} must be in {text_embedding_tasks}." f"Got {value}"
            )
        return value

    @field_validator("data_split")
    def validate_data_split(cls, value: str, info: ValidationInfo):
        """
        Validate data split type is heterogeneous or homogeneous
        :param value: Field validator
        :param info: Instance of data split type
        :return: Validated attack type or raises exception.
        """
        data_split_types = ["heterogeneous", "homogeneous"]
        if value not in data_split_types:
            raise ValueError(
                f"Under general configuration: {info.field_name} must be in {data_split_types}." f"Got {value}"
            )
        return value

    @field_validator("alpha")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of General class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under general configuration: {info.field_name} must be positive. Got {value}")
        return value

    @field_validator("one_hot_modalities")
    def validate_one_hot_modalities(cls, value):
        if value is None:
            return []

        if not all(isinstance(v, str) for v in value):
            raise ValueError("One hot modalities must be a list of strings")

        return value


class Backend(BaseModel):
    client_resources: dict[str, float]


class Config(BaseModel):
    server: Server
    client: Client
    model: Model
    general: General
    backend: Backend
    config_path: Path

    def __init__(self, config_path: Path) -> None:
        if config_path.is_file():
            with open(config_path) as f:
                config = safe_load(f)
            config.update({"config_path": config_path})
            super().__init__(**config)
        else:
            raise FileNotFoundError("Error: yaml config file not found.")


PROJECT_NAME = "Multimodal FL Project"
FOLDER_DIR = Path(__file__).parent.parent
config_file = FOLDER_DIR / "config.yaml"
settings = Config(config_file)
