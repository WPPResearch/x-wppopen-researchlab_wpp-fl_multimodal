import ast
import base64
import copy
import json
import os
import re
import uuid
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig, HttpOptions
from google.oauth2 import service_account
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from skorch import NeuralNetClassifier

load_dotenv()


class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Get index of max probability
        max_idx = torch.argmax(x, dim=-1)
        # Convert to one-hot
        return F.one_hot(max_idx, num_classes=self.num_classes).float()


class GeneralisedJointModel_3cls_feat_drop(nn.Module):
    def __init__(
        self,
        input_dims,
        hidden_embed_dim=512,
        embed_dim=512,
        hidden_dim=256,
        dropout_prob=0.3,
        modality_dropout_prob=0.2,
    ):
        """
        input_dims: list of ints, each is the raw dimension of a modality
        embed_dim: dimension to project each modality into
        """
        super().__init__()

        # Create a projection layer for each modality
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_embed_dim),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                )
                for dim in input_dims
            ]
        )

        self.modality_dropout_prob = modality_dropout_prob

        # MLP takes concatenated embeddings from all modalities
        total_embed_dim = embed_dim * len(input_dims)
        self.mlp = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, raw_modalities_dict):
        """
        raw_modalities_dict: A dictionary where keys are modality names (strings)
                         and values are tensors, each shape [batch_size, input_dim_i].
        """
        projected = []
        num_modalities = len(self.projections)

        # Get batch size from the first modality (assuming all have the same batch size)
        batch_size = raw_modalities_dict[list(raw_modalities_dict.keys())[0]].shape[0]
        device = raw_modalities_dict[list(raw_modalities_dict.keys())[0]].device

        if self.training and self.modality_dropout_prob > 0:
            # Create a Bernoulli distribution for each (sample, modality) pair
            # Shape: (batch_size, num_modalities)
            modality_mask = (
                (torch.empty(batch_size, num_modalities).uniform_(0, 1) > self.modality_dropout_prob).float().to(device)
            )
        else:
            modality_mask = torch.ones(batch_size, num_modalities).to(device)

        for i, mod in enumerate(raw_modalities_dict.keys()):
            x = raw_modalities_dict[mod]
            proj = self.projections[i]

            embedded_modality = F.normalize(proj(x), dim=-1)  # Shape: (batch_size, embed_dim)

            # Apply modality dropout:
            # We need to expand modality_mask[:, i] to match embed_dim
            # (batch_size, 1) * (batch_size, embed_dim) -> (batch_size, embed_dim)
            embedded_modality = embedded_modality * modality_mask[:, i].unsqueeze(1)

            projected.append(embedded_modality)

        # Concatenate along feature dimension
        embeddings = torch.cat(projected, dim=-1)

        score = self.mlp(embeddings)
        return score


def collate_fn_skorch_generalised_3cls(batch, modalities):
    # Case 1: batch contains (features, label) tuples
    if all(isinstance(item, tuple) and len(item) == 2 for item in batch):
        inputs, labels = zip(*batch)
    else:
        # Case 2: batch contains only features
        inputs = batch
        labels = None

    # Combine batch items into a single dictionary
    batch_inputs = {mod: torch.stack([inp[i] for inp in inputs]) for i, mod in enumerate(modalities)}

    # Convert labels if present
    if labels is not None:
        try:
            labels = torch.tensor(labels, dtype=torch.long)
        except Exception:
            labels = None

    return {"raw_modalities_dict": batch_inputs}, labels


class NeuralNetClassifierWithProjected(NeuralNetClassifier):

    def predict(self, X, return_projected=False):
        self.module_.eval()
        with torch.no_grad():
            if isinstance(X, torch.utils.data.dataset.Subset):
                if hasattr(self, "iterator_valid__collate_fn"):
                    X, _ = self.iterator_valid__collate_fn(X)
                elif hasattr(self, "collate_fn"):
                    X, _ = self.collate_fn(X)
                else:
                    raise RuntimeError("No collate_fn found for prediction.")

                X = X["raw_modalities_dict"]
                # print("[DEBUG] Features type:", type(X))
            if return_projected:
                if not isinstance(X, dict):
                    X, _ = self.iterator_valid__collate_fn(X)
                score, projected = self.module_.forward(X["raw_modalities_dict"], return_projected=True)
                return score.cpu().numpy(), [p.cpu().numpy() for p in projected]
            else:
                y_pred = self.forward(X)
                y_pred_labels = y_pred.argmax(dim=1).cpu().numpy()

                return y_pred_labels


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """
    Extract relevant columns given a column pattern
    """

    def __init__(self, column_name_pattern, skip_encoding=False):
        self.column_name_pattern = column_name_pattern
        self.skip_encoding = skip_encoding

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if self.skip_encoding:
            return X[[c for c in X.columns if f"{self.column_name_pattern}_emb" in c]]
        else:
            return X[self.column_name_pattern]


class EmbeddingEncoder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        column_name_pattern: str,
        text_model="text-embedding-005",
        skip_encoding: bool = False,
        verbose: bool = False,
        encoding_dimension: int = 512,
    ):
        self.column_name_pattern = column_name_pattern
        self.text_model = text_model
        self.skip_encoding = skip_encoding
        self.verbose = verbose
        self._client = None
        self.encoding_dimension = encoding_dimension
        zcta_path = Path("data/zcta_embeddings.csv")
        if not zcta_path.exists():
            raise ValueError(
                f"Required file missing: {zcta_path.resolve()}. Please ensure it exists in the data directory."
            )
        self.pdfm_emb = pd.read_csv(zcta_path)

    def __getstate__(self):
        """
        Custom method to exclude non-serializable objects during pickling.
        """
        state = self.__dict__.copy()
        if "_client" in state:
            del state["_client"]  # Do NOT pickle the client instance
        return state

    def __setstate__(self, state):
        """
        Restore the object's state after unpickling.
        """
        self.__dict__.update(state)
        self._client = None  # Client will be None after loading, needs to be set via set_client()

    def set_client(self, client_instance):
        """Sets the GenAI client instance after the pipeline is loaded."""
        self._client = client_instance
        if self.verbose:
            print(f"Client set for EmbeddingEncoder for pattern: {self.column_name_pattern}")

    def fit(self, X, y=None):
        """
        Load the pre-trained text embedding model.
        """
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform text columns into embeddings.

        Parameters:
        -----------
        X : pd.DataFrame
            Input DataFrame containing text columns.

        Returns:
        --------
        np.ndarray
            Array of shape (n_samples, total_embedding_dim) containing concatenated embeddings.
        """

        if self.skip_encoding:
            return X

        if self.verbose:
            print("Encoder transforming data.")

        # Generate embeddings for the text data
        embeddings = []
        for val in X.to_list():
            if val:
                if self.column_name_pattern == "geo":
                    embedding = pdfm_encode(val, self.pdfm_emb)
                    self.encoding_dimension = 330
                else:
                    embedding = (
                        self._client.models.embed_content(
                            model=self.text_model,
                            contents=val,
                            config=EmbedContentConfig(
                                task_type="CLASSIFICATION",
                                output_dimensionality=self.encoding_dimension,
                            ),
                        )
                        .embeddings[0]
                        .values
                    )
                embeddings.append(embedding)
            else:
                embeddings.append(np.random.rand(self.encoding_dimension))

        column_names = [f"{X.name}_emb_{i:03}" for i in range(1, self.encoding_dimension + 1)]
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.columns = column_names
        return df_embeddings


def get_vertexai_client():
    """
    Authenticates with Google Vertex AI using a base64-encoded service account key from the environment variable
    'GCP_SA_BASE64_ENCODED_KEY' and returns an authenticated Vertex AI client.
    """
    gcp_sa_key_base64_str = os.getenv("GCP_SA_BASE64_ENCODED_KEY")
    gcp_sa_key_decoded_json_bytes = base64.b64decode(gcp_sa_key_base64_str.encode("utf-8"))
    gcp_sa_key_decoded_json_str = gcp_sa_key_decoded_json_bytes.decode("utf-8")
    gcp_sa_key_json = json.loads(gcp_sa_key_decoded_json_str)
    credentials = service_account.Credentials.from_service_account_info(
        gcp_sa_key_json, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    vertexai_client = genai.Client(http_options=HttpOptions(api_version="v1"), credentials=credentials)
    print("Vertex AI client authenticated successfully.")
    return vertexai_client


def pdfm_encode(zipcodes, emb_df):
    zipcodes = ast.literal_eval(zipcodes)
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


class TensorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name_pattern, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.column_name_pattern = column_name_pattern

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        # Select columns matching the pattern and extract them as a NumPy array
        pattern = rf"^{self.column_name_pattern}_emb_\d+$"
        selected_columns = [col for col in X.columns if re.match(pattern, col)]
        if selected_columns:
            emb_array = X.loc[:, selected_columns].fillna(0).to_numpy(dtype=np.float32)

        else:
            # if emb is provided means the column already contains all the embedding
            single_col = f"{self.column_name_pattern}_emb"
            if single_col not in X.columns:
                raise ValueError(f"No embedding columns found for pattern '{self.column_name_pattern}'")
            emb_array = np.stack(X[single_col].fillna(0).apply(lambda x: np.array(x, dtype=np.float32)).values)

        emb_array = np.array(emb_array, dtype=np.float32, copy=True)
        emb_tensor = torch.from_numpy(emb_array).to(self.device)
        encoded_dict = dict(zip(X.index, emb_tensor))
        X[f"{self.column_name_pattern}_enc"] = X.index.map(lambda x: encoded_dict[x])

        if selected_columns:
            X.drop(selected_columns, axis=1, inplace=True)
        else:
            X.drop(single_col, axis=1, inplace=True)

        return X


def write_json(obj, filepath: Union[str, Path], indent=4):
    json.dump(obj, open(filepath, "w"), indent=indent)


def read_json(filepath: Union[str, Path]):
    return json.load(open(filepath, "r"))


class GenAIPipeline(Pipeline):
    """
    A custom pipeline class that wraps sklearn.pipeline.Pipeline
    and provides a method to inject an already initialized GenAI client.
    """

    def __init__(self, steps, model_dir):
        super().__init__(steps)
        # Find the skorch step and store its name for easy access
        self._skorch_step_name = None
        self.model_dir = model_dir
        for name, step in self.steps:
            if isinstance(step, NeuralNetClassifier):
                self._skorch_step_name = name
                self.state_dict = step.module.state_dict()
                break
        self._pipeline_uuid = str(uuid.uuid4())

    def setup_genai_clients(self, client_instance):
        """
        Injects the provided GenAI client instance into all EmbeddingEncoder steps
        within the pipeline.
        """
        # Traverse the pipeline to find and set clients in EmbeddingEncoder steps
        feature_union_step = self.named_steps.get("features")
        if feature_union_step and isinstance(feature_union_step, FeatureUnion):
            for _, sub_pipeline in feature_union_step.transformer_list:
                # Each sub_pipeline is a Pipeline object (from create_text_pipeline)
                embedding_encoder_step = sub_pipeline.named_steps.get("embedding_encoder")
                if embedding_encoder_step and isinstance(embedding_encoder_step, EmbeddingEncoder):
                    embedding_encoder_step.set_client(client_instance)
        else:
            print("Warning: No 'features' step or it's not a FeatureUnion found in the pipeline.")

        print("All GenAI clients in pipeline initialized.")

    def to(self, device):
        """
        Moves all PyTorch modules within the pipeline to the specified device.
        This is useful if you want to change the device after initial loading.
        """
        classifier_step = self.named_steps.get("classifier")
        if classifier_step and isinstance(classifier_step, NeuralNetClassifier):
            if hasattr(classifier_step, "module_") and classifier_step.module_ is not None:
                classifier_step.module_.to(device)
                print(f"Classifier module moved to {device}")
            else:
                print("Warning: Classifier module not initialized, cannot move.")

            if hasattr(classifier_step, "criterion_") and classifier_step.criterion_ is not None:
                # Only move if it's a torch.nn.Module (e.g., CrossEntropyLoss)
                if isinstance(classifier_step.criterion_, torch.nn.Module):
                    classifier_step.criterion_.to(device)
                    print(f"Classifier criterion moved to {device}")
            else:
                print("Warning: Classifier criterion not initialized, cannot move.")

            # Update skorch's internal device tracking
            classifier_step.set_params(device=device)
        else:
            print("Warning: No 'classifier' step or it's not a NeuralNetClassifier found in the pipeline.")
        return self

    def save(self, joblib_filepath):
        """
        Saves the pipeline to joblib_filepath, and its skorch model weights
        to a .pt file in the same directory.
        """
        # Determine the directory where the .joblib and .pt files will reside
        save_dir = os.path.dirname(joblib_filepath)
        if not save_dir:  # If joblib_filepath is just a filename, dirname is empty
            save_dir = "."  # Use current directory
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        # Temporarily store the save_dir so __getstate__ can use it
        self._temp_save_dir_for_getstate = save_dir

        joblib.dump(self, joblib_filepath)

        # Clean up the temporary attribute after saving
        del self._temp_save_dir_for_getstate
        print(f"Pipeline and skorch weights saved to {joblib_filepath} and its directory.")

    def __getstate__(self):
        skorch_net = self.named_steps.get(self._skorch_step_name)

        if not skorch_net or not hasattr(skorch_net, "module_") or skorch_net.module_ is None:
            print("No initialized skorch classifier found for custom serialization. Using default.")
            state = super().__getstate__()
            state["_pytorch_model_params_path"] = None
            return state

        # Get the save_dir from the temporary attribute set by the .save() method
        save_dir = getattr(self, "_temp_save_dir_for_getstate", os.getcwd())
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

        pytorch_model_params_filename = f"data/{self.model_dir}/skorch_params_{self._pipeline_uuid}.pt"
        # Store the ABSOLUTE path to ensure it's found regardless of CWD at load time
        absolute_pytorch_model_params_path = os.path.join(save_dir, pytorch_model_params_filename)

        skorch_net.save_params(f_params=absolute_pytorch_model_params_path)
        print(f"Saved skorch model params to {absolute_pytorch_model_params_path}")

        skorch_net_for_serialization = copy.deepcopy(skorch_net)

        if hasattr(skorch_net_for_serialization, "module_"):
            skorch_net_for_serialization.module_ = None
        if hasattr(skorch_net_for_serialization, "criterion_") and isinstance(
            skorch_net_for_serialization.criterion_, torch.nn.Module
        ):
            skorch_net_for_serialization.criterion_ = None

        base_state = super().__getstate__()

        new_steps = []
        for name, step in base_state["steps"]:
            if name == self._skorch_step_name:
                new_steps.append((name, skorch_net_for_serialization))
            else:
                new_steps.append((name, step))
        base_state["steps"] = new_steps

        # Store the ABSOLUTE path in the pipeline's state
        base_state["_pytorch_model_params_path"] = absolute_pytorch_model_params_path
        base_state["_pipeline_uuid"] = self._pipeline_uuid

        return base_state

    def __setstate__(self, state):
        # 1. Determine the current inference device
        INFERENCE_DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        print(f"Deserializing pipeline for inference on device: {INFERENCE_DEVICE}")

        # 2. Restore the basic pipeline state
        super().__setstate__(state)

        # Restore the UUID
        self._pipeline_uuid = state.get("_pipeline_uuid", str(uuid.uuid4()))

        # Retrieve the skorch step name
        self._skorch_step_name = None
        for name, step in self.steps:
            if isinstance(step, NeuralNetClassifier):
                self._skorch_step_name = name
                break

        pytorch_model_params_path = state.get("_pytorch_model_params_path")

        if not self._skorch_step_name or not pytorch_model_params_path:
            print("No skorch classifier or parameter path found for custom deserialization. Using default.")
            return

        # Check if the file exists using the absolute path stored in the state
        if not os.path.exists(pytorch_model_params_path):
            print(
                f"Error: Skorch model parameters file not found at {pytorch_model_params_path}. "
                "This is critical. Ensure the .pt file was saved correctly and is accessible."
            )
            # IMPORTANT: Return here if the file is genuinely missing.
            # This prevents subsequent AttributeErrors.
            return

        # 3. Get the skorch NeuralNetClassifier instance (which currently has module_=None)
        skorch_net = self.named_steps[self._skorch_step_name]

        # --- CRITICAL FIX: Initialize module and criterion BEFORE loading params ---
        skorch_net.initialize_module()
        skorch_net.initialize_criterion()
        print("Skorch module and criterion re-initialized.")

        # Set the device on the skorch_net. skorch's load_params will then use this device.
        skorch_net.set_params(device=INFERENCE_DEVICE)
        print(f"Set skorch_net device to {INFERENCE_DEVICE} before loading params.")

        # 4. Load the parameters using the stored path.
        print(f"Loading skorch model params from {pytorch_model_params_path}")
        skorch_net.load_params(f_params=pytorch_model_params_path)

        # Set to evaluation mode
        if hasattr(skorch_net, "module_") and skorch_net.module_ is not None:
            skorch_net.module_.eval()
        else:
            print("Warning: skorch module_ was not initialized after loading params.")
