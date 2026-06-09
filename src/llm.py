import base64
import json
import os
import warnings

import vertexai
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions
from google.oauth2 import service_account

from src.utils import now_to_str

load_dotenv()


def get_vertexai_client():
    """
    Authenticates with Google Vertex AI using a base64-encoded service account key from
    the environment variable 'GCP_SA_BASE64_ENCODED_KEY' and returns an authenticated Vertex AI client.

    Returns:
        genai.Client: An authenticated Vertex AI client instance.

    Example:
        vertexai_client = get_vertexai_client()
        response = vertexai_client.models.generate_content(
            model="gemini-2.0-flash-lite-001",
            contents="Explain how AI works in a few words"
        )
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="vertexai.*")
    gcp_sa_key_base64_str = os.getenv("GCP_SA_BASE64_ENCODED_KEY")
    if not gcp_sa_key_base64_str:
        raise ValueError("GCP_SA_BASE64_ENCODED_KEY environment variable is not set. Please set it in your .env file.")
    scopes = os.getenv("SCOPES")
    gcp_sa_key_decoded_json_bytes = base64.b64decode(gcp_sa_key_base64_str.encode("utf-8"))
    gcp_sa_key_decoded_json_str = gcp_sa_key_decoded_json_bytes.decode("utf-8")
    gcp_sa_key_json = json.loads(gcp_sa_key_decoded_json_str)
    credentials = service_account.Credentials.from_service_account_info(gcp_sa_key_json, scopes=[scopes])
    vertexai.init(credentials=credentials)  # Used for TextEmbeddingModel
    vertexai_client = genai.Client(http_options=HttpOptions(api_version="v1"), credentials=credentials)
    print(f"{now_to_str()} - Vertex AI client authenticated successfully.")
    return vertexai_client
