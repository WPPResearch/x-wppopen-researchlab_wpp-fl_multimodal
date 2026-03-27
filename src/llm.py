import base64
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial

import vertexai
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.oauth2 import service_account
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from src.helper import load_prompt, now_to_str, read_cache, write_cache
from src.schemas import IntentSignals, LLMRecommendation, PlatformSegment

TEXT_EMBEDDING_MODEL = "text-embedding-005"
TEXT_EMBEDDING_DIMENSIONALITY = 512
TEXT_EMBEDDING_TASK = (
    "CLUSTERING"  # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#task_type
)
GEN_MODEL = "gemini-2.5-flash-lite-preview-09-2025"
CONTENT_CONFIG = types.GenerateContentConfig(
    temperature=1.0,
    response_mime_type="application/json",
)
MAX_RETRIES = 2
NUM_RECOMMENDATIONS = IntentSignals.model_json_schema()["properties"]["n_examples"]["maximum"]
GEMINI_WORKERS = 5
NUM_MAPPED_AUDIENCES = 4

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


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
    gcp_sa_key_base64_str = os.getenv("GCP_SA_BASE64_ENCODED_KEY")
    gcp_sa_key_decoded_json_bytes = base64.b64decode(gcp_sa_key_base64_str.encode("utf-8"))
    gcp_sa_key_decoded_json_str = gcp_sa_key_decoded_json_bytes.decode("utf-8")
    gcp_sa_key_json = json.loads(gcp_sa_key_decoded_json_str)
    credentials = service_account.Credentials.from_service_account_info(gcp_sa_key_json, scopes=SCOPES)
    vertexai.init(credentials=credentials)  # Used for TextEmbeddingModel
    vertexai_client = genai.Client(http_options=HttpOptions(api_version="v1"), credentials=credentials)
    print(f"{now_to_str()} - Vertex AI client authenticated successfully.")
    return vertexai_client


def embed_texts(
    texts, model=TEXT_EMBEDDING_MODEL, dimensionality=TEXT_EMBEDDING_DIMENSIONALITY, task=TEXT_EMBEDDING_TASK
) -> list[list[float]]:
    """Embeds texts with the text-embedding-005 model and the vertexai SDK. Maximum of 250 texts per request.
    Returns:
        A list of lists containing the embedding vectors for each input text
    """
    _ = get_vertexai_client()
    model = TextEmbeddingModel.from_pretrained(model)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    embeddings = model.get_embeddings(inputs, output_dimensionality=dimensionality)
    return [embedding.values for embedding in embeddings]


def llm_response(prompt: str, task="recommendation"):
    """Generates a response from the Vertex AI model using the provided prompt."""
    vertexai_client = get_vertexai_client()
    for attempt in range(MAX_RETRIES):
        try:
            response = vertexai_client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config=CONTENT_CONFIG,
            )
            result = json.loads(re.sub("(```)(json)*(.+)(```)", "\\3", response.text, flags=re.DOTALL))
            # Validating the response against the appropriate pydantic schema
            if task == "recommendation":
                [LLMRecommendation.model_validate(cluster) for cluster in result]
            elif task == "audience_mapping":
                [PlatformSegment.model_validate(cluster) for cluster in result]
            return result
        except Exception as e:
            if "429" in str(e):
                print(f"{now_to_str()} - Resource exhausted error, retrying.. (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(0.2 * (2**attempt))
            else:
                print(f"{now_to_str()} - Error: {str(e)}. Retrying.. (attempt {attempt + 1}/{MAX_RETRIES})")
    raise RuntimeError(f"{now_to_str()} - Max retries exceeded for Vertex AI generate_content")


@lru_cache(maxsize=6000)
def get_recommendation(user_request: str) -> list[dict]:
    """
    Returns a response of a vertex_ai client. Uses lru_cache for speed.
    """
    prompt = load_prompt(prompt_loc="files/recommendation_prompt.txt")
    user_request = {**json.loads(user_request), "n_examples": int(NUM_RECOMMENDATIONS / GEMINI_WORKERS)}
    prompt += "\n" + json.dumps(user_request)

    try:
        cached_response = read_cache(json.dumps(user_request))
        if cached_response:
            print(f"{now_to_str()} - CacheAcross_Hit. Returning cached response.")
            return cached_response
    except Exception as e:
        print(f"{now_to_str()} - Error reading from cache: {e}")

    if GEMINI_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=GEMINI_WORKERS) as executor:
            prompts = [prompt] * GEMINI_WORKERS
            responses = list(executor.map(llm_response, prompts))
            response = sorted([d for r in responses for d in r], key=lambda x: x["score"], reverse=True)
    else:
        responses = llm_response(prompt)
        response = sorted([r for r in responses], key=lambda x: x["score"], reverse=True)

    write_cache(user_request, response)
    print(f"{now_to_str()} - Storing response to disk cache.")
    return response


@lru_cache(maxsize=6000)
def map_audience(user_request: str) -> list[dict]:
    """
    Returns a response of a vertex_ai client. Uses lru_cache for speed.
    """
    prompt = load_prompt(prompt_loc="files/audience_mapping_prompt.txt")
    user_request = json.loads(user_request)
    user_request_ = {k: v for k, v in user_request.items() if k in ["audience", "platform", "estimated_reach"]}
    user_request_["n_examples"] = NUM_MAPPED_AUDIENCES
    prompt += "\n" + json.dumps(user_request_)

    try:
        cached_response = read_cache(json.dumps(user_request_))
        if cached_response:
            print(f"{now_to_str()} - CacheAcross_Hit. Returning cached response.")
            return cached_response
    except Exception as e:
        print(f"{now_to_str()} - Error reading from cache: {e}")

    responses = llm_response(prompt, task="audience_mapping")
    write_cache(user_request_, responses)
    print(f"{now_to_str()} - Storing response to disk cache.")
    return responses


# @lru_cache(maxsize=6000)
def get_representative_desc(par_list: list) -> list[dict]:
    """
    Returns a response of a vertex_ai client. Uses lru_cache for speed.
    """
    prompts = []
    prompt = load_prompt(prompt_loc="files/representative_prompt.txt")
    for par in par_list:
        prompts.append(prompt + "\n" + str(par))

    # try:
    #     cached_response = read_cache(par_list)
    #     if cached_response:
    #         print(f"{now_to_str()} - CacheAcross_Hit. Returning cached response.")
    #         return cached_response
    # except Exception as e:
    #     print(f"{now_to_str()} - Error reading from cache: {e}")

    if GEMINI_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=GEMINI_WORKERS) as executor:
            llm_response_with_task = partial(llm_response, task="rep")
            responses = list(executor.map(llm_response_with_task, prompts))

    else:
        responses = llm_response(prompt, task="rep")

    write_cache(par_list, responses)
    print(f"{now_to_str()} - Storing response to disk cache.")
    return responses
