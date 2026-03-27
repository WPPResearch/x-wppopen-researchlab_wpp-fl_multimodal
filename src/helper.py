import json
import os
import re
import subprocess
import sys
from datetime import datetime
from functools import lru_cache

import numpy as np

import src.schemas
from src.auth import hash256

CACHE_DIR = "data/.apicache/"
EXPLANATION_PHRASINGS = "files/explanation_phrasings.txt"
POSITIVE_NEGATIVE_SWITCH = json.load(open("files/positive_negative_switch.json"))


def now_to_str(just_numbers=False):
    if just_numbers:
        return re.sub("[^0-9]", "", str(datetime.now()).split(".")[0])
    return str(datetime.now()).split(".")[0]


@lru_cache(maxsize=2000)
def load_prompt(prompt_loc="files/recommendation_prompt.txt", positive=True):
    """Loads the prompt and injects the values

    Parameters
    ----------
    prompt_loc : str, optional
        The location of the prompt txt file, by default "files/recommendation_prompt.txt"

    Returns
    -------
    str
        The prompt ready to be used
    """
    with open(prompt_loc, "r") as f:
        prompt = f.read()
    for attribute, values in src.schemas.ALLOWED_REQUEST_VALS.items():
        prompt = prompt.replace(f"<<{attribute}>>", str(values))
    for attribute, values in POSITIVE_NEGATIVE_SWITCH["positive" if positive else "negative"].items():
        prompt = prompt.replace(f"<<{attribute}>>", str(values))
    return prompt


def add_explanation(response: dict) -> dict:
    """Compiles the responses by replacing placeholders in the response text with actual values."""

    def score_to_proba(score):
        return int(45 + 40 * score) + int(str(score).split(".")[1][0])

    def random_proba(score):
        return int((score * 0.3) * 100) + int(str(score)[-1])

    for i, resp in enumerate(response):
        score = resp.get("score")

        resp["explanation"] = (
            f"This cluster is ranked #{i + 1} among the top recommended clusters."
            f"\nIt has a {score_to_proba(score)}% probability of leading to high performance."
            + (f"\nThe #1 cluster has a probability of " f"{score_to_proba(response[0]['score'])}%." if i != 0 else "")
            + f"\nA random cluster has a probability of {random_proba(response[0]['score'])}%."
        )

    return response


def add_coordinates(response: list[dict]) -> list[dict]:
    scores = [resp["score"] for resp in response]
    angles = (np.linspace(0, 2 * np.pi, len(scores), endpoint=False) + np.random.uniform(0, 2 * np.pi)) % (2 * np.pi)
    radii = 1 - np.array(scores)  # Higher scores closer to center
    x = (radii * np.cos(angles)).round(3)
    y = (radii * np.sin(angles)).round(3)

    coords = list(zip(x.tolist(), y.tolist()))
    for i, (x, y) in enumerate(coords):
        response[i]["coordinates_xy"] = (x, y)
    return response


def add_performance(response: list[dict]) -> list[dict]:
    """Adds performance value to the response based on the scores."""

    def score_to_performance(score):
        return int(35 + 60 * score + 2 * np.sin(score * int(str(score).split(".")[1][0])))

    for resp in response:
        score = resp.get("score")
        if score is not None:
            resp["performance"] = score_to_performance(score)

    return response


@lru_cache(maxsize=6000)
def read_cache(user_request) -> list[dict] | None:
    """Reads and parses JSON from the cache file if it exists."""
    hashed_user_request = hash256(user_request, times=1, n=20)
    cache_file = f"{CACHE_DIR}/{hashed_user_request}"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.loads(f.read())["response"]
    else:
        return None


def write_cache(user_request, response):
    """Serializes response as JSON and writes to the cache file."""
    hashed_user_request = hash256(json.dumps(user_request), times=1, n=20)
    cache_file = f"{CACHE_DIR}/{hashed_user_request}"
    os.makedirs(CACHE_DIR, exist_ok=True)
    content = {"request": user_request, "response": response}
    with open(cache_file, "w") as f:
        f.write(json.dumps(content))


def execute_cronjob():
    command = "python scripts/generate_response_examples.py"
    subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stdout)
