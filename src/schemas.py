import json
from typing import List

from pydantic import BaseModel, Field, model_validator

REQUEST_SCHEMA = {
    "brand": "",
    "subbrand": "",
    "product": "",
    "geo": [],
    "audience": [],
    "platform": "",
    "spend": "",
    "creative": "",
}

REQUIRED_REQUEST_VALUES = ["objective", "brand"]

with open("files/allowed_request_vals.json", "r") as f:
    ALLOWED_REQUEST_VALS = json.load(f)

with open("files/request_examples.json", "r") as f:
    request_examples = json.load(f)
    INTENT_SIGNALS_REQUEST_EXAMPLES = request_examples["INTENT_SIGNALS_REQUEST"]
    MAP_AUDIENCE_REQUEST_EXAMPLES = request_examples["MAP_AUDIENCES_REQUEST"]


class IntentSignals(BaseModel):
    objective: str = Field(..., alias="objective")  # required
    brand: str = Field(..., alias="brand")  # required
    subbrand: str = Field(default="", alias="subbrand")  # optional
    product: str = Field(default="", alias="product")  # optional
    geo: List[str] = Field(default=[], alias="geo")  # optional
    audience: List[str] = Field(default=[], alias="audience")  # optional
    platform: str = Field(default="", alias="platform")  # optional
    spend: str = Field(default="", alias="spend")  # optional
    creative: str = Field(default="", alias="creative")  # optional
    n_examples: int = Field(default=3, ge=1, le=6, alias="n_examples")  # required

    @model_validator(mode="before")
    def check(self) -> "IntentSignals":
        # Only validate objective and brand
        for required in REQUIRED_REQUEST_VALUES:
            value = self.get(required)
            allowed = ALLOWED_REQUEST_VALS.get(required, [])
            if not value or value not in allowed:
                raise ValueError(f"{required} is required and must be one of: {allowed}")
        return self


# Adding this schema for micro level validation of the Gemini's response to retry if it fails
class LLMRecommendation(BaseModel):
    name: str
    brand: str
    subbrand: str
    product: str
    geo: List[str]
    audience: List[str]
    platform: str
    spend: str
    creative: str
    score: float
    estimated_reach: int

    @model_validator(mode="before")
    def check(self) -> "LLMRecommendation":
        value = self.get("brand")
        allowed = ALLOWED_REQUEST_VALS.get("brand", [])
        if not value or value not in allowed:
            raise ValueError(f"brand is required and must be one of: {allowed}")
        return self


class AudienceCluster(BaseModel):
    objective: str
    name: str
    brand: str
    subbrand: str
    product: str
    geo: List[str]
    audience: List[str]
    platform: str
    spend: str
    creative: str
    estimated_reach: int
    score: float
    performance: int
    explanation: str
    coordinates_xy: tuple[float, float]


class PlatformSegment(BaseModel):
    title: str
    description: str
    filters: dict
    estimated_reach: int
    platform: str
