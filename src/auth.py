import hashlib

from fastapi import HTTPException, Security, status
from fastapi.security import api_key


def hash256(string, times=2, n=12):
    """Creates sha256 hashes"""
    s = string
    for i in range(times):
        s = hashlib.sha256(s.encode()).hexdigest()[::-1]
    return s[:n]


def load_users(location="data/.USERS"):
    """Function that loads users"""
    with open(location, "r") as f:
        codes = [u for u in f.read().split("\n") if u]
    return codes


api_key_header = api_key.APIKeyHeader(name="X-API-KEY")


async def validate_api_key(key: str = Security(api_key_header)):
    h = hash256(key)
    if h not in load_users() + load_users(location="data/.ADMIN"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized - API Key is wrong or doesn't possess the right privilleges",
        )
    return None


async def validate_admin_api_key(key: str = Security(api_key_header)):
    h = hash256(key)
    if h not in load_users(location="data/.ADMIN"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized - API Key is wrong or doesn't possess the right privilleges",
        )
    return None
