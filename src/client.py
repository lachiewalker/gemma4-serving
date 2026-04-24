import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL_ID = os.getenv("MODEL_ID", "RedHatAI/gemma-4-31B-it-FP8-block")
BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")


def get_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)
