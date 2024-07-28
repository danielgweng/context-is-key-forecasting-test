"""
Configuration variables for the benchmark

"""

import os

# Model weight storage
MODEL_STORAGE_PATH = os.environ.get("STARCASTER_MODEL_STORE", "./models")
if not os.path.exists(MODEL_STORAGE_PATH):
    os.makedirs(MODEL_STORAGE_PATH)

# Evaluation configuration
DEFAULT_N_SAMPLES = 50
RESULT_CACHE_PATH = os.environ.get("STARCASTER_RESULT_CACHE", "./results")

# OpenAI configuration
OPENAI_USE_AZURE = (
    os.environ.get("STARCASTER_OPENAI_USE_AZURE", "False").lower() == "true"
)
OPENAI_API_KEY = os.environ.get("STARCASTER_OPENAI_API_KEY", "")
OPENAI_API_VERSION = os.environ.get("STARCASTER_OPENAI_API_VERSION", None)
OPENAI_AZURE_ENDPOINT = os.environ.get("STARCASTER_OPENAI_AZURE_ENDPOINT", None)


DATA_STORAGE_PATH = os.environ.get("STARCASTER_DATA_STORE", "benchmark/data")

DOMINICK_STORAGE_PATH = os.environ.get(
    "STARCASTER_DOMINICK_STORE", os.path.join(DATA_STORAGE_PATH, "dominicks")
)
DOMINICK_CSV_PATH = os.path.join(
    DOMINICK_STORAGE_PATH, "filtered_dominick_grocer_sales.csv"
)
DOMINICK_JSON_PATH = os.path.join(DOMINICK_STORAGE_PATH, "grocer_sales_influences.json")
