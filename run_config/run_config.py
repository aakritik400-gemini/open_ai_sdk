# run_config.py
import os
import logging
from dotenv import load_dotenv

from agents import AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# -----------------------------
# LOGGER SETUP
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Load ENV
# -----------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    logger.error("GEMINI_API_KEY not found in .env file")
    raise ValueError("GEMINI_API_KEY not found")

logger.info("GEMINI_API_KEY loaded successfully")

# -----------------------------
# Initialize Gemini Client
# -----------------------------
try:
    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    logger.info("Gemini client initialized")
except Exception as e:
    logger.exception("Failed to initialize Gemini client")
    raise e

# -----------------------------
# Initialize Model
# -----------------------------
try:
    model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=client
    )
    logger.info("Model created")
except Exception as e:
    logger.exception("Model creation failed")
    raise e

# -----------------------------
# Create RunConfig
# -----------------------------
run_config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)
logger.info("RunConfig created successfully")