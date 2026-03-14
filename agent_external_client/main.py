import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# ----------------------------
# Logger Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    logger.error("GEMINI_API_KEY not found in .env")
    raise ValueError("GEMINI_API_KEY not found")

logger.info("Environment variables loaded successfully")

# ----------------------------
# Gemini Client
# ----------------------------
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

logger.info("Gemini client initialized")

# ----------------------------
# Model
# ----------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-flash-lite-latest",
    openai_client=client
)

logger.info("Model loaded: gemini-flash-lite-latest")

# ----------------------------
# Run Configuration
# ----------------------------
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# ----------------------------
# Agent
# ----------------------------
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=model
)

logger.info("Agent initialized")

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

class Query(BaseModel):
    question: str


@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Gemini Agent API Running Successfully"}


@app.post("/ask")
async def ask_agent(query: Query):

    logger.info(f"Received question: {query.question}")

    try:
        result = await Runner.run(
            agent,
            query.question,
            run_config=config
        )

        logger.info("Agent response generated successfully")

        return {"response": result.final_output}

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return {"error": str(e)}