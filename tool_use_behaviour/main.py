import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    function_tool
)

from agents.run import RunConfig

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Load env
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    logger.error("GEMINI_API_KEY not found")
    raise ValueError("GEMINI_API_KEY not found")

logger.info("Environment variables loaded")

# Gemini client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

logger.info("Gemini client initialized")

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

logger.info("Model loaded")

# Tool
@function_tool
def get_weather(city: str) -> str:
    """Returns weather information for a city"""
    return f"The weather in {city} is sunny"

# Run config
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# Agent with tool behavior
agent = Agent(
    name="Weather Assistant",
    instructions="You help users with weather information.",
    model=model,
    tools=[get_weather],
    tool_use_behavior="stop_on_first_tool"
)

logger.info("Agent initialized")

# FastAPI
app = FastAPI()

class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Gemini Agent API Running"}


@app.post("/ask")
async def ask_agent(query: Query):

    logger.info(f"Question: {query.question}")

    try:
        result = await Runner.run(
            agent,
            query.question,
            run_config=config
        )

        return {"response": result.final_output}

    except Exception as e:
        logger.error(str(e))
        return {"error": str(e)}