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
    WebSearchTool
)

from agents.run import RunConfig

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found")

logger.info("Environment variables loaded")

# Gemini client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

# Run config
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# Agent with Web Search Tool
agent = Agent(
    name="Research Assistant",
    instructions="""
    You are a helpful research assistant.
    Use web search when the user asks about current events,
    latest information, news, or facts from the internet.
    """,
    model=model,
    tools=[WebSearchTool()],
    tool_use_behavior="run_llm_again"
)

logger.info("Agent initialized with WebSearchTool")

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Web Search Agent API Running"}


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