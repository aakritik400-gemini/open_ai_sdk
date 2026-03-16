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
from agents.tools import ToolSearchTool

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Load Environment
# -----------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# -----------------------------
# Gemini Client
# -----------------------------
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

logger.info("Gemini client initialized")

# -----------------------------
# Model
# -----------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

# -----------------------------
# RunConfig
# -----------------------------
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# -----------------------------
# Custom Tool Example
# -----------------------------
@function_tool
def get_product_price(product: str) -> str:
    """Returns product price from store"""
    
    products = {
        "laptop": "$900",
        "mouse": "$25",
        "keyboard": "$70"
    }

    return products.get(product.lower(), "Product not found")


# -----------------------------
# Tool Search Tool
# -----------------------------
tool_search = ToolSearchTool()

# -----------------------------
# Agent
# -----------------------------
agent = Agent(
    name="Shopping Assistant",
    instructions="""
    You are a shopping assistant.
    Use tools when necessary to answer questions.
    """,
    tools=[get_product_price, tool_search]
)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

class Query(BaseModel):
    question: str


@app.post("/ask")
async def ask_agent(query: Query):

    logger.info(f"Question: {query.question}")

    result = await Runner.run(
        agent,
        query.question,
        run_config=config
    )

    return {
        "answer": result.final_output
    }