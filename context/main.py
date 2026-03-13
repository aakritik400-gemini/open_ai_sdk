import os
import logging
from dataclasses import dataclass
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    function_tool,
)

from agents.run import RunConfig


# -----------------------------
# LOGGER SETUP
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------
# Load ENV
# -----------------------------
logger.info("Loading environment variables...")

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    logger.error("GEMINI_API_KEY not found in .env file")
    raise ValueError("GEMINI_API_KEY not found")

logger.info("GEMINI_API_KEY loaded successfully")


# -----------------------------
# Gemini Client
# -----------------------------
try:
    logger.info("Initializing Gemini client...")

    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    logger.info("Gemini client initialized")

except Exception as e:
    logger.exception("Failed to initialize Gemini client")
    raise e


# -----------------------------
# Model
# -----------------------------
try:
    logger.info("Creating model...")

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash-001",
        openai_client=client
    )

    logger.info("Model created")

except Exception as e:
    logger.exception("Model creation failed")
    raise e


config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

logger.info("RunConfig created")


# -----------------------------
# Data Classes
# -----------------------------
@dataclass
class Product:
    name: str
    price: float


@dataclass
class StoreContext:
    products: List[Product]


logger.info("Creating product context")

context = StoreContext(
    products=[
        Product("Laptop", 1000),
        Product("Mouse", 25),
        Product("Keyboard", 75),
    ]
)

logger.info("Context created successfully")


# -----------------------------
# Tools
# -----------------------------
@function_tool
async def list_products(wrapper: RunContextWrapper[StoreContext]) -> str:

    logger.info("Tool called: list_products")

    result = "Available Products:\n"

    for p in wrapper.context.products:
        result += f"- {p.name}: ${p.price}\n"

    return result


@function_tool
async def calculate_total(wrapper: RunContextWrapper[StoreContext], product_name: str, quantity: int) -> str:

    logger.info(f"Tool called: calculate_total | {product_name} | {quantity}")

    for p in wrapper.context.products:
        if p.name.lower() == product_name.lower():
            total = p.price * quantity
            return f"Total price for {quantity} {p.name} is ${total}"

    return "Product not found."


@function_tool
async def recommend_product(wrapper: RunContextWrapper[StoreContext]) -> str:

    logger.info("Tool called: recommend_product")

    cheapest = min(wrapper.context.products, key=lambda x: x.price)

    return f"I recommend {cheapest.name} because it costs only ${cheapest.price}"


# -----------------------------
# Agent
# -----------------------------
try:
    logger.info("Creating agent...")

    agent = Agent[StoreContext](
        name="Ecommerce Assistant",
        instructions="""
        You help users explore products and calculate prices.
        Use tools whenever necessary.
        """,
        tools=[list_products, calculate_total, recommend_product],
        model=model
    )

    logger.info("Agent created successfully")

except Exception as e:
    logger.exception("Agent creation failed")
    raise e


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()

logger.info("FastAPI app created")


class Query(BaseModel):
    question: str


@app.get("/")
def home():
    logger.info("Home endpoint called")
    return {"message": "Agent API Running "}


@app.post("/ask")
async def ask_agent(query: Query):

    logger.info(f"Received question: {query.question}")

    try:

        result = await Runner.run(
            starting_agent=agent,
            input=query.question,
            context=context,
            run_config=config
        )

        logger.info("Agent response generated")

        return {"response": result.final_output}

    except Exception as e:

        logger.exception("Agent execution failed")

        return {
            "error": str(e)
        }