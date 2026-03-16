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
    ModelSettings
)

from agents.run import RunConfig


# -----------------------------
# Logger
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Load ENV
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


# -----------------------------
# Model
# -----------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)


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


context = StoreContext(
    products=[
        Product("Laptop", 1000),
        Product("Mouse", 25),
        Product("Keyboard", 75),
    ]
)


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
async def calculate_total(
    wrapper: RunContextWrapper[StoreContext],
    product_name: str,
    quantity: int
) -> str:

    logger.info("Tool called: calculate_total")

    for p in wrapper.context.products:
        if p.name.lower() == product_name.lower():
            total = p.price * quantity
            return f"Total price for {quantity} {p.name} is ${total}"

    return "Product not found."


# -----------------------------
# Agent (FORCED TOOL USE)
# -----------------------------
agent = Agent[StoreContext](
    name="Ecommerce Assistant",
    instructions="""
    Always use tools to answer user queries.
    Never generate product data yourself.
    """,
    tools=[list_products, calculate_total],

    model=model,

    model_settings=ModelSettings(
        tool_choice="required"   # here i have used the concept of forced tool use, which means the agent will be forced to use the tools provided to answer the queries. It will not be able to generate any output on its own without using the tools.
    )
)


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()


class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Forced Tool Agent Running "}


@app.post("/ask")
async def ask_agent(query: Query):

    logger.info(f"Question: {query.question}")

    result = await Runner.run(
        starting_agent=agent,
        input=query.question,
        context=context,
        run_config=config
    )

    return {"response": result.final_output}