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
    model="gemini-flash-lite-latest",
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
    """List all available products"""

    logger.info("Tool: list_products")

    result = "Available Products:\n"

    for p in wrapper.context.products:
        result += f"- {p.name}: ${p.price}\n"

    return result


@function_tool
async def calculate_total(
    wrapper: RunContextWrapper[StoreContext],
    product_name: str,
    quantity: int = 1
) -> str:
    """Calculate total price of a product"""

    logger.info("Tool: calculate_total")

    for p in wrapper.context.products:
        if p.name.lower() == product_name.lower():
            total = p.price * quantity
            return f"Total price for {quantity} {p.name} is ${total}"

    return "Product not found."


@function_tool
async def recommend_product(wrapper: RunContextWrapper[StoreContext]) -> str:
    """Recommend cheapest product"""

    logger.info("Tool: recommend_product")

    cheapest = min(wrapper.context.products, key=lambda x: x.price)

    return f"I recommend {cheapest.name} because it costs only ${cheapest.price}"


# -----------------------------
# Specialized Agents
# -----------------------------
product_agent = Agent[StoreContext](
    name="Product Agent",
    instructions="""
You are responsible for product listing queries.
Use the list_products tool to answer user questions.
""",
    tools=[list_products],
    model=model
)


billing_agent = Agent[StoreContext](
    name="Billing Agent",
    instructions="""
You are responsible for pricing queries.
Use calculate_total tool when the user asks about price or cost.
If quantity is not provided assume quantity = 1.
""",
    tools=[calculate_total],
    model=model
)


recommend_agent = Agent[StoreContext](
    name="Recommendation Agent",
    instructions="""
You recommend products to customers.
Use the recommend_product tool to give suggestions.
""",
    tools=[recommend_product],
    model=model
)


# -----------------------------
# Router Agent (Main Agent)
# -----------------------------
store_agent = Agent[StoreContext](
    name="Store Assistant",
    instructions="""
You are the main ecommerce assistant that routes user requests. you can only use tools given below not allowed to answer from external website or your own knowledge.

Routing Rules:

Always handoff to the correct agent instead of answering directly.
""",
    handoffs=[product_agent, billing_agent, recommend_agent],
    model=model
)


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()


class Query(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Store Agent Running 🚀"}


@app.post("/ask")
async def ask_agent(query: Query):

    logger.info(f"Question: {query.question}")

    result = await Runner.run(
        starting_agent=store_agent,
        input=query.question,
        context=context,
        run_config=config
    )

    return {"response": result.final_output}