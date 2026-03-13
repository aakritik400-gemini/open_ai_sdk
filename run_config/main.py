# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List
import logging

from agents import Agent, Runner, RunContextWrapper, function_tool
from run_config import run_config 

# -----------------------------
# LOGGER SETUP
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

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
    return "\n".join([f"- {p.name}: ${p.price}" for p in wrapper.context.products])

@function_tool
async def calculate_total(wrapper: RunContextWrapper[StoreContext], product_name: str, quantity: int) -> str:
    for p in wrapper.context.products:
        if p.name.lower() == product_name.lower():
            return f"Total for {quantity} {p.name} is ${p.price * quantity}"
    return "Product not found."

@function_tool
async def recommend_product(wrapper: RunContextWrapper[StoreContext]) -> str:
    cheapest = min(wrapper.context.products, key=lambda x: x.price)
    return f"I recommend {cheapest.name} for ${cheapest.price}"

# -----------------------------
# Agent
# -----------------------------
agent = Agent[StoreContext](
    name="Ecommerce Assistant",
    instructions="You help users explore products and calculate prices. Use tools whenever necessary.",
    tools=[list_products, calculate_total, recommend_product],
    model=run_config.model  # Using model from the shared RunConfig
)

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Agent API Running"}

@app.post("/ask")
async def ask_agent(query: Query):
    result = await Runner.run(
        starting_agent=agent,
        input=query.question,
        context=context,
        run_config=run_config  
    )
    return {"response": result.final_output}