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
    input_guardrail
)
from agents.run import RunConfig

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ENV
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
logger.info("GEMINI_API_KEY loaded successfully")

# Gemini Client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-001",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# Data Classes
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

@input_guardrail
async def ecommerce_guardrail(wrapper: RunContextWrapper[StoreContext], agent, user_input: str) -> str:
    """Check user input for safety"""
    logger.info(f"Guardrail checking input: {user_input}")

    blocked_keywords = ["hack", "attack", "bomb", "drugs", "politics"]

    if any(word in user_input.lower() for word in blocked_keywords):
        # Raise ValueError to trigger tripwire
        raise ValueError("Query blocked by safety guardrail")

    if len(user_input.strip()) == 0:
        raise ValueError("Empty query not allowed")

    # Always return the user_input string if it's safe
    return user_input

# Tools
@function_tool
async def list_products(wrapper: RunContextWrapper[StoreContext]) -> str:
    logger.info("Tool called: list_products")
    result = "Available Products:\n"
    for p in wrapper.context.products:
        result += f"- {p.name}: ${p.price}\n"
    return result

@function_tool
async def calculate_total(wrapper: RunContextWrapper[StoreContext], product_name: str, quantity: int) -> str:
    logger.info("Tool called: calculate_total")
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

# Fallback handoff tool using function_tool
@function_tool
async def fallback_tool(wrapper: RunContextWrapper[StoreContext], user_input: str) -> str:
    logger.info(f"Fallback tool triggered for: {user_input}")
    return f"Sorry, I cannot handle this request directly. You asked: '{user_input}'"

# Agent
agent_prompt = """
You are a helpful ecommerce assistant.
- Always try to use tools first to answer questions.
- Tools available: list_products, calculate_total, recommend_product
- If the query cannot be answered with tools, use the fallback_tool.
- Guardrails are in place to block unsafe inputs.
- Always be polite and concise.
"""

agent = Agent[StoreContext](
    name="Guarded Ecommerce Assistant",
    instructions=agent_prompt,
    tools=[list_products, calculate_total, recommend_product, fallback_tool],
    input_guardrails=[ecommerce_guardrail],
    model=model
)

# FastAPI
app = FastAPI()

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Guarded Ecommerce Agent with Handoffs Running 🚀"}

@app.post("/ask")
async def ask_agent(query: Query):
    logger.info(f"Received query: {query.question}")

    try:
        result = await Runner.run(
            starting_agent=agent,
            input=query.question,
            context=context,
            run_config=config
        )
        return {"response": result.final_output}

    except Exception as e:
        # Don't assume 'tripwire_triggered', just log the exception
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}