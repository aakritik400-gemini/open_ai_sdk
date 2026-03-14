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
    input_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered
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

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found")

logger.info("GEMINI_API_KEY loaded successfully")


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
# Guardrail
# -----------------------------
@input_guardrail(run_in_parallel=False)
async def ecommerce_guardrail(
    ctx: RunContextWrapper[StoreContext],
    agent: Agent,
    user_input: str
) -> GuardrailFunctionOutput:

    logger.info(f"Guardrail checking input: {user_input}")

    blocked_keywords = ["hack", "attack", "bomb", "drugs", "politics"]

    blocked = any(word in user_input.lower() for word in blocked_keywords)

    return GuardrailFunctionOutput(
        output_info=user_input,
        tripwire_triggered=blocked
    )


# -----------------------------
# Tools
# -----------------------------
@function_tool
async def list_products(wrapper: RunContextWrapper[StoreContext]) -> str:
    """Return all available products"""

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
    """Calculate total price"""

    logger.info("Tool called: calculate_total")

    for p in wrapper.context.products:
        if p.name.lower() == product_name.lower():
            total = p.price * quantity
            return f"Total price for {quantity} {p.name} is ${total}"

    return "Product not found."


@function_tool
async def recommend_product(wrapper: RunContextWrapper[StoreContext]) -> str:
    """Recommend cheapest product"""

    logger.info("Tool called: recommend_product")

    cheapest = min(wrapper.context.products, key=lambda x: x.price)

    return f"I recommend {cheapest.name} for ${cheapest.price}"


# -----------------------------
# Agent Prompt
# -----------------------------
agent_prompt = """
You are an ecommerce assistant.

Rules:
- Use list_products to show products
- Use calculate_total to compute price
- If quantity not provided assume quantity = 1
- Use recommend_product for suggestions

Always rely on tools.
"""


# -----------------------------
# Agent
# -----------------------------
agent = Agent[StoreContext](
    name="Guarded Ecommerce Agent",
    instructions=agent_prompt,
    tools=[list_products, calculate_total, recommend_product],
    input_guardrails=[ecommerce_guardrail],
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
    return {"message": "Ecommerce Agent Running 🚀"}


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

    except InputGuardrailTripwireTriggered:

        return {
            "error": "Query blocked by guardrail"
        }

    except Exception as e:

        logger.error(str(e))

        return {
            "error": str(e)
        }