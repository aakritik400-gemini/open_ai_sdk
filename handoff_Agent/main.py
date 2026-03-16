import os
import logging
from dataclasses import dataclass
from typing import List, Optional

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


# ---------------------------------------------------
# Logger Configuration
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("store-agent")


# ---------------------------------------------------
# Load ENV
# ---------------------------------------------------
logger.info("Loading environment variables")

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found")

logger.info("Environment loaded successfully")


# ---------------------------------------------------
# Gemini Client
# ---------------------------------------------------
logger.info("Initializing Gemini client")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

logger.info("Gemini client initialized")


# ---------------------------------------------------
# Model
# ---------------------------------------------------
logger.info("Loading model gemini-2.5-flash")

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

logger.info("Model configured")


# ---------------------------------------------------
# Data Classes
# ---------------------------------------------------
@dataclass
class Product:
    name: str
    price: float


@dataclass
class StoreContext:
    products: List[Product]


logger.info("Creating store context")

context = StoreContext(
    products=[
        Product("Laptop", 1000),
        Product("Mouse", 25),
        Product("Keyboard", 75),
    ]
)

logger.info("Store context created")


# ---------------------------------------------------
# Tools
# ---------------------------------------------------
@function_tool
async def list_products(wrapper: RunContextWrapper[StoreContext]) -> str:
    """Return list of products"""

    logger.info("TOOL → list_products called")

    result = "Available products:\n"

    for p in wrapper.context.products:
        result += f"- {p.name}: ${p.price}\n"

    return result


@function_tool
async def calculate_total(
    wrapper: RunContextWrapper[StoreContext],
    product_name: str,
    quantity: int = 1
) -> str:
    """Calculate total price"""

    logger.info(f"TOOL → calculate_total | product={product_name} quantity={quantity}")

    for p in wrapper.context.products:

        if p.name.lower() == product_name.lower():

            total = p.price * quantity

            logger.info(f"Calculated total = ${total}")

            return f"Total price for {quantity} {p.name} is ${total}"

    return "Product not found."


@function_tool
async def recommend_product(
    wrapper: RunContextWrapper[StoreContext],
    max_price: Optional[float] = None
) -> str:
    """Recommend product optionally under price"""

    logger.info(f"TOOL → recommend_product | max_price={max_price}")

    products = wrapper.context.products

    if max_price is not None:
        products = [p for p in products if p.price <= max_price]

    if not products:
        return "No products found in that price range."

    best = min(products, key=lambda x: x.price)

    return f"I recommend {best.name} for ${best.price}"


# ---------------------------------------------------
# Agents
# ---------------------------------------------------
logger.info("Creating Product Agent")

product_agent = Agent[StoreContext](
    name="Product Agent",
    handoff_description="Handles product list queries",
    instructions="""
You handle product listing queries.

Rules:
- Always use the list_products tool.
- Return the tool output to the user.
""",
    tools=[list_products],
    model=model
)


logger.info("Creating Billing Agent")

billing_agent = Agent[StoreContext](
    name="Billing Agent",
    handoff_description="Handles price and billing questions",
    instructions="""
You handle pricing questions.

Rules:
- Always use calculate_total tool when price is requested.
- If quantity missing assume 1.
- Return the tool output to the user.
""",
    tools=[calculate_total],
    model=model
)


logger.info("Creating Recommendation Agent")

recommend_agent = Agent[StoreContext](
    name="Recommendation Agent",
    handoff_description="Handles product recommendation questions",
    instructions="""
You recommend products.

Rules:
- Always use recommend_product tool.
- If the user specifies a price limit, pass it as max_price.
- Return tool result directly.
""",
    tools=[recommend_product],
    model=model
)


# ---------------------------------------------------
# Router Agent
# ---------------------------------------------------
logger.info("Creating Router Agent")

store_agent = Agent[StoreContext](
    name="Store Router",
    instructions="""
You are a routing agent.

Your job is ONLY to delegate user requests to the correct agent.

Rules:
- Never answer questions yourself.
- Always perform exactly ONE handoff.
- After the handoff the delegated agent will answer.

Routing Guide:

Product Agent:
Questions about available products or product list.

Billing Agent:
Questions about price, cost, or totals.

Recommendation Agent:
Questions about suggestions, recommendations,
or what product to buy.
""",
    handoffs=[
        product_agent,
        billing_agent,
        recommend_agent,
    ],
    model=model
)

logger.info("Router agent ready")


# ---------------------------------------------------
# FastAPI
# ---------------------------------------------------
app = FastAPI()


class Query(BaseModel):
    question: str


@app.get("/")
def home():

    logger.info("Health check called")

    return {"message": "Store Agent Running 🚀"}


@app.post("/ask")
async def ask_agent(query: Query):

    logger.info("====================================")
    logger.info(f"User Question → {query.question}")

    try:

        result = await Runner.run(
            starting_agent=store_agent,
            input=query.question,
            context=context,
            run_config=config
        )

        logger.info(f"Final Response → {result.final_output}")

        return {"response": result.final_output}

    except Exception as e:

        logger.exception("Agent execution failed")

        return {"error": str(e)}
