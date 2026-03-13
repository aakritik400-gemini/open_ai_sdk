"""
Gemini Agent Continuous Conversation Demo
"""
import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunContextWrapper, function_tool
from agents.run import RunConfig
from dataclasses import dataclass
from typing import List

# Load environment
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")

# Initialize client and model
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model=gemini_model,
    openai_client=client
)
run_config = RunConfig(model=model, model_provider=client, tracing_disabled=True)

# Data classes for context
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

# Define tools
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

# Create agent
agent = Agent[StoreContext](
    name="Ecommerce Assistant",
    instructions="You help users explore products and calculate totals. Use tools whenever necessary.",
    tools=[list_products, calculate_total, recommend_product],
    model=model
)

# Multi-turn conversation loop
async def continuous_conversation():
    conversation_history = []  # keep track of all previous messages
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending conversation.")
            break

        # Prepare input including previous conversation
        input_list = conversation_history + [{"role": "user", "content": user_input}]

        # Run agent
        result = await Runner.run(agent, input=input_list, context=context, run_config=run_config)
        print(f"Agent: {result.final_output}")

        # Update conversation history
        conversation_history = result.to_input_list()

# Run the loop
if __name__ == "__main__":
    asyncio.run(continuous_conversation())