import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.lifecycle import RunHooks

# ----------------------------
# Logger Configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("store-agent")

# ----------------------------
# Load Environment Variables
# ----------------------------
logger.info("Loading environment variables")

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    logger.error("GEMINI_API_KEY not found")
    raise ValueError("GEMINI_API_KEY not found")

logger.info("Environment loaded successfully")

# ----------------------------
# Gemini Client
# ----------------------------
logger.info("Initializing Gemini client")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

logger.info("Gemini client initialized")

# ----------------------------
# Model
# ----------------------------
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

logger.info("Model loaded: gemini-2.5-flash")

# ----------------------------
# Hooks
# ----------------------------
class MyRunHooks(RunHooks):

    async def before_run(self, context, agent, input):
        logger.info("HOOK → Before agent run")
        logger.info(f"Agent: {agent.name}")
        logger.info(f"Input: {input}")

    async def after_run(self, context, agent, output):
        logger.info("HOOK → After agent run")
        logger.info(f"Output: {output.final_output}")

    async def on_error(self, context, agent, error):
        logger.error("HOOK → Error during run")
        logger.error(str(error))


hooks = MyRunHooks()

# ----------------------------
# Run Config
# ----------------------------
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# ----------------------------
# Agent
# ----------------------------
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=model
)

logger.info("Agent initialized")

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Gemini Agent API")

class Query(BaseModel):
    question: str


@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Gemini Agent API Running Successfully"}


@app.post("/ask")
async def ask_agent(query: Query):

    logger.info(f"Received question: {query.question}")

    try:
        result = await Runner.run(
            agent,
            query.question,
            run_config=config,
            hooks=hooks   
        )

        logger.info("Agent response generated")

        return {
            "question": query.question,
            "response": result.final_output
        }

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return {"error": str(e)}