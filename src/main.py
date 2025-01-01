from fastapi import FastAPI
from core.logger_config import logger
from api.api import api_router


app = FastAPI(title="LLM Inference App")

app.include_router(api_router)

logger.info("Starting LLM Inference App")
