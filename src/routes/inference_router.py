from fastapi import APIRouter, HTTPException

from modules.schemas import LLMRequest, LLMResponse
from core.logger_config import logger


inference_router = APIRouter(tags=["Next Token Prediction"])


@inference_router.post("/", response_model=LLMResponse)
async def llm_inference(request: LLMRequest) -> str:

    try:

        query = request.text

        response = "Hi, " + query

        return LLMResponse(text=response)

    except Exception as e:
        logger.error(
            f"An error occurred while processing the request: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")
