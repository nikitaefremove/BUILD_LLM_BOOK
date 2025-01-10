from fastapi import APIRouter, HTTPException

from schemas.inference_schemas import LLMRequest, LLMResponse
from core.logger_config import logger

from utils.generate_text import generation_pipeline


inference_router = APIRouter(tags=["Inference of an Untrained Language Model"])


@inference_router.post("/", response_model=LLMResponse)
async def llm_inference(request: LLMRequest) -> str:

    try:

        query = request.text

        response = generation_pipeline(text=query)

        return LLMResponse(text=response)

    except Exception as e:
        logger.error(
            f"An error occurred while processing the request: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")
