from fastapi import APIRouter, HTTPException

from schemas.inference_schemas import LLMRequest, LLMResponse
from core.logger_config import logger

from utils.generate_text import generation_pipeline


inference_router = APIRouter(tags=["Inference of an Untrained Language Model"])


@inference_router.post("/", response_model=LLMResponse)
async def llm_inference(request: LLMRequest) -> str:
    """
    This module contains an endpoint for performing language model inference.

    Args:
        request (LLMRequest): A request object containing a `text` field with the input query for inference.

    Returns:
        LLMResponse: A response object containing the generated text from the language model.
    """

    try:
        query = request.text
        response = generation_pipeline(text=query)

        return LLMResponse(text=response)

    except Exception as e:
        logger.error(
            f"An error occurred while processing the request: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")
