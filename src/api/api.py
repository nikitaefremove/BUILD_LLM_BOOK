from fastapi import APIRouter
from routes.inference_router import inference_router

api_router = APIRouter()

api_router.include_router(inference_router, prefix="/inference")
