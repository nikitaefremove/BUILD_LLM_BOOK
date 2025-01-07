from fastapi import APIRouter
from routers.inference_router import inference_router


api_router = APIRouter()

api_router.include_router(inference_router, prefix="/inference")
