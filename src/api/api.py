from fastapi import APIRouter
from routers.inference_router import inference_router
from routers.train_model_router import training_router


api_router = APIRouter()

api_router.include_router(inference_router, prefix="/inference")
api_router.include_router(training_router, prefix="/training")