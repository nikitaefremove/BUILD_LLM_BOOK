import torch
import tiktoken

from fastapi import APIRouter, HTTPException
from core.logger_config import logger
from schemas.training_schemas import TrainingRequest, TrainingResponse

from core.model_config import GPT_CONFIG_124M
from utils.models import GPTModel

from utils.train_model_simple import train_model_simple
from utils.vocab import vocab
from utils.train_val_loaders import train_loader, val_loader


training_router = APIRouter(tags=["Simple LLM Training Process"])


@training_router.post("/", response_model=TrainingResponse)
def train_llm_simple(request: TrainingRequest):

    try:
        start_context = request.start_context
        learning_rate = request.learning_rate
        num_epoch = request.num_epoch

        torch.manual_seed(123)
        device = "mps" if torch.mps.is_available() else "cpu"
        print(f"Device: {device}")
        logger.info(f"Device: {device}")

        tokenizer = tiktoken.get_encoding("gpt2")

        model = GPTModel(GPT_CONFIG_124M)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.1
        )

        train_losses, val_losses, tokens_seen = train_model_simple(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=num_epoch,
            eval_freq=5,
            eval_iter=5,
            start_context=start_context,
            tokenizer=tokenizer,
        )

        return TrainingResponse(text="The model training process is complete")

    except Exception as e:
        logger.error(
            f"An error occurred while processing the request: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal Server Error")
