import torch

from transformers import GPT2Model

from model.architecture import GPTModel
from core.model_config import GPT_CONFIG_355M
from utils.load_model.load_weights import load_weights


BASE_CONFIG = GPT_CONFIG_355M

model_names = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl",
}

CHOOSE_MODEL = "gpt2-medium (355M)"

gpt_hf = GPT2Model.from_pretrained(
    model_names[CHOOSE_MODEL], cache_dir="fine-tuning-instruct/checkpoints"
)
gpt_hf.eval()

device = torch.device("mps" if torch.mps.is_available() else "cpu")

gpt = GPTModel(BASE_CONFIG)


print("Before loading weights:")
print(gpt.pos_emb.weight[:5]) 

load = load_weights(gpt, gpt_hf)

torch.save(gpt.state_dict(), "fine-tuning-instruct/model/weights/gpt_with_hf_weights.pth")

print("After loading weights:")
print(gpt.pos_emb.weight[:5])
