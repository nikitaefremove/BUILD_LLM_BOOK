from pydantic import BaseModel


class LLMRequest(BaseModel):
    text: str


class LLMResponse(BaseModel):
    text: str
