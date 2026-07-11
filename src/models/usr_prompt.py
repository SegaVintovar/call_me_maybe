from pydantic import BaseModel, model_validator
from typing import Self


class UserPrompt(BaseModel):
    prompt: str

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")
        return self
