from pydantic import BaseModel, model_validator
from typing import Self


class Param(BaseModel):
    name: str
    tp: str

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if not self.name:
            raise ValueError("parameter name is not missing")
        if self.tp not in ["string", "number", "integer", "boolean"]:
            raise ValueError(f"Unknown data type: {self.tp}")
        return self
