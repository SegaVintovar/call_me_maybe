from pydantic import BaseModel, Field, model_validator
from typing import Self
from .param import Param


class Function(BaseModel):
    name: str = Field(min_length=4)
    description: str
    parameters: list[Param]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if not self.name.startswith("fn_"):
            raise ValueError(
                "Function model validation error: "
                "function has to start with 'fn_'")
        if not self.parameters:
            raise ValueError(
                "Function model validation error:"
                "no parameters were given")
        return self


# class FuncDef(BaseModel):
#     """
#     Here we are storing func definition
#     """
#     name: str = Field(pattern=r"^fn_")
#     description: str
#     parameters: dict
#     # or "number" or "string"
#     returns: dict

#     @model_validator(mode="after")
#     def post(self):
#         if self.returns["type"] not in ["string", "number"]:
#             raise ValueError(
#                 "Function return type can be only 'string' or 'number'")
#         return self
