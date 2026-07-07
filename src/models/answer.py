from pydantic import BaseModel, Field


class Answer(BaseModel):
    """
    prompt is user prompt
    name is func name
    params are function prams with chosen values
    """
    # def __init__(self, prompt: str, name: str = "", params: dict = {}):
    #     self.prompt = prompt
    #     self.name = name
    #     self.params = params
    prompt: str
    name: str = Field(min_length=3)
    params: dict = {}
