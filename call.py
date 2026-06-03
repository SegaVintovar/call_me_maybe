from llm_sdk import Small_LLM_Model
import json
import sys
from typing import cast, Any, Literal
import math
import heapq
import re
from pydantic import BaseModel, Field, model_validator
from dataclasses import dataclass

class Answer(BaseModel):
    """
    prompt is user prompt
    name is func name
    params are function prams with chosen values
    """
    def __init__(self, prompt: str, name: str = "", params: dict = {}):
        self.prompt = prompt
        self.name = name
        self.params = params

    @model_validator(mode="after")
    def post(self):
        ...


class UserPrompt(BaseModel):
    """
    Storing user prompt here
    """
    prompt: str


class FuncDef(BaseModel):
    """
    Here we are storing func definition
    """
    name: str = Field(pattern=r"^fn_")
    description: str
    parameters: dict
    # or number or
    returns: str["number" | "string"]

    @model_validator(mode="after")
    def post(self):
        if self.returns not in ["string", "number"]:
            raise ValueError(
                "Function resturn type can be only 'string' or 'number'")
        return self


@dataclass
class AiProcessor(BaseModel):
    """
    Dataclass is needed for post init
    """
    def __init__(self,
                 model: Small_LLM_Model,
                 path_to_prompts: str =
                 "data/input/function_calling_tests.json",
                 path_to_functions: str =
                 "data/input/functions_definition.json"
                 ) -> None:
        self.model = model
        self.path_to_prompts = path_to_prompts
        self.path_to_functions = path_to_functions
        self.func_list: list[FuncDef] = []
        self.func_name_list: list[str] = []
        self.user_prompts: list[UserPrompt] = []
        self.user_prompts_str: list[str] = []

    @model_validator
    def post_valid(self):
        if not self.path_to_functions.endswith(".json"):
            raise ValueError(
                "Invalid path to the function defenition\
                      file, it has to end with '.json'"
                )
        if not self.path_to_prompts.endswith(".json"):
            raise ValueError(
                "Invalid path to the file with prompts,",
                " it has to end with '.json'"
                )

    def __post_init__(self) -> None:
        try:
            with open(self.path_to_prompts) as f:
                all_prompts = json.load(f)
                for p in all_prompts:
                    self.user_prompts.append(UserPrompt(prompt=p["prompt"]))
                    self.user_prompts_str.append(p["prompt"])
            with open(self.path_to_functions) as f:
                all_fn_defs = json.load(f)
                for fn in all_fn_defs:
                    name = fn["name"]
                    description = fn["description"]
                    params = fn["parameters"]
                    returns = fn["returns"]
                    self.func_list.append(
                        FuncDef(name, description, params, returns))
                    self.func_name_list.append(name)

        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            exit(1)
        except PermissionError as e:
            print(str(e), file=sys.stderr)
            exit(1)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            exit(1)
        except Exception as e:
            print(str(e), file=sys.stderr)
            exit(1)

    def run(self):
        ...

    def build_first_prompt(self, user_prompt):
        return f'choose right function\n\
    Available functions:{self.func_list}\n\
    User prompt: {user_prompt}\n' +\
            r'Answer only with function name: [{"name": "'

    def generate_text(
            model: Small_LLM_Model,
            prompt_text: str,
            fn_list: list[str],
            max_new_tokens=12
            ) -> str:

        input_ids = model.encode(prompt_text)[0].tolist()
        gen_tokens: list[float] = []

        for _ in range(max_new_tokens):
            logits = model.get_logits_from_input_ids(input_ids)
            # here i need to pick a valid logit,
            # not the one with highest probability

            max_logit = max(logits)

            next_token_id = logits.index(max_logit)
            gen_tokens.append(next_token_id)
            if model.decode(gen_tokens) in fn_list:
                break
            input_ids.append(next_token_id)
        print(input_ids)
        print()
        print(gen_tokens)
        return model.decode(gen_tokens)

def generate_text(
        model: Small_LLM_Model,
        prompt_text: str,
        fn_list: list[str],
        max_new_tokens=12
        ) -> str:

    input_ids = model.encode(prompt_text)[0].tolist()
    gen_tokens: list[float] = []

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        # here i need to pick a valid logit,
        # not the one with highest probability

        max_logit = max(logits)

        next_token_id = logits.index(max_logit)
        gen_tokens.append(next_token_id)
        if model.decode(gen_tokens) in fn_list:
            break
        input_ids.append(next_token_id)
    print(input_ids)
    print()
    print(gen_tokens)
    return model.decode(gen_tokens)


def build_first_prompt(functions, user_prompt):
    return f'choose right function\n\
Available functions:{functions}\n\
User prompt: {user_prompt}\n' +\
        r'Answer only with function name: [{"name": '





def main():

    s_llm = Small_LLM_Model()

    try:
        with open("data/input/functions_definition.json") as func_def:
            funs = json.load(func_def, object_hook=dict[str, Any])
        with open("data/input/function_calling_tests.json") as usr_input:
            prompts_json = json.load(usr_input)
        with open(s_llm.get_path_to_vocab_file()) as v:
            vocab = json.load(v)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        exit(1)
    except PermissionError as e:
        print(str(e), file=sys.stderr)
        exit(1)
    except Exception as e:
        print(str(e), file=sys.stderr)
        exit(1)

    function_names: list = [p["name"] for p in funs]
    prompts_list = [p["prompt"] for p in prompts_json]
    # print(prompts_list[8])
    result = []
    for prompt in prompts_list:
        tmp = {}
        p = build_first_prompt(function_names, prompt)
        # p += ai_help(prompt)
        print(p)
        text = generate_text(s_llm, p, function_names)
        tmp["prompt"] = prompt
        for fn in function_names:
            if fn in text:
                text = fn
                break
        tmp["name"] = text
        result.append(tmp)
    print(result)

    with open("text.txt", "w") as f:
        f.write(str(result))


if __name__ == "__main__":
    main()

# fun_d = json.loads(funcs)
# json_user_prompts = json.loads(prompts)


# tmp = s_llm.get_path_to_vocab_file()
# with open(file=tmp) as vocab_path:
#     print(vocab_path.read())
# for prompt in json_user_prompts:
#     # print(prompt)
    


#     prompt = build_function_call_prompt(fun_d[0], prompt["prompt"])
#     print(generate_text(s_llm, prompt))
#     print(prompt)
    # s_llm.encode(prompt)[0].tolist()
    # logits = s_llm.get_logits_from_input_ids(
    #     )
#     # prompts)[0].tolist()
#     # stable softmax in pure Python
#     max_logit = max(logits)
#     exp_scores = [math.exp(x - max_logit) for x in logits]
#     total = sum(exp_scores)
#     probs = [x / total for x in exp_scores]

#     top5 = heapq.nlargest(15, enumerate(probs), key=lambda item: item[1])

#     for token_id, prob in top5:
#         print(token_id, s_llm.decode([token_id]), prob)



# # pass here funcs and prompts

# # receive an answer and validate it

# # store it into output files


# [
#     {
#         "prompt": "What is the sum of 2 and 3?",
#         "name": "fn_add_numbers",
#         "parameters": {"a": 2.0, "b": 3.0}
#     },
#     {
#         "prompt": "Reverse the string 'hello'",
#         "name": "fn_reverse_string",
#         "parameters": {"s": "hello"}
#     }
# ]


# valid_token_set_fn_choose =
class Tokenvalidation():
    valid_chars = r"0123456789abcdefghijklmnopqrstuvwxyz[]{},_."

    def __init__(self, fn_list: list[str]):
        self.fn_list = fn_list


def ai_help(user_prompt: str) -> str:
    return r"[{'name': "