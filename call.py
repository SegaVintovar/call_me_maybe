from llm_sdk import Small_LLM_Model
import json
import sys
from typing import cast
import math
import heapq


class Tokenvalidation():
    valid_chars = r"0123456789abcdefghijklmnopqrstuvwxyz[]{},"
    def __init__(self, fn_list: list[str]):
        self.fn_list = fn_list
    


def generate_text(model: Small_LLM_Model, prompt_text, max_new_tokens=64) -> str:
    input_ids = model.encode(prompt_text)[0].tolist()

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        # 
        next_token_id = max(range(len(logits)), key=lambda i: logits[i])
        
        input_ids.append(next_token_id)

    return model.decode(input_ids)

def build_function_call_prompt(functions, user_prompt):
    return f"""
                You are a function selection engine.
                to answer users prompt,
                choose exactly one function from the list below.

                Available functions:
                {json.dumps(functions, indent=2)}

                Return only valid JSON in this format:
                [
                    {
                    "prompt": "What is the sum of 2 and 3?",
                    "name": "fn_add_numbers",
                    "parameters": {"a": 2.0, "b": 3.0}
                    },
                    {
                    "prompt": "Reverse the string 'hello'",
                    "name": "fn_reverse_string",
                    "parameters": {"s": "hello"}
                    }
                ]

                User prompt:
                {user_prompt}
                """

s_llm = Small_LLM_Model()

try:
    with open("data/input/functions_definition.json") as func_def:
        funcs = func_def.read()
    with open("data/input/function_calling_tests.json") as usr_input:
        prompts = usr_input.read()
    with open(s_llm.get_path_to_vocab_file()) as v:
        # vocab = v.read()
        vocab = json.load(v)
        print(vocab)
except FileNotFoundError as e:
    print(str(e), file=sys.stderr)
    exit(1)
except PermissionError as e:
    print(str(e), file=sys.stderr)
    exit(1)
except Exception as e:
    print(str(e), file=sys.stderr)
    exit(1)


# vocab = vocab.strip(r"{}")
# vocab_list = vocab.split(",")

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

