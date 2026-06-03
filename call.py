from llm_sdk import Small_LLM_Model
import json
import sys
from typing import cast, Any
import math
import heapq
import re

# valid_token_set_fn_choose = 

class Tokenvalidation():
    valid_chars = r"0123456789abcdefghijklmnopqrstuvwxyz[]{},_"

    def __init__(self, fn_list: list[str]):
        self.fn_list = fn_list


def generate_text(
        model: Small_LLM_Model, prompt_text, max_new_tokens=12) -> str:

    input_ids = model.encode(prompt_text)[0].tolist()
    prompt_l = len(input_ids)
    # start_ids = model.encode(ai_help(prompt_text))[0].tolist()
    # input_ids += start_ids
    
    # print(input_ids)

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        # here i need to pick a valid logit,
        # not the one with highest probability
        # next_token_id = max(range(len(logits)), key=lambda i: logits[i])
        # print(logits[:10])
        max_logit = max(logits)
        next_token_id = logits.index(max_logit)
        print(model.decode(next_token_id))
        input_ids.append(next_token_id)

    return model.decode(input_ids)


def build_first_prompt(functions, user_prompt):
    return f'choose right function\n\
Available functions:{functions}\n\
User prompt: {user_prompt}\n' +\
r'Answer only with function name: '


def ai_help(user_prompt: str) -> str:
    return r"[{'prompt': " + user_prompt


s_llm = Small_LLM_Model()

try:
    with open("data/input/functions_definition.json") as func_def:
        # funcs = func_def.read()
        # print("funcs before json load\n", funcs, type(funcs))
        funs = json.load(func_def, object_hook=dict[str, Any])
        # print("funcs after json load\n", funs, type(funs))
    with open("data/input/function_calling_tests.json") as usr_input:
        # prompts = usr_input.read()
        # print("prompts before json load\n", prompts, type(prompts))
        prompts_json = json.load(usr_input)
        # print("prompts after json load\n", prompts_json, type(prompts_json))
    with open(s_llm.get_path_to_vocab_file()) as v:
        # vocab = v.read()
        vocab = json.load(v)
        # print(vocab)
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
print(prompts_list[8])
p = build_first_prompt(function_names, prompts_list[8])
p += ai_help(prompts_list[8])
print(p)
text = generate_text(s_llm, p)
for fn in function_names:
    if fn in text:
        ai_fn_choice = fn
print(ai_fn_choice)
i = 0
for ch in text:
    if ch == "[":
        break
    i += 1
with open("text.txt", "w") as f:
    f.write(text)
print(text[i:])

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

