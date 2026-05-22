from llm_sdk.llm_sdk import Small_LLM_Model
import json
import sys


funcs = None
try:
    with open("data/input/functions_definition.json") as func_def:
        funcs = func_def.read()
    with open("data/input/function_calling_tests.json") as usr_input:
        prompts = usr_input.read()
except FileNotFoundError as e:
    print(str(e), file=sys.stderr)
    exit(1)
except PermissionError as e:
    print(str(e), file=sys.stderr)
    exit(1)
except Exception as e:
    print(str(e), file=sys.stderr)
    exit(1)

fun_d = json.loads(funcs)
print(fun_d)


# pass here funcs and prompts

# receive an answer and validate it

# store it into output files

