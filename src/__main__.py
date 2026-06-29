import sys
from .engine.engine import AiProcessor
from llm_sdk import Small_LLM_Model
import json


# 1. Check arguments
path_to_fn_defs = "data/input/functions_definition.json"
path_to_usr_prmpts = "data/input/function_calling_tests.json"
path_to_output = "data/output/function_calls.json"
args = sys.argv[1:]
if len(args) >= 1:
    i = 0
    while i < len(args):
        if args[i].startswith("--functions_definition"):
            path_to_fn_defs = args[i + 1]
            i += 1
        elif args[i].startswith("--input"):
            path_to_usr_prmpts = args[i + 1]
            i += 1
        elif args[i].startswith("--output"):
            path_to_output = args[i + 1]
            i += 1
        i += 1

try:
    with open(path_to_fn_defs, 'r') as file:
        func_defs = json.load(file)
    # print("File data =", data)
except FileNotFoundError:
    print(f"Error: Could not find {path_to_fn_defs} file.", file=sys.stderr)
    exit(1)
except json.JSONDecodeError as e:
    print(f"Invalid JSON syntax in {path_to_fn_defs} file:", e)
except Exception as e:
    print(str(e), file=sys.stderr)
    exit(1)

try:
    with open(path_to_usr_prmpts, 'r') as file:
        user_prompts = json.load(file)
    # print("File data =", data)
except FileNotFoundError:
    print(f"Error: Could not find {path_to_usr_prmpts} file.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Invalid JSON syntax in {path_to_usr_prmpts} file:",
          e,
          file=sys.stderr)
except Exception as e:
    print(e, file=sys.stderr)
    exit(1)

# try:
#     with open(path_to_output, 'w') as output_file:
#         data = json.load(output_file)
#     # print("File data =", data)
# except Exception:
#     print("Error: Could not create output file.")


# get all of the fn_def and user_prompts
model = Small_LLM_Model()
my_ai = AiProcessor(
    model,
    path_to_usr_prmpts,
    path_to_fn_defs,
    path_to_output
    )
# mb prepare for start/to answer?instead of post init

my_ai.run()
