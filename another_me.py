from llm_sdk import Small_LLM_Model
import json
import sys
import re


def build_function_call_prompt(functions, user_prompt):
    return f"""
You are a function selection engine.
Choose exactly one function from the list below.

Available functions:
{json.dumps(functions, indent=2)}

Return only valid JSON in this format:
{{
  "name": "function_name",
  "parameters": {{
    "arg1": "value"
  }}
}}

User prompt:
{user_prompt}
"""


def generate_text(model, prompt_text, max_new_tokens=128):
    input_ids = model.encode(prompt_text)[0].tolist()

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

        # greedy decoding: pick the highest-scoring next token
        next_token_id = max(range(len(logits)), key=lambda i: logits[i])
        input_ids.append(next_token_id)

    return model.decode(input_ids)


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


def validate_call(obj, functions):
    function_map = {f["name"]: f for f in functions}

    if "name" not in obj:
        raise ValueError("Missing name")
    if "parameters" not in obj:
        raise ValueError("Missing parameters")

    name = obj["name"]
    if name not in function_map:
        raise ValueError(f"Unknown function: {name}")

    schema = function_map[name]["parameters"]
    params = obj["parameters"]

    for key in schema:
        if key not in params:
            raise ValueError(f"Missing parameter: {key}")

    return True


def main():
    try:
        with open("data/input/functions_definition.json") as func_def:
            funcs = json.load(func_def)
        with open("data/input/function_calling_tests.json") as usr_input:
            prompts = json.load(usr_input)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    model = Small_LLM_Model()
    print("After model init")
    for item in prompts:
        prompt_text = build_function_call_prompt(funcs, item["prompt"])
        reply = generate_text(model, prompt_text, max_new_tokens=128)

        print("RAW MODEL OUTPUT:")
        print(reply)

        # try:
        #     obj = extract_json(reply)
        #     validate_call(obj, funcs)
        #     print("VALID JSON:")
        #     print(json.dumps(obj, indent=2))
        # except Exception as e:
        #     print("INVALID OUTPUT:", e)


if __name__ == "__main__":
    main()