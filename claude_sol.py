from llm_sdk import Small_LLM_Model
import json
import re

def generate_text(model: Small_LLM_Model, prompt: str, max_new_tokens: int = 300) -> str:
    """Greedy decode: repeatedly pick the highest-logit next token."""
    input_ids = model._tokenizer.encode(prompt, add_special_tokens=False)
    eos_id = model._tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

        # Greedy: pick the token with the highest logit
        next_token_id = logits.index(max(logits))

        if next_token_id == eos_id:
            break

        input_ids.append(next_token_id)

    # Decode only the newly generated tokens (not the prompt)
    prompt_length = len(model._tokenizer.encode(prompt, add_special_tokens=False))
    generated_ids = input_ids[prompt_length:]
    return model._tokenizer.decode(generated_ids, skip_special_tokens=True)


def build_prompt(user_prompt: str) -> str:
    """
    Instruct the model to return ONLY a JSON object describing
    a function and its parameters — not an answer to the question.
    """
    return f"""You are a function-planning assistant.
Given a user request, respond with ONLY a valid JSON object (no explanation, no markdown) in this exact format:
{{
  "function": "<function_name>",
  "parameters": {{
    "<param1>": <value1>,
    "<param2>": <value2>
  }}
}}

Choose a fitting function name (e.g. calculate_average, convert_currency, sort_list, search_web).
Infer parameter names and values from the request.

User request: {user_prompt}

JSON:"""


def extract_json(text: str) -> dict:
    """Pull the first {{ ... }} block out of the generated text and parse it."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return json.loads(match.group())


def get_function_call(user_prompt: str) -> dict:
    model = Small_LLM_Model()  # loads Qwen3-0.6B by default

    prompt = build_prompt(user_prompt)
    raw_output = generate_text(model, prompt, max_new_tokens=200)

    print(f"[Raw model output]:\n{raw_output}\n")

    return extract_json(raw_output)


# --- Main ---
if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    result = get_function_call(user_prompt)
    print("[Parsed JSON result]:")
    print(json.dumps(result, indent=2))