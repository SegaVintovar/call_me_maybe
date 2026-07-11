*This project has been created as part of the 42 curriculum by vsudak*

# Call Me Maybe

## Description

This is a function calling tool that translates natural language
prompts into structured function calls. Given a question like "What is the sum of 40 and
2?", your solution should not return 42, but instead provide:
• The function name: fn_add_numbers
• The arguments: {"a": 40, "b": 2}
My implementation uses constrained decoding and guarantees 100% valid JSON
output, ensuring near-perfect reliability even with a small 0.6B parameter model.

## Instructions

1. make install
2. make run or uv run python -m src [–functions_definition <function_definition_file>]
[–input <input_file>] [–output <output_file>]

## Resources

[https://www.lmsys.org/blog/2024-02-05-compressed-fsm/?ref=aidancooper.co.uk](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/?ref=aidancooper.co.uk)
[https://www.w3schools.com/python/](https://www.w3schools.com/python/)

### AI usage

AI helped me to understand basic generation loop:

def generate_text(model, prompt_text, max_new_tokens=128):
    input_ids = model.encode(prompt_text)[0].tolist()

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

        # greedy decoding: pick the highest-scoring next token
        next_token_id = max(range(len(logits)), key=lambda i: logits[i])
        input_ids.append(next_token_id)

    return model.decode(input_ids)

## Algorithm explanation

I process prompt by prompt in two stages. 1st stage deciedes which function to use.
At the 2nd stage I know which parameters do I need to extract and what types they are.
Depending on the stage and datatype, I am masking invalid logits.

## Design decisions

I made this design, because this was my first idea and it worked out.
Also, I do not think that letting AI to generate entire JSON file is a good idea.
Pydantic BaseModel can be used not only for validation, but also for dumping instances into dict objects. 

## Performance analysis

Given prompt and fn_def set is being processed in betwenn 2 and 3 minutes.

## Challenges faced

Understanding and implementing constrained decoding 

## Testing strategy

I was breaking everything regarding the input: prompts and fn_defenitions. Errors that occured were fixed, so the program is never crashed
