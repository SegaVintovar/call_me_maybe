from src.models.func_def import Function
from src.models.usr_prompt import UserPrompt
from src.models.answer import Answer
from dataclasses import dataclass
from typing import Any
import json
import os
from llm_sdk import Small_LLM_Model


class Engine():
    def __init__(
            self,
            usr_prompts: list[UserPrompt],
            func_defs: list[Function]
            ) -> None:
        self.usr_prompts = usr_prompts
        self.func_defs = func_defs

    def generate() -> None:
        ...


@dataclass
class AiProcessor():
    """
    Dataclass is needed for post init
    """
    def __init__(self,
                 model: Small_LLM_Model,
                 function_defenitions_d: list[dict],
                 user_prompts_d: dict,
                 #  path_to_prompts: str =
                 #  "data/input/function_calling_tests.json",
                 #  path_to_functions: str =
                 #  "data/input/functions_definition.json",
                 path_to_output: str = "data/output/function_calls.json"
                 ) -> None:

        self.model = model
        self.fn_defs_d = function_defenitions_d
        self.user_prompts_d = user_prompts_d

        self.path_to_output = path_to_output

        self.func_list: list[Function] = []
        self.user_prompts: list[UserPrompt] = []

        # do I need it here
        self.func_name_list: list[str] = []
        self.user_prompts_str: list[str] = []

        self.answers: list[Answer] = []
        self.vocab: dict
        self.vocab_invert: dict = {}
        self.__post_init__()

    def __post_init__(self) -> None:
        # try:

        for p in self.user_prompts_d:
            self.user_prompts.append(UserPrompt(prompt=p["prompt"]))
            self.user_prompts_str.append(p["prompt"])
        # with open(self.path_to_functions) as f:
        #     all_fn_defs = json.load(f, object_hook=dict[str, Any])

        for fn in self.fn_defs_d:
            name = fn["name"]
            self.func_list.append(Function(**fn))
            # useful later
            self.func_name_list.append(name)

        with open(self.model.get_path_to_vocab_file(), "r") as v:
            vocab = json.load(v)
        self.vocab = vocab
        for k, v in self.vocab.items():
            self.vocab_invert[v] = k

    def run(self):
        # run stage 1 and then stage 2
        # use Answer class to store results
        for p in self.user_prompts_d[1:3]:
            self.process(p["prompt"])

        # version for development
        # self.process(self.user_prompts_d[0])
        self.compile_json()
    
    def process(self, prompt: str):
        answer = self.stage1(prompt)
        self.stage2(answer)

    def stage1(self, prompt) -> Answer:
        """
        Here we are prompting AI to make it choose an function
        """
        ans = ""
        print(f"\nAnalyzing user prompt: {prompt}")
        p = self.build_first_prompt(prompt)
        valid_tokens = self.what_is_valid_fn_name()
        text = self.generate_text(p, valid_tokens)
        print("generated text: ", text)
        for fn in self.func_name_list:
            if fn in text:
                print(f"SOLUTION FOUND\n==============\n\n{fn}\n")

                ans = Answer(prompt=prompt, name=fn, params={})
                self.answers.append(ans)
                break
        return ans

    def build_first_prompt(self, user_prompt):
        return ('Choose right function\n'
                f'Available functions: {self.func_name_list}\n'
                f'User prompt: {user_prompt}\n'
                'Answer only with function name: ')

    def what_is_valid_fn_name(self) -> set:
        result = set()
        for fn in self.func_name_list:
            tmp = self.model.encode(fn)[0].tolist()
            for t in tmp:
                result.add(t)
        print("valid tokens: ", result)
        return result

    def stage2(self, ans: Answer):
        # according to the choosen function, define parameters
        # by using function defenition and user prompt
        # constrain the generation by parameter type
        # and it presence in the prompt
        fn_name = ans.name
        prompt = ans.prompt
        for fn in self.func_list:
            if fn.name == fn_name:
                parameters = fn.parameters
                break
        # print((type(parameters)))
        for param, tp in parameters.items():
            # ask LLM to generate each aparameter separetly
            p = self.build_second_prompt(prompt, param, tp["type"])
            if tp["type"] == "number":
                valid_tokens = {1, 2, 3}
            elif tp["type"] == "string":
                valid_tokens = {4, 5, 6}
            elif tp["type"] == "boolean":
                valid_tokens = {
                    (self.model.encode("True")[0].tolist() +
                     self.model.encode("False")[0].tolist())
                    }
            ans.params[param] = self.generate_text(p, valid_tokens, stage=2)

            ...

    def build_second_prompt(
            self, prompt: str, parameters: str, type: str) -> str:

        return ("find all parameters for the function call"
                f"{parameters}")

    def mask_logits(self, logits, valid_tokens) -> list[float]:
        ...

    def generate_text(
            self,
            prompt_text: str,
            valid_tokens: set,
            max_new_tokens: int = 20,
            stage: int = 1
            ) -> str:
        # should it handle different stages ?
        # now it is for fn_name gen
        input_ids = self.model.encode(prompt_text)[0].tolist()
        gen_tokens: list[float] = []
        vt = valid_tokens.copy()

        for _ in range(max_new_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            
            # here i need to pick a valid logit,
            # not the one with highest probability

            for i in range(len(logits)):
                # print(self.vocab_invert[i])
                if i not in vt:
                    logits[i] = float("-inf")

            next_token_id = max(range(len(logits)), key=lambda i: logits[i])

            # vt.discard(next_token_id)
            
            gen_tokens.append(next_token_id)
            if stage == 1:
                for fn in self.func_name_list:
                    if fn in self.model.decode(gen_tokens):
                        return self.model.decode(gen_tokens)
            else:
                ...
            input_ids.append(next_token_id)

        return self.model.decode(gen_tokens)

    def compile_json(self):
        result = []
        print((a.name for a in self.answers))
        for ans in self.answers:
            tmp = {}
            tmp["prompt"] = ans.prompt
            tmp["name"] = ans.name
            result.append(tmp)
        print(result)
        p = self.path_to_output.split("/")
        name = p.pop(-1)
        path = "/".join(p)

        os.makedirs(path, exist_ok=True)
        with open((path + "/" + name), mode="w") as f:
            f.write(json.dumps(result, indent=2))


def main():
    # here we are checking for the given arguments

    s_llm = Small_LLM_Model()

    ai = AiProcessor(s_llm)
    ai.run()


if __name__ == "__main__":
    main()

# for prompt in self.user_prompts_d:
#     self.process(prompt["prompt"])
#         # here we are performing stage 1 and 2 on the prompt
#         # and storing the result into the list of Answers
