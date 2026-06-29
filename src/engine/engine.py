from src.models.func_def import Function
from src.models.usr_prompt import UserPrompt
from src.models.answer import Answer
from dataclasses import dataclass
from typing import Any
import json
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
                 path_to_prompts: str =
                 "data/input/function_calling_tests.json",
                 path_to_functions: str =
                 "data/input/functions_definition.json",
                 path_to_output: str = "data/output/function_calls.json"
                 ) -> None:

        self.model = model
        self.path_to_prompts = path_to_prompts
        self.path_to_functions = path_to_functions
        self.path_to_output = path_to_output
        self.func_list: list[Function] = []
        # do i need it here
        self.func_name_list: list[str] = []
        self.user_prompts: list[UserPrompt] = []
        # do I need it here
        self.user_prompts_str: list[str] = []
        self.answers: list[Answer] = []
        self.vocab: dict
        self.__post_init__()

    def __post_init__(self) -> None:
        # try:
        with open(self.path_to_prompts) as f:
            all_prompts = json.load(f)
            for p in all_prompts:
                self.user_prompts.append(UserPrompt(prompt=p["prompt"]))
                self.user_prompts_str.append(p["prompt"])
        with open(self.path_to_functions) as f:
            all_fn_defs = json.load(f, object_hook=dict[str, Any])
            for fn in all_fn_defs:
                name = fn["name"]
                description = fn["description"]
                params = fn["parameters"]
                returns = fn["returns"]
                print(fn)
                self.func_list.append(
                    Function(**fn)
                )
                # self.func_list.append(
                #     FuncDef(
                #         name=name,
                #         description=description,
                #         parameters=params,
                #         returns=returns)
                #         )
                self.func_name_list.append(name)

        with open(self.model.get_path_to_vocab_file(), "r") as v:
            vocab = json.load(v)
        self.vocab = vocab

    def stage1(self):
        """
        Here we are prompting AI to make it choose an function
        """
        for p in self.user_prompts_str:
            print(f"\nAnalyzing user prompt: {p}")
            prompt = self.build_first_prompt(p)
            text = self.generate_text(prompt)
            for fn in self.func_name_list:
                if fn in text:
                    print(f"SOLUTION FOUND\n==============\n\n{fn}\n")
                    self.answers.append(
                        Answer(prompt=p, name=fn, params={}))
                    break
            # tmp break
            break
        ...

    def stage2(self):
        # according to the choosen function, define parameters
        # by using function defenition and user prompt
        # constrain the generation by parameter type
        # and it presence in the prompt
        ...

    def run(self):
        # run stage 1 and then stage 2
        # use Answer class to store results
        self.stage1()
        # self.stage2()
        self.compile_json()

        ...

    def build_first_prompt(self, user_prompt):
        return f'choose right function\n\
Available functions:{self.func_name_list}\n\
User prompt: {user_prompt}\n' +\
            "       "\
            r'Answer only with function name: [{"name": "'

    def generate_text(self, prompt_text: str, max_new_tokens=12) -> str:
        # handle different stages 
        input_ids = self.model.encode(prompt_text)[0].tolist()
        gen_tokens: list[float] = []

        for _ in range(max_new_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)

            # here i need to pick a valid logit,
            # not the one with highest probability

            # max_logit = max(logits)
            for i, t in enumerate(self.vocab):
                # print(i, t)
                if not t.startswith("fn_"):
                    logits[i] = float("-inf")
            # next_token_id = logits.index(max_logit)
            next_token_id = max(range(len(logits)), key=lambda i: logits[i])
            gen_tokens.append(next_token_id)
            for fn in self.func_name_list:
                if fn in self.model.decode(gen_tokens):
                    break
            # if self.model.decode(gen_tokens) in self.func_name_list:
            #     break
            input_ids.append(next_token_id)

        return self.model.decode(gen_tokens)

    def compile_json(self):
        result = []
        for ans in self.answers:
            tmp = {}
            tmp["prompt"] = ans.prompt
            tmp["name"] = ans.name
            result.append(tmp)
        print(result)
        path = "data/output/"
        name = "function_calling_results.json"
        os.makedirs(path, exist_ok=True)
        with open((path + name), "w") as f:
            f.write(json.dumps(result, indent=2))

    def another_method(self):
        for a in self.answers:
            fn = a.name
            for f in self.func_list:
                if f.name == fn:
                    ...
                    break


def main():
    # here we are checking for the given arguments

    s_llm = Small_LLM_Model()

    ai = AiProcessor(s_llm)
    ai.run()

if __name__ == "__main__":
    main()
