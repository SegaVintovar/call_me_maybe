from src.models.func_def import Function
from src.models.usr_prompt import UserPrompt
from src.models.answer import Answer
from src.models.param import Param
from dataclasses import dataclass
import json
import os
from llm_sdk import Small_LLM_Model  # type: ignore
import time
import string
import sys


@dataclass
class AiProcessor():
    """
    Tool calling system

    Takes: LLM model, tool defenitions and user prompts

    Returns: function calls in JSON format
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

        self.valid_number_tokens: set[int] = set()
        self.valid_string_tokens: set[int] = set()
        self.valid_boolean_tokens: set[int] = set()
        self.stop_tokens: set[int] = set()

        self.__post_init__()

    def __post_init__(self) -> None:
        try:

            for p in self.user_prompts_d:
                try:
                    self.user_prompts.append(UserPrompt(prompt=p["prompt"]))
                    self.user_prompts_str.append(p["prompt"])
                except Exception:
                    raise Exception("UserPrompt validation error")
            for fn in self.fn_defs_d:
                n = fn["name"]
                description = fn["description"]
                returns = fn["returns"]
                parameters = [
                    Param(
                        name=k,
                        tp=v["type"]
                        ) for k, v in fn["parameters"].items()
                        ]
                self.func_list.append(
                    Function(
                        name=n,
                        description=description,
                        parameters=parameters,
                        returns=returns))
                # useful later
                self.func_name_list.append(n)
        except Exception as e:
            print("Validation Error: ", str(e), file=sys.stderr)
            exit(1)

        with open(self.model.get_path_to_vocab_file(), "r") as v:
            vocab = json.load(v)
        self.vocab = vocab
        for k, v in self.vocab.items():
            self.vocab_invert[v] = k

        valid_number_chars = [
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", ".", ";"]
        stop_chars = [' "', '"', ";"]

        for c in stop_chars:
            self.stop_tokens.add(self.model.encode(c)[0].tolist()[0])

        for ch in valid_number_chars:
            self.valid_number_tokens.add(self.model.encode(ch)[0].tolist()[0])

        for prompt in self.user_prompts_str:
            vt = self.model.encode(prompt)[0].tolist()
            for t in vt:
                self.valid_string_tokens.add(t)
        self.valid_string_tokens.add(self.model.encode('"')[0].tolist()[0])
        for c in string.printable:
            self.valid_string_tokens.add(self.model.encode(c)[0].tolist()[0])

    def run(self) -> None:
        """
        Method to process all given prompts"""
        # run stage 1 and then stage 2
        # use Answer class to store results
        start = time.time()
        for p in self.user_prompts_d:
            self._process(p["prompt"])
        print("Prompt processing took: ", time.time() - start, " seconds")
        self._compile_json()

    def _process(self, prompt: str) -> None:
        answer = self._stage1(prompt)
        self._stage2(answer)

    def _stage1(self, prompt: str) -> Answer:
        """
        Here we are prompting AI to make it choose an function
        """

        print(f"\nAnalyzing user prompt: {prompt}")
        p = self._build_first_prompt(prompt)
        valid_tokens = self._what_is_valid_fn_name()
        mt = 6
        text = self._generate_text(p, valid_tokens, max_new_tokens=mt)
        print("generated fn_name text: ", text)
        for fn in self.func_name_list:
            if fn in text:
                print(f"SOLUTION FOUND\n==============\n\n{fn}\n")

                ans = Answer(prompt=prompt, name=fn, params={})
                self.answers.append(ans)

                break
        return ans

    def _build_first_prompt(self, user_prompt: str) -> str:
        # funcs = {func["name"]: func["description"] for func in self.fn_defs_d}

        return ('Choose right function\n'
                f'Available functions: {self.fn_defs_d}\n'
                f'User prompt: {user_prompt}\n'
                'Answer only with function name: ')

    def _what_is_valid_fn_name(self) -> set:
        result = set()
        for fn in self.func_name_list:
            tmp = self.model.encode(fn)[0].tolist()
            for t in tmp:
                result.add(t)
        print("valid tokens: ", result)
        return result

    def _stage2(self, ans: Answer) -> None:
        # according to the choosen function, define parameters
        # by using function defenition and user prompt
        # constrain the generation by parameter type
        # and it presence in the prompt
        par: str | float
        fn_name = ans.name
        usr_prompt = ans.prompt
        # self.valid_string_tokens.add(self.model.encode(usr_prompt)[0].tolist())
        for fn in self.func_list:
            if fn.name == fn_name:
                parameters = fn.parameters
                function = fn
                break
        # print((type(parameters)))
        tmp = ""
        for param in parameters:
            # ask LLM to generate each aparameter separetly
            p = self._build_second_prompt(
                usr_prompt, function, param.name, param.tp, tmp)
            # print("second prompt: ", p)
            mnt = 10
            if param.tp == "number" or param.tp == "integer":
                valid_tokens = self.valid_number_tokens
            elif param.tp == "string":
                valid_tokens = self.valid_string_tokens
                if param.name == "replacement":
                    mnt = 2
                else:
                    mnt = 15
            elif param.tp == "boolean":
                mnt = 1
                valid_tokens = {
                    self.model.encode("True")[0].tolist()[0],
                    self.model.encode("False")[0].tolist()[0]
                    }
            # here i need to generate only parameter by parameter
            par = self._generate_text(
                p, valid_tokens, max_new_tokens=mnt, stage=2)
            print("generated parameter: ", param.name, "=", par)
            tmp = param.name + "=" + par
            par = par.strip()
            if param.tp == "number":
                # try
                par = float(par)
            elif param.tp == "integer":
                par = int(float(par))
            elif param.tp == "string":
                par = par.strip()
            ans.params[param.name] = par

    def _build_second_prompt(
            self,
            prompt: str,
            function: Function,
            parameter: str,
            type: str,
            already_gen: str
            ) -> str:

        # the best prompt so far
        if type == "string":
            start = '="'
        else:
            start = "="
        prmpt = (
            "Find parameters for the function call in the user prompt\n"
            f"{function.__dict__}\n"
            f"User prompt: {prompt}\n"
            "End generation with ;\n"
            f"Function call: {function.name}({already_gen} {parameter}{start}"
        )

        return prmpt

    def _generate_text(
            self,
            prompt_text: str,
            valid_tokens: set,
            max_new_tokens: int = 20,
            stage: int = 1
            ) -> str:

        input_ids = self.model.encode(prompt_text)[0].tolist()
        gen_tokens: list[float] = []
        vt = valid_tokens.copy()
        stop_tokens = self.stop_tokens

        for _ in range(max_new_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)

            for i in range(len(logits)):
                if i not in vt:
                    logits[i] = float("-inf")

            next_token_id = max(range(len(logits)), key=lambda i: logits[i])

            gen_tokens.append(next_token_id)
            if stage == 1:
                for fn in self.func_name_list:
                    if fn in self.model.decode(gen_tokens):
                        return str(self.model.decode(gen_tokens))
            else:
                # for c in '";':
                #     stop_tokens.add(self.model.encode(c)[0].tolist()[0])
                # stop_tokens.add(self.model.encode(' "')[0].tolist()[0])
                if next_token_id in stop_tokens:
                    return str(self.model.decode(gen_tokens[:-1]))
            input_ids.append(next_token_id)

        return str(self.model.decode(gen_tokens))

    def _compile_json(self) -> None:
        result = []

        for ans in self.answers:
            tmp: dict[str, str | float | dict | None] = {
                "prompt": None, "name": None, "parameters": None
                }
            tmp["prompt"] = ans.prompt
            tmp["name"] = ans.name
            tmp["parameters"] = ans.params
            result.append(tmp)
        # print(result)
        p = self.path_to_output.split("/")
        name = p.pop(-1)
        path = "/".join(p)

        os.makedirs(path, exist_ok=True)
        with open((path + "/" + name), mode="w") as f:
            f.write(json.dumps(result, indent=2))
