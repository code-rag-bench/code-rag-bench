"""LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code
https://arxiv.org/abs/2403.07974

The LiveCodeBench dataset is a contamination-fre code generation benchmark with 
problems collected between May 2023 and February 2024. 

Homepage: https://livecodebench.github.io/
"""

import os, json
from time import time
from eval.base import Task
from eval.utils import extract_code_pieces
from eval.tasks.custom_metrics.io_eval import codegen_metrics

_CITATION = """
@misc{jain2024livecodebench,
      title={LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code}, 
      author={Naman Jain and King Han and Alex Gu and Wen-Ding Li and Fanjia Yan and Tianjun Zhang and Sida Wang and Armando Solar-Lezama and Koushik Sen and Ion Stoica},
      year={2024},
      eprint={2403.07974},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
"""
INSTRUCTION = (
    "### Instruction: You will be given a question (problem specification) and "
    "will generate a correct Python program that matches the specification and passes all tests. "
    "You will NOT return anything except for the program.\n\n"
)
FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution "
    "to the problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters as follows."
)
FUNC_PATH = "eval/tasks/lcb_examples/func.json"
func = json.load(open(FUNC_PATH, 'r'))
STDIN_PATH = "eval/tasks/lcb_examples/stdin.json"
stdin = json.load(open(STDIN_PATH, 'r'))


class LCB(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    def __init__(
        self, dataset_path: str = None, dataset_name: str = None, data_files: dict = None, 
        cache_dir: str = None, topk_docs: int = 5, tokenizer: str = None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            data_files=data_files,
            cache_dir=cache_dir,
            stop_words=["\n### Question", "if __name__", "# Write", "# Test", "\nprint"],
            requires_execution=True,
        )
        self.topk_docs = topk_docs

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 400
        ), "please ensure you have the latest version of LiveCodeBench dataset, try deleting its old cache"
        return dataset #.select([i for i in range(10)])

    def get_example_prompt(self, example, has_starter_code: bool):
        prompt = ""
        prompt += "### Question\n" + example["question"] + "\n\n"
        if has_starter_code:
            prompt += "### Starter Code\n" + example["sample_code"] + "\n\n"
        prompt += "### Answer\n\n" + example["answer"]
        if example["answer"]: prompt += "\n\n"
        return prompt

    def get_prompt(self, doc, instruct_mode: bool = True, return_dict: bool = False):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        # [1] instruction: following LiveCodeBench/lcb_runner/prompts/code_generation.py
        if instruct_mode: 
            prompt = INSTRUCTION
            prompt += f"Question:\n{doc['question_content']}\n\n"
            if doc["starter_code"]:
                prompt += f"### Instruction: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
                prompt += f"```python\n{doc['starter_code']}\n```\n\n"
            else:
                prompt += f"### Instruction: {FORMATTING_WITHOUT_STARTER_CODE}\n"
                prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
            prompt += f"### Response:\n\n"
        else: # [2] one-shot example
            if doc["starter_code"]: examples_json = func
            else: examples_json = stdin
            has_starter_code = len(doc["starter_code"]) > 0
            prompt = self.get_example_prompt(examples_json[0], has_starter_code)
            prompt += self.get_example_prompt({
                "question": doc["question_content"],
                "sample_code": doc["starter_code"],
                "answer": "",
            }, has_starter_code)

        # [3] add retrieved contexts
        context = doc.get("docs", [])
        if len(context) > 0:
            if isinstance(context[0], dict):
                context = "\n".join([ctx["text"] for ctx in context[: self.topk_docs]])
            else:
                context = '\n'.join(context[: self.topk_docs])
        else:
            context = ""
        if return_dict:
            return {"context": context, "prompt": prompt}
        prompt = context + "\n" + prompt
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["public_test_cases"])

    def postprocess_generation(self, generation, idx, new_tokens_only=False):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt_dict = self.get_prompt(self.dataset["test"][idx], return_dict=True)
        prompt = prompt_dict["context"] + '\n' + prompt_dict["prompt"]
        if not new_tokens_only:
            generation = generation[len(prompt):]

        gen_lines = generation.split('\n')
        question_indices = [i for i,l in enumerate(gen_lines) if l.startswith("### Question")]
        if len(question_indices) > 1:
            s = question_indices[1]
        else:
            s = 0
        if len(question_indices) < 3:
            generation = '\n'.join(gen_lines[s: ])
        else:
            e = question_indices[2]
            generation = '\n'.join(gen_lines[s: e])
        # generation = self._stop_at_stop_token(generation, self.stop_words)
        if "### Answer" in generation:
            offset = len("### Answer")
            answer_index = generation.rindex("### Answer")
        else:
            offset = 0
            answer_index = 0
        generation = generation[answer_index + offset:].lstrip()
        if "### Question" in generation: 
            question_index = generation.index("### Question")
            generation = generation[:question_index].rstrip()
        generation = generation.rstrip().rstrip('</s>')
        if '```python' in generation:
            generation = '\n'.join(extract_code_pieces(generation, prefix='```python', return_all=True))
        elif '```' in generation:
            generation = '\n'.join(extract_code_pieces(generation, prefix='```', return_all=True))
        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        metrics, results, final_metadata = codegen_metrics(
            examples=self.get_dataset(),
            generations=generations,
            k_list=[1],
        )
        return metrics
