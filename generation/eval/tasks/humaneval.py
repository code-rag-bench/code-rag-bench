"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

import os
from transformers import AutoTokenizer
from eval.base import Task
from eval.tasks.custom_metrics.code_eval import compute_code_eval
from eval.utils import extract_generation_code, extract_code_pieces

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"humaneval": create_task(True), "humaneval-unstripped": create_task(False)}


def create_task(strip_prompt):
    class HumanEval(GeneralHumanEval):
        def __init__(self, **kwargs):
            super().__init__(strip_prompt, **kwargs)

    return HumanEval


class GeneralHumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    def __init__(
        self, strip_prompt, k=[1, 10, 100], num_workers=16, timeout=3.0, topk_docs: int = 5,
        dataset_path: str = None, dataset_name: str = None, data_files: dict = None,
        cache_dir: str = None, tokenizer: str = None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            data_files=data_files,
            cache_dir=cache_dir,
            stop_words=["\nclass", "<file_sep>", "if __name__", "\nprint(", "\ndef"],
            requires_execution=True,
        )
        self.strip_prompt = strip_prompt
        self.k = k
        self.num_workers = num_workers
        self.timeout = timeout
        self.topk_docs = topk_docs
        if (tokenizer is not None) and ("deepseek" in tokenizer):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = None

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.strip_prompt:
            prompt = doc["prompt"].strip()
        else:
            prompt = doc["prompt"]
        prompt = '```python\n' + prompt

        context = doc.get("docs", "")
        if len(context) > 0:
            if isinstance(context, list):
                if isinstance(context[0], dict):
                    context = "\n".join([ctx["text"] for ctx in context[: self.topk_docs]])
                else:
                    context = "\n".join(context[: self.topk_docs])
            prompt = context + '\n\nPlease complete the following function based on the example above:\n' + prompt

        # deepseek-coder-33b needs special prompt prefix to work properly                                         
        if (self.tokenizer is not None) and (self.tokenizer.name_or_path == "deepseek-ai/deepseek-coder-33b-instruct"):
            prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], 
                tokenize=False, add_generation_prompt=True
            )
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    def postprocess_generation(self, generation, idx, new_tokens_only=False):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        if not new_tokens_only:  # for hf models
            full_prompt = self.get_prompt(self.dataset["test"][idx])
            generation = generation[len(full_prompt):]
            generation = self._stop_at_stop_token(generation, self.stop_words)
            generation = self.dataset["test"][idx]["prompt"] + generation
            generation = extract_generation_code(generation, self.dataset["test"][idx]["prompt"])
        else: # for api models, set `new_tokens_only`
            generation = extract_code_pieces(generation)
        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
            k=self.k,
            num_workers=self.num_workers,
            timeout=self.timeout,
        )
        return results
