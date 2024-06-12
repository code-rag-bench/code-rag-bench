"""Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""

from transformers import AutoTokenizer
from eval.base import Task
from eval.utils import extract_code_pieces
from eval.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""


class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    def __init__(
        self, dataset_path: str = None, dataset_name: str = None, data_files: dict = None, 
        cache_dir: str = None, topk_docs: int = 5, tokenizer: AutoTokenizer = None, 
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            data_files=data_files,
            cache_dir=cache_dir,
            stop_words=["\nprint", "\n>>> ", "\n**", "\nclass", "# Write a", "# Test", "<EOS_TOKEN>"],
            requires_execution=True,
        )
        self.topk_docs = topk_docs
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        # add test cases to indicate the name of the function to be generated
        test_example = '\n'.join([f"{test}" for test in doc["test_list"]])
        prompt = f"# {doc['text']}\nTest cases:\n" + test_example + '\nCode:'

        context = doc.get("docs", "")
        if len(context) > 0:
            if isinstance(context, list):
                if isinstance(context[0], dict):
                    context = "\n".join([ctx["text"] for ctx in context[: self.topk_docs]])
                else:
                    context = "\n".join(context[: self.topk_docs])

        if hasattr(self, "tokenizer") and (self.tokenizer.name_or_path == "deepseek-ai/deepseek-coder-33b-instruct"):
            prompt = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], 
                tokenize=False, add_generation_prompt=True
            )
        prompt = context + "\n" + prompt
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])

    def postprocess_generation(self, generation, idx, new_tokens_only=True):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        if not new_tokens_only:
            prompt = self.get_prompt(self.dataset["test"][idx])
            generation = generation[len(prompt) :]
        generation = self._stop_at_stop_token(generation, self.stop_words)
        if "```python\n" in generation:
            generation = extract_code_pieces(generation, prefix="```python")
        elif "```\n" in generation:
            generation = extract_code_pieces(generation, prefix="```")
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
        )
        return results
