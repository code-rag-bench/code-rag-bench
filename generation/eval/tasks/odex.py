"""Execution-Based Evaluation for Open Domain Code Generation
https://arxiv.org/pdf/2212.10481.pdf
The ODEX dataset includes 945 NL-to-Code generation pairs with 1,707 
human-written test cases. ODEX involves NL intents in four natural languages: 
with 439, 90, 164, and 252 samples in English, Spanish, Japanese, and Russian.
https://github.com/zorazrw/odex
"""

from eval.base import Task
from eval.utils import extract_code_pieces
from eval.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = """
@article{wang2022execution,
         title={Execution-Based Evaluation for Open-Domain Code Generation},
         author={Zhiruo Wang, Shuyan Zhou, Daniel Fried, Graham Neubig},
         journal={arXiv preprint arXiv:2212.10481},
         year={2022}
}
"""

def create_all_tasks():
    """Creates a dictionary of tasks from multiple languages
    :return: {language: task}
        e.g. {en: Task, en: Task, ja: Task, ru: Task}
    """
    return {f"odex-{lang}": create_task(lang) for lang in ["en", "es", "ja", "ru"]}


def create_task(lang):
    class ODEX(GeneralODEX):
        def __init__(self, **kwargs):
            super().__init__(lang, **kwargs)

    return ODEX



class GeneralODEX(Task):

    # DATASET_PATH = "neulab/odex"
    # DATASET_NAME = None

    def __init__(
        self, lang, strip_prompt=True, k=[1, 10, 100], num_workers=16, timeout=3.0,
        dataset_path: str = None, dataset_name: str = None, data_files: dict = None, 
        cache_dir: str = None, topk_docs: int = 5, tokenizer: str = None,
    ):
        # self.DATASET_NAME = lang
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            data_files=data_files,
            cache_dir=cache_dir,
            stop_words=["###", "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"],
            requires_execution=True,
       )
        self.lang = lang
        self.strip_prompt = strip_prompt
        self.k = k
        self.num_workers = num_workers
        self.timeout = timeout
        self.topk_docs = topk_docs

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc, return_dict: bool = False):
        """Builds the prompt for the LM to generate from."""
        function_head, function_prefix = doc["prompt"].split("\n")
        docstr = f'    """{doc["intent"]}\n    """'
        code_body = function_prefix.replace("\t", " " * 4)
        prompt = "\n".join([function_head, docstr, code_body])

        context = doc.get("docs", "")
        if len(context) > 0:
            if isinstance(context, list):
                if isinstance(context[0], dict):
                    context = "\n".join([ctx["text"] for ctx in context[:self.topk_docs]])
                else:
                    context = "\n".join(context[:self.topk_docs])
            elif not isinstance(context, str):
                context = ""
            instruction = "Please refer to the following documentation to generate the code:\n"
            context = instruction + context
        else:
            context = ""

        if return_dict:
            return {"prompt": prompt, "context": context}
        prompt = context + "\n" + prompt
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(
            [
                doc["test_start"],
                "".join(doc["test"]),
                "",
                f"check({doc['entry_point']})",
            ]
        )

    def postprocess_generation(self, generation, idx, new_tokens_only=False):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
            (not used for ODEX)
        :return: str
        """
        prompt_dict = self.get_prompt(self.dataset["test"][idx], return_dict=True)
        prompt = prompt_dict["context"] + '\n' + prompt_dict["prompt"]
        if not new_tokens_only:
            generation = generation[len(prompt):]
            generation = self._stop_at_stop_token(generation, self.stop_words)
            generation = prompt_dict["prompt"] + generation
        else:
            if "```python\n" in generation:
                generation = extract_code_pieces(generation, prefix="```python")
            elif "```\n" in generation:
                generation = extract_code_pieces(generation, prefix="```")
            generation = self._stop_at_stop_token(generation, self.stop_words)
        return generation.rstrip()

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
