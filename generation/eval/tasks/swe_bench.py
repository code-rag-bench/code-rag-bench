"""SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
https://arxiv.org/abs/2310.06770

The SWE-bench dataset released by Princeton includes repository-level GitHub issues with a problem statememt,
base commit, and a patch, along with other metadata.
They were collected from real issues from top-stared GitHub repositories.

Homepage: https://www.swebench.com/
"""

import os
import json
import shutil
from git import Repo
from eval.base import Task
from eval.utils import extract_code_pieces

EXAMPLE_PATCH = """
I need you to solve this issue by generating a single patch file that I can apply directly to this repository using git apply. Please respond with a single patch file in the following format.
<patch>
--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
 def euclidean(a, b):
-    while b:
-        a, b = b, a % b
-    return a
+    if b == 0:
+        return a
+    return euclidean(b, a % b)
 
 
 def bresenham(x0, y0, x1, y1):
     points = []
     dx = abs(x1 - x0)
     dy = abs(y1 - y0)
-    sx = 1 if x0 < x1 else -1
-    sy = 1 if y0 < y1 else -1
-    err = dx - dy
+    x, y = x0, y0
+    sx = -1 if x0 > x1 else 1
+    sy = -1 if y0 > y1 else 1
 
-    while True:
-        points.append((x0, y0))
-        if x0 == x1 and y0 == y1:
-            break
-        e2 = 2 * err
-        if e2 > -dy:
+    if dx > dy:
+        err = dx / 2.0
+        while x != x1:
+            points.append((x, y))
             err -= dy
-            x0 += sx
-        if e2 < dx:
-            err += dx
-            y0 += sy
+            if err < 0:
+                y += sy
+                err += dx
+            x += sx
+    else:
+        err = dy / 2.0
+        while y != y1:
+            points.append((x, y))
+            err -= dx
+            if err < 0:
+                x += sx
+                err += dy
+            y += sy
 
+    points.append((x, y))
     return points
</patch>
"""

_CITATION = """
@inproceedings{
    jimenez2024swebench,
    title={{SWE}-bench: Can Language Models Resolve Real-world Github Issues?},
    author={Carlos E Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik R Narasimhan},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=VTF8yNQM66}
}
"""


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"swebench": create_task(True), "swebench-lite": create_task(False)}


def create_task(strip_prompt):
    class SWEbench(GeneralSWEbench):
        def __init__(self, **kwargs):
            super().__init__(strip_prompt, **kwargs)

    return SWEbench


class GeneralSWEbench(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    def __init__(
        self, strip_prompt, k=[1, 10, 100], num_workers=16, timeout=3.0,
        dataset_path: str = None, dataset_name: str = None, data_files: dict = None, 
        cache_dir: str = None, topk_docs: int = 5, tokenizer: str = None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            data_files=data_files,
            cache_dir=cache_dir,
            stop_words=["\n<issue>"],
            requires_execution=True,
        )
        self.strip_prompt = strip_prompt
        self.k = k
        self.num_workers = num_workers
        self.timeout = timeout
        self.topk_docs = topk_docs

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if "text" in doc: # oracle file
            prompt = doc["text"]
            if self.strip_prompt: 
                prompt = prompt.strip()
            return prompt

        prompt = "You will be provided with a partial code base and an issue statement explaining a problem to resolve.\n"
        prompt += "<issue>\n" + doc["problem_statement"] + "\n</issue>\n"
        
        # get retrieved contexts
        context = doc.get("docs", "")
        if len(context) > 0:
            context = "\n".join([
                f"\n[start of {ctx['title']}]\n" + '\n'.join([f"{idx} {line}" for idx,line in enumerate(ctx["text"].split('\n'))]) + f"\n[end of {ctx['title']}]\n"
                for ctx in context[: self.topk_docs]
            ])
            context = "\n<code>\n" + context + "\n</code>\n"
            prompt += context
        prompt += EXAMPLE_PATCH
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return

    def postprocess_generation(self, generation, idx, new_tokens_only=False):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        if not new_tokens_only:
            prompt = self.get_prompt(self.dataset["test"][idx])
            generation = generation[len(prompt) :]
        generation = self._stop_at_stop_token(generation, self.stop_words)
        if "```python\n" in generation:
            generation = extract_code_pieces(generation, prefix="```python")
        elif "```\n" in generation:
            generation = extract_code_pieces(generation, prefix="```")
        return {
            "instance_id": self.dataset["test"][idx]["instance_id"],
            "model_name_or_path": "test",
            "model_patch": generation,
        }

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        # Please refer to the documentation for swebench evaluation
        raise NotImplementedError
