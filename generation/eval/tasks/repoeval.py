"""RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation
https://aclanthology.org/2023.emnlp-main.151/

The RepoEval dataset released by Microsoft includes repository-level code generation problems. 
"""
import os
import time
import json
import re
import numpy as np
from tqdm import tqdm
from eval.base import Task
from eval.tasks.custom_metrics.repoeval_ESEM import (
    process_prediction, compute_EM, compute_ES
)
from eval.tasks.custom_metrics.repoeval_execution import (
    copy_all_repos, setup_repos, check_tests, eval_generation
)

_CITATION = """
@article{zhang2023repocoder,
  title={RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation},
  author={Fengji Zhang and Bei Chen and Yue Zhang and Jacky Keung and Daoguang Zan and Yi Mao and Jian-Guang Lou and Weizhu Chen},
  journal={EMNLP},
  year={2023}
}
"""


STOP_WORDS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"]

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {
        f"repoeval-{split}": create_task(split)
        for split in ["api", "line", "function"]
    }


def create_task(split):
    class RepoEval(GeneralRepoEval):
        def __init__(self, **kwargs):
            super().__init__(split, **kwargs)

    return RepoEval


def get_retrieved_prompt(docs):
    """Builds the retrieved prompt based on a list of docs"""
    start_line = "Here are some relevant code fragments from other files of the repo:"
    sep_line = "--------------------------------------------------"
    intro_line = "the below code fragment can be found in:"
    
    title_block = intro_line + '\n' + '__TITLE__' + '\n' + sep_line
    
    retrieved_prompt = start_line + '\n' + sep_line + '\n'
    for doc in docs:
        title, text = doc['title'], doc['text']
        retrieved_prompt += title_block.replace('__TITLE__', title) + '\n'
        retrieved_prompt += doc['text'] + '\n' + sep_line + '\n'
    
    # add '# ' to each line except for the last line
    retrieved_prompt = "\n".join(
        [ "# " + x for x in retrieved_prompt.split('\n')[:-1]]
    ) + '\n' 
    return retrieved_prompt


class GeneralRepoEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    def __init__(
        self, split, k=[1, 10, 100], num_workers=16, timeout=3.0, topk_docs: int = 2, 
        dataset_path: str = None, dataset_name: str = None, data_files: dict = None, 
        cache_dir: str = None, args=None, tokenizer=None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            data_files=data_files,
            cache_dir=cache_dir,
            stop_words=STOP_WORDS,
            requires_execution=args.allow_code_execution if args else False,
        )
        self.split = split
        self.k = k
        self.num_workers = num_workers
        self.timeout = timeout
        self.topk_docs = topk_docs
        
        self.setup_repoeval = args.setup_repoeval if args else False
        self.metric_output_path = args.metric_output_path if args else os.getcwd()
        self.repoeval_input_repo_dir = args.repoeval_input_repo_dir \
            if args else "../retrieval/output/repoeval/repositories/function_level"
        self.repoeval_cache_dir = args.repoeval_cache_dir if args else "tmp"

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def preprocess_all_data(self, **kwargs):
        """ concate the prompt and retrieved docs here. Save the new prompts as self.dataset["test"]['processed_prompt'], which is a list of str """
        
        if "processed_prompt" in self.dataset["test"].column_names:
            return
        
        required_keys = ['tokenizer', 'remove_linebreak', 'add_linebreak', 'max_length_input']
        assert len([x for x in required_keys if x in kwargs]) == len(required_keys), "missing arguments in preprocessing"
        tokenizer, remove_linebreak, add_linebreak, max_length_input= kwargs['tokenizer'], kwargs['remove_linebreak'], kwargs['add_linebreak'], kwargs['max_length_input']
        
        prompts = self.dataset["test"]['prompt']
        
        if remove_linebreak:
            # remove the last linebreak for starcoder2
            prompts = [x if x[-1]!='\n' else x.rstrip() for x in prompts]
        
        if add_linebreak:
            print("Adding linebreaks to the end of the prompts ..")
            prompts = [x+'\n' for x in prompts]
        
        tokenizer.truncation_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        max_doc_num = max([len(x) for x in self.dataset["test"]["docs"]])
        if self.topk_docs == 0 or max_doc_num == 0:
            start = time.time()
            print(f"Preprocessing infile prompts ..")
            tokenized_prompts = tokenizer(prompts, truncation=True, padding=True, max_length=max_length_input)
            clean_prompts = tokenizer.batch_decode(tokenized_prompts.input_ids, skip_special_tokens=True)
            end = time.time()
            print(f"finished preprocessing with {end-start}s!")
            
            self.dataset["test"] = self.dataset["test"].add_column('processed_prompt', clean_prompts)
        else:
            # raise NotImplementedError("Currently only support generation w/o retrieval") # TBD
            docs_list = self.dataset["test"]["docs"]
            retrieved_prompts = [get_retrieved_prompt(docs[:self.topk_docs]) for docs in docs_list]
            # full_prompts = [r + p for r, p in zip(retrieved_prompts, prompts)]
            
            retrieved_max_length_input = infile_max_length_input = max_length_input // 2 - 2
            
            # retrieved prompts
            start = time.time()
            print(f"Preprocessing retrieved docs ({self.topk_docs} per example) ..")
            tokenizer.truncation_side = 'right'
            tokenized_retrieved_prompts = tokenizer(retrieved_prompts, truncation=True, padding=True, max_length=retrieved_max_length_input)
            clean_retrieved_prompts = tokenizer.batch_decode(tokenized_retrieved_prompts.input_ids, skip_special_tokens=True)
            
            # infile prompts 
            print(f"Preprocessing infile prompts ..")
            tokenizer.truncation_side = 'left'
            tokenized_prompts = tokenizer(prompts, truncation=True, padding=True, max_length=infile_max_length_input)
            clean_prompts = tokenizer.batch_decode(tokenized_prompts.input_ids, skip_special_tokens=True)
            
            full_prompts = [r + '\n\n' + p for r, p in zip(clean_retrieved_prompts, clean_prompts)]
            
            # test 
            print(f"test preprocessing ..")
            tokenzied_full_prompts = tokenizer(full_prompts, truncation=False, padding=True)
            assert len(tokenzied_full_prompts.input_ids[0]) <= max_length_input
            end = time.time()
            
            print(f"finished preprocessing with {end-start}s!")
            
            self.dataset["test"] = self.dataset["test"].add_column('processed_prompt', full_prompts)

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if "processed_prompt" in doc: 
            return doc["processed_prompt"]
        prompt = doc["prompt"]
        retrieved_docs = doc.get("docs", [])
        if len(retrieved_docs) > 0:
            context = get_retrieved_prompt(docs=retrieved_docs[: self.topk_docs])
            prompt = context + '\n\n' + prompt
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        if "reference" in doc:
            return [doc["reference"]]
        else:
            return [doc["metadata"]["ground_truth"]]

    def postprocess_generation(self, generation, idx, new_tokens_only=False):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        if not new_tokens_only:
            prompt = self.get_prompt(self.dataset["test"][idx])
            generation = generation[len(prompt) :]
            return self._stop_at_stop_token(generation, self.stop_words)
        else:
            return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        # extract code blocks 
        CODE_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"
        def extract_code(text: str, pattern: str = CODE_BLOCK_PATTERN):
            match = re.findall(pattern, text, flags=re.DOTALL)
            return match[0][1] if match else text
        
        generations = [[extract_code(x[0])] for x in generations]
        
        EM_scores, ES_scores = [], []
        clean_references, clean_generations = [], []
        for ref, gen in zip(references, generations):
            clean_ref, clean_gen = process_prediction(ref[0], gen[0])
            EM_scores.append(compute_EM(clean_ref, clean_gen))
            ES_scores.append(compute_ES(clean_ref, clean_gen))
            
            clean_references.append(clean_ref)
            clean_generations.append(clean_gen)
        
        import evaluate
        bleu = evaluate.load("bleu")
        bleu_results = bleu.compute(
            references=[[x] for x in clean_references],
            predictions=clean_generations,
        )
        
        results = {
            "bleu_results": bleu_results,
            "EM": np.mean(EM_scores),
            "ES": np.mean(ES_scores)
        }
        
        if self.split == "function" and self.requires_execution:
            # evaluate execution accuracy 
            metadata = self.dataset["test"]['metadata']
            
            setup_success = True
            if self.setup_repoeval:
                print("Running setup for RepoEval-func ..")
                setup_repos(input_dir=self.repoeval_input_repo_dir, output_dir=self.repoeval_cache_dir)
                print("Validating tests for RepoEval-func ..")
                setup_success = check_tests(output_dir=self.repoeval_cache_dir)
            else:
                copy_all_repos(input_dir=self.repoeval_input_repo_dir, output_dir=self.repoeval_cache_dir)
                
            if setup_success:
                print("Running evaluation for RepoEval-func ..")
                assert len(generations) == len(references) == len(metadata)
                
                tmp_output_path = self.metric_output_path + '.intermediate'
                if os.path.exists(tmp_output_path):
                    execution_results = json.load(open(tmp_output_path, 'r'))
                else:
                    execution_results = {}
                
                new_generation_count = 0
                for i, (gen, ref, meta) in enumerate(tqdm(zip(generations, references, metadata), total=len(generations))):
                    gen, ref = gen[0], ref[0]
                    repo = meta["fpath_tuple"][0]
                    task_id = meta["task_id"]
                    
                    if task_id in execution_results and execution_results[task_id] != "timeout":
                        continue
                    
                    return_result = eval_generation(
                        gen, ref, meta, return_output=False, eval_relevant_test_only=True,
                        input_dir=self.repoeval_input_repo_dir, output_dir=self.repoeval_cache_dir,
                    )
                    execution_results[task_id] = return_result
                    new_generation_count += 1
                    
                    if new_generation_count % 5 == 0 and new_generation_count > 0:
                        print(f"Saving intermediate results to {tmp_output_path} ..")
                        json.dump(execution_results, open(tmp_output_path, 'w'), indent=4)
                        
                print(f"Saving intermediate results to {tmp_output_path} ..")
                json.dump(execution_results, open(tmp_output_path, 'w'), indent=4)
                
                results["Pass@1"] = np.mean([1 if x == "success" else 0 for x in execution_results.values()])
                results["Num_computed"] = len(execution_results)
                results["Num_timeout"] = sum([1 if x == "timeout" else 0 for x in execution_results.values()])
        
        return results
