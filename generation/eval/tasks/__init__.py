import inspect
from pprint import pprint

from . import (ds1000, humaneval, live_code_bench, mbpp, odex, repoeval, swe_bench)

TASK_REGISTRY = {
    **ds1000.create_all_tasks(),
    **humaneval.create_all_tasks(),
    "lcb": live_code_bench.LCB,
    "mbpp": mbpp.MBPP,
    **odex.create_all_tasks(),
    **repoeval.create_all_tasks(),
    **swe_bench.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args):
    kwargs = {}
    if args.dataset_path is not None:
        kwargs["dataset_path"] = args.dataset_path
    if args.dataset_name is not None:
        kwargs["dataset_name"] = args.dataset_name
    if args.data_files is not None:
        kwargs["data_files"] = args.data_files
    kwargs["cache_dir"] = args.cache_dir
    if task_name == "repoeval-function":
        kwargs["args"] = args
    kwargs["topk_docs"] = args.topk_docs
    kwargs["tokenizer"] = args.tokenizer if hasattr(args, "tokenizer") else None
    return TASK_REGISTRY[task_name](**kwargs)