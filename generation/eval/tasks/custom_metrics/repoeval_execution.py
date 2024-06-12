import os 
import json
import subprocess

REPOs = [
    "amazon-science_patchcore-inspection",
    "deepmind_tracr",
    "facebookresearch_omnivore",
    "google_lightweight_mmm",
    "lucidrains_imagen-pytorch",
    "maxhumber_redframes",
]

REPOEVAL_ENV_NAME = "repoeval"
CONDA_INIT_COMMAND = f"conda init bash ; source ~/.bashrc ; conda activate {REPOEVAL_ENV_NAME}"

relevant_test_file = "eval/tasks/custom_metrics/repoeval_task_id2tests.json"
task_id2tests = json.load(open(relevant_test_file))


def copy_all_repos(
    input_dir="../retrieval/output/repoeval/repositories/function_level", 
    output_dir = "scripts/repoeval", 
):
    """copy all repos to output_dir"""
    for repo in REPOs:
        repo_path = os.path.join(input_dir, repo)
        output_repo_path = os.path.join(output_dir, repo)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Copying {repo} to {output_dir}")
        subprocess.call(f"cp -r {repo_path} {output_dir}", shell=True)


def setup_repos(
    input_dir="../retrieval/output/repoeval/repositories/function_level", 
    output_dir = "scripts/repoeval", 
):
    """copy all repos to output_dir, run setup.py to install the repo as a package"""
    
    copy_all_repos(input_dir, output_dir)
    
    orig_working_dir = os.getcwd()
    
    for repo in REPOs:
        output_repo_path = os.path.join(output_dir, repo)

        # switch the working dir to the repo 
        os.chdir(output_repo_path)
            
        print(f"Running setup for {repo}")
        command = CONDA_INIT_COMMAND + f" ;pip install -e ."
        subprocess.call(command, shell=True)
            
        # switch back the working dir
        os.chdir(orig_working_dir)
        print(f"Switching back the working dir to", os.getcwd())


def check_tests(
    output_dir = "scripts/repoeval", 
):
    """copy all repos to output_dir, run tests, and return True if all tests pass, False otherwise"""
    orig_working_dir = os.getcwd()
        
    repo2return_code = {}
    for repo in REPOs:
        output_repo_path = os.path.join(output_dir, repo)
        
        os.chdir(output_repo_path)
        
        print(f"Running tests for {repo}")
        command = CONDA_INIT_COMMAND + f" ; pytest"
        return_code = subprocess.call(command, shell=True)
        repo2return_code[repo] = return_code
        print(f"Return code: {return_code}")
        
        os.chdir(orig_working_dir)
        
    failed_repos = [repo for repo, return_code in repo2return_code.items() if return_code != 0]
    if failed_repos:
        print(f"Tests failed for {failed_repos}")
        return False 
    else:
        return True
    
    
def postprocess_by_line(generation, target):
    target_lines = [line for line in target.split('\n') if line.split() and line.split()[0]!='#']
    generation_lines = [line for line in generation.split('\n') if line.split() and line.split()[0]!='#'][:len(target_lines)]
    return generation_lines
    
    
def postprocess_by_function(generation, target):
    first_token = target.split()[0]
    function_indent = target.split(first_token)[0]
    generation_lines = []
    for line in generation.split('\n'):
        if line.split() and line.split()[0]!='#':
            first_token = line.split()[0]
            indent = line.split(first_token)[0]
            if len(indent) < len(function_indent):
                break
            generation_lines.append(line)
    return generation_lines
    
    
def eval_generation(
    generation, target, metadata,
    input_dir="../retrieval/output/repoeval/repositories/function_level", 
    output_dir = "scripts/repoeval",
    return_output = False,
    eval_relevant_test_only = False,
    stop_at_the_first_failed_test = True,
):
    """
    Check whether the generation passes the tests or not. 
    If eval_relevant_test_only is True, only run the tests that are relevant to the generation.
    Return "success" or "failed". If return_output, return the output of the tests.
    """
    
    orig_working_dir = os.getcwd()
    
    local_file_path = '/'.join(metadata["fpath_tuple"])
    input_file_path = os.path.join(input_dir, local_file_path)
    output_file_path = os.path.join(output_dir, local_file_path)
    
    repo = metadata["fpath_tuple"][0]
    input_repo_path = os.path.join(input_dir, repo)
    output_repo_path = os.path.join(output_dir, repo)
    
    task_id = metadata["task_id"]
    metadata["tests"] = task_id2tests.get(task_id, [])
    
    # get start and end line ids
    start_line_id = metadata["line_no"]
    target = target.rstrip() if target[-1] == '\n' else target
    target_line_num = len(target.split('\n'))
    end_line_id = start_line_id + target_line_num
    
    # replace the lines in the repo with the generated code
    file_lines = [line.rstrip() for line in open(input_file_path, 'r')]
    input_file_content = '\n'.join(file_lines)
            
    generation_lines = postprocess_by_function(generation, target)
    file_lines[start_line_id:end_line_id] = generation_lines
    new_file_content = '\n'.join(file_lines)
    
    clean_input_file_content = '\n'.join([line for line in input_file_content.split('\n') if line.split() and line.split()[0]!='#'])
    clean_new_file_content = '\n'.join([line for line in new_file_content.split('\n') if line.split() and line.split()[0]!='#'])
    
    if clean_input_file_content == clean_new_file_content:
        if return_output:
            return "success", ""
        return "success"
    
    with open(output_file_path, 'w') as f:
        f.write('\n'.join(file_lines))
    
    # run pytest, record the return_code
    os.chdir(output_repo_path)
    
    if not eval_relevant_test_only or "tests" not in metadata or len(metadata["tests"]) == 0:
        try:
            command = CONDA_INIT_COMMAND + f" ; pytest"
            if stop_at_the_first_failed_test:
                command = command + " -x"
            ret = subprocess.run(command, shell=True, capture_output=True, timeout=600)
            output = ret.stdout.decode("utf8", errors="replace") if ret.stdout else ""
            return_code = ret.returncode
            if return_code == 5:
                return_code = 0 # no tests collected
        except subprocess.TimeoutExpired:
            print('!'*50, "Timeout", '!'*50)
            output = "timeout"
            return_code = "timeout"
    else:
        output, return_code = "", 0
        for test in metadata["tests"]:
            try:
                command = CONDA_INIT_COMMAND + f" ; pytest {test}"
                if stop_at_the_first_failed_test:
                    command = command + " -x"
                ret = subprocess.run(command, shell=True, capture_output=True, timeout=200)
                out = ret.stdout.decode("utf8", errors="replace") if ret.stdout else ""
                ret_code = ret.returncode
                if ret_code == 5:
                    ret_code = 0 # no tests collected
            except subprocess.TimeoutExpired:
                print('!'*50, "Timeout", '!'*50)
                out = "timeout"
                ret_code = "timeout"
            
            output += out + '\n'
            if ret_code != 0:
                return_code = ret_code
                if not return_output:
                    break
    
    os.chdir(orig_working_dir)
    
    # copy the original file to the output_dir
    subprocess.call(f"cp {input_file_path} {output_file_path}", shell=True)
    
    execution_result = "failed"
    if return_code == 0:
        execution_result = "success"
    elif return_code == 'timeout':
        execution_result = "timeout"
    if return_output:
        return execution_result, output
    return execution_result


def extract_failed_tests(output):
    """Extract the failed tests from the output of pytest (used to extract relevant tests for each example)."""
    lines = output.split('\n')
    for lid, line in enumerate(lines):
        if "=========================== short test summary info ============================" in line:
            break 
        
    failed_tests = []
    for line in lines[lid+1:]:
        if line[:6] != "ERROR " and line[:7] != "FAILED ":
            break 
        filename = line.split(' ')[1].split('::')[0]
        failed_tests.append(filename)
            
    return failed_tests


def get_relevant_tests():
    """Get the relevant tests for each example in the repoeval dataset."""
    from tqdm import tqdm 
    data_file = "../retrieval/results/repoeval-function-gt.jsonl"
    data = [json.loads(line) for line in open(data_file, 'r')]
    output_file = "eval/tasks/custom_metrics/repoeval_task_id2tests.json"
    
    if os.path.exists(output_file):
        task_id2tests = json.load(open(output_file, 'r'))
    else:
        task_id2tests = {}
        
    new_generation_count = 0
    for example_id, example in enumerate(tqdm(data)):
        target = example["reference"]
        metadata = example["metadata"]
        task_id = example['metadata']['task_id']
        
        generation = f"assert 1 == 0" # which will cause error 
        
        if task_id in task_id2tests and len(task_id2tests[task_id]) > 0:
            continue
        
        execution_result, output = eval_generation(
            generation, target, metadata, return_output=True, eval_relevant_test_only=False, stop_at_the_first_failed_test=False,
        )
        task_id2tests[task_id] = extract_failed_tests(output)
        print(task_id, task_id2tests[task_id])
        new_generation_count += 1
        
        if new_generation_count % 5 == 0 and new_generation_count > 0:
            print(f"Saving {new_generation_count} examples to {output_file}..")
            json.dump(task_id2tests, open(output_file, 'w'), indent=4)
    json.dump(task_id2tests, open(output_file, 'w'), indent=4)


def sanity_check():
    """Sanity check: check whether the target passes all the tests."""
    from tqdm import tqdm 
    data_file = "../retrieval/results/repoeval-function-gt.jsonl"
    data = [json.loads(line) for line in open(data_file, 'r')]
    
    failed_count = 0
    for example_id, example in enumerate(tqdm(data)):
        generation = example["reference"]
        target = example["reference"]
        metadata = example["metadata"]
        task_id = example['metadata']['task_id']
        repo = metadata["fpath_tuple"][0]
        
        execution_result = eval_generation(generation, target, metadata)
        if execution_result == "failed":
            print("[test failed]", task_id, metadata["tests"])
            failed_count += 1
    
    return failed_count

if __name__ == "__main__":
    setup_repos()
    ret = check_tests()
    print("Setup for the environment:", ret)

    # sanity_check()
    # get_relevant_tests()
