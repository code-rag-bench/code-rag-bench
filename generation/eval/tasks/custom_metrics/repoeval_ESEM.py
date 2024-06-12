import editdistance
import json
import numpy as np

def remove_comments(code):
    clean_lines = []
    comment_flag = False
    for line in code.split('\n'):
        if line.lstrip()[:3] in ['"""', "'''"]:
            if not comment_flag:
                comment_flag = True 
        if line.rstrip()[-3:] in ['"""', "'''"]:
            if comment_flag:
                comment_flag = False
        if comment_flag:
            continue
        if not line.lstrip().startswith('#'):
            clean_lines.append(line)
    return '\n'.join(clean_lines)

def process_prediction(target, prediction):
    target = remove_comments(target)
    prediction = remove_comments(prediction)
    target_lines = [line for line in target.split('\n') if line.split() and line.split()[0]!='#']
    target_str = '\n'.join(target_lines)
    prediction_lines = [line for line in prediction.split('\n') if line.split() and line.split()[0]!='#'][:len(target_lines)]
    prediction_str = '\n'.join(prediction_lines)
    return target_str, prediction_str

def compute_EM(target, prediction):
    EM_score = target == prediction 
    return EM_score 

def compute_ES(target, prediction):
    ES_score = 1 - (editdistance.eval(target, prediction) / max(len(target), len(prediction)))
    return ES_score
